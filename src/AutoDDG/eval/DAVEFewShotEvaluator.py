import os
from abc import abstractmethod
from typing import Any, Dict

import torch
from torch.nn import DataParallel

from AutoDDG.DAVE.models.dave import build_model as build_model

from ..DAVE.utils.data import resize
from .BaseFewShotEvaluator import BaseFewShotEvaluator


class DAVEFewShotEvaluator(BaseFewShotEvaluator):
    """Specialized evaluator for the DAVE model."""

    def __init__(
        self,
        config: Dict[str, Any],
        job,
    ):
        super().__init__(config=config, job=job)

    def _infer(self, img, boxes, keep_side):
        """Perform inference for the DAVE model."""
        # Resize input image and boxes
        img_tensor, boxes_tensor, scale = resize(img, boxes)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        boxes_tensor = boxes_tensor.unsqueeze(0).to(self.device)

        # Perform inference
        den_map, _, _, pred = self.model(img_tensor, bboxes=boxes_tensor)

        # Rescale predictions back to original
        scaled_boxes = pred.box.cpu() / torch.tensor(
            [scale[0], scale[1], scale[0], scale[1]]
        )

        # Filter predictions by side
        mid_x = img.width / 2
        filtered_boxes = []
        for b in scaled_boxes:
            cx = (b[0] + b[2]) / 2
            if (keep_side == "left" and cx < mid_x) or (
                keep_side == "right" and cx > mid_x
            ):
                filtered_boxes.append(b.tolist())

        density_map_sum = den_map.sum().item()
        return filtered_boxes, density_map_sum

    @abstractmethod
    def _build_model(self):
        model = DataParallel(
            build_model(self.config).to(self.device),
            device_ids=[self.device],
            output_device=self.device,
        )
        model.load_state_dict(
            torch.load(os.path.join(self.config.model_path, "DAVE_3_shot.pth"))[
                "model"
            ],
            strict=False,
        )
        pretrained_dict_feat = {
            k.split("feat_comp.")[1]: v
            for k, v in torch.load(
                os.path.join(self.config.model_path, "verification.pth")
            )["model"].items()
            if "feat_comp" in k
        }
        model.module.feat_comp.load_state_dict(pretrained_dict_feat)
        model.eval()

        return model
