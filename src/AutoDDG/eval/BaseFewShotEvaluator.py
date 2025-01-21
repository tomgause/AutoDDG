import json
import os
from typing import Any, Dict

import matplotlib
from datumaro import Dataset

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import torch
from datumaro.components.annotation import AnnotationType
from PIL import Image
from tqdm import tqdm

from ..utils.draw import draw_bounding_boxes_ui
from ..utils.logging import setup_logger


class BaseFewShotEvaluator:
    """
    A base evaluator for few-shot counting tasks. Handles:
      - Reference image loading and processing
      - Overlaying bounding boxes onto the input images
      - Dataset iteration, metrics, and visualization
      - Delegates only inference logic to subclasses via infer()
    """

    def __init__(self, config: Dict[str, Any], job):
        self._config = config
        self._job = job
        self.logger = setup_logger(self.__class__.__name__, self.job.id)
        self.dataset = self._build_datumaro_dataset()
        self.dataset = list(self.dataset)[100:110]

        self.device = self._setup_device()
        self.model = self._build_model()

        # Reference-related settings
        self.reference_crop_fn = self.default_reference_crop_fn
        self.ref_bounding_boxes = self.config.ref_bounding_boxes
        self.reference_image = None
        self.reference_image_full = None
        self.left_half = None

        self._init_reference()

    @property
    def config(self):
        return self._config

    @property
    def job(self):
        return self._job

    def _setup_device(self):
        if self.config.device:
            if self.config.device == "cuda":
                self.device = 0
        else:
            self.device = "cpu"
        torch.cuda.set_device(self.device)
        self.device = torch.device(self.device)

    def _build_datumaro_dataset(self):
        return Dataset.import_from(
            self.config.datumaro_path,
            format=self.config.datumaro_format,
        )

    def _init_reference(self):
        """Load and process the reference image and bounding boxes."""
        if not self.config.ref_image_path or not os.path.exists(
            self.config.ref_image_path
        ):
            self.logger.warning(
                f"No valid reference image path: {self.config.ref_image_path}"
            )
            return

        self.reference_image_full = Image.open(
            os.path.join(self.config.ref_image_path)
        ).convert("RGB")
        self.reference_image = self.reference_image_full
        w, h = self.reference_image_full.size
        self.left_half = self.reference_image_full.crop((0, 0, w // 2, h))
        self.logger.debug(f"Loaded reference image from: {self.config.ref_image_path}")

        if not self.config.ref_bounding_boxes:
            self.ref_bounding_boxes = self._check_or_draw_ref_boxes(self.left_half)
        else:
            self.ref_bounding_boxes = torch.tensor(self.config.ref_bounding_boxes)

    def _check_or_draw_ref_boxes(self, ref_img):
        """If no bounding boxes are provided, allow the user to draw them."""
        self.logger.info("No bounding boxes found for reference; launching UI.")
        bbs = draw_bounding_boxes_ui(ref_img, self.logger)
        if not bbs:
            self.logger.warning("No bounding boxes drawn by user!")
            return torch.zeros((0, 4))
        return torch.tensor(bbs, dtype=torch.float32)

    @staticmethod
    def default_reference_crop_fn(ref_image):
        """Split the reference image into left half and compute offset for right half."""
        w, h = ref_image.width, ref_image.height
        left_ref = ref_image.crop((0, 0, w // 2, h))
        offset_right = w - (w // 2)
        return left_ref, offset_right

    def overlay_reference(self, base_img, overlay_img, offset_x, drawn_bboxes):
        """Overlay reference bounding boxes and shift them for the base image."""
        base_copy = base_img.copy()
        base_copy.paste(overlay_img, (offset_x, 0))

        # Shift bounding boxes
        boxes = drawn_bboxes.clone()
        boxes[:, [0, 2]] += offset_x
        return base_copy, boxes

    @torch.no_grad()
    def evaluate(self):
        """Iterate over dataset and evaluate using model-specific inference."""
        if self.reference_image is None:
            self.logger.error("Cannot evaluate without a reference image.")
            return [], None

        left_ref, offset_right = self.reference_crop_fn(self.reference_image)

        all_results = []
        losses = []
        total_items = len(self.dataset)

        # Iterate over the dataset as a generator using tqdm for a progress bar
        for item in tqdm(self.dataset, total=total_items, desc="Evaluating"):
            img_path = item.media.path
            inf_image = Image.open(img_path).convert("RGB")

            # Overlay reference on the left and infer the right
            right_img, right_boxes = self.overlay_reference(
                inf_image, left_ref, offset_x=0, drawn_bboxes=self.ref_bounding_boxes
            )
            right_preds, den_left = self._infer(right_img, right_boxes, "right")

            # Overlay reference on the right and infer the left
            left_img, left_boxes = self.overlay_reference(
                inf_image,
                left_ref,
                offset_x=offset_right,
                drawn_bboxes=self.ref_bounding_boxes,
            )
            left_preds, den_right = self._infer(left_img, left_boxes, "left")

            # Combine predictions and compute loss
            final_boxes = right_preds + left_preds
            total_dmap_count = round(den_left + den_right, 2)

            gt_point_count = sum(
                1 for ann in item.annotations if ann.type == AnnotationType.points
            )
            item_loss = abs(len(final_boxes) - gt_point_count)
            losses.append(item_loss)

            result = {
                "item_id": item.id,
                "boxes": final_boxes,
                "dmap_count": total_dmap_count,
                "gt_point_count": gt_point_count,
                "loss": item_loss,
            }
            all_results.append(result)

            if self.config.save_annotated_images:
                vis_path = os.path.join(
                    self.job.output_dir, f"visualizations/{item.id}"
                )
                self._save_result_visual(
                    inf_image, final_boxes, gt_point_count, vis_path
                )

        return self._save_evaluation_results(losses=losses, all_results=all_results)

    def _save_evaluation_results(self, losses, all_results):
        overall_loss = sum(losses) / len(losses) if losses else None
        self.logger.info(f"Evaluation complete. MAE: {overall_loss}")

        # Ensure the output directory exists
        os.makedirs(self.job.output_dir, exist_ok=True)

        # Define file paths
        results_path = os.path.join(self.job.output_dir, "all_results.json")
        loss_path = os.path.join(self.job.output_dir, "overall_loss.json")

        # Save results to JSON files
        with open(results_path, "w") as results_file:
            json.dump(all_results, results_file, indent=4)

        with open(loss_path, "w") as loss_file:
            json.dump({"MAE": overall_loss}, loss_file, indent=4)

        return all_results, overall_loss

    def _infer(self, img, boxes, keep_side):
        """
        Subclasses must implement inference logic.
        Args:
            img (PIL.Image): The input image after overlaying the reference.
            boxes (torch.Tensor): Shifted reference bounding boxes.
            keep_side (str): Side ('left' or 'right') for predictions.
        Returns:
            predicted_boxes (list of lists): Predictions in (x1, y1, x2, y2).
            density_map_sum (float): Total density map sum (optional).
        """
        raise NotImplementedError("Subclasses must implement _infer().")

    def _save_result_visual(self, pil_img, boxes, gt_point_count, save_path):
        """Visualize results."""
        fig, ax = plt.subplots()
        ax.imshow(pil_img)
        for b in boxes:
            x1, y1, x2, y2 = b
            ax.plot(
                [x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=2, color="red"
            )
        ax.set_title(f"Boxes: {len(boxes)} | GT: {gt_point_count}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)

    def _build_model(config):
        raise NotImplementedError("Subclasses must implement _build_model().")
