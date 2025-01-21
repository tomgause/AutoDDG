import torch
from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import IDataset
from torchvision import ops
from tqdm import tqdm

from ..GeCo.utils.data import resize_and_pad
from .BaseFewShotEvaluator import BaseFewShotEvaluator


class GeCoFewShotEvaluator(BaseFewShotEvaluator):
    """
    A specialized evaluator for GeCo in a few-shot scenario.
    Now, all images are in the same subset (e.g. "train"), but some have
    BBox annotations (exemplars) and some have Points annotations (test items).
    """

    def __init__(self, dataset: IDataset, model, device="cpu"):
        super().__init__(dataset, model, device=device)
        self.final_exemplars = None

    def prepare_exemplars(self):
        """
        Gathers bounding boxes from all items (in a single subset) to produce a
        single bank of exemplar embeddings for few-shot inference.
        """
        self.logger.info("Preparing exemplars from bounding-box-labeled items...")

        # Collect all items that have at least one BBox annotation
        bbox_items = []
        for item in self.dataset:
            bboxes = [
                ann for ann in item.annotations if ann.type == AnnotationType.bbox
            ]
            if bboxes:
                bbox_items.append(item)

        all_exemplar_embs = []
        for item in bbox_items:
            img_torch = self.load_image_as_tensor(item)
            _, H, W = img_torch.shape

            # Convert bounding boxes to tensor
            bboxes_torch = self.convert_boxes_to_tensor(item.annotations, (H, W))

            img, bboxes, scale = resize_and_pad(
                img_torch, bboxes_torch, full_stretch=False
            )
            img = img.unsqueeze(0).to(self.device)
            bboxes = bboxes.unsqueeze(0).to(self.device)

            with torch.no_grad():
                ex_embs = self.model.extract_exemplar_embeddings(img, bboxes)
                all_exemplar_embs.extend(ex_embs.cpu())

        if not all_exemplar_embs:
            self.logger.error("No bounding boxes found; exemplars will be empty!")
            # Create a dummy shape so that inference won't fail
            self.final_exemplars = torch.zeros(
                (1, 0, getattr(self.model, "emb_dim", 256))
            )
            return

        self.final_exemplars = torch.cat(all_exemplar_embs, dim=1)
        self.logger.info(f"Final exemplar shape: {self.final_exemplars.shape}")

    def evaluate(self):
        """
        Perform the full evaluation:
          1) Build or load exemplar embeddings (from BBox-labeled items).
          2) Run inference on point-labeled images.
          3) Compute and log a simple loss metric.
        """
        self.logger.info("Starting GeCoFewShot evaluation.")

        # 1) Prepare exemplar embeddings
        self.prepare_exemplars()
        if self.final_exemplars is None:
            self.logger.warning("No exemplars found; evaluation aborted.")
            return

        # 2) Filter all items that have Points annotations
        self.logger.info("Running inference on point-labeled items...")
        point_items = []
        for item in self.dataset:
            points = [
                ann for ann in item.annotations if ann.type == AnnotationType.points
            ]
            if points:
                point_items.append(item)

        predictions_list = []
        gt_points_list = []

        for item in tqdm(point_items[0:2]):
            img_torch = self.load_image_as_tensor(item)
            img, bboxes, scale = resize_and_pad(
                img_torch,
                torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                full_stretch=False,
            )
            img = img.unsqueeze(0).to(self.device)

            # Ground-truth points
            points_ann = [
                ann for ann in item.annotations if ann.type == AnnotationType.points
            ]
            gt_points_list.append(points_ann)

            with torch.no_grad():
                outputs, _, _, _, _ = self.model(
                    img=img,
                    bboxes=None,
                    exemplars=self.final_exemplars.to(self.device),
                )

            del _
            idx = 0
            thr = 4
            keep = ops.nms(
                outputs[idx]["pred_boxes"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
                ],
                outputs[idx]["box_v"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
                ],
                0.5,
            )
            boxes = (
                outputs[idx]["pred_boxes"][
                    outputs[idx]["box_v"] > outputs[idx]["box_v"].max() / thr
                ]
            )[keep]
            bboxes = torch.clamp(boxes, 0, 1)
            predictions_list.append(bboxes)

        # 3) Compute and report a simple loss
        self.logger.debug(gt_points_list)
        self.logger.debug(predictions_list)
        loss_value = self.compute_loss(predictions_list, gt_points_list)
        self.logger.info(f"Evaluation complete. Loss: {loss_value:.4f}")
        self.visualize_examples(predictions_list, point_items, num_examples=10)
        return loss_value
