import os

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import IDataset
from PIL import Image
from torchvision import transforms as T

from ..utils.logging import setup_logging


# --------------------------------------------------------------------
# BaseFewShotEvaluator
# --------------------------------------------------------------------
class BaseFewShotEvaluator:
    """
    A base evaluator for few-shot tasks. Provides common methods:
      - Loading images from Datumaro DatasetItems
      - Basic image pre-processing
      - Generic evaluation loop structure
    """

    def __init__(self, dataset: IDataset, model, device="cpu", logger=None):
        """
        Args:
            dataset (IDataset): A Datumaro dataset containing items for evaluation.
            model: The model to evaluate.
            device (str): The device for PyTorch operations ('cpu' or 'cuda').
            logger (logging.Logger): Optional logger to use (otherwise a new one is created).
        """

        self.dataset = dataset
        self.model = model
        self.device = device
        if logger is None:
            self.logger = setup_logging(__name__)
        else:
            self.logger = logger

    def load_image_as_tensor(self, item) -> torch.Tensor:
        """
        Converts a Datumaro DatasetItem's image into a (C, H, W) PyTorch tensor.
        """
        assert item.media is not None, f"No media found for item ID={item.id}."
        img_path = item.media.path  # This is where Datumaro stores the path
        self.logger.debug(f"Loading image from {img_path} for item ID={item.id}")
        img_torch = T.ToTensor()(Image.open(img_path).convert("RGB"))
        return img_torch

    def convert_boxes_to_tensor(self, annotations, image_shape) -> torch.Tensor:
        """
        Convert bounding box annotations into a (B, N, 4) tensor normalized to [0,1].
        Each box is in the format (xmin, ymin, xmax, ymax).
        """
        bounding_boxes = []
        height, width = image_shape[:2]

        for ann in annotations:
            if ann.type == AnnotationType.bbox:
                x1 = ann.x / width
                y1 = ann.y / height
                x2 = (ann.x + ann.w) / width
                y2 = (ann.y + ann.h) / height
                bounding_boxes.append([x1, y1, x2, y2])

        if not bounding_boxes:
            return torch.zeros((1, 0, 4), dtype=torch.float32)

        bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        return bboxes

    def compute_loss(self, predictions, ground_truth):
        """
        A placeholder loss function. In a real scenario, you might compute
        bounding box regression loss, classification loss, etc.
        """
        pred_counts = [len(pred_dict) for pred_dict in predictions]
        gt_counts = [len(g) for g in ground_truth]  # each item is a list of points
        diffs = [abs(pc - gc) for pc, gc in zip(pred_counts, gt_counts)]
        return sum(diffs) / len(diffs) if diffs else 0.0

    def evaluate(self):
        """
        Skeleton for the evaluation logic. Subclasses must implement their own approach
        to data splitting, reference usage, etc.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    def visualize_examples(
        self, predictions_list, items, output_dir="visualizations", num_examples=20
    ):
        """
        Save a few examples of predictions vs. ground-truth to disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, (bboxes, item) in enumerate(zip(predictions_list, items)):
            if idx >= num_examples:
                break

            # For demonstration, we assume item.media.data is a (H,W,C) numpy array
            img = item.media.data
            H, W, _ = img.shape

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)

            # If you used normalized [0..1] boxes, multiply by the width/height
            bboxes = bboxes.cpu()
            for box in bboxes:
                x1, y1 = box[0] * W, box[1] * H
                x2, y2 = box[2] * W, box[3] * H
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.5,
                    edgecolor="g",
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Show ground-truth points
            for ann in item.annotations:
                if ann.type == AnnotationType.points:
                    x_coords = ann.points[0::2]
                    y_coords = ann.points[1::2]
                    ax.scatter(x_coords, y_coords, c="red", marker="x", s=80)

            plt.axis("off")

            save_name = f"{item.id}.png"
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
