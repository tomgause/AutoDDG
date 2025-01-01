import os
import random

import matplotlib.pyplot as plt
import torch
from datumaro.components.dataset import AnnotationType, IDataset
from PIL import Image

from ..DAVE.utils.data import resize
from ..utils.draw import draw_bounding_boxes_ui
from .BaseFewShotEvaluator import BaseFewShotEvaluator


class DAVEFewShotEvaluator(BaseFewShotEvaluator):
    """
    An evaluator for the DAVE model that uses a 'reference image' strategy:
      - Load (or let user define) bounding boxes from a reference image
      - Overlay these reference boxes onto the new image (on left & right)
      - Combine predicted boxes from the model, compute DMAP, etc.
      - Compute a simple box-vs-points loss for each item and overall.
    """

    def __init__(
        self,
        dataset: IDataset,
        model,
        device="cpu",
        reference_image_path=None,
        reference_crop_fn=None,
        save_visualizations=False,
        logger=None,
    ):
        super().__init__(dataset, model, device=device, logger=logger)

        self.model = model
        self.save_visualizations = save_visualizations
        self.reference_image_path = reference_image_path
        self.reference_crop_fn = reference_crop_fn or self.default_reference_crop_fn

        # Attempt to load the reference image
        self.reference_image = None
        if reference_image_path:
            self.reference_image = Image.open(reference_image_path).convert("RGB")
            self.logger.debug(f"Loaded reference image from: {reference_image_path}")
        else:
            self.logger.warning(
                f"No valid reference_image_path provided: {reference_image_path}"
            )

        self.reference_image_full = Image.open(reference_image_path).convert("RGB")
        w, h = self.reference_image_full.size
        self.left_half = self.reference_image_full.crop((0, 0, w // 2, h))

        # Attempt to find bounding boxes for reference
        self.ref_bounding_boxes = torch.tensor(
            [
                (
                    29.528638028638056,
                    52.69480519480521,
                    51.693140193140266,
                    85.08907758907753,
                ),
                (
                    32.086080586080584,
                    119.18831168831167,
                    54.250582750582794,
                    141.35281385281382,
                ),
                (
                    11.626540126540135,
                    172.0421245421245,
                    40.61088911088916,
                    192.50166500166495,
                ),
                (
                    36.34848484848487,
                    276.8972693972694,
                    65.3328338328339,
                    291.38944388944384,
                ),
                (
                    52.54562104562103,
                    366.4077589077589,
                    79.8250083250083,
                    386.01481851481844,
                ),
                (
                    76.41508491508489,
                    411.58924408924406,
                    94.31718281718281,
                    436.3111888111888,
                ),
                (
                    17.59390609390607,
                    451.65584415584414,
                    45.72577422577422,
                    471.26290376290376,
                ),
                (
                    157.40076590076592,
                    461.88561438561436,
                    188.0900765900766,
                    480.6401931401931,
                ),
                (
                    132.67882117882118,
                    355.32550782550777,
                    157.40076590076592,
                    380.8999333999334,
                ),
                (
                    210.2545787545788,
                    279.4547119547119,
                    239.23892773892783,
                    302.47169497169494,
                ),
                (
                    164.22061272061273,
                    212.96120546120545,
                    196.61488511488517,
                    233.4207459207459,
                ),
                (
                    183.82767232767242,
                    133.68048618048613,
                    205.9921744921745,
                    164.36979686979686,
                ),
                (
                    148.02347652347657,
                    62.07209457209456,
                    180.417748917749,
                    90.20396270396265,
                ),
                (
                    136.0887445887446,
                    262.40509490509487,
                    163.36813186813185,
                    287.97952047952043,
                ),
            ]
        )
        if not self.ref_bounding_boxes:
            self.ref_bounding_boxes = self._check_or_draw_ref_boxes(self.left_half)

    def _check_or_draw_ref_boxes(self, ref_img):
        self.logger.info("No bounding boxes found for reference; launching UI.")
        bbs = draw_bounding_boxes_ui(ref_img, self.logger)
        if not bbs:
            self.logger.warning("No bounding boxes drawn by user!")
            return torch.zeros((0, 4))
        return torch.tensor(bbs, dtype=torch.float32)

    @staticmethod
    def default_reference_crop_fn(ref_image):
        w, h = ref_image.width, ref_image.height
        left_ref = ref_image.crop((0, 0, w // 2, h))
        offset_right = w - (w // 2)
        return left_ref, offset_right

    def overlay_and_infer(
        self, base_img, overlay_img, offset_x, drawn_bboxes, keep_side
    ):
        base_copy = base_img.copy()
        base_copy.paste(overlay_img, (offset_x, 0))

        # Shift drawn reference boxes
        boxes = drawn_bboxes.clone()
        boxes[:, [0, 2]] += offset_x

        # Resize to model input
        img_tensor, boxes_tensor, scale = resize(base_copy, boxes)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        boxes_tensor = boxes_tensor.unsqueeze(0).to(self.device)

        den_map, _, tblr, pred = self.model(img_tensor, bboxes=boxes_tensor)
        scaled_boxes = pred.box.cpu() / torch.tensor(
            [scale[0], scale[1], scale[0], scale[1]]
        )

        # Filter predicted boxes by 'left' or 'right' half
        keep = []
        mid_x = base_copy.width / 2
        for b in scaled_boxes:
            cx = (b[0] + b[2]) / 2
            if keep_side == "left" and cx < mid_x:
                keep.append(b)
            elif keep_side == "right" and cx > mid_x:
                keep.append(b)

        return den_map, keep

    def evaluate(self):
        if not self.reference_image:
            self.logger.error("Cannot evaluate without a reference image. Exiting.")
            return

        left_ref, offset_right = self.reference_crop_fn(self.reference_image)

        all_results = []
        losses = []  # keep track of per-item loss

        from itertools import islice

        # Set the random seed for reproducibility
        random.seed(42)

        # Generate random indices to sample
        dataset_length = len(self.dataset)
        sample_indices = set(random.sample(range(dataset_length), dataset_length))
        total_gt_points = 0
        total_predicted_boxes = 0

        # Iterate over the dataset and evaluate only sampled indices
        for idx, item in enumerate(islice(self.dataset, dataset_length)):
            if idx in sample_indices:
                self.logger.info(
                    f"Evaluating item {item.id} ({len(sample_indices)}/{dataset_length})"
                )

                # Convert item to PIL
                img_path = item.media.path
                inf_image = Image.open(img_path).convert("RGB")

                # Overlay left -> keep right
                den_left, right_preds = self.overlay_and_infer(
                    inf_image,
                    left_ref,
                    offset_x=0,
                    drawn_bboxes=self.ref_bounding_boxes,
                    keep_side="right",
                )

                # Overlay right -> keep left
                den_right, left_preds = self.overlay_and_infer(
                    inf_image,
                    left_ref,
                    offset_x=offset_right,
                    drawn_bboxes=self.ref_bounding_boxes,
                    keep_side="left",
                )

                final_boxes = right_preds + left_preds
                total_dmap_count = round(
                    den_left.sum().item() + den_right.sum().item(), 2
                )

                # Count ground-truth points
                gt_point_count = sum(
                    1 for ann in item.annotations if ann.type == AnnotationType.points
                )

                # Simple loss: absolute difference
                item_loss = abs(len(final_boxes) - gt_point_count)
                losses.append(item_loss)

                self.logger.debug(f"Item {item.id}: {len(final_boxes)} predicted boxes")

                all_results.append(
                    {
                        "item_id": item.id,
                        "boxes": final_boxes,
                        "dmap_count": total_dmap_count,
                        "gt_point_count": gt_point_count,
                        "loss": item_loss,
                    }
                )

                # Optionally visualize
                if self.save_visualizations:
                    vis_path = f"visualizations/{item.id}.png"
                    self._save_result_visual(
                        inf_image, final_boxes, gt_point_count, vis_path
                    )
                total_gt_points += gt_point_count
                total_predicted_boxes += len(final_boxes)
                sample_indices.remove(idx)  # Remove to speed up checks
                if not sample_indices:  # Stop early if all indices are processed
                    break

        # Filter out None values from losses
        filtered_losses = [loss for loss in losses if loss is not None]

        # Calculate overall loss
        overall_loss = (
            sum(filtered_losses) / len(filtered_losses) if filtered_losses else None
        )
        self.logger.info(f"Evaluation complete. Mean box-point loss: {overall_loss}")
        self.logger.info(
            f"Total GT point count: {total_gt_points}\n"
            f"Total predicted boxes: {total_predicted_boxes}"
        )

        return all_results, overall_loss

    def _save_result_visual(self, pil_img, boxes, gt_point_count, save_path):
        fig, ax = plt.subplots()
        ax.imshow(pil_img)
        for b in boxes:
            x1, y1, x2, y2 = b
            ax.plot(
                [x1, x1, x2, x2, x1],
                [y1, y2, y2, y1, y1],
                linewidth=2,
                color="red",
            )
        ax.set_title(f"Combined boxes: {len(boxes)} | GT: {gt_point_count}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
