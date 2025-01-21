import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import DataParallel

from models.dave import build_model
from utils.arg_parser import get_argparser
from utils.data import resize

bounding_boxes = []
fig = None
ax = None
ix, iy = 0, 0


def on_click(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    fig.canvas.mpl_connect("button_release_event", on_release)


def on_release(event):
    global ix, iy
    x, y = event.xdata, event.ydata
    width, height = x - ix, y - iy
    rect = patches.Rectangle((ix, iy), width, height, edgecolor="r", facecolor="none")
    ax.add_patch(rect)
    bounding_boxes.append((ix, iy, ix + width, iy + height))
    plt.draw()


@torch.no_grad()
def demo(args):
    inf_img_path = "material/4896_15232.jpg"
    ref_img_path = "material/4896_17408.jpg"
    global fig, ax

    # Prepare device and model
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, "DAVE_3_shot.pth"))["model"],
        strict=False,
    )
    pretrained_dict_feat = {
        k.split("feat_comp.")[1]: v
        for k, v in torch.load(os.path.join(args.model_path, "verification.pth"))[
            "model"
        ].items()
        if "feat_comp" in k
    }
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    model.eval()

    # Open images
    inf_image = Image.open(inf_img_path).convert("RGB")
    ref_image = Image.open(ref_img_path).convert("RGB")

    # Only the left half of the reference
    w, h = ref_image.width, ref_image.height
    left_ref = ref_image.crop((0, 0, w // 2, h))

    # 1) Collect bounding boxes from the left_ref
    global bounding_boxes
    bounding_boxes = []
    fig, ax = plt.subplots(1)
    ax.imshow(left_ref)
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.title("Draw boxes on LEFT half of reference, then close window")
    plt.show()
    drawn_bboxes = torch.tensor(bounding_boxes)

    def overlay_and_infer(base_image, overlay_image, offset_x, keep_side):
        """Overlay half-ref on base_image at offset_x, run inference, keep boxes from keep_side."""
        # Make overlay copy
        base_copy = base_image.copy()
        base_copy.paste(overlay_image, (offset_x, 0))

        # Shift drawn boxes by offset_x
        boxes = drawn_bboxes.clone()
        boxes[:, [0, 2]] += offset_x

        # Resize and run inference
        img_tensor, boxes_tensor, scale = resize(base_copy, boxes)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        boxes_tensor = boxes_tensor.unsqueeze(0).to(device)

        den_map, _, tblr, pred = model(img_tensor, bboxes=boxes_tensor)
        scaled_boxes = pred.box.cpu() / torch.tensor(
            [scale[0], scale[1], scale[0], scale[1]]
        )

        # Keep boxes only from specified side
        final = []
        mid_x = base_copy.width / 2
        for b in scaled_boxes:
            cx = (b[0] + b[2]) / 2
            if keep_side == "left" and cx < mid_x:
                final.append(b)
            elif keep_side == "right" and cx > mid_x:
                final.append(b)
        return den_map, final

    # 2) Overlay on the left side of the inference image, keep boxes from the right
    den_left, right_side_preds = overlay_and_infer(
        inf_image, left_ref, offset_x=0, keep_side="right"
    )

    # 3) Overlay again on the right side of the inference image, keep boxes from the left
    offset_right = inf_image.width - left_ref.width
    den_right, left_side_preds = overlay_and_infer(
        inf_image, left_ref, offset_x=offset_right, keep_side="left"
    )

    # Combine final boxes
    final_boxes = right_side_preds + left_side_preds

    # Display final results
    plt.figure()
    plt.imshow(inf_image)
    for b in final_boxes:
        plt.plot(
            [b[0], b[0], b[2], b[2], b[0]],
            [b[1], b[3], b[3], b[1], b[1]],
            linewidth=2,
            color="red",
        )
    total_count = round(den_left.sum().item() + den_right.sum().item(), 1)
    plt.title(
        f"Combined box count: {len(final_boxes)} | Combined dmap count: {total_count}"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DAVE", parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
