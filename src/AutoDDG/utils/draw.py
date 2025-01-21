import matplotlib.patches as patches
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Helper UI for bounding box drawing (if needed)
# --------------------------------------------------------------------
def draw_bounding_boxes_ui(ref_image, logger):
    """
    If the reference image doesn't have bounding boxes, opens a matplotlib
    UI to let the user draw them. Returns a list of (xmin, ymin, xmax, ymax).
    """
    bounding_boxes = []
    fig, ax = plt.subplots()
    ax.imshow(ref_image)
    plt.title("Draw boxes, then close window")

    def on_click(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        fig.canvas.mpl_connect("button_release_event", on_release)

    def on_release(event):
        global ix, iy
        x, y = event.xdata, event.ydata
        width, height = x - ix, y - iy
        rect = patches.Rectangle(
            (ix, iy), width, height, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        bounding_boxes.append((ix, iy, ix + width, iy + height))
        plt.draw()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    logger.debug(f"User-drawn bounding boxes: {bounding_boxes}")
    return bounding_boxes
