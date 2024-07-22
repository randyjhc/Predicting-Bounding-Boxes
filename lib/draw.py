import os, re, time, json, zipfile
import numpy as np
import cv2
from matplotlib import pyplot as plt


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5
):
    """
    Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
    """

    image_width = image.shape[1]
    image_height = image.shape[0]
    cv2.rectangle(
        image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness
    )


def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=5):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(
            image,
            boxes[i, 1],
            boxes[i, 0],
            boxes[i, 3],
            boxes[i, 2],
            color[i],
            thickness,
        )


def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=5):
    """
    Draws bounding boxes on image (numpy array).

    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: a list of strings for each bounding box.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """

    draw_bounding_boxes_on_image(image, boxes, color, thickness)

    return image


def config_plt():
    """
    Matplotlib config
    """
    plt.rc("image", cmap="gray")
    plt.rc("grid", linewidth=0)
    plt.rc("xtick", top=False, bottom=False, labelsize="large")
    plt.rc("ytick", left=False, right=False, labelsize="large")
    plt.rc("axes", facecolor="F8F8F8", titlesize="large", edgecolor="white")
    plt.rc("text", color="a8151a")
    plt.rc("figure", facecolor="F0F0F0")  # Matplotlib fonts
    MATPLOTLIB_FONT_DIR = os.path.join(
        os.path.dirname(plt.__file__), "mpl-data/fonts/ttf"
    )

'''
# utility to display a row of digits with their predictions
def display_digits_with_boxes(
    images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False
):

    n = len(images)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if len(pred_bboxes) > i:
            bbox = pred_bboxes[i]
            bbox = [
                bbox[0] * images[i].shape[1],
                bbox[1] * images[i].shape[0],
                bbox[2] * images[i].shape[1],
                bbox[3] * images[i].shape[0],
            ]
            bboxes_to_plot.append(bbox)

        if len(bboxes) > i:
            bbox = bboxes[i]
            if bboxes_normalized == True:
                bbox = [
                    bbox[0] * images[i].shape[1],
                    bbox[1] * images[i].shape[0],
                    bbox[2] * images[i].shape[1],
                    bbox[3] * images[i].shape[0],
                ]
            bboxes_to_plot.append(bbox)

        img_to_draw = draw_bounding_boxes_on_image_array(
            image=images[i],
            boxes=np.asarray(bboxes_to_plot),
            color=[(255, 0, 0), (0, 255, 0)],
        )
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if iou[i][0] < iou_threshold:
                color = "red"
            ax.text(
                0.2,
                -0.3,
                "iou: %s" % (np.round(iou[i][0], 2)),
                color=color,
                transform=ax.transAxes,
            )


# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(
        history.history[metric_name],
        color="green",
        label="train_" + metric_name,
        linestyle="dashed",
    )
    plt.plot(
        history.history["val_" + metric_name], color="green", label="val_" + metric_name
    )
    plt.legend()
'''
    