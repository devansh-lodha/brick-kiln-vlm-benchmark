# src/utils/visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import List, Dict

def plot_bounding_boxes(
    image: Image.Image,
    predictions: List[Dict] = None,
    ground_truths: List[Dict] = None,
    title: str = "Detections",
    pred_color: str = 'red',
    gt_color: str = 'lime'
) -> None:
    """
    Plots bounding boxes for predictions and ground truths on an image.

    Args:
        image (Image.Image): The PIL image to draw on.
        predictions (List[Dict], optional): A list of prediction dicts.
            Each dict should have a 'box' key with [xmin, ymin, xmax, ymax]
            and optionally a 'label' key.
        ground_truths (List[Dict], optional): A list of ground truth dicts.
            Each dict should have a 'box' key with [xmin, ymin, xmax, ymax].
        title (str, optional): The title for the plot.
        pred_color (str, optional): Color for prediction boxes.
        gt_color (str, optional): Color for ground truth boxes.
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    ax.set_title(title, fontsize=16)
    ax.axis('off')

    # Plot ground truth boxes
    if ground_truths:
        for gt in ground_truths:
            box = gt['box']
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor=gt_color, facecolor='none', label='Ground Truth'
            )
            ax.add_patch(rect)

    # Plot prediction boxes
    if predictions:
        for pred in predictions:
            box = pred['box']
            label = pred.get('label', '')
            conf = pred.get('confidence', None)
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor=pred_color, facecolor='none', label='Prediction'
            )
            ax.add_patch(rect)
            if label:
                display_text = f"{label}"
                if conf:
                    display_text += f": {conf:.2f}"
                ax.text(xmin, ymin - 10, display_text, color='white',
                        bbox=dict(facecolor=pred_color, alpha=0.6, pad=1))

    # Create a single legend
    handles = []
    if ground_truths:
        handles.append(patches.Patch(color=gt_color, label='Ground Truth'))
    if predictions:
        handles.append(patches.Patch(color=pred_color, label='Prediction'))

    if handles:
        plt.legend(handles=handles)

    plt.show()