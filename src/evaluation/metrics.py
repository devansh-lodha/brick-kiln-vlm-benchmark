# src/evaluation/metrics.py

import numpy as np
import pandas as pd
import re
import json
from typing import List, Dict, Tuple

def parse_florence2_output(raw_text: str, class_names: List[str]) -> List[Dict]:
    """Parses the <OD> output from Florence-2 into a list of detections."""
    detections = []
    # Pattern: <OD> class_name<loc_xxxx><loc_xxxx>...
    pattern = r"<OD>(.*?)(<loc_\d{4}>){4}"
    # This is a simplified parser. The actual thesis might have used a more robust one.
    # For now, we assume a simple split logic.
    try:
        results = re.findall(r'([^<]+)<loc_(\d+?)><loc_(\d+?)><loc_(\d+?)><loc_(\d+?)>', raw_text)
        for res in results:
            label, y1, x1, y2, x2 = res
            label = label.strip()
            if label in class_names:
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "class_id": class_names.index(label),
                    "confidence": 0.99 # Assume high confidence
                })
    except Exception:
        return []
    return detections


def parse_paligemma_output(raw_text: str, class_names: List[str]) -> List[Dict]:
    """Parses PaliGemma's output format."""
    detections = []
    # Pattern: <loc_xxxx><loc_xxxx><loc_xxxx><loc_xxxx> class_name
    try:
        results = re.findall(r'<loc_(\d+?)><loc_(\d+?)><loc_(\d+?)><loc_(\d+?)> ([\w\s]+)', raw_text)
        for res in results:
            y1, x1, y2, x2, label = res
            label = label.strip()
            if label in class_names:
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "class_id": class_names.index(label),
                    "confidence": 0.99
                })
    except Exception:
        return []
    return detections


def parse_qwen_output(raw_text: str, class_names: List[str]) -> List[Dict]:
    """Parses Qwen's JSON-like output."""
    detections = []
    try:
        # Clean up markdown fences
        if raw_text.startswith("```json"):
            raw_text = raw_text.strip("```json\n").strip("`")
        
        data = json.loads(raw_text)
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label in class_names:
                detections.append({
                    "box": obj.get("bbox_2d"),
                    "label": label,
                    "class_id": class_names.index(label),
                    "confidence": 0.99
                })
    except (json.JSONDecodeError, AttributeError):
        return []
    return detections


def parse_vlm_output_to_boxes(raw_text: str, class_names: List[str]) -> List[Dict]:
    """A general parser that tries multiple formats."""
    # This acts as a dispatcher for different model outputs
    if "<OD>" in raw_text:
        return parse_florence2_output(raw_text, class_names)
    elif "<loc_" in raw_text:
        return parse_paligemma_output(raw_text, class_names)
    elif "{" in raw_text:
        return parse_qwen_output(raw_text, class_names)
    return []


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def calculate_metrics(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """
    Calculates TP, FP, FN for a single image.
    """
    tp, fp, fn = 0, 0, 0
    gt_matched = [False] * len(ground_truths)

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            # Assumes single class for now
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = len(ground_truths) - sum(gt_matched)

    return {"tp": tp, "fp": fp, "fn": fn}


def get_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1