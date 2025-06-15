# src/data/data_utils.py

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple

def obb_to_aabb(obb_coords: List[float]) -> List[float]:
    """
    Converts an oriented bounding box (OBB) to an axis-aligned bounding box (AABB).

    Args:
        obb_coords (List[float]): A flat list of 8 coordinates [x1, y1, x2, y2, ...].

    Returns:
        List[float]: A list of 4 coordinates [xmin, ymin, xmax, ymax] for the AABB.
    """
    if len(obb_coords) != 8:
        raise ValueError("OBB coordinates must be a list of 8 floats.")

    x_coords = obb_coords[0::2]
    y_coords = obb_coords[1::2]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]

def parse_yolo_obb_label(label_path: str) -> List[Dict]:
    """
    Parses a YOLO-OBB format label file.

    Each line is expected to be: class_idx x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

    Args:
        label_path (str): The path to the YOLO-OBB .txt file.

    Returns:
        List[Dict]: A list of objects, each with 'class_index' and 'obb_coords'.
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 9:
                objects.append({
                    "class_index": int(float(parts[0])),
                    "obb_coords": [float(c) for c in parts[1:]],
                })
    return objects

def convert_to_coco_format(image_dir: str, label_dir: str, output_json_path: str, class_names: List[str]):
    """
    Converts a dataset from YOLO-OBB format to COCO JSON format.

    This is a conceptual implementation. The actual logic would involve iterating
    through all images and their corresponding label files, creating entries for
    images, annotations, and categories in a structured JSON file as expected
    by frameworks like Maestro.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing YOLO-OBB .txt labels.
        output_json_path (str): Path to save the output COCO JSON file.
        class_names (List[str]): A list of class names, where the index corresponds
                                 to the class ID in the YOLO files.
    """
    print("Conceptual function: Converting YOLO-OBB annotations to COCO JSON format...")

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create category entries
    for i, name in enumerate(class_names):
        coco_format['categories'].append({"id": i, "name": name})

    annotation_id = 1
    image_id = 1
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])

    for image_name in tqdm(image_files, desc="Converting to COCO"):
        image_path = os.path.join(image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        # Add image entry
        coco_format['images'].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Add annotation entries
        yolo_objects = parse_yolo_obb_label(label_path)
        for obj in yolo_objects:
            # Convert normalized OBB to un-normalized AABB for COCO
            unnormalized_obb = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(obj['obb_coords'])]
            aabb = obb_to_aabb(unnormalized_obb)
            x, y, w, h = aabb[0], aabb[1], aabb[2] - aabb[0], aabb[3] - aabb[1]

            coco_format['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj['class_index'],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [obj['obb_coords']] # Storing OBB in segmentation
            })
            annotation_id += 1

        image_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO JSON successfully created at {output_json_path}")