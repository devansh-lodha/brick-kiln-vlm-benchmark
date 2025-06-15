# scripts/1_run_zero_shot_eval.py

import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

# Adjust the path to import from the src directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predict import load_model_for_inference, predict_vlm
from src.evaluation.metrics import parse_vlm_output_to_boxes, calculate_metrics, get_precision_recall_f1
from src.data.data_utils import parse_yolo_obb_label, obb_to_aabb

# --- Models to evaluate in zero-shot setting ---
ZERO_SHOT_MODELS = {
    "Florence-2-large-ft": "microsoft/Florence-2-large-ft",
    "PaliGemma-3B-PT": "google/paligemma-3b-pt-448",
    "Qwen2.5-VL-3B": "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    "YoloE": "yoloe-l", # Placeholder for YoloE logic which is different
}

# --- Prompts for different models/tasks ---
PROMPTS = {
    "OD": "detect brick kiln with chimney",
    "Florence-2-large-ft": "<OD>", # Florence-2 uses special tokens for tasks
    "PaliGemma-3B-PT": "detect brick kiln with chimney",
    "Qwen2.5-VL-3B": "Identify and locate brick kiln. Provide bounding boxes in JSON format.",
}

def run_zero_shot_evaluation(test_data_dir: str, output_dir: str):
    """
    Runs zero-shot evaluation for a predefined list of VLMs on a test set.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(test_data_dir, 'images')
    label_dir = os.path.join(test_data_dir, 'labels')
    image_paths = sorted(glob(os.path.join(image_dir, '*.png'))) # Assuming PNG format

    results = []

    for model_name, model_id in ZERO_SHOT_MODELS.items():
        print(f"\n--- Evaluating Zero-Shot: {model_name} ---")

        if model_name == "YoloE":
            # YoloE has a different inference mechanism, handled separately.
            # Placeholder for its evaluation logic.
            print("Skipping YoloE for now. Requires a different evaluation loop.")
            continue

        model, processor = load_model_for_inference(model_id)
        
        total_tp, total_fp, total_fn = 0, 0, 0
        prompt = PROMPTS.get(model_name, PROMPTS["OD"])

        for image_path in tqdm(image_paths, desc=f"Inferring with {model_name}"):
            image = Image.open(image_path).convert("RGB")
            
            # Get ground truth
            label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
            gt_objects_raw = parse_yolo_obb_label(label_path)
            
            w, h = image.size
            ground_truths = []
            for obj in gt_objects_raw:
                unnorm_obb = [c * w if i % 2 == 0 else c * h for i, c in enumerate(obj['obb_coords'])]
                ground_truths.append({'box': obb_to_aabb(unnorm_obb)})

            # Get predictions
            raw_output = predict_vlm(model, processor, image, prompt)
            predictions = parse_vlm_output_to_boxes(raw_output, class_names=['brick kiln with chimney'])

            # Calculate metrics for the image
            metrics = calculate_metrics(predictions, ground_truths, iou_threshold=0.5)
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']

        # Calculate overall metrics for the model
        precision, recall, f1 = get_precision_recall_f1(total_tp, total_fp, total_fn)
        results.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
        })

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    print("\n--- Zero-Shot Evaluation Summary ---")
    print(results_df.to_markdown(index=False))
    results_df.to_csv(os.path.join(output_dir, "zero_shot_results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zero-Shot VLM Evaluation.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Path to the test data directory containing 'images' and 'labels' subfolders.")
    parser.add_argument("--output_dir", type=str, default="results/zero_shot", help="Directory to save the evaluation results.")
    
    args = parser.parse_args()
    run_zero_shot_evaluation(args.test_data_dir, args.output_dir)