# scripts/4_run_cross_domain_eval.py

import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predict import load_model_for_inference, predict_vlm
from src.evaluation.metrics import parse_vlm_output_to_boxes, calculate_metrics, get_precision_recall_f1
from src.data.data_utils import parse_yolo_obb_label, obb_to_aabb

PROMPT = "detect brick kiln with chimney"

def run_cross_domain_evaluation(model_path: str, test_data_dir: str, output_dir: str):
    """
    Evaluates a fine-tuned model on a cross-domain test dataset to measure generalization.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(test_data_dir, 'images')
    label_dir = os.path.join(test_data_dir, 'labels')
    image_paths = sorted(glob(os.path.join(image_dir, '*.png')))

    print(f"\n--- Cross-Domain Evaluation ---")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Test Region: {os.path.basename(test_data_dir)}")
    
    model, processor = load_model_for_inference(model_path)

    total_tp, total_fp, total_fn = 0, 0, 0

    for image_path in tqdm(image_paths, desc="Evaluating Cross-Domain"):
        image = Image.open(image_path).convert("RGB")
        
        gt_objects_raw = parse_yolo_obb_label(os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt'))
        w, h = image.size
        ground_truths = [{'box': obb_to_aabb([c * w if i % 2 == 0 else c * h for i, c in obj['obb_coords']])} for obj in gt_objects_raw]

        raw_output = predict_vlm(model, processor, image, PROMPT)
        predictions = parse_vlm_output_to_boxes(raw_output, class_names=['brick kiln with chimney'])

        metrics = calculate_metrics(predictions, ground_truths)
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']

    precision, recall, f1 = get_precision_recall_f1(total_tp, total_fp, total_fn)
    
    result = {
        "Model Path": model_path,
        "Test Region": os.path.basename(test_data_dir),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    results_df = pd.DataFrame([result])
    print("\n--- Cross-Domain Evaluation Summary ---")
    print(results_df.to_markdown(index=False))
    
    output_filename = os.path.join(output_dir, f"cross_domain_eval_{os.path.basename(model_path)}_on_{os.path.basename(test_data_dir)}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Domain VLM Evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model trained on the source domain.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Path to the test data directory from the target domain.")
    parser.add_argument("--output_dir", type=str, default="results/cross_domain", help="Directory to save the evaluation results.")
    
    args = parser.parse_args()
    run_cross_domain_evaluation(args.model_path, args.test_data_dir, args.output_dir)