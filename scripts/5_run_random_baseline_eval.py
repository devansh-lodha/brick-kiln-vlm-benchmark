# scripts/5_run_random_baseline_eval.py

import os
import argparse
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import calculate_metrics, get_precision_recall_f1

def run_random_baseline(results_csv_path: str, image_data_dir: str, output_csv: str, num_runs: int = 10):
    """
    Performs a random baseline evaluation by shifting predicted bounding boxes.

    Args:
        results_csv_path (str): Path to a CSV file containing 'image_name', 'ground_truths', and 'predictions' columns.
        image_data_dir (str): Path to the directory containing the test images.
        output_csv (str): Path to save the aggregated random baseline results.
        num_runs (int): Number of random runs to average over.
    """
    results_df = pd.read_csv(results_csv_path)
    image_dir = os.path.join(image_data_dir, 'images')

    all_run_metrics = []

    for i in tqdm(range(num_runs), desc="Running Random Baselines"):
        run_tp, run_fp, run_fn = 0, 0, 0
        
        for _, row in results_df.iterrows():
            image_name = row['image_name']
            # Convert string representations to lists of dicts/boxes
            try:
                # This assumes boxes are stored as strings of lists
                gt_boxes = eval(row['ground_truths'])
                pred_boxes = eval(row['predictions'])
            except:
                continue

            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                continue
            
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            randomized_boxes = []
            for box in pred_boxes:
                x_min, y_min, x_max, y_max = box['box']
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                # Max shift possible without pushing the box out of bounds
                x_shift_max = img_width - box_width - x_min
                y_shift_max = img_height - box_height - y_min
                
                random_x_shift = random.uniform(-x_min, x_shift_max)
                random_y_shift = random.uniform(-y_min, y_shift_max)
                
                randomized_boxes.append({
                    'box': [
                        x_min + random_x_shift,
                        y_min + random_y_shift,
                        x_max + random_x_shift,
                        y_max + random_y_shift
                    ]
                })

            metrics = calculate_metrics(randomized_boxes, gt_boxes)
            run_tp += metrics['tp']
            run_fp += metrics['fp']
            run_fn += metrics['fn']
        
        precision, recall, f1 = get_precision_recall_f1(run_tp, run_fp, run_fn)
        all_run_metrics.append({'precision': precision, 'recall': recall, 'f1_score': f1})

    # Aggregate results
    final_df = pd.DataFrame(all_run_metrics)
    summary = {
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Mean": [final_df['precision'].mean(), final_df['recall'].mean(), final_df['f1_score'].mean()],
        "Std": [final_df['precision'].std(), final_df['recall'].std(), final_df['f1_score'].std()],
    }
    summary_df = pd.DataFrame(summary)

    print("\n--- Random Baseline Summary ---")
    print(summary_df.to_markdown(index=False))
    summary_df.to_csv(output_csv, index=False)
    print(f"Random baseline results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Baseline Evaluation.")
    parser.add_argument("--results_csv_path", type=str, required=True, help="Path to CSV from a previous evaluation run.")
    parser.add_argument("--image_data_dir", type=str, required=True, help="Path to the test data directory.")
    parser.add_argument("--output_csv", type=str, default="results/random_baseline.csv", help="Path to save the output CSV.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of random runs for statistical stability.")
    
    args = parser.parse_args()
    run_random_baseline(args.results_csv_path, args.image_data_dir, args.output_csv, args.num_runs)