# scripts/2_run_finetuning.py

import argparse
import yaml
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finetuning.finetune_florence2 import train_florence2
from src.finetuning.finetune_paligemma import train_paligemma
from src.finetuning.finetune_qwen import train_qwen

# Mapping from model family name to training function
TRAINING_DISPATCHER = {
    "florence2": train_florence2,
    "paligemma": train_paligemma,
    "qwen": train_qwen,
}

def run_finetuning(config_path: str):
    """
    Loads a configuration file and starts the fine-tuning process for the specified model.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_family = config.get("model_family")
    if not model_family or model_family not in TRAINING_DISPATCHER:
        raise ValueError(f"Config must specify a 'model_family' from: {list(TRAINING_DISPATCHER.keys())}")

    training_function = TRAINING_DISPATCHER[model_family]
    training_function(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Vision-Language Model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for the training run.")
    
    args = parser.parse_args()
    run_finetuning(args.config)