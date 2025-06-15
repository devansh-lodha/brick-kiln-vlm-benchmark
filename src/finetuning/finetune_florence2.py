# src/finetuning/finetune_florence2.py

import os
from maestro.trainer.models.florence_2.core import train as maestro_train
from typing import Dict, Any

def train_florence2(config: Dict[str, Any]) -> None:
    """
    Wrapper function to fine-tune a Florence-2 model using Maestro.

    Args:
        config (Dict[str, Any]): A dictionary containing all the necessary
                                 parameters for training, including model_id,
                                 dataset path, epochs, learning rate, etc.
    """
    print("--- Starting Florence-2 Fine-tuning ---")
    print(f"  Model ID: {config.get('model_id')}")
    print(f"  Dataset: {config.get('dataset')}")
    print(f"  Epochs: {config.get('epochs')}")
    print(f"  Output Directory: {config.get('output_dir')}")
    print("-----------------------------------------")

    # Ensure output directory exists
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    # Maestro's train function expects a specific config structure.
    # We ensure our passed config matches this.
    try:
        maestro_train(config)
        print("--- Florence-2 Fine-tuning Completed Successfully ---")
    except Exception as e:
        print(f"--- Florence-2 Fine-tuning Failed ---")
        print(f"Error: {e}")
        raise