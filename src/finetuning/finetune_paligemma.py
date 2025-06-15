# src/finetuning/finetune_paligemma.py

import os
from maestro.trainer.models.paligemma_2.core import train as maestro_train_paligemma
from typing import Dict, Any

def train_paligemma(config: Dict[str, Any]) -> None:
    """
    Wrapper function to fine-tune a PaliGemma model using Maestro.

    Args:
        config (Dict[str, Any]): A dictionary containing all the necessary
                                 parameters for training.
    """
    print("--- Starting PaliGemma Fine-tuning ---")
    print(f"  Model ID: {config.get('model_id')}")
    print(f"  Dataset: {config.get('dataset')}")
    print(f"  Epochs: {config.get('epochs')}")
    print(f"  Output Directory: {config.get('output_dir')}")
    print("-----------------------------------------")

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    try:
        maestro_train_paligemma(config)
        print("--- PaliGemma Fine-tuning Completed Successfully ---")
    except Exception as e:
        print(f"--- PaliGemma Fine-tuning Failed ---")
        print(f"Error: {e}")
        raise