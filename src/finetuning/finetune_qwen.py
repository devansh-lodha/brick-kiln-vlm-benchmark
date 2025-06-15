# src/finetuning/finetune_qwen.py

import os
import gc
import torch
from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.trainer import UnslothVisionDataCollator
from typing import Dict, Any, List
from datasets import load_dataset

def train_qwen(config: Dict[str, Any]) -> None:
    """
    Wrapper function to fine-tune a Qwen2.5-VL model using Unsloth.

    Args:
        config (Dict[str, Any]): A dictionary containing training parameters.
    """
    print("--- Starting Qwen2.5-VL Fine-tuning with Unsloth ---")
    
    # 1. Load Model and Tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config['model_id'],
        load_in_4bit=config.get('load_in_4bit', True),
        device_map="auto"
    )
    
    # 2. Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=True,
    )
    FastVisionModel.for_training(model)

    # 3. Load Dataset
    # This assumes a specific conversational format dataset.
    # The actual data loading logic would be more complex and is represented here conceptually.
    dataset = load_dataset(config['dataset_path'])
    train_dataset = dataset['train']

    # 4. Set up Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        dataset_text_field="messages", # Assumes a 'messages' field in the dataset
        args=TrainingArguments(
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            warmup_steps=config.get('warmup_steps', 5),
            num_train_epochs=config['epochs'],
            learning_rate=config.get('learning_rate', 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=config['output_dir'],
            remove_unused_columns=False,
        ),
    )

    # 5. Start Training
    print("Starting training...")
    trainer.train()
    print("--- Qwen2.5-VL Fine-tuning Completed ---")

    # 6. Save model
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    print(f"Model saved to {config['output_dir']}")

    # Clean up memory
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()