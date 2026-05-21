"""DPO training with LoRA using TRL's DPOTrainer."""

import argparse
import logging
import os

import torch
import yaml
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess_function(example):
    """Convert raw data fields to DPO chat format."""
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(train_file: str) -> Dataset:
    """Load training data from jsonl and apply preprocessing."""
    dataset = Dataset.from_json(train_file)
    logger.info(f"Loaded {len(dataset)} training examples from {train_file}")
    dataset = dataset.map(preprocess_function)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="DPO training with LoRA.")
    parser.add_argument("--config", type=str, default="train/config.yaml", help="Path to YAML config")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training jsonl")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Override model from config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir from config")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate from config")
    parser.add_argument("--beta", type=float, default=None, help="Override beta from config")
    parser.add_argument("--init_adapter", type=str, default=None, help="Path to a pre-trained PEFT adapter to start from (e.g. SFT adapter for DDPO)")
    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.beta is not None:
        config["beta"] = args.beta

    model_name = config["model_name_or_path"]
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {config['output_dir']}")

    # Build DPOConfig
    dpo_config = DPOConfig(
        output_dir=config["output_dir"],
        beta=config.get("beta", 0.1),
        loss_type=config.get("loss_type", ["sigmoid"]),
        # loss_weight_field=config.get("loss_weight_field", None),
        max_length=config.get("max_length", 2048),
        learning_rate=config.get("learning_rate", 5e-6),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        bf16=config.get("bf16", True),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", None),
        save_only_model=config.get("save_only_model", False),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        label_smoothing=config.get("label_smoothing", 0.0),
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name", "dpo_training"),
        remove_unused_columns=False,
    )

    # Build LoRA config
    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.init_adapter:
        logger.info(f"Loading SFT adapter from {args.init_adapter}...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None  # adapter already attached; DPOTrainer will continue training it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_train_data(args.train_file)

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Resume from checkpoint if one exists in output_dir
    last_checkpoint = None
    if os.path.isdir(config["output_dir"]):
        last_checkpoint = get_last_checkpoint(config["output_dir"])
        if last_checkpoint is None:
            logger.warning(
                f"output_dir {config['output_dir']} exists but contains no checkpoint-* directory; starting fresh."
            )
        else:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Train
    logger.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    trainer.save_model(config["output_dir"])
    logger.info(f"Model saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
