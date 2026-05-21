"""SFT pre-training step for DDPO: fine-tune on chosen responses."""

import argparse
import json
import logging
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_train_data(train_file: str) -> Dataset:
    """Load {"prompt", "completion"} JSONL and return as a messages-format dataset."""
    rows = []
    with open(train_file) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                rows.append({
                    "messages": [
                        {"role": "user", "content": ex["prompt"]},
                        {"role": "assistant", "content": ex["completion"]},
                    ]
                })
    dataset = Dataset.from_list(rows)
    logger.info(f"Loaded {len(dataset)} SFT examples from {train_file}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="SFT training with LoRA (pre-step for DDPO).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training jsonl with {prompt, completion}")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model_name_or_path"]
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {config['output_dir']}")

    sft_config = SFTConfig(
        output_dir=config["output_dir"],
        max_length=config.get("max_length", 2048),
        learning_rate=config.get("learning_rate", 1e-5),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        bf16=config.get("bf16", True),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 20),
        save_only_model=config.get("save_only_model", False),
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name", "sft_training"),
        completion_only_loss=True,
        packing=False,
    )

    peft_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3's chat template defaults to enable_thinking=True, which injects
    # <think></think> tokens into every assistant turn. Patch apply_chat_template
    # so all internal TRL calls (boundary detection, formatting) use thinking=False.
    if "qwen" in model_name.lower():
        _orig_apply = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda *a, **kw: _orig_apply(*a, **{"enable_thinking": False, **kw})

    dataset = load_train_data(args.train_file)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    last_checkpoint = None
    if os.path.isdir(config["output_dir"]):
        last_checkpoint = get_last_checkpoint(config["output_dir"])
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(config["output_dir"])
    logger.info(f"Model saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
