from __future__ import annotations
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType


load_dotenv()


@dataclass
class Settings:
    model_name: str = os.getenv("MODEL_ID", "google/gemma-4-E2B")
    dataset_name: str = os.getenv("DATASET_ID", "iamshnoo/alpaca-cleaned-bengali")
    output_dir: str = os.getenv("OUTPUT_DIR", "./google/gemma-4-E2B-finetuned")
    max_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))


@dataclass
class LoraSettings:
    rank: int = int(os.getenv("LORA_RANK", "16"))
    alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "3"))
    per_device_train_batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    lr_scheduler_type: str = os.getenv("LR_SCHEDULER", "cosine")
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.05"))
    bf16: bool = os.getenv("USE_BF16", "true").lower() == "true"
    logging_steps: int = int(os.getenv("LOGGING_STEPS", "25"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "200"))


@dataclass
class WandbConfig:
    project: str = os.getenv("WANDB_PROJECT", "lora-peft-finetuning")
    entity: str | None = os.getenv("WANDB_ENTITY", None)
    run_name: str | None = os.getenv("WANDB_RUN_NAME", None)
    enabled: bool = os.getenv("WANDB_ENABLED", "false").lower() == "true"


def get_settings() -> Settings:
    return Settings()

def get_lora_settings() -> LoraSettings:
    return LoraSettings()

def get_training_config() -> TrainingConfig:
    return TrainingConfig()

def get_wandb_config() -> WandbConfig:
    return WandbConfig()

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

def get_lora_config() -> LoraConfig:
    lora = get_lora_settings()
    return LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        init_lora_weights="gaussian",
        target_modules=lora.target_modules,
        lora_dropout=lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
