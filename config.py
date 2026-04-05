from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import os

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType


load_dotenv()

@dataclass
class Settings:
    model_name: str = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-1.7B")
    dataset_name: str = os.getenv("DATASET_ID", "yahma/alpaca-cleaned")
    output_dir: str = os.getenv("OUTPUT_DIR", "./smollm2-1.7b-finetuned")
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    lora_rank: int = int(os.getenv("LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    max_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))


def get_settings() -> Settings:
    return Settings()

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def get_lora_config() -> LoraConfig:
    settings = get_settings()
    return LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

