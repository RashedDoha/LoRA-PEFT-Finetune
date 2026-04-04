from transformers import AutoTokenizer
from config import get_settings

def get_tokenizer():
    config = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

