from transformers import AutoModelForCausalLM
from config import get_settings, get_bnb_config

def get_model():
    config = get_settings()
    model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=get_bnb_config(), device_map="auto")
    model.config.use_cache = False
    return model
