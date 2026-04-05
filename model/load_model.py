from peft import get_peft_model
from transformers import AutoModelForCausalLM
from config import get_settings, get_bnb_config, get_lora_config

def get_model():
    config = get_settings()
    model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=get_bnb_config(), device_map="auto")
    model.config.use_cache = False
    return model

def load_peft_model(model):
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    return model