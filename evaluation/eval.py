import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import get_settings, get_bnb_config
from prompt import get_prompt_template


def load_eval_model(model_id: str, adapter_path: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path else model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 200,
) -> str:
    prompt_template = get_prompt_template()
    prompt = prompt_template.format(
        instruction=instruction,
        input=input_text,
        output="",
    ).rstrip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
