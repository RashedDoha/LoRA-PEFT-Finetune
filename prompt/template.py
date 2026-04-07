PROMPT_TEMPLATE = """### নির্দেশনা:
{instruction}

### ইনপুট:
{input}

### উত্তর:
{output}"""

def get_prompt_template():
    return PROMPT_TEMPLATE

def format_example(example):
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example.get("input", ""),
            output=example["output"],
        )
    }

