import sys
from evaluation import (
    load_eval_model,
    generate_response,
    get_eval_examples,
    compute_perplexity,
    compute_rouge,
    compute_bertscore,
)
from config import get_settings, get_wandb_config
from tracking import init_wandb, finish_wandb


def evaluate(adapter_path: str | None = None) -> dict:
    settings = get_settings()
    wandb_config = get_wandb_config()

    model, tokenizer = load_eval_model(settings.model_name, adapter_path)
    examples = get_eval_examples()

    predictions = []
    references = []
    for ex in examples:
        pred = generate_response(model, tokenizer, ex["instruction"], ex.get("input", ""))
        predictions.append(pred)
        references.append(ex["reference"])

    results = {
        "rouge": compute_rouge(predictions, references),
        "bertscore": compute_bertscore(predictions, references),
        "perplexity": compute_perplexity(model, tokenizer, references)["mean_perplexity"],
    }

    init_wandb(wandb_config, settings=settings)
    if wandb_config.enabled:
        import wandb
        wandb.log(results)
    finish_wandb()

    return results


if __name__ == "__main__":
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = evaluate(adapter_path)
    for metric, value in results.items():
        print(f"{metric}: {value}")
