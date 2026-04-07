import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def compute_perplexity(model, tokenizer, texts: list[str]) -> dict:
    """Measures how well the model predicts the reference responses.
    Lower perplexity indicates better performance.
    """
    perplexities = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()
        perplexities.append(perplexity)

    return {
        "mean_perplexity": np.mean(perplexities),
        "per_example": perplexities,
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Measures n-gram overlap between generated and reference responses.
    Higher ROUGE scores indicate better performance.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores: dict[str, list] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return {
        "precision": round(P.mean().item(), 4),
        "recall":    round(R.mean().item(), 4),
        "f1":        round(F1.mean().item(), 4),
    }

