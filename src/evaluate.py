import nltk
import numpy as np
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True
)


def compute_rouge(predictions, references):
    """Average ROUGE scores (as percentages) over all articles."""
    results = {"rouge1": [], "rouge2": [], "rougeLsum": []}
    for pred, ref in zip(predictions, references):
        # rougeLsum expects one sentence per line
        pred = "\n".join(nltk.sent_tokenize(pred))
        ref = "\n".join(nltk.sent_tokenize(ref))
        scores = _scorer.score(ref, pred)
        for key in results:
            results[key].append(scores[key].fmeasure)
    return {key: round(100 * np.mean(vals), 2) for key, vals in results.items()}