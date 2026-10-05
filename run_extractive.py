from src.data import load_test_samples
from src.extractive import lead_k, textrank
from src.evaluate import compute_rouge

samples = load_test_samples(100)
articles = [s["article"] for s in samples]
references = [s["reference"] for s in samples]

methods = {
    "Lead-3": lead_k,
    "TextRank": textrank,
}

# Show one example from each method
print("ARTICLE (start):", articles[0][:300], "\n")
print("REFERENCE:", references[0], "\n")

for name, fn in methods.items():
    predictions = [fn(a) for a in articles]
    print(f"{name} example:", predictions[0], "\n")
    print(f"{name} ROUGE:", compute_rouge(predictions, references), "\n")