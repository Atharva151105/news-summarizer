import time
from src.data import load_test_samples
from src.abstractive import load_summarizer, summarize_batch
from src.evaluate import compute_rouge

# Start with just 10 articles: BART is slow on a CPU
samples = load_test_samples(10)
articles = [s["article"] for s in samples]
references = [s["reference"] for s in samples]

print("Loading BART (first time downloads about 1.6 GB)...")
summarizer = load_summarizer()

start = time.time()
predictions = summarize_batch(summarizer, articles)
print(f"Done in {time.time() - start:.0f} seconds\n")

print("REFERENCE:", references[0], "\n")
print("BART:", predictions[0], "\n")
print("BART ROUGE:", compute_rouge(predictions, references))