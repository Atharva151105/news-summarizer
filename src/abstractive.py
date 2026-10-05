import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use the GPU if there is one, otherwise the CPU (your laptop = CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_summarizer(model_name: str = "facebook/bart-large-cnn"):
    # The tokenizer turns text into numbers; the model turns numbers into a summary
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()  # we are only generating, not training
    return tokenizer, model


def summarize_batch(summarizer, articles, max_length=80, min_length=30, batch_size=4):
    tokenizer, model = summarizer
    summaries = []

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]

        # Convert text to numbers, cutting anything longer than 1024 tokens
        inputs = tokenizer(
            batch,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():  # saves memory, since we're not training
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
            )

        summaries.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        print(f"  summarized {min(i + batch_size, len(articles))}/{len(articles)}")

    return summaries