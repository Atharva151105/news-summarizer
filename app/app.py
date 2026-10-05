import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.data import clean_article
from src.extractive import lead_k, textrank
from src.abstractive import load_summarizer, summarize_batch

# Abstractive models load the first time they are used, then stay in memory
_cache = {}

ABSTRACTIVE = {
    "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
    # Add your own model here after we upload it to the Hugging Face Hub:
    # "T5-small fine-tuned (ours)": "YOUR_USERNAME/t5-small-cnn",
}
EXTRACTIVE = {"Lead-3": lead_k, "TextRank": textrank}


def summarize(text, mode, extractive_method, abstractive_model, num_sentences):
    if not text or len(text.split()) < 40:
        return "Please paste a longer article (at least 40 words)."

    text = clean_article(text)

    if mode == "Extractive":
        return EXTRACTIVE[extractive_method](text, k=int(num_sentences))

    ckpt = ABSTRACTIVE[abstractive_model]
    if ckpt not in _cache:
        _cache[ckpt] = load_summarizer(ckpt)
    return summarize_batch(_cache[ckpt], [text], batch_size=1)[0]


with gr.Blocks(title="News Summarizer") as demo:
    gr.Markdown("# News Article Summarizer\nExtractive (copies sentences) vs abstractive (writes new ones).")
    text = gr.Textbox(lines=12, label="Paste a news article")
    mode = gr.Radio(["Extractive", "Abstractive"], value="Extractive", label="Mode")
    with gr.Row():
        ext = gr.Dropdown(list(EXTRACTIVE), value="Lead-3", label="Extractive method")
        n = gr.Slider(1, 6, value=3, step=1, label="Sentences (extractive only)")
        abs_model = gr.Dropdown(list(ABSTRACTIVE), value=list(ABSTRACTIVE)[0], label="Abstractive model")
    out = gr.Textbox(label="Summary", lines=6)
    gr.Button("Summarize").click(summarize, [text, mode, ext, abs_model, n], out)

if __name__ == "__main__":
    demo.launch()