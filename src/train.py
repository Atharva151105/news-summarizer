import nltk
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from src.data import clean_article, clean_summary
from src.evaluate import compute_rouge

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MODEL = "google-t5/t5-small"
N_TRAIN = 5000      # training articles
N_VAL = 200         # validation articles
MAX_IN = 512        # max input tokens
MAX_OUT = 96        # max summary tokens
PREFIX = "summarize: "

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
train = ds["train"].shuffle(seed=42).select(range(N_TRAIN))
val = ds["validation"].select(range(N_VAL))


def preprocess(batch):
    inputs = [PREFIX + clean_article(a) for a in batch["article"]]
    model_inputs = tok(inputs, max_length=MAX_IN, truncation=True)
    labels = tok(
        text_target=[clean_summary(h) for h in batch["highlights"]],
        max_length=MAX_OUT,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
val_tok = val.map(preprocess, batched=True, remove_columns=val.column_names)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # -100 marks padding; swap it back so we can decode
    preds = np.where(preds != -100, preds, tok.pad_token_id)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    pred_text = tok.batch_decode(preds, skip_special_tokens=True)
    label_text = tok.batch_decode(labels, skip_special_tokens=True)
    return compute_rouge(pred_text, label_text)


args = Seq2SeqTrainingArguments(
    output_dir="t5-small-cnn",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
    generation_max_length=80,
    logging_steps=50,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
    processing_class=tok,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("t5-small-cnn-final")
tok.save_pretrained("t5-small-cnn-final")
print("Saved to t5-small-cnn-final")