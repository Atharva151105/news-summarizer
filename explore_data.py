from datasets import load_dataset
import pandas as pd

# Download the news dataset (first time takes a few minutes)
ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
print(ds)

# Look at one article and its human-written summary
sample = ds["train"][0]
print("\nARTICLE (first 600 characters):")
print(sample["article"][:600])
print("\nSUMMARY:")
print(sample["highlights"])

# Count words in 2000 articles
df = pd.DataFrame(ds["train"].select(range(2000)))
df["article_words"] = df["article"].str.split().str.len()
df["summary_words"] = df["highlights"].str.split().str.len()
print("\nLength statistics:")
print(df[["article_words", "summary_words"]].describe())