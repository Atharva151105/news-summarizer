import re
from datasets import load_dataset


def clean_article(text: str) -> str:
    """Remove the dateline at the start, e.g. 'LONDON, England (Reuters) --'."""
    # Pattern: optional text, then (CNN) or (Reuters) etc., then --
    text = re.sub(r"^.{0,100}?\((CNN|Reuters|AP|EW\.com)\)\s*(--|-)?\s*", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_summary(text: str) -> str:
    """Fix the ' .' quirk and join the highlight lines into one paragraph."""
    text = text.replace(" .\n", ". ").replace("\n", " ")
    text = text.replace(" .", ".")
    return re.sub(r"\s+", " ", text).strip()


def load_test_samples(n: int = 100):
    """Load n test articles with cleaned text. Used for all our evaluations."""
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    ds = ds.select(range(n))
    return [
        {
            "id": row["id"],
            "article": clean_article(row["article"]),
            "reference": clean_summary(row["highlights"]),
        }
        for row in ds
    ]


if __name__ == "__main__":
    samples = load_test_samples(3)
    for s in samples:
        print("ARTICLE:", s["article"][:200])
        print("REFERENCE:", s["reference"])
        print("-" * 50)