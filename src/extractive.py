import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the sentence-splitting data (only downloads once)
for pkg in ["punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)


def split_sentences(text: str):
    return nltk.sent_tokenize(text)


def lead_k(text: str, k: int = 3) -> str:
    """Baseline: the first k sentences."""
    return " ".join(split_sentences(text)[:k])


def textrank(text: str, k: int = 3) -> str:
    """Pick the k most 'central' sentences, keep them in original order."""
    sentences = split_sentences(text)
    if len(sentences) <= k:
        return " ".join(sentences)

    try:
        # Turn each sentence into a vector of word weights
        vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:  # happens if a text has no usable words
        return " ".join(sentences[:k])

    # How similar is every sentence to every other sentence?
    sim = cosine_similarity(vectors)
    np.fill_diagonal(sim, 0)  # a sentence shouldn't vote for itself
    if sim.sum() == 0:
        return " ".join(sentences[:k])

    # Run PageRank on the sentence graph
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph)

    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return " ".join(sentences[i] for i in sorted(best))