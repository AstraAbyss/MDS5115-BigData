import re
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Document:
    text: str
    covariates: Dict[str, float]  # document-level features (incl. intercept)

@dataclass
class Corpus:
    docs: List[Document]
    vocab: List[str]
    # doc-term matrix (list of dict for sparsity: {v_idx: count})
    dtm: List[Dict[int, int]]

# Basic tokenizer: lower, remove punctuation, split by whitespace
punct_table = str.maketrans('', '', string.punctuation)

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(punct_table)
    tokens = re.split(r"\s+", text)
    return [t for t in tokens if t]


def build_vocabulary(docs: List[str], max_vocab: int = 5000, min_df: int = 5) -> List[str]:
    freq: Dict[str, int] = {}
    df: Dict[str, int] = {}
    for text in docs:
        toks = tokenize(text)
        seen = set()
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
            if w not in seen:
                df[w] = df.get(w, 0) + 1
                seen.add(w)
    # filter by doc freq
    filtered = [w for w, d in df.items() if d >= min_df]
    # sort by frequency
    filtered.sort(key=lambda w: freq.get(w, 0), reverse=True)
    vocab = filtered[:max_vocab]
    return vocab


def vectorize_corpus(docs: List[Document], vocab: List[str]) -> List[Dict[int, int]]:
    index: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
    dtm: List[Dict[int, int]] = []
    for doc in docs:
        counts: Dict[int, int] = {}
        for w in tokenize(doc.text):
            if w in index:
                idx = index[w]
                counts[idx] = counts.get(idx, 0) + 1
        dtm.append(counts)
    return dtm
