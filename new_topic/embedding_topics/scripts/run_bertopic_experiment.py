#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding-based Topic Modeling Experiment (BERTopic) on sts_hotel_sample_1k
- Loads hotel review sample (1k)
- Preprocesses text (English), uses Reviewer_Score as sentiment proxy (sent_score)
- Trains BERTopic with sentence-transformers embeddings
- For each topic, computes Top-15 hotwords under three sentiment subsets:
  negative (10th percentile), average (all docs of the topic), positive (90th percentile), via c-TF-IDF.
- Computes topic quality metrics (coherence c_v, exclusivity approx)
- Maps topics to original STS results and computes Jaccard overlaps
- Saves outputs CSVs in experiments/embedding_topics/outputs
"""
import os
import sys
import json
import math
import re
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# Ensure reproducibility
import random
random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")

# NLTK for preprocessing
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# BERTopic and embeddings
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# gensim for coherence
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Paths
DATA_CSV = "sts_hotel_sample_1k/data/Hotel_Reviews_1k_sample.csv"
ORIG_STS_PATH = "sts_hotel_sample_1k/outputs/topic_hotwords_sts.csv"
OUT_DIR = "experiments/embedding_topics/outputs"
FIG_DIR = "experiments/embedding_topics/figs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Preprocess
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

CLEAN_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    if s is None:
        return ""
    # Common placeholders to drop
    s = s.strip()
    s = s.replace('No Negative', '').replace('No Positive', '')
    s = s.replace("hotel was", "")  # light noise removal (optional)
    # Lowercase and remove non-letters
    s = s.lower()
    s = CLEAN_RE.sub(' ', s)
    # Tokenize, remove stopwords, lemmatize (simple)
    tokens = [lemmatizer.lemmatize(tok) for tok in s.split() if tok not in STOPWORDS and len(tok) > 2]
    s2 = ' '.join(tokens)
    s2 = MULTISPACE_RE.sub(' ', s2).strip()
    return s2


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # Compose a single text field by concatenating positive and negative
    # If any side is missing, use the other
    def compose(row):
        pos = row.get('Positive_Review', '')
        neg = row.get('Negative_Review', '')
        if pd.isna(pos): pos = ''
        if pd.isna(neg): neg = ''
        return (str(pos) + ' ' + str(neg)).strip()

    df['raw_text'] = df.apply(compose, axis=1)
    df['text_clean'] = df['raw_text'].apply(clean_text)
    # Sentiment proxy: Reviewer_Score (0-10 scale)
    if 'Reviewer_Score' in df.columns:
        df['sent_score'] = pd.to_numeric(df['Reviewer_Score'], errors='coerce')
    else:
        # Fallback: VADER (if no rating), but in this dataset we have ratings
        df['sent_score'] = np.nan
    # Keep doc_id if exists else create
    if 'doc_id' not in df.columns:
        df['doc_id'] = np.arange(1, len(df) + 1)
    # Filter out empty cleaned texts
    df = df[~df['text_clean'].isna()]
    df = df[df['text_clean'].str.len() > 0]
    df = df.reset_index(drop=True)
    return df


def fit_bertopic(docs: List[str]) -> Tuple[BERTopic, List[int], np.ndarray]:
    # Embedding model: English
    model_name = 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(docs, batch_size=32, show_progress_bar=True)
    vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language='english',
        min_topic_size=10,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics, probs


def build_global_vectorizer(docs: List[str]) -> Tuple[CountVectorizer, TfidfTransformer, np.ndarray]:
    vect = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)
    X = vect.fit_transform(docs)
    tfidf = TfidfTransformer(norm=None, smooth_idf=True, use_idf=True)
    tfidf.fit(X)
    return vect, tfidf, X


def agg_ctfidf_for_subset(X, tfidf: TfidfTransformer, subset_idx: np.ndarray) -> np.ndarray:
    if subset_idx.size == 0:
        return np.zeros(X.shape[1])
    # Aggregate counts across subset
    counts = X[subset_idx, :].sum(axis=0)  # 1 x V matrix
    # Transform with global IDF
    from scipy.sparse import csr_matrix
    counts_csr = csr_matrix(counts)
    ctfidf = tfidf.transform(counts_csr)  # 1 x V
    return np.asarray(ctfidf.todense()).flatten()


def top_n_words(vect: CountVectorizer, scores: np.ndarray, n: int = 15) -> List[Tuple[str, float]]:
    vocab = np.array(vect.get_feature_names_out())
    order = np.argsort(-scores)
    top_idx = order[:n]
    return [(vocab[i], float(scores[i])) for i in top_idx]


def compute_quality_metrics(texts_tokens: List[List[str]], avg_topic_words: Dict[int, List[str]], ctfidf_avg_per_topic: Dict[int, np.ndarray], vect: CountVectorizer) -> pd.DataFrame:
    dictionary = Dictionary(texts_tokens)
    coherence_rows = []
    # Build per-topic exclusivity
    # Stack avg ctfidf vectors to compute per-word exclusivity
    K = len(avg_topic_words)
    V = len(vect.get_feature_names_out())
    mat = np.zeros((K, V), dtype=float)
    for k, vec in ctfidf_avg_per_topic.items():
        mat[k, :] = vec
    epsilon = 1e-9
    for k in avg_topic_words:
        topic_words = avg_topic_words[k]
        # Coherence c_v
        cm = CoherenceModel(topics=[topic_words], texts=texts_tokens, dictionary=dictionary, coherence='c_v')
        coh = float(cm.get_coherence())
        # Exclusivity: mean s_k / sum_{t!=k} s_t over top words
        vocab = np.array(vect.get_feature_names_out())
        word2idx = {w: i for i, w in enumerate(vocab)}
        ex_vals = []
        for w in topic_words:
            i = word2idx.get(w)
            if i is None:
                continue
            s_k = mat[k, i]
            s_others = mat[:, i].sum() - s_k
            ex_vals.append(float(s_k / (s_others + epsilon)))
        exclusivity = float(np.mean(ex_vals)) if ex_vals else 0.0
        coherence_rows.append({'topic_id': k, 'coherence_cv': coh, 'exclusivity': exclusivity})
    return pd.DataFrame(coherence_rows)


def tokenize_for_gensim(s: str) -> List[str]:
    return s.split()


def map_to_sts(bertopic_avg: Dict[int, List[Tuple[str, float]]], sts_df: pd.DataFrame) -> pd.DataFrame:
    # STS DF has columns: topic_id, sentiment_setting, rank, word, probability
    sts_avg = sts_df[sts_df['sentiment_setting'] == 'average'].copy()
    # Build STS per-topic word-prob dict
    sts_topics = sorted(sts_avg['topic_id'].unique())
    sts_wordsets = {}
    sts_weightvecs = {}
    # Build vocabulary union across both sides for cosine
    for t in sts_topics:
        sub = sts_avg[sts_avg['topic_id'] == t]
        words = list(sub['word'])
        probs = list(sub['probability'].astype(float))
        sts_wordsets[t] = set(words)
        sts_weightvecs[t] = dict(zip(words, probs))
    rows = []
    for k, tw in bertopic_avg.items():
        bw_words = [w for w, s in tw]
        bw_scores = [s for w, s in tw]
        bw_set = set(bw_words)
        # Jaccard with each STS topic
        best_jacc = -1.0
        best_topic = None
        best_cos = -1.0
        for t in sts_topics:
            jacc = len(bw_set & sts_wordsets[t]) / float(len(bw_set | sts_wordsets[t]))
            # Cosine over union vocabulary
            vocab_union = list(bw_set | sts_wordsets[t])
            vw_idx = {w: i for i, w in enumerate(vocab_union)}
            vec_b = np.zeros(len(vocab_union))
            vec_s = np.zeros(len(vocab_union))
            # Fill vec_b
            b_map = dict(tw)
            for w, i in vw_idx.items():
                vec_b[i] = float(b_map.get(w, 0.0))
                vec_s[i] = float(sts_weightvecs[t].get(w, 0.0))
            if np.linalg.norm(vec_b) == 0 or np.linalg.norm(vec_s) == 0:
                cos = 0.0
            else:
                cos = float(cosine_similarity(vec_b.reshape(1, -1), vec_s.reshape(1, -1))[0][0])
            if jacc > best_jacc or (jacc == best_jacc and cos > best_cos):
                best_jacc = jacc
                best_cos = cos
                best_topic = t
        rows.append({'bertopic_topic_id': k, 'sts_topic_id': int(best_topic) if best_topic is not None else -1,
                     'cosine_similarity': best_cos, 'jaccard_avg': best_jacc})
    return pd.DataFrame(rows)


def compute_overlap_jaccard(bertopic_hot: pd.DataFrame, sts_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    # bertopic_hot: columns [topic_id, sentiment_setting, rank, word, score]
    result_rows = []
    for _, row in mapping_df.iterrows():
        bk = int(row['bertopic_topic_id'])
        sk = int(row['sts_topic_id'])
        if sk == -1:
            continue
        for setting in ['negative', 'average', 'positive']:
            bw = set(bertopic_hot[(bertopic_hot['topic_id'] == bk) & (bertopic_hot['sentiment_setting'] == setting)]['word'].tolist())
            sw = set(sts_df[(sts_df['topic_id'] == sk) & (sts_df['sentiment_setting'] == setting)]['word'].tolist())
            if len(bw | sw) == 0:
                jacc = 0.0
            else:
                jacc = len(bw & sw) / float(len(bw | sw))
            result_rows.append({'bertopic_topic_id': bk, 'sts_topic_id': sk, 'sentiment_setting': setting, 'jaccard': jacc})
    return pd.DataFrame(result_rows)


def main():
    df = load_data()
    docs = df['text_clean'].tolist()
    print(f"Loaded {len(docs)} docs for modeling.\n")

    # Fit BERTopic
    topic_model, topics, probs = fit_bertopic(docs)
    df['topic'] = topics
    # Extract probability of the assigned topic per document
    topic_probs = []
    if probs is None:
        topic_probs = [np.nan] * len(df)
    else:
        # probs is an (n_docs, n_topics) matrix; assigned probability is probs[i, topic_i] if topic_i != -1
        for i, t in enumerate(topics):
            if probs is None or t == -1:
                topic_probs.append(np.nan)
            else:
                try:
                    topic_probs.append(float(probs[i][t]))
                except Exception:
                    topic_probs.append(np.nan)
    df['topic_prob'] = topic_probs
    # Persist doc-topic assignments
    df_assign = df[['doc_id', 'topic', 'topic_prob', 'sent_score']]
    df_assign.to_csv(os.path.join(OUT_DIR, 'bertopic_doc_topics.csv'), index=False)

    # Global vectorizer & IDF on full corpus for c-TF-IDF computation
    vect, tfidf, X = build_global_vectorizer(docs)

    # Tokenized texts for coherence
    texts_tokens = [tokenize_for_gensim(s) for s in docs]

    # Prepare outputs for hotwords
    hot_rows = []

    # Cache avg ctfidf per topic for exclusivity
    ctfidf_avg_per_topic = {}
    avg_topic_words = {}

    valid_topics = sorted([t for t in set(topics) if t != -1])
    print(f"Found {len(valid_topics)} topics (excluding outlier -1).\n")

    for k in valid_topics:
        # Subset docs in topic k
        idx_k = np.where(df['topic'].values == k)[0]
        if idx_k.size == 0:
            continue
        sent_k = df.iloc[idx_k]['sent_score'].values
        # Percentiles per topic subset
        q10 = np.nanpercentile(sent_k, 10)
        q90 = np.nanpercentile(sent_k, 90)
        mean_k = np.nanmean(sent_k)
        # Build subsets
        idx_neg = idx_k[np.where(sent_k <= q10)[0]]
        idx_pos = idx_k[np.where(sent_k >= q90)[0]]
        idx_avg = idx_k  # Use all docs as average setting representative

        # Compute c-TF-IDF for each subset
        vec_neg = agg_ctfidf_for_subset(X, tfidf, idx_neg)
        vec_avg = agg_ctfidf_for_subset(X, tfidf, idx_avg)
        vec_pos = agg_ctfidf_for_subset(X, tfidf, idx_pos)

        # Save for exclusivity
        ctfidf_avg_per_topic[k] = vec_avg
        avg_topic_words[k] = [w for w, s in top_n_words(vect, vec_avg, n=15)]

        for setting, vec_scores in [('negative', vec_neg), ('average', vec_avg), ('positive', vec_pos)]:
            topw = top_n_words(vect, vec_scores, n=15)
            for r, (w, s) in enumerate(topw, start=1):
                hot_rows.append({'topic_id': k, 'sentiment_setting': setting, 'rank': r, 'word': w, 'score': float(s)})

    hot_df = pd.DataFrame(hot_rows)
    out_hot_path = os.path.join(OUT_DIR, 'topic_hotwords_bertopic.csv')
    hot_df.to_csv(out_hot_path, index=False)
    print(f"Saved hotwords to {out_hot_path}\n")

    # Quality metrics
    qual_df = compute_quality_metrics(texts_tokens, avg_topic_words, ctfidf_avg_per_topic, vect)
    qual_path = os.path.join(OUT_DIR, 'topic_quality_metrics.csv')
    qual_df.to_csv(qual_path, index=False)
    print(f"Saved quality metrics to {qual_path}\n")

    # Mapping to original STS results if available
    mapping_path = os.path.join(OUT_DIR, 'topic_mapping.csv')
    overlap_path = os.path.join(OUT_DIR, 'topic_overlap_jaccard.csv')
    if os.path.exists(ORIG_STS_PATH):
        sts_df = pd.read_csv(ORIG_STS_PATH)
        # Build bertopic avg dict
        bertopic_avg = {}
        for k in valid_topics:
            sub = hot_df[(hot_df['topic_id'] == k) & (hot_df['sentiment_setting'] == 'average')]
            bertopic_avg[k] = list(zip(sub['word'].tolist(), sub['score'].astype(float).tolist()))
        mapping_df = map_to_sts(bertopic_avg, sts_df)
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Saved topic mapping to {mapping_path}\n")
        # Overlap Jaccard per sentiment setting
        overlap_df = compute_overlap_jaccard(hot_df, sts_df, mapping_df)
        overlap_df.to_csv(overlap_path, index=False)
        print(f"Saved overlap Jaccard to {overlap_path}\n")
    else:
        # Create empty mapping file with note
        pd.DataFrame([], columns=['bertopic_topic_id','sts_topic_id','cosine_similarity','jaccard_avg']).to_csv(mapping_path, index=False)
        print("Original STS hotwords not found; mapping file created empty.\n")


if __name__ == '__main__':
    main()
