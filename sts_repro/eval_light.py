import os
import json
import math
import numpy as np
from typing import List, Dict, Tuple

from sts_repro.sts.data import Document, build_vocabulary, vectorize_corpus
from sts_repro.sts.anchor import AnchorInitializer
from sts_repro.sts.model import STSModel
from sts_repro.sts.em import EMRunner
from sts_repro.sts.variational import VariationalEstimator

OUTPUT_DIR = 'output'


def load_yelp_reviews(n_docs: int = 400) -> List[Document]:
    """Try to load Yelp Review Full via datasets; fallback to small local sample."""
    try:
        from datasets import load_dataset
        ds = load_dataset('yelp_review_full', split=f'train[:{n_docs}]')
        docs: List[Document] = []
        for row in ds:
            text = row['text']
            stars = int(row['label']) + 1  # 0..4 -> 1..5
            covs = {'intercept': 1.0, 'stars': float(stars)}
            docs.append(Document(text=text, covariates=covs))
        return docs
    except Exception:
        # Fallback small sample
        samples = [
            ("Great food and friendly staff. Loved the pizza!", 5),
            ("Service was slow, but the burger tasted fine.", 3),
            ("Horrible experience. Dirty tables and cold fries.", 1),
            ("Amazing pasta, cozy atmosphere. Will come again.", 5),
            ("Average coffee. Price is okay.", 3),
            ("Terrible service. Manager was rude.", 1),
            ("Nice patio seating and fresh salad.", 4),
            ("Waited too long. Food arrived cold.", 2),
            ("Delicious tacos! Highly recommend.", 5),
            ("Mask policy not followed. Very disappointing.", 1),
            ("Good takeout experience, fast pickup.", 4),
            ("Mediocre taste, decent price.", 3),
        ]
        docs = [Document(text=t, covariates={'intercept': 1.0, 'stars': float(s)}) for t, s in samples]
        return docs


def to_matrix(docs: List[Document]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted(list(docs[0].covariates.keys()))
    X = np.zeros((len(docs), len(keys)))
    for i, d in enumerate(docs):
        for j, k in enumerate(keys):
            X[i, j] = d.covariates[k]
    return X, keys


def top_words_indices(vocab: List[str], beta_matrix: np.ndarray, topn: int = 10) -> List[List[int]]:
    K, V = beta_matrix.shape
    tops = []
    for k in range(K):
        b = beta_matrix[k]
        idx = np.argsort(b)[-topn:][::-1]
        tops.append(list(map(int, idx)))
    return tops


def compute_umass_coherence(dtm: List[Dict[int, int]], top_word_sets: List[List[int]], eps: float = 1.0) -> List[float]:
    """Approximate UMass coherence using document co-occurrence.
    C_umass(T) = avg_{i<j} log( (D(w_i,w_j) + eps) / D(w_j) )
    """
    D = len(dtm)
    # Build doc frequency for each word and pair co-occurrence
    V = max(max(s) for s in top_word_sets) + 1 if top_word_sets else 0
    df = np.zeros(V, dtype=np.int32)
    # presence boolean per doc for words of interest
    pres_list: List[Dict[int, int]] = []
    for d in range(D):
        counts = dtm[d]
        pres = {v: 1 for v in counts.keys()}
        pres_list.append(pres)
        for v in pres.keys():
            if v < V:
                df[v] += 1
    coherences = []
    for S in top_word_sets:
        pairs = []
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                wi = S[i]; wj = S[j]
                # co-doc count
                co = 0
                for d in range(D):
                    pi = pres_list[d].get(wi, 0)
                    pj = pres_list[d].get(wj, 0)
                    if pi and pj:
                        co += 1
                denom = df[wj] if wj < len(df) else 0
                val = math.log((co + eps) / (denom + eps))
                pairs.append(val)
        coherences.append(float(np.mean(pairs)) if pairs else 0.0)
    return coherences


def pearsonr_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x); y = np.asarray(y)
    # subtract mean
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom == 0:
        return 0.0, 1.0
    r = float(np.dot(xm, ym) / denom)
    n = len(x)
    # p-value approximation using t-statistic
    if n > 3:
        t = r * math.sqrt((n-2) / max(1e-12, 1-r*r))
        # approximate two-sided p via survival function of t with df=n-2; use normal approx
        # this is a rough estimate sufficient for sanity check
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        p = 1.0
    return r, p


def heldout_ll_perplexity(model: STSModel, ve: VariationalEstimator,
                          dtm: List[Dict[int, int]], X: np.ndarray) -> Tuple[float, float]:
    """Compute average word log-likelihood per token and perplexity on given set."""
    V = model.V
    total_tokens = 0
    ll_sum = 0.0
    for d in range(len(dtm)):
        c_d = np.zeros(V)
        for v, c in dtm[d].items():
            c_d[v] = c
            total_tokens += c
        a_opt, _, _ = ve.optimize_ad(c_d, X[d], np.zeros(2*model.K))
        a_p = a_opt[:model.K]
        a_s = a_opt[model.K:]
        theta = model.softmax(a_p)
        S = np.zeros(V)
        for k in range(model.K):
            b = model.beta_topic(k, a_s[k])
            S += theta[k] * b
        S = np.maximum(S, 1e-12)
        ll_sum += float((c_d * np.log(S)).sum())
    avg_ll_per_token = ll_sum / max(1, total_tokens)
    perplexity = float(np.exp(-avg_ll_per_token))
    return avg_ll_per_token, perplexity


def run_eval(seed: int = 0, n_docs: int = 400, K: int = 6, max_em_iter: int = 8):
    np.random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs = load_yelp_reviews(n_docs=n_docs)
    texts = [d.text for d in docs]
    vocab = build_vocabulary(texts, max_vocab=2000, min_df=5)
    dtm = vectorize_corpus(docs, vocab)
    X, cov_keys = to_matrix(docs)

    model = STSModel(K=K, V=len(vocab), ix=X.shape[1], random_state=seed)
    ai = AnchorInitializer(K=K, vocab_size=len(vocab))
    beta0, anchors = ai.initialize(dtm)

    runner = EMRunner(model, max_em_iter=max_em_iter, tol=1e-4, group_size=30)
    runner.initialize_kappa_with_anchor(beta0)
    A_opt, Phis = runner.run(dtm, X)

    # Build topic beta with median sentiment per topic
    a_s_med = np.median(A_opt[:, K:(2*K)], axis=0)
    beta_mat = np.zeros((K, len(vocab)))
    for k in range(K):
        beta_mat[k] = model.beta_topic(k, a_s_med[k])
    top_sets = top_words_indices(vocab, beta_mat, topn=10)

    # Coherence
    coherences = compute_umass_coherence(dtm, top_sets, eps=1.0)
    coh_mean = float(np.mean(coherences))
    coh_median = float(np.median(coherences))

    # Sentiment-rating correlation
    stars = X[:, 1] if X.shape[1] >= 2 else np.zeros(len(X))
    pearsons = []
    pvals = []
    for k in range(K):
        r, p = pearsonr_with_p(A_opt[:, K+k], stars)
        pearsons.append(float(r)); pvals.append(float(p))
    pearson_mean = float(np.mean(pearsons))

    # Held-out split: 80/20
    D = len(dtm)
    idx = np.arange(D)
    np.random.shuffle(idx)
    split = int(0.8 * D)
    train_idx = idx[:split]; test_idx = idx[split:]
    dtm_train = [dtm[i] for i in train_idx]
    X_train = X[train_idx]
    dtm_test = [dtm[i] for i in test_idx]
    X_test = X[test_idx]

    # Fit on train: reuse current model as a proxy (already fit on full),
    # but to be strict, refit a fresh model on train only
    model_t = STSModel(K=K, V=len(vocab), ix=X.shape[1], random_state=seed)
    beta0_t, _ = ai.initialize(dtm_train)
    runner_t = EMRunner(model_t, max_em_iter=max_em_iter, tol=1e-4, group_size=30)
    runner_t.initialize_kappa_with_anchor(beta0_t)
    A_train_opt, _ = runner_t.run(dtm_train, X_train)
    ve_t = VariationalEstimator(model_t)
    ll_train, ppl_train = heldout_ll_perplexity(model_t, ve_t, dtm_train, X_train)
    ll_test, ppl_test = heldout_ll_perplexity(model_t, ve_t, dtm_test, X_test)

    metrics = {
        'seed': seed,
        'K': K,
        'n_docs': D,
        'coherence_per_topic': coherences,
        'coherence_mean': coh_mean,
        'coherence_median': coh_median,
        'pearson_per_topic': pearsons,
        'pearson_pvals': pvals,
        'pearson_mean': pearson_mean,
        'heldout_train_ll_per_token': ll_train,
        'heldout_test_ll_per_token': ll_test,
        'heldout_train_perplexity': ppl_train,
        'heldout_test_perplexity': ppl_test,
    }
    with open(os.path.join(OUTPUT_DIR, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(metrics, ensure_ascii=False, separators=(',', ':')))
    return metrics


def run_stability(seed_base: int = 0) -> Dict[str, float]:
    # baseline
    m0 = run_eval(seed=seed_base, n_docs=200)
    # perturbed
    m1 = run_eval(seed=seed_base + 1, n_docs=200, max_em_iter=6)
    return {
        'coherence_mean_base': m0['coherence_mean'],
        'coherence_mean_perturbed': m1['coherence_mean'],
        'coherence_mean_delta': m1['coherence_mean'] - m0['coherence_mean'],
        'pearson_mean_base': m0['pearson_mean'],
        'pearson_mean_perturbed': m1['pearson_mean'],
        'pearson_mean_delta': m1['pearson_mean'] - m0['pearson_mean'],
        'heldout_ppl_base': m0['heldout_test_perplexity'],
        'heldout_ppl_perturbed': m1['heldout_test_perplexity'],
        'heldout_ppl_delta': m1['heldout_test_perplexity'] - m0['heldout_test_perplexity'],
    }


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Try installing datasets if needed
    try:
        import datasets  # noqa
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets', 'scipy'], check=False)
    metrics = run_eval(seed=0, n_docs=200)
    stability = run_stability(seed_base=0)
    summary = {'metrics': metrics, 'stability': stability}
    with open(os.path.join(OUTPUT_DIR, 'eval_summary.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False, separators=(',', ':')))
    print('Saved metrics to output/eval_metrics.json and output/eval_summary.json')
