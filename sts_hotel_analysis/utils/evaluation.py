import numpy as np
from typing import List, Tuple
from .sts_model import STSModel


def infer_doc_counts(model: STSModel, c_counts: np.ndarray, x_d: np.ndarray,
                     a0: np.ndarray, maxiter: int = 200) -> Tuple[np.ndarray, float]:
    """Infer a doc latent a using custom counts, return a_opt and doc log-likelihood for heldout counts.
    """
    K = model.K
    # Build a custom f using provided counts
    def f_doc(a: np.ndarray) -> float:
        a_p = a[:K]
        a_s = a[K:]
        theta = model.softmax(a_p)
        beta_list = [model._beta_topic(a_s[k], k) for k in range(K)]
        mix = np.zeros(model.V)
        for k in range(K):
            mix += theta[k] * beta_list[k]
        mix = np.clip(mix, 1e-12, None)
        ll = np.sum(c_counts * np.log(mix))
        mu = x_d.dot(model.Gamma)
        diff = a - mu
        try:
            Sigma_inv = np.linalg.inv(model.Sigma)
        except Exception:
            Sigma_inv = np.linalg.pinv(model.Sigma)
        prior_quad = -0.5 * diff.T.dot(Sigma_inv).dot(diff)
        return ll + prior_quad
    from scipy.optimize import minimize
    res = minimize(lambda a: -f_doc(a), a0, method='BFGS', options={'maxiter': maxiter})
    a_opt = res.x
    return a_opt, f_doc(a_opt)


def heldout_log_likelihood(model: STSModel, holdout_frac_docs: float = 0.1, seed: int = 42) -> float:
    """Compute held-out log-likelihood by removing half words from 10% docs and evaluating.
    Returns average per-heldout-word log-likelihood.
    """
    rng = np.random.default_rng(seed)
    D = model.D
    idx_docs = np.arange(D)
    n_hold = max(1, int(D * holdout_frac_docs))
    hold_docs = rng.choice(idx_docs, size=n_hold, replace=False)
    total_ll = 0.0
    total_words = 0
    for d in hold_docs:
        c = model.dtm[d, :].astype(float)
        held = np.floor(c / 2.0)
        train = c - held
        a0 = np.concatenate([model.alpha_p[d, :], model.alpha_s[d, :]])
        a_opt, _ = infer_doc_counts(model, train, model.X[d, :], a0)
        # Evaluate log-likelihood on held-out words
        a_p = a_opt[:model.K]
        a_s = a_opt[model.K:]
        theta = model.softmax(a_p)
        beta_list = [model._beta_topic(a_s[k], k) for k in range(model.K)]
        mix = np.zeros(model.V)
        for k in range(model.K):
            mix += theta[k] * beta_list[k]
        mix = np.clip(mix, 1e-12, None)
        ll = np.sum(held * np.log(mix))
        total_ll += ll
        total_words += int(held.sum())
    avg_ll = total_ll / max(1, total_words)
    return float(avg_ll)


def topic_coherence_umass(model: STSModel, topn: int = 10) -> float:
    """Compute UMass coherence averaged over topics."""
    D, V = model.D, model.V
    # Binary presence per doc per word
    B = (model.dtm > 0).astype(int)
    # Precompute word doc freq
    df_w = B.sum(axis=0)
    # Compute coherence per topic at a_s=0 baseline
    coherences = []
    for k in range(model.K):
        beta = model._beta_topic(0.0, k)
        idxs = np.argsort(beta)[::-1][:topn]
        score = 0.0
        count = 0
        for i in range(topn):
            for j in range(i+1, topn):
                w_i = idxs[i]; w_j = idxs[j]
                # co-doc frequency
                D_ij = int((B[:, w_i] * B[:, w_j]).sum())
                D_j = int(df_w[w_j])
                score += np.log((D_ij + 1) / (D_j + 1e-9))
                count += 1
        coherences.append(score / max(1, count))
    return float(np.mean(coherences))
