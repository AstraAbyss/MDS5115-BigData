import numpy as np
from typing import Tuple
from sklearn.decomposition import NMF


def nmf_initialize(dtm: np.ndarray, K: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NMF to get initial document-topic weights (W) and topic-word (H).
    Returns:
      theta_init: D x K normalized topic proportions per doc
      kappa_t_init: K x V initial topic baseline coefficients
      m_v: V baseline log frequency
    """
    D, V = dtm.shape
    # Add small constant to avoid zero issues
    dtm_float = dtm.astype(float) + 1e-6
    model = NMF(n_components=K, init='nndsvda', random_state=random_state, max_iter=500)
    W = model.fit_transform(dtm_float)  # D x K
    H = model.components_               # K x V
    # Normalize W to topic proportions per doc
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    theta_init = W / row_sums
    # Baseline m_v as log corpus freq
    corpus_counts = dtm.sum(axis=0)
    m_v = np.log(corpus_counts + 1.0)
    # Initial kappa_t: center log(H)
    logH = np.log(H + 1e-6)
    kappa_t_init = logH - logH.mean(axis=1, keepdims=True)
    return theta_init, kappa_t_init, m_v


def theta_to_alpha_p(theta: np.ndarray) -> np.ndarray:
    """Map simplex theta (D x K) to unconstrained alpha^p (D x K).
    Since softmax is invariant to constant shifts, using log(theta) is acceptable as an inverse up to shift.
    """
    return np.log(theta + 1e-12)
