import numpy as np
from typing import List, Dict, Tuple

"""
Anchor words initializer following separability-based Successive Projections Algorithm (SPA).
This is used as fast deterministic initialization for baseline topic-word distribution \hat{beta}_{k,v}
(Arora et al. 2012/2013). For reproducibility we implement a practical SPA variant:
- Build word co-occurrence matrix Q (row-normalized)
- Run SPA to pick K anchors
- Recover topic-word distributions by nonnegative least squares with simplex constraints

Note: Full robustness enhancements in Arora et al. are complex; SPA is a standard approach
under separability and provides consistent anchors when assumptions hold.
"""

class AnchorInitializer:
    def __init__(self, K: int, vocab_size: int, random_state: int = 42):
        self.K = K
        self.V = vocab_size
        self.random_state = random_state

    @staticmethod
    def build_cooccurrence(dtm: List[Dict[int, int]], V: int) -> np.ndarray:
        # word co-occurrence across documents: Q = sum_d (c_d c_d^T) / sum_d sum_j c_{d,j}
        Q = np.zeros((V, V), dtype=np.float64)
        total_words = 0
        for counts in dtm:
            # sparse vector
            idxs = list(counts.keys())
            vals = np.array([counts[i] for i in idxs], dtype=np.float64)
            total_words += vals.sum()
            # outer product for sparse rows
            for a_pos, i in enumerate(idxs):
                vi = vals[a_pos]
                # diagonal
                Q[i, i] += vi * vi
                for b_pos in range(a_pos + 1, len(idxs)):
                    j = idxs[b_pos]
                    vj = vals[b_pos]
                    v = vi * vj
                    Q[i, j] += v
                    Q[j, i] += v
        if total_words > 0:
            Q /= total_words
        # Row-normalize to sum to 1
        row_sums = Q.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        Q = Q / row_sums
        return Q

    @staticmethod
    def successive_projections(Q: np.ndarray, K: int) -> List[int]:
        # SPA: pick rows with largest norms after orthogonal projections
        V = Q.shape[0]
        anchors: List[int] = []
        R = Q.copy()
        for _ in range(K):
            # pick argmax row norm
            norms = np.linalg.norm(R, axis=1)
            # mask already selected
            for a in anchors:
                norms[a] = -np.inf
            j = int(np.argmax(norms))
            anchors.append(j)
            # project R onto orthogonal complement of row j
            rj = R[j:j+1, :]  # shape (1, V)
            denom = float(np.dot(rj, rj.T))
            if denom <= 1e-12:
                continue
            proj = (R @ rj.T) / denom  # shape (V,1)
            R = R - proj @ rj
        return anchors

    @staticmethod
    def recover_topics(Q: np.ndarray, anchors: List[int]) -> np.ndarray:
        # Represent each row Q[i] as convex combination of anchor rows Q[a_k]
        # Solve NNLS with simplex constraint: weights >=0, sum=1
        K = len(anchors)
        V = Q.shape[0]
        A = Q[anchors, :]  # K x V
        topics = np.zeros((K, V), dtype=np.float64)
        # For numeric stability add small ridge
        ridge = 1e-6
        AtA = A @ A.T + ridge * np.eye(K)
        AtA_inv = np.linalg.inv(AtA)
        At = A.T
        for k in range(K):
            # set topic k initially as anchor row normalized
            vk = A[k].copy()
            s = vk.sum()
            if s > 0:
                vk /= s
            topics[k] = vk
        # Improve by projecting rows onto anchor convex hull and mixing
        # For each word v, compute barycentric coordinates of Q[:,v] with anchors
        # Then use these to set beta_k,v as proportional to anchor mixture.
        Y = Q  # V x V
        for v in range(V):
            y = Y[:, v]  # contributions of word v
            # Solve least squares for weights w in R^K to fit y ≈ sum_k w_k * A_k
            # Closed form: w = (AtA_inv) @ (A @ y)
            rhs = A @ y  # K
            w = AtA_inv @ rhs
            # project onto simplex: clip >=0, normalize
            w = np.maximum(w, 0)
            s = w.sum()
            if s <= 1e-12:
                w = np.ones(K) / K
            else:
                w = w / s
            # set beta_k,v ∝ w_k (later renormalize per topic)
            for k in range(K):
                topics[k, v] = w[k]
        # renormalize topics rows to sum to 1
        topics = np.maximum(topics, 1e-12)
        topics = topics / topics.sum(axis=1, keepdims=True)
        return topics

    def initialize(self, dtm: List[Dict[int, int]]) -> Tuple[np.ndarray, List[int]]:
        Q = self.build_cooccurrence(dtm, self.V)
        anchors = self.successive_projections(Q, self.K)
        beta0 = self.recover_topics(Q, anchors)  # K x V
        return beta0, anchors
