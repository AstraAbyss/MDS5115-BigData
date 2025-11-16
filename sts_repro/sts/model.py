import numpy as np
from typing import Dict, List, Tuple

class STSModel:
    def __init__(self, K: int, V: int, ix: int, random_state: int = 0):
        self.K = K
        self.V = V
        self.ix = ix  # number of covariates incl. intercept
        self.rs = np.random.RandomState(random_state)
        # Parameters Γ: (ix) x (2K)
        self.Gamma = self.rs.normal(0, 0.1, size=(ix, 2*K))
        # Σ: 2K x 2K (start diagonal)
        self.Sigma = np.diag(self.rs.gamma(shape=1.0, scale=1.0, size=2*K))
        # Topic-word parameters κ^{(t)} and κ^{(s)}: K x V
        self.kappa_t = np.zeros((K, V))
        self.kappa_s = np.zeros((K, V))
        # Baseline m_v (log word freq) length V
        self.m_v = np.zeros(V)

    def set_baseline_m(self, corpus_counts: np.ndarray):
        # corpus_counts: length V counts
        total = corpus_counts.sum()
        if total <= 0:
            self.m_v = np.zeros_like(corpus_counts, dtype=np.float64)
            return
        p = corpus_counts / total
        p = np.maximum(p, 1e-12)
        # Normalize so sum exp(m_v) = 1 ⇒ m_v = log p_v
        self.m_v = np.log(p)

    @staticmethod
    def softmax(a: np.ndarray) -> np.ndarray:
        z = a - a.max()
        e = np.exp(z)
        return e / e.sum()

    def beta_topic(self, k: int, a_s_k: float) -> np.ndarray:
        # β_{k,v}(a_s_k) = softmax_v( m_v + κ_t[k,v] + κ_s[k,v]*a_s_k )
        z = self.m_v + self.kappa_t[k] + self.kappa_s[k] * a_s_k
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def beta_doc(self, a_s: np.ndarray) -> np.ndarray:
        # returns V x K matrix B_d
        B = np.zeros((self.V, self.K))
        for k in range(self.K):
            b = self.beta_topic(k, a_s[k])
            B[:, k] = b
        return B

    def f_objective(self, c_d: np.ndarray, x_d: np.ndarray, a_d: np.ndarray) -> float:
        # a_d: length 2K: [a_p (K), a_s (K)]
        a_p = a_d[:self.K]
        a_s = a_d[self.K:]
        theta = self.softmax(a_p)
        # S_v = sum_k theta_k * beta_k,v(a_s[k])
        S = np.zeros(self.V)
        for k in range(self.K):
            b = self.beta_topic(k, a_s[k])
            S += theta[k] * b
        S = np.maximum(S, 1e-12)
        ll_words = (c_d * np.log(S)).sum()
        # prior term
        mu = x_d @ self.Gamma  # length 2K
        diff = a_d - mu
        prior = -0.5 * diff.T @ np.linalg.inv(self.Sigma) @ diff
        return ll_words + prior

    def grad_objective(self, c_d: np.ndarray, x_d: np.ndarray, a_d: np.ndarray) -> np.ndarray:
        # Analytical gradient
        a_p = a_d[:self.K]
        a_s = a_d[self.K:]
        theta = self.softmax(a_p)
        # precompute beta and expectations
        beta = np.zeros((self.K, self.V))
        E_kappa = np.zeros(self.K)
        for k in range(self.K):
            b = self.beta_topic(k, a_s[k])
            beta[k] = b
            E_kappa[k] = (b * self.kappa_s[k]).sum()
        S = (theta[:, None] * beta).sum(axis=0)
        S = np.maximum(S, 1e-12)
        # grad wrt a_p
        grad_p = np.zeros(self.K)
        # For each j: sum_v c_dv (1/S_v) * sum_k theta_k (delta_{kj}-theta_j) beta_kv
        for j in range(self.K):
            # compute sum_k theta_k (δ_kj - theta_j) beta_kv
            term = beta[j] * theta[j]  # when k=j, δ_kj=1 ⇒ theta_j * beta_jv
            # subtract theta_j * sum_k theta_k beta_kv = theta_j * S_v
            term -= theta[j] * S
            grad_p[j] = (c_d * (term / S)).sum()
        # grad wrt a_s
        grad_s = np.zeros(self.K)
        for j in range(self.K):
            # ∂β_jv/∂a_s_j = β_jv (κ_s[j,v] - E_kappa[j])
            d_b = beta[j] * (self.kappa_s[j] - E_kappa[j])
            # ∂S_v/∂a_s_j = theta_j * d_b
            term = theta[j] * d_b
            grad_s[j] = (c_d * (term / S)).sum()
        # add prior gradient: ∂prior/∂a = -Σ^{-1}(a - mu)
        mu = x_d @ self.Gamma
        invS = np.linalg.inv(self.Sigma)
        prior_grad = -invS @ (a_d - mu)
        grad = np.concatenate([grad_p, grad_s]) + prior_grad
        return grad

    def hessian_objective(self, c_d: np.ndarray, x_d: np.ndarray, a_d: np.ndarray) -> np.ndarray:
        # Use numerical Hessian (central differences) for robustness on small corpora
        eps = 1e-4
        D = 2 * self.K
        H = np.zeros((D, D))
        base_grad = self.grad_objective(c_d, x_d, a_d)
        for i in range(D):
            a_up = a_d.copy(); a_up[i] += eps
            a_dn = a_d.copy(); a_dn[i] -= eps
            g_up = self.grad_objective(c_d, x_d, a_up)
            g_dn = self.grad_objective(c_d, x_d, a_dn)
            H[:, i] = (g_up - g_dn) / (2*eps)
        # add prior Hessian: -Σ^{-1}
        invS = np.linalg.inv(self.Sigma)
        H += -invS
        return H

    def theta(self, a_p: np.ndarray) -> np.ndarray:
        return self.softmax(a_p)

    def phi_expected(self, c_d: np.ndarray, a_d: np.ndarray) -> np.ndarray:
        # Expected word-topic assignment counts φ_{d,v,k} = c_{d,v} * θ_k * β_{k,v} / S_v
        a_p = a_d[:self.K]
        a_s = a_d[self.K:]
        theta = self.softmax(a_p)
        beta = np.zeros((self.K, self.V))
        for k in range(self.K):
            beta[k] = self.beta_topic(k, a_s[k])
        S = (theta[:, None] * beta).sum(axis=0)
        S = np.maximum(S, 1e-12)
        phi = np.zeros((self.V, self.K))
        for k in range(self.K):
            phi[:, k] = c_d * theta[k] * beta[k] / S
        return phi  # V x K
