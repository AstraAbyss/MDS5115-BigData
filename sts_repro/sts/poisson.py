import numpy as np
from typing import List, Dict, Tuple

class PoissonAggregator:
    def __init__(self, group_size: int = 50, max_iter: int = 10):
        self.group_size = group_size
        self.max_iter = max_iter

    def make_groups(self, D: int) -> List[List[int]]:
        idxs = list(range(D))
        groups = [idxs[i:i+self.group_size] for i in range(0, D, self.group_size)]
        return groups

    def update_kappa(self, kappa_t_k: np.ndarray, kappa_s_k: np.ndarray,
                     phi_d_vk: List[np.ndarray], alpha_s_dk: np.ndarray,
                     m_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For one topic k: estimate κ^{(t)}_{k,v} (intercept) and κ^{(s)}_{k,v} (slope) via Poisson GLM.
        Inputs:
          - phi_d_vk: list over documents of φ_{d, :, k} (V-length expected counts)
          - alpha_s_dk: array length D of α^{(s)}_{d,k}
          - m_v: baseline log freq (length V)
        Returns updated (kappa_t_k, kappa_s_k)
        """
        D = len(phi_d_vk)
        V = len(m_v)
        groups = self.make_groups(D)
        # Initialize α_k for each group
        alpha_group = np.zeros(len(groups))
        # Start with κ_s = current values; iterate aggregation adjustment
        kappa_s = kappa_s_k.copy()
        kappa_t = kappa_t_k.copy()
        for it in range(self.max_iter):
            # compute group counts and offsets
            phi_g_vk = []  # list of (V-length) counts per group
            phi_g_k = []   # total counts per group
            for gi, g in enumerate(groups):
                phi_sum = np.zeros(V)
                total_k = 0.0
                for d in g:
                    phi = phi_d_vk[d]
                    phi_sum += phi
                    total_k += phi.sum()
                phi_g_vk.append(phi_sum)
                phi_g_k.append(total_k)
            phi_g_k = np.array(phi_g_k)
            # update alpha_group per Proposition 2
            # weights u_d depend on current kappa_s; approximate with current values
            for gi, g in enumerate(groups):
                num = 0.0
                den = 0.0
                for d in g:
                    phi = phi_d_vk[d]
                    # Contribution weight ~ φ_{d,k} * exp(κ_s·α_{d,k}) ; approximate by exp(mean κ_s * α)
                    factor = np.clip(np.mean(kappa_s) * alpha_s_dk[d], -10, 10)
                    w_d = phi.sum() * np.exp(factor)
                    num += w_d * alpha_s_dk[d]
                    den += w_d
                alpha_group[gi] = (num / den) if den > 1e-12 else 0.0
            # Fit V independent Poisson regressions with features [1, alpha_group]
            # IRLS for Poisson GLM: log μ = offset + β0 + β1 * α_group
            # Prepare design matrix
            X = np.column_stack([np.ones(len(groups)), alpha_group])  # G x 2
            for v in range(V):
                y = np.array([phi_g_vk[gi][v] for gi in range(len(groups))])  # counts
                offset = np.log(phi_g_k + 1e-12) + m_v[v]
                # initialize beta
                beta = np.array([kappa_t[v], kappa_s[v]])
                for _ in range(10):
                    eta = offset + X @ beta
                    eta = np.clip(eta, -10, 10)
                    mu = np.exp(eta)

                    W = mu  # variance=mu
                    z = eta + (y - mu) / mu
                    # solve weighted least squares: (X^T W X) beta = X^T W z
                    WX = X * W[:, None]
                    XtWX = X.T @ WX + 1e-6*np.eye(2)
                    XtWz = X.T @ (W * z)
                    try:
                        beta_new = np.linalg.solve(XtWX + 1e-8*np.eye(2), XtWz)
                    except np.linalg.LinAlgError:
                        break
                    if np.max(np.abs(beta_new - beta)) < 1e-5:
                        beta = beta_new
                        break
                    beta = beta_new
                kappa_t[v] = beta[0]
                kappa_s[v] = beta[1]
        return kappa_t, kappa_s
