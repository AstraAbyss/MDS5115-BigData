import numpy as np
from typing import List, Dict, Tuple
from .variational import VariationalEstimator
from .poisson import PoissonAggregator

class EMRunner:
    def __init__(self, model, max_em_iter: int = 20, tol: float = 1e-5,
                 group_size: int = 50):
        self.m = model
        self.max_em_iter = max_em_iter
        self.tol = tol
        self.ve = VariationalEstimator(model)
        self.pa = PoissonAggregator(group_size=group_size)
        self.history = []

    def initialize_kappa_with_anchor(self, beta0: np.ndarray):
        # β0: K x V baseline distributions
        # Map to κ_t initial via log-odds relative to m_v
        # κ_t[k,v] ≈ log β0[k,v] - m_v
        for k in range(self.m.K):
            self.m.kappa_t[k] = np.log(np.maximum(beta0[k], 1e-12)) - self.m.m_v
        # κ_s start small random noise
        self.m.kappa_s = np.random.normal(0, 1e-3, size=self.m.kappa_s.shape)

    def run(self, dtm: List[Dict[int, int]], X: np.ndarray):
        D = len(dtm)
        V = self.m.V
        K = self.m.K
        # Initialize a_d for each doc: concatenate a_p (zeros) and a_s (scaled rating or zeros)
        A = np.zeros((D, 2*K))
        # baseline counts
        corpus_counts = np.zeros(V)
        for d in range(D):
            for v, c in dtm[d].items():
                corpus_counts[v] += c
        self.m.set_baseline_m(corpus_counts)
        # EM loop
        last_elbo = -np.inf
        for it in range(self.max_em_iter):
            # E-step: per document optimize a_d and collect φ
            A_opt = np.zeros_like(A)
            Covs = []
            Phis = []
            elbo_sum = 0.0
            for d in range(D):
                # c_d vector
                c_d = np.zeros(V)
                for v, c in dtm[d].items():
                    c_d[v] = c
                a0 = A[d]
                a_opt, cov_opt, phi, obj = self.ve.e_step_doc(c_d, X[d], a0)
                A_opt[d] = a_opt
                Covs.append(cov_opt)
                Phis.append(phi)
                # ELBO_d ≈ f(a*) + 0.5 (log|cov| - log|Σ|)
                try:
                    logdet_cov = np.linalg.slogdet(cov_opt)[1]
                except np.linalg.LinAlgError:
                    logdet_cov = -1e3
                try:
                    logdet_S = np.linalg.slogdet(self.m.Sigma)[1]
                except np.linalg.LinAlgError:
                    logdet_S = -1e3
                elbo_sum += obj + 0.5*(logdet_cov - logdet_S)
            # M-step: update Γ via linear regression, Σ via covariance update
            # Stack regressions for 2K outputs
            # Γ: (ix x 2K) solve least squares X Γ = A_opt
            XtX = X.T @ X
            XtY = X.T @ A_opt
            try:
                Gamma_new = np.linalg.solve(XtX + 1e-8*np.eye(XtX.shape[0]), XtY)
            except np.linalg.LinAlgError:
                Gamma_new = self.m.Gamma
            self.m.Gamma = Gamma_new
            # Σ: average of Covs + residual outer products
            residuals = A_opt - X @ self.m.Gamma
            S_new = np.zeros_like(self.m.Sigma)
            for d in range(D):
                S_new += Covs[d] + np.outer(residuals[d], residuals[d])
            S_new /= D
            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(S_new)
            except np.linalg.LinAlgError:
                S_new = S_new + 1e-3*np.eye(S_new.shape[0])
                eigvals, eigvecs = np.linalg.eigh(S_new)
            eigvals = np.maximum(eigvals, 1e-6)
            self.m.Sigma = (eigvecs * eigvals) @ eigvecs.T
            # Update κ via Poisson aggregation for each topic
            # Prepare φ_d_vk and α_s_dk
            for k in range(K):
                phi_d_vk = [Phis[d][:, k] for d in range(D)]
                alpha_s_dk = A_opt[:, K + k]
                kt_new, ks_new = self.pa.update_kappa(self.m.kappa_t[k].copy(),
                                                      self.m.kappa_s[k].copy(),
                                                      phi_d_vk, alpha_s_dk,
                                                      self.m.m_v.copy())
                self.m.kappa_t[k] = kt_new
                self.m.kappa_s[k] = ks_new
            # Convergence check
            self.history.append(elbo_sum)
            if it > 0 and abs(elbo_sum - last_elbo)/ (abs(last_elbo)+1e-9) < self.tol:
                break
            last_elbo = elbo_sum
            # Update A for next iteration
            A = A_opt
        return A_opt, Phis
