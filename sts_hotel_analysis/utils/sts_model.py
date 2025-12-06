import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.optimize import minimize
import statsmodels.api as sm

class STSModel:
    def __init__(self, K: int, vocab: List[str], X: np.ndarray, dtm: np.ndarray, m_v: np.ndarray,
                 kappa_t_init: np.ndarray, kappa_s_init: np.ndarray, alpha_p_init: np.ndarray, alpha_s_init: np.ndarray):
        """
        Args:
          K: number of topics
          vocab: list of V terms
          X: (D x (ix+1)) covariate matrix including intercept
          dtm: (D x V) document-term counts
          m_v: (V,) baseline log frequency
          kappa_t_init: (K x V) initial topic baseline coeffs
          kappa_s_init: (K x V) initial topic sentiment-discourse coeffs
          alpha_p_init: (D x K) initial a^(p)
          alpha_s_init: (D x K) initial a^(s)
        """
        self.K = K
        self.vocab = vocab
        self.V = len(vocab)
        self.D = dtm.shape[0]
        self.X = X
        self.dtm = dtm
        self.m_v = m_v
        self.kappa_t = kappa_t_init.copy()
        self.kappa_s = kappa_s_init.copy()
        self.alpha_p = alpha_p_init.copy()
        self.alpha_s = alpha_s_init.copy()
        if self.alpha_p.shape[0] != self.D:
            self.alpha_p = self.alpha_p[:self.D,:]
        if self.alpha_s.shape[0] != self.D:
            self.alpha_s = self.alpha_s[:self.D,:]
        if self.X.shape[0] != self.D:
            self.X = self.X[:self.D,:]
        # Regression coefficients (ix+1 x 2K)
        self.Gamma = np.zeros((X.shape[1], 2*K))
        # Covariance Sigma (2K x 2K)
        self.Sigma = np.eye(2*K) * 0.5
        # Per-doc posterior covariances (Laplace approx inverse Hessian)
        self.Upsilon_list = [np.eye(2*K) for _ in range(self.D)]
        # Cache expected assignments
        self.phi_dvk = None  # (D x V x K)

    @staticmethod
    def softmax(a: np.ndarray) -> np.ndarray:
        z = a - np.max(a)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12)

    def _beta_topic(self, a_s_k: float, k: int) -> np.ndarray:
        # β_{k,v}(a_s_k) for given topic k across v
        logits = self.m_v + self.kappa_t[k, :] + self.kappa_s[k, :] * a_s_k
        # Softmax over vocabulary v
        b = np.exp(logits - np.max(logits))
        b = b / (np.sum(b) + 1e-12)
        return b  # shape (V,)

    def _compute_phi_expected(self, d: int, theta_d: np.ndarray, beta_kv_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Returns phi_dvk (V x K) and phi_dk (K,)
        c = self.dtm[d, :]  # (V,)
        mix = np.zeros(self.V)
        for k in range(self.K):
            mix += theta_d[k] * beta_kv_list[k]
        mix = np.clip(mix, 1e-12, None)
        phi_dvk = np.zeros((self.V, self.K))
        for k in range(self.K):
            # φ_{d,v,k} = c_{d,v} * θ_k β_kv / sum_j θ_j β_jv
            phi_dvk[:, k] = c * (theta_d[k] * beta_kv_list[k] / mix)
        phi_dk = phi_dvk.sum(axis=0)
        return phi_dvk, phi_dk

    def _f_doc(self, a: np.ndarray, d: int) -> float:
        # a: (2K,) vector for doc d
        a_p = a[:self.K]
        a_s = a[self.K:]
        theta = self.softmax(a_p)
        beta_list = [self._beta_topic(a_s[k], k) for k in range(self.K)]
        c = self.dtm[d, :]
        mix = np.zeros(self.V)
        for k in range(self.K):
            mix += theta[k] * beta_list[k]
        mix = np.clip(mix, 1e-12, None)
        ll = np.sum(c * np.log(mix))
        # Prior term (Normal with mean X_d Gamma and cov Sigma)
        mu = self.X[d, :].dot(self.Gamma)
        diff = a - mu
        try:
            Sigma_inv = np.linalg.inv(self.Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(self.Sigma)
        prior_quad = -0.5 * diff.T.dot(Sigma_inv).dot(diff)
        return ll + prior_quad

    def e_step(self, maxiter: int = 200) -> None:
        ups_list = []
        phi_all = np.zeros((self.D, self.V, self.K))
        for d in range(self.D):
            a0 = np.concatenate([self.alpha_p[d, :], self.alpha_s[d, :]])
            res = minimize(lambda a: -self._f_doc(a, d), a0, method='BFGS', options={'maxiter': maxiter})
            a_opt = res.x
            # Update alpha_p and alpha_s
            self.alpha_p[d, :] = a_opt[:self.K]
            self.alpha_s[d, :] = a_opt[self.K:]
            # Hessian inverse approximation from BFGS
            try:
                Hinv = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else res.hess_inv.todense()
            except Exception:
                Hinv = np.eye(2*self.K)
            # Upsilon = - Hessian^{-1}
            ups_list.append(-np.array(Hinv))
            # Expected assignments φ
            theta = self.softmax(self.alpha_p[d, :])
            beta_list = [self._beta_topic(self.alpha_s[d, k], k) for k in range(self.K)]
            phi_dvk, _ = self._compute_phi_expected(d, theta, beta_list)
            phi_all[d, :, :] = phi_dvk
        self.Upsilon_list = ups_list
        self.phi_dvk = phi_all

    def m_step_update_gamma_sigma(self) -> Dict[str, np.ndarray]:
        # Update Gamma by OLS for each of 2K dimensions
        D, P = self.X.shape
        Y = np.hstack([self.alpha_p, self.alpha_s])  # D x 2K
        Gamma = np.zeros((P, 2*self.K))
        se = np.zeros((P, 2*self.K))
        pvals = np.zeros((P, 2*self.K))
        for j in range(2*self.K):
            model = sm.OLS(Y[:, j], self.X)
            res = model.fit()
            Gamma[:, j] = res.params
            se[:, j] = res.bse
            pvals[:, j] = res.pvalues
        self.Gamma = Gamma
        # Update Sigma
        Sigma = np.zeros((2*self.K, 2*self.K))
        for d in range(D):
            mu = self.X[d, :].dot(self.Gamma)
            diff = np.concatenate([self.alpha_p[d, :], self.alpha_s[d, :]]) - mu
            Sigma += self.Upsilon_list[d] + np.outer(diff, diff)
        Sigma = Sigma / D
        # Regularize to ensure positive definiteness
        w, V = np.linalg.eigh(Sigma)
        w_reg = np.clip(w, 1e-6, None)
        Sigma = (V * w_reg).dot(V.T)
        self.Sigma = Sigma
        return {'Gamma': Gamma, 'Sigma': Sigma, 'Gamma_se': se, 'Gamma_p': pvals}

    def m_step_update_kappa(self, maxiter_per_word: int = 100, aggregate: bool = False, group_size: int = 10) -> Dict[str, np.ndarray]:
        # Update kappa_t and kappa_s via Poisson GLM per topic & word
        D = self.D
        kappa_t_new = np.zeros_like(self.kappa_t)
        kappa_s_new = np.zeros_like(self.kappa_s)
        se_s = np.zeros_like(self.kappa_s)
        # Precompute per doc theta & phi totals per topic
        theta_all = np.zeros((D, self.K))
        beta_all = [[None]*self.K for _ in range(D)]
        phi_dk_all = np.zeros((D, self.K))
        for d in range(D):
            theta = self.softmax(self.alpha_p[d, :])
            theta_all[d, :] = theta
            beta_list = [self._beta_topic(self.alpha_s[d, k], k) for k in range(self.K)]
            beta_all[d] = beta_list
            _, phi_dk = self._compute_phi_expected(d, theta, beta_list)
            phi_dk_all[d, :] = np.clip(phi_dk, 1e-9, None)
        # Iterate topics
        for k in range(self.K):
            a_s_k = self.alpha_s[:, k]
            for v in range(self.V):
                # Response y_d = φ_{d,v,k}
                y = self.phi_dvk[:, v, k]
                # Offset = log(φ_{d,k}) + m_v
                offset = np.log(phi_dk_all[:, k] + 1e-9) + self.m_v[v]
                X_po = np.column_stack([np.ones(D), a_s_k])
                try:
                    model = sm.GLM(y, X_po, family=sm.families.Poisson(), offset=offset)
                    res = model.fit(maxiter=maxiter_per_word, disp=0)
                    params = res.params
                    cov = res.cov_params()
                    kappa_t_new[k, v] = params[0]
                    kappa_s_new[k, v] = params[1]
                    se_s[k, v] = np.sqrt(cov[1,1]) if cov is not None else np.nan
                except Exception:
                    # Fallback: keep old params
                    kappa_t_new[k, v] = self.kappa_t[k, v]
                    kappa_s_new[k, v] = self.kappa_s[k, v]
                    se_s[k, v] = np.nan
        self.kappa_t = kappa_t_new
        self.kappa_s = kappa_s_new
        return {'kappa_t': kappa_t_new, 'kappa_s': kappa_s_new, 'kappa_s_se': se_s}

    def compute_elbo(self) -> float:
        # Approximate ELBO = sum_d [ f(a*) + 0.5 (log|Upsilon_d| - log|Sigma|) ]
        total = 0.0
        try:
            signS, logdetS = np.linalg.slogdet(self.Sigma)
            if signS <= 0:
                logdetS = np.log(np.abs(np.linalg.det(self.Sigma)) + 1e-9)
        except Exception:
            logdetS = 0.0
        for d in range(self.D):
            a = np.concatenate([self.alpha_p[d, :], self.alpha_s[d, :]])
            fval = self._f_doc(a, d)
            try:
                signU, logdetU = np.linalg.slogdet(self.Upsilon_list[d])
                if signU <= 0:
                    logdetU = np.log(np.abs(np.linalg.det(self.Upsilon_list[d])) + 1e-9)
            except Exception:
                logdetU = 0.0
            total += fval + 0.5 * (logdetU - logdetS)
        return float(total)

    def top_words(self, topn: int = 10, a_s_level: float = 0.0) -> List[List[Tuple[str, float]]]:
        # Return top words per topic for given a_s level
        out = []
        for k in range(self.K):
            beta = self._beta_topic(a_s_level, k)
            idxs = np.argsort(beta)[::-1][:topn]
            out.append([(self.vocab[i], float(beta[i])) for i in idxs])
        return out
