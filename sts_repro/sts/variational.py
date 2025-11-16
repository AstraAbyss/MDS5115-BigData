import numpy as np
from typing import List, Dict, Tuple
from .model import STSModel

class VariationalEstimator:
    def __init__(self, model: STSModel, max_iter: int = 50, tol: float = 1e-5):
        self.m = model
        self.max_iter = max_iter
        self.tol = tol

    def optimize_ad(self, c_d: np.ndarray, x_d: np.ndarray, a0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        # alternating updates via quasi-Newton (L-BFGS-like using numerical Hessian)
        a = a0.copy()
        prev_obj = -np.inf
        for it in range(self.max_iter):
            obj = self.m.f_objective(c_d, x_d, a)
            if it > 0 and abs(obj - prev_obj) / (abs(prev_obj) + 1e-9) < self.tol:
                break
            prev_obj = obj
            g = self.m.grad_objective(c_d, x_d, a)
            H = self.m.hessian_objective(c_d, x_d, a)
            # Solve Newton step: H * delta = -g -> delta = -H^{-1} g
            try:
                delta = -np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), g)
            except np.linalg.LinAlgError:
                delta = -g * 1e-2  # fallback small step
            # backtracking line search
            step = 1.0
            for _ in range(10):
                a_new = a + step * delta
                obj_new = self.m.f_objective(c_d, x_d, a_new)
                if obj_new >= obj:
                    a = a_new
                    prev_obj = obj_new
                    break
                step *= 0.5
        # Covariance at mode ≈ -H^{-1}
        H = self.m.hessian_objective(c_d, x_d, a)
        try:
            cov = -np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.eye(len(a)) * 1e-3
        return a, cov, prev_obj

    def e_step_doc(self, c_d: np.ndarray, x_d: np.ndarray, a0: np.ndarray):
        a_opt, cov_opt, obj = self.optimize_ad(c_d, x_d, a0)
        # φ expected counts
        phi = self.m.phi_expected(c_d, a_opt)
        return a_opt, cov_opt, phi, obj
