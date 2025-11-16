import numpy as np
from sts_repro.sts.model import STSModel

# Basic test: shapes and a forward pass

def test_shapes():
    K, V, ix = 3, 50, 2
    m = STSModel(K, V, ix)
    m.set_baseline_m(np.ones(V))
    a_s = np.zeros(K)
    B = m.beta_doc(a_s)
    assert B.shape == (V, K)
    # theta softmax
    th = m.theta(np.zeros(K))
    assert abs(th.sum() - 1.0) < 1e-9
    # f objective
    c_d = np.zeros(V); c_d[:10] = 1
    x_d = np.ones(ix)
    a_d = np.zeros(2*K)
    val = m.f_objective(c_d, x_d, a_d)
    assert np.isfinite(val)

if __name__ == '__main__':
    test_shapes()
    print('basic tests passed')
