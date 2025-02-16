import functools
import numpy as np
import lib.util as util

def linear_predict(w, xs):
    return np.einsum('j, ij->i', w, util.force_arr(xs))

@util.jit_nogil()
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

BATCH_SIZE = 50000

@util.jit_nogil()
def _logistic_regression_inner(
        T: int, lambda_: float,
        s: np.ndarray, t0: int, accp: bool,
        xys: np.ndarray, 
        w: np.ndarray, 
        out: np.ndarray):
    T = int(T)
    t = int(t0)
    for i in range(s.shape[0]):
        y_tx_t = xys[s[i]]
        w *= (1 - 1/t)
        w += (1/(t*lambda_) * _sigmoid(-util.numba_dot(w, y_tx_t))) * y_tx_t
        if accp: out += w / T
        t += 1

def logistic_regression(params, xs, ys):
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    T = int(params['T'])
    lambda_ = float(params['lambda'])
    g = util.seeded_rng(params.get('seed', None))
    
    es = xs * ys.reshape([-1, 1])
    # E stands for uppercase sigma
    w = util.zeros([es.shape[1]])
    out = np.zeros_like(w)
    
    for t0 in range(1, T+1, BATCH_SIZE):
        s = g.integers(0, xs.shape[0], min(T+1-t0, BATCH_SIZE))
        _logistic_regression_inner(
            T, lambda_,
            s, t0, params['accp'],
            es, w, out)
    return functools.partial(linear_predict, w if not params['accp'] else out)
