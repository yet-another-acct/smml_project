import numpy as np
import math
import functools
import lib.util as util

@util.jit_nogil()
def _gaussian_kernel(gamma, gs: np.ndarray, xs: np.ndarray, out: np.ndarray):
    for i in range(gs.shape[0]):
        g = gs[i]
        for j in range(xs.shape[0]):
            x = xs[j]
            distsq = 0.0
            # compute squared distance
            for l in range(g.shape[0]):
                distsq += (g[l] - x[l])*(g[l] - x[l])
            # and then the rest.
            out[i][j] = math.exp(-gamma*distsq)
            
@util.jit_nogil()
def _poly_kernel(d: float, gs: np.ndarray, xs: np.ndarray, out: np.ndarray):
    for i in range(gs.shape[0]):
        g = gs[i]
        for j in range(xs.shape[0]):
            x = xs[j]
            out[i][j] = 1 + util.numba_dot(g, x)**d
            
def gaussian_kernel(gamma, gs, xs, out=None):
    gs = np.atleast_2d(util.force_arr(gs))
    xs = np.atleast_2d(util.force_arr(xs))
    
    out = out or util.zeros([gs.shape[0], xs.shape[0]])
    _gaussian_kernel(float(gamma), gs, xs, out)
    return out

def using_distance_based_gamma(A, params, xs, ys):
    xs = util.force_arr(xs)
    indices = util.seeded_rng(params.get('seed', None)).permutation(np.arange(xs.shape[0]))
    diffs = xs - xs[indices, :]
    sigma1 = np.mean(np.linalg.norm(diffs, axis=1))
    sigma = params.get('rho', 1) * sigma1
    return A(params | {'gamma': 1/(2*sigma*sigma)}, xs, ys)

def with_gaussian_kernel(A, params, xs, ys):
    params_ = params | {'kernel': functools.partial(gaussian_kernel, params['gamma'])}
    return A(params_, xs, ys)
    
def poly_kernel(d, gs, xs, out=None):
    gs = np.atleast_2d(util.force_arr(gs))
    xs = np.atleast_2d(util.force_arr(xs))
    out = out or util.zeros([gs.shape[0], xs.shape[0]])
    _poly_kernel(float(d), gs, xs, out)
    return out

def with_poly_kernel(A, params, xs, ys):
    params_ = params | {'kernel': functools.partial(poly_kernel, params['degree'])}
    return A(params_, xs, ys)

def kernelized_predict(kernel, alphas, gs, xs):
    return np.einsum("i, ij->j", alphas, kernel(gs, xs))
