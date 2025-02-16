import functools
import numpy as np
import lib.util as util
import lib.kernels as kernels
import tempfile
import pathlib
import uuid

def linear_predict(w, xs):
    return np.einsum('j, ij->i', w, util.force_arr(xs))

@util.jit_nogil()
def _perceptron_inner(T: int, xys: np.ndarray, out: np.ndarray):
    T = int(T)
    m = int(xys.shape[0])
    lasterr = 0
    for t in range(T):
        t_in_epoch = t % m
        xy_t = xys[t_in_epoch]
        if float(util.numba_dot(xy_t, out)) <= 0:
            out += xy_t
            lasterr = t
        # if we ran through all elements in the training set with no update,
        # we converged.
        elif t - lasterr == m:
            return

def perceptron(params, xs, ys):
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    T = int(params['T'])
    # x_t and y_t always come in the y_t*x_t form, so we can precompute that upfront.
    xys = xs * ys.reshape([-1, 1]) 
    out = np.zeros_like(xs[0])
    _perceptron_inner(T, xys, out)
    return functools.partial(linear_predict, out)

@util.jit_nogil()
def _kernel_perceptron_inner(T, K_xs, ys, alphas):
    m = int(K_xs.shape[0])
    lasterr = 0
    for t in range(T):
        t_in_epoch = t % m
        K_t = K_xs[t_in_epoch]
        # as mentioned in the report, I allow `alphas` to have negative coefficients
        if float(util.numba_dot(alphas, K_t))*ys[t_in_epoch] <= 0:
            alphas[t_in_epoch] += ys[t_in_epoch]
            lasterr = t
        elif t - lasterr == m:
            return

def kernel_perceptron(params, xs, ys):
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    kernel = params['kernel']
    # We could cache this, but it's more trouble than it's worth 
    # given that the iterations dominate the runtime.
    # We're close to the limit beyond which naively storing this matrix in memory
    # is unfeasible, but I will happily apply the optimization if I can.
    K_xs = kernel(xs, xs)
    alphas = util.zeros([K_xs.shape[0]])
    _kernel_perceptron_inner(int(params['T']), K_xs, ys, alphas)
        
    return functools.partial(kernels.kernelized_predict, kernel, alphas, xs)
