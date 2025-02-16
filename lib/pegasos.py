import functools
import numpy as np
import lib.util as util
import lib.kernels as kernels

def linear_predict(w, xs):
    return np.einsum('j, ij->i', w, util.force_arr(xs))

# avoid allocating a lot of memory for the choices of S
BATCH_SIZE = 50000

@util.jit_nogil()
def _pegasos_inner(
        T: int, lambda_: float, accp: bool,
        s: np.ndarray, t0: int, xys: np.ndarray, 
        Ev_t: np.ndarray, out: np.ndarray):
    T = int(T)
    t = int(t0)
    for i in range(s.shape[0]):
        xy_t = xys[s[i]]
        # Given that the magnitude of the separator is irrelevant, I take the freedom of moving
        # the division by lambda and t from the updates to the comparison and subsequently move it
        # to the right, in the knowledge that the behavior is the same as long as lambda != 0.
        if float(np.dot(Ev_t, xy_t)) <= t*lambda_:
            Ev_t += xy_t
        if accp: out += Ev_t/T
        t += 1

def pegasos(params, xs, ys):
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    T = int(params['T'])
    lambda_ = float(params['lambda'])
    accp = bool(params.get('accp', False))
    g = util.seeded_rng(params.get('seed', None))
    # similarly to perceptron, x_t and y_t always come in the y_t*x_t form, so we precompute it.
    xys = xs * ys.reshape([-1, 1])
    out = np.zeros_like(xs[0])
    # E stands for uppercase sigma here; v_t is the single update
    Ev_t = np.zeros_like(out)
    
    for t0 in range(1, T+1, BATCH_SIZE):
        s = g.integers(0, xs.shape[0], min(T+1-t0, BATCH_SIZE))
        _pegasos_inner(
            T, lambda_, accp,
            s, t0, xys, 
            Ev_t, out)
    return functools.partial(linear_predict, out if accp else Ev_t)

@util.jit_nogil()
def _kernel_pegasos_inner(
        T: int, lambda_: float, accp: bool,
        s: np.ndarray, t0: int, 
        K_xs: np.ndarray, ys: np.ndarray,
        alphas: np.ndarray, out: np.ndarray):
    t = t0
    for i in range(s.shape[0]):
        s_t = s[i]
        # similar considerations to pegasos w.r.t. lambda's placement in the inequality
        # and to kernelized perceptron w.r.t. allowing the alphas to go negative
        if np.dot(alphas, K_xs[s_t])*ys[s_t] <= t*lambda_:
            alphas[s_t] += ys[s_t]
        if accp: out += alphas/T
        t += 1
        
def kernel_pegasos(params, xs, ys):
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    
    kernel = params['kernel']
    T = int(params['T'])
    lambda_ = float(params['lambda'])
    accp = bool(params.get('accp', False))
    g = util.seeded_rng(params.get('seed', None))
    
    # Similar considerations to kernel perceptron in terms of the matrix being precomputed
    K_xs = kernel(xs, xs)
    alphas = util.zeros([K_xs.shape[0]])
    out = np.zeros_like(alphas)
    
    for t0 in range(1, T+1, BATCH_SIZE):
        s = g.integers(0, xs.shape[0], min(T+1-t0, BATCH_SIZE))
        _kernel_pegasos_inner(
            T, lambda_, accp,
            s, t0,
            K_xs, ys,
            alphas, out)
        
    return functools.partial(kernels.kernelized_predict, kernel, out if accp else alphas, xs)

def norm_based_lambda(maybe_kernel, xs):
    # if no kernel is specified, use ||x||^2 = <x, x>
    inner = maybe_kernel or (lambda x, x_: np.dot(x, x_))
    return np.percentile([inner(x, x) for x in xs], 98)

def using_norm_based_lambda(A, params, xs, ys):
    maybe_kernel = params.get('kernel', None)
    Tprime = params["T'"]
    lambda_W_98 = norm_based_lambda(maybe_kernel, xs)
    return A(params | {'lambda': lambda_W_98*np.log(Tprime)/Tprime}, xs, ys)
