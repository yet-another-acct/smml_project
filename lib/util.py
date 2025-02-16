import functools
import numpy as np
import numpy.random as np_random
import numba

def ensure_iterable(x):
    if not hasattr(x, '__iter__'):
        return [x]
    return x

def pipeline(*stages):
    # I would've loved it if lambdas were pickleable, but alas.
    [f, *prev_stages] = ensure_iterable(stages[::-1])
    for [p, *args] in map(ensure_iterable, prev_stages):
        f = functools.partial(p, *args, f)
    return f

def merge(*dicts):
    out = {}
    for d in dicts:
        out |= d
    return out

def jit_nogil(**kwargs): 
    return numba.jit(nopython=True, nogil=True, parallel=False, **kwargs)

# the naive implementation by numba turned out to be faster than `np.dot` on my machine,
# but I will stick to the definition by `numpy` for potentially better numerical accuracy.
@jit_nogil()
def numba_dot(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b)

# mostly because of the better numerical accuracy and greater representable maximum value
float_type = np.float64

# ensure that all my arrays are of the right kind

def force_arr(x, dtype=None, order=None): 
    return np.asarray(
        x, 
        dtype= dtype or float_type, 
        order= order or 'C')

def zeros(shape, dtype=None, order=None): 
    return np.zeros(
        shape, 
        dtype= dtype or float_type, 
        order= order or 'C')

def ensure_output(out, shape, dtype=None, order=None):
    if out is None: return zeros(shape, dtype=dtype, order=order)
    out = force_arr(out, dtype=dtype, order=order)
    if (out.ndim != len(shape)): raise ValueError({'expected_ndims': len(shape), 'got': out.ndim})
    return out

def seeded_rng(seed):
    # use the same RNG and the same seed everywhere
    return np_random.Generator(np_random.MT19937(seed))
