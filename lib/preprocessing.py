import functools
import numpy as np
import lib.util as util
import numba

# I represent a learning algorithm as a function A of a params dictionary,
# a matrix of features xs, and a vector of labels ys.

# Whatever variation applied on top of the original algorithm is represented
# as a function which takes the algorithm and its parameters, and possibly
# more arguments before them. I would've rather returned new algorithms,
# but inner functions aren't serializable with `pickle`,
# so instead I do something to a similar effect using `partial`.

# THe preprocessing steps are all very light compared to the wrapped learning algorithm,
# so I will favor expressiveness and simplicity for the grid search over caching of preprocessing steps.

def with_defaults(opts, A, params, xs, ys):
    return A(opts | params, xs, ys)
    
def preprocessed_predict(pp, h, xs):
    return h(pp(xs))

#### Standardization ####

def standardize(mu, sigma, xs, out=None):
    out = np.subtract(xs, mu, out)
    out = np.divide(out, sigma, out, where=np.not_equal(sigma, 0))
    return out

def with_standardization(A, params, xs, ys):
    if params.get('standardize', False):
        # as mentioned in the paper, we learn the standardization parameters
        # from the training set.
        mu = np.mean(xs, axis=0)
        sigma = np.std(xs, axis=0)
        standardizer = functools.partial(standardize, mu, sigma)
        h = A(params, standardizer(xs), ys)
        return functools.partial(preprocessed_predict, standardizer, h)
    return A(params, xs, ys)

#### Feature Removal ####

def keep_features(mask, xs):
    xs = util.force_arr(xs)
    take_all = slice(None)
    out = xs.__getitem__((*[take_all]*(xs.ndim-1), mask))
    return out

def with_feature_removal(A, params, xs, ys):
    xs = util.force_arr(xs)
    if removed := params.get('remove_features', None):
        mask = [d for d in range(xs.shape[1]) if d not in removed]
        removal = functools.partial(keep_features, mask)
        h = A(params, removal(xs), ys)
        return functools.partial(preprocessed_predict, removal, h)
    return A(params, xs, ys)

#### Bias ####

def add_bias(xs):
    xs = util.force_arr(xs)
    return np.concatenate([xs, np.broadcast_to(1, [*xs.shape[:-1], 1])], axis=np.ndim(xs)-1)

def with_bias(A, params, xs, ys):
    if params.get('bias', None):
        h = A(params, add_bias(xs), ys)
        return functools.partial(preprocessed_predict, add_bias, h)
    return A(params, xs, ys)

#### Degree 2 Feature Map ####

@util.jit_nogil()
def _degree2_feature_map_inner(xs: np.ndarray, out: np.ndarray):
    # I implement this rather tha the general case for the sake of simplicity
    # (and also because the third degree feature map is already borderline unusable)
    for r in range(xs.shape[0]):
        x = xs[r]
        o = out[r]
        
        l = x.shape[0]
        for i in range(l):
            o[i] = x[i]*x[i]
        for i in range(x.shape[0]):
            for j in range(i):
                o[l] = x[i]*x[j]
                l += 1
                
def degree2_feature_map(xs, out=None):
    xs = util.force_arr(xs)
    l = xs.shape[1]
    out = out or util.zeros([xs.shape[0], (l*(l+1))//2])
    _degree2_feature_map_inner(xs, out)
    return out

def with_degree2_feature_map(A, params, xs, ys):
    h = A(params, degree2_feature_map(xs), ys)
    return functools.partial(preprocessed_predict, degree2_feature_map, h)

#### Regressor to Classifier ####

def predict_as_binary_classifier(h, xs):
    return np.sign(h(xs))

def regressor_to_binary_classifier(A, params, xs, ys):
    h = A(params, xs, ys)
    return functools.partial(predict_as_binary_classifier, h)
