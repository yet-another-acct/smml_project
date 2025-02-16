# why is explicit parallelism with *Python threads* better than implicit parallelism?
import os

import pandas
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import lib.grid_search as grid_search
import lib.util as util
import lib.preprocessing as preprocessing
import lib.perceptron as perceptron
import lib.pegasos as pegasos
import lib.kernels as kernels
import lib.logistic as logistic

def main(max_concurrent, root, dataset):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # you may have to tweak this.
    MAX_CONCURRENT = max_concurrent or os.cpu_count()//2 + 2
    ROOT = root or 'results'

    data = util.force_arr(pandas.read_csv(dataset).to_numpy())

    features = np.ascontiguousarray(data[:, :-1])
    labels = np.ascontiguousarray(data[:, -1])

    def epochs2Ts(epoch_counts): return [len(data)*ec for ec in epoch_counts]

    with_our_preprocessing = [
        preprocessing.with_feature_removal,
        preprocessing.with_standardization,
        preprocessing.with_bias]

    def simple_grid_element(key, elts):
        return [{key: e} for e in elts]
    def simple_grid(*k2es):
        return [simple_grid_element(k, es) for [k, es] in k2es]

    def z1l_5f_cv_grid_search(root, algo, A, grid):
        print(f"==== RUNNING GRID SEARCH FOR {algo} ====")
        return grid_search.cv_grid_search(
            root, algo, 5, np.not_equal,
            A, grid, features, labels, 
            max_concurrent=MAX_CONCURRENT)

    
    def plot_comparison_chart(ax: plt.Axes, xs, ys):
        xs = util.force_arr(xs)
        ys = util.force_arr(ys)
        order = np.argsort(xs, kind='stable')
        ax.plot(xs[order], ys[order])

    def plot_comparison_chart(ax: plt.Axes, xs, ys):
        xs = util.force_arr(xs)
        ys = util.force_arr(ys)
        o = xs.argsort(kind='stable')
        ax.plot(xs[o], ys[o])
        
    def feature_comparisons(features):
        d = features.shape[1]
        [_, axes] = plt.subplots(d, d, squeeze=False)
        for i in range(d):
            xs = features[:, i].squeeze()
            for j in range(d):
                ax: plt.Axes = axes[i, j]
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                ys = features[:, j].squeeze()
                plot_comparison_chart(ax, xs, ys)

    if False: feature_comparisons(features)

    
    def interesting_feature_comparisons(features):
        interesting = [[2, 5], [2, 9], [5, 9]]
        [_, axes] = plt.subplots(len(interesting))
        for i in range(len(interesting)):
            [xf, yf] = interesting[i]
            plot_comparison_chart(axes[i], features[:, xf].squeeze(), features[:, yf].squeeze())
            
    if False: interesting_feature_comparisons(features)

    # there seems to be some weak linear correlation

    
    def search_perceptron():
        A = util.pipeline(
            *with_our_preprocessing,
            preprocessing.regressor_to_binary_classifier,
            perceptron.perceptron)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50, 500, 5000, 50000])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]])
        return z1l_5f_cv_grid_search(
            ROOT, 'perceptron', A, grid)

    search_perceptron()

    
    def search_perceptron_d2fm():
        A =  util.pipeline(
            *with_our_preprocessing,
            preprocessing.with_degree2_feature_map,
            preprocessing.regressor_to_binary_classifier,
            perceptron.perceptron)
        grid = simple_grid(
                ['T', epochs2Ts([5, 500, 50000])],
                ['standardize', [False, True]],
                ['bias', [True]],
                ['remove_features', [None, {5, 9}]])
        return z1l_5f_cv_grid_search(
            ROOT, 'perceptron_d2fm', A, grid)

    search_perceptron_d2fm()

    
    # we will use the very original seed of 420 everywhere randomness is involved to ensure reproducibility.
    def search_pegasos():
        A = util.pipeline(
        [preprocessing.with_defaults, {'seed': 420}],
        *with_our_preprocessing,
        pegasos.using_norm_based_lambda,
        preprocessing.regressor_to_binary_classifier,
        pegasos.pegasos)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50, 500, 5000])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['accp', [False, True]],
            ["T'", epochs2Ts([5, 500, 50000])])
        return z1l_5f_cv_grid_search(
            ROOT, 'pegasos', A, grid)
        
    search_pegasos()

    
    def search_pegasos_d2fm():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            preprocessing.with_degree2_feature_map,
            pegasos.using_norm_based_lambda,
            preprocessing.regressor_to_binary_classifier,
            pegasos.pegasos)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50, 500])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['accp', [False, True]],
            ["T'", epochs2Ts([5, 500, 50000])])
        return z1l_5f_cv_grid_search(
            ROOT, 'pegasos_d2fm', A, grid)

    search_pegasos_d2fm()

    
    def search_kernel_perceptron_poly():
        A = util.pipeline(
            *with_our_preprocessing,
            kernels.with_poly_kernel,
            preprocessing.regressor_to_binary_classifier,
            perceptron.kernel_perceptron)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['degree', [2, 3, 4]])
        z1l_5f_cv_grid_search(
            ROOT, 'kernel_perceptron_poly', A, grid)

    search_kernel_perceptron_poly()

    
    def search_kernel_perceptron_gaussian():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            kernels.using_distance_based_gamma,
            kernels.with_gaussian_kernel,
            preprocessing.regressor_to_binary_classifier,
            perceptron.kernel_perceptron)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50])],
            ['standardize', [False, True]],
            ['bias', [False]], # no bias as it's irrelevant to the Gaussian kernel
            ['remove_features', [None, {5, 9}]],
            ['rho', [0.125, 0.25, 0.5, 1, 2, 4, 8]])
        z1l_5f_cv_grid_search(
            ROOT, 'kernel_perceptron_gaussian', A, grid)

    search_kernel_perceptron_gaussian()

    
    def search_kernel_pegasos_poly():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            kernels.with_poly_kernel,
            pegasos.using_norm_based_lambda,
            preprocessing.regressor_to_binary_classifier,
            pegasos.kernel_pegasos)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50])],
            ['accp', [False, True]],
            ['standardize', [True]], # we amply proved standardization is right w/ poly feature maps
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['degree', [2, 3, 4]],
            ["T'", epochs2Ts([5, 500, 50000])])
        return z1l_5f_cv_grid_search(
            ROOT, 'kernel_pegasos_poly', A, grid)

    search_kernel_pegasos_poly()

    
    def search_kernel_pegasos_gaussian():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            kernels.using_distance_based_gamma,
            kernels.with_gaussian_kernel,
            pegasos.using_norm_based_lambda,
            preprocessing.regressor_to_binary_classifier,
            pegasos.kernel_pegasos)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50])],
            ['accp', [False, True]],
            ['standardize', [False, True]],
            ['bias', [False]],
            ['remove_features', [None, {5, 9}]],
            ['rho', [0.25, 0.5, 1]],
            ["T'", epochs2Ts([5, 500, 50000])])
        z1l_5f_cv_grid_search(
            ROOT, 'kernel_pegasos_gaussian', A, grid)
        
    search_kernel_pegasos_gaussian()

    
    def search_logistic():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            pegasos.using_norm_based_lambda,
            preprocessing.regressor_to_binary_classifier,
            logistic.logistic_regression)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50, 500])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['accp', [False, True]],
            ["T'", epochs2Ts([5, 500, 50000])])
        
        return z1l_5f_cv_grid_search(
            ROOT, 'logistic', A, grid)
        
    search_logistic()

    
    def search_logistic_d2fm():
        A = util.pipeline(
            [preprocessing.with_defaults, {'seed': 420}],
            *with_our_preprocessing,
            preprocessing.with_degree2_feature_map,
            pegasos.using_norm_based_lambda,
            preprocessing.regressor_to_binary_classifier,
            logistic.logistic_regression)
        grid = simple_grid(
            ['T', epochs2Ts([5, 50, 500])],
            ['standardize', [False, True]],
            ['bias', [True]],
            ['remove_features', [None, {5, 9}]],
            ['accp', [False, True]],
            ["T'", epochs2Ts([5, 500, 50000])])
        return z1l_5f_cv_grid_search(
            ROOT, 'logistic_d2fm', A, grid)
        
    search_logistic_d2fm()

