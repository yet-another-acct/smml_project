# ad hoc, simple grid-search w/ multithreading
import itertools
import numpy as np
import functools
import lib.util as util
import concurrent.futures as futures
import time
import os
import pickle
import shutil
import pathlib
import tempfile

def run_job(results, errs, job):
    try:
        results.append(job())
    except BaseException as e:
        errs.append(e)
        
def run_jobs(jobs, max_concurrent=None):
    max_concurrent = max_concurrent or os.cpu_count()
    exe = futures.ThreadPoolExecutor(max_workers=max_concurrent)
    todo = [*jobs]
    processing: set[futures.Future] = set()
    errs = []
    results = []

    with futures.ThreadPoolExecutor(max_workers=max_concurrent) as exe:
        try:
            while not errs and (todo or processing):
                to_submit = max_concurrent - len(processing)
                for j in todo[:to_submit]: processing.add(exe.submit(j))
                todo = todo[to_submit:]
                
                [done, processing] = futures.wait(
                    processing, timeout=0.3, return_when=futures.FIRST_EXCEPTION)
                done: list[futures.Future]
                for f in done:
                    if f.exception() or f.cancelled():
                        errs.append(f.exception() or futures.CancelledError("A job was cancelled."))
                    else:
                        results.append(f.result())
                time.sleep(0.3)
        except BaseException as e:
            errs.append(e)
    
    if errs:
        for f in processing: f.cancel()
        raise BaseExceptionGroup("errors in running loop or in jobs", errs)
    return results

def cv_grid_search(root, algo, k, loss, A, grid, xs, ys, max_concurrent=None):
    # about as ad-hoc as you may ever like it to be...
    xs = util.force_arr(xs)
    ys = util.force_arr(ys)
    
    root = pathlib.Path(root)
    if not root.is_dir():
        raise IOError({
            "message": "expected root to be a dir",
            "root": str(root.relative_to(pathlib.Path(".")))})
    root.joinpath(algo).mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_res_path = pathlib.Path(td).joinpath(str(int(time.time())))
        edges = [i*((len(xs)+k-1)//k) for i in range(k)] + [len(xs)]
                
        test_xs = [xs[edges[i]:edges[i+1]] for i in range(k)]
        test_ys = [ys[edges[i]:edges[i+1]] for i in range(k)]
        
        train_xs = [np.concatenate([xs[:edges[i]], xs[edges[i+1]:]]) for i in range(k)]
        train_ys = [np.concatenate([ys[:edges[i]], ys[edges[i+1]:]]) for i in range(k)]
        
        def evaluate_fold(grid_entry, fold, params):
            h = A(params, train_xs[fold], train_ys[fold])
            out = {
                'grid_entry': grid_entry,
                'params': params,
                'fold': fold,
                'test_loss': float(np.mean(loss(test_ys[fold], h(test_xs[fold])))),
                'train_loss': float(np.mean(loss(train_ys[fold], h(train_xs[fold])))),
                'predictor': h}
            entry_path = tmp_res_path.joinpath(f"entry_{grid_entry+1}/")
            entry_path.mkdir(parents=True, exist_ok=True)
            with open(entry_path.joinpath(f"fold_{fold+1}.pkl"), 'wb') as f:
                print(out | {'predictor': "<elided>"})
                pickle.dump(out, f, protocol=4)
            
        jobs = []
        for [grid_entry, param_combination] in enumerate(itertools.product(*grid)):
            params = util.merge(*param_combination)
            for fold in range(k):
                jobs.append(functools.partial(evaluate_fold, grid_entry, fold, params))
        run_jobs(jobs, max_concurrent=max_concurrent)
        # ensure that the results are written to the destination only on success
        shutil.move(tmp_res_path, root.joinpath(algo))
        return None

