#!/usr/bin/env python
"""Microbenchmark for CMIknn shuffle surrogate parallelization.

Run from the repository root:

    PYTHONPATH=. python scripts/benchmark_cmiknn_shuffle.py

This script is not part of the pytest suite; it prints timing results for
PR documentation.
"""

from __future__ import print_function

import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tigramite.independence_tests.cmiknn import CMIknn


def _time_shuffle(array, xyz, seed, sig_samples, n_jobs, workers=1):
    cmi = CMIknn(
        seed=seed,
        knn=0.2,
        sig_samples=sig_samples,
        shuffle_neighbors=5,
        n_jobs=n_jobs,
        workers=workers,
        verbosity=0,
    )
    value = cmi.get_dependence_measure(array, xyz)
    start = time.perf_counter()
    cmi.get_shuffle_significance(array, xyz, value)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    seed = 42
    sig_samples = 500
    sample_sizes = [500, 2000]

    print("CMIknn shuffle surrogate benchmark")
    print("sig_samples = %d" % sig_samples)
    print("-" * 60)

    for T in sample_sizes:
        dim = 4
        array = np.random.default_rng(seed).standard_normal((dim, T))
        xyz = np.array([0, 1, 2, 2])

        serial_time = _time_shuffle(array, xyz, seed, sig_samples, n_jobs=1)
        parallel_time = _time_shuffle(array, xyz, seed, sig_samples, n_jobs=-1)

        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        print("T = %d" % T)
        print("  serial (n_jobs=1):   %.3f s" % serial_time)
        print("  parallel (n_jobs=-1): %.3f s" % parallel_time)
        print("  speedup:             %.2fx" % speedup)
        print()


if __name__ == "__main__":
    main()
