#!/usr/bin/env python3
"""
bench_cpu_frozen_tree.py

Benchmarks the CBRNG frozen-tree sampler's CPU path
(cbrng_frozen_tree_sampler_v4.run_cpu) and writes timing JSON in the format
plot_On_scaling.py expects, so the measured CPU curve drops straight into
panels (a) and (b):

    on_timing.json  : per-sample time (us) vs n        (panel a)
    oM_timing.json  : total wall-clock (s) vs M, n=100 & n=1000  (panel b)

Run:
    python bench_cpu_frozen_tree.py                  # default sweep
    python bench_cpu_frozen_tree.py --M 100000 --reps 5
    python bench_cpu_frozen_tree.py --out-prefix cpu   # -> cpu_on_timing.json, cpu_oM_timing.json

Notes
-----
* Calls run_cpu(...) directly (no subprocess), so timing reflects only the
  sampling loop, matching result.seconds from cbrng_frozen_tree_sampler_v4.
* For n < 30, run_cpu also builds a 2**n leaf-count histogram (do_counts=True
  inside the module); this is why small/medium n can look noisier than the
  pure O(n) trend (e.g. the n=24 hiccup noted in plot_On_scaling.py) -- it's
  the count-array allocation, not the sampler itself. We keep this rather
  than patching the sampler, so the benchmark reflects the exact code path
  used elsewhere in the paper.
* A warm-up call at each n (excluded from timing) absorbs first-call effects
  (e.g. numpy import/alloc warm caches) so the reported time is steady-state.
* per_sample_us = wall_time_s / M * 1e6, throughput = M / wall_time_s.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np

from cbrng_frozen_tree_sampler_v4 import run_cpu, default_state_bits


def time_once(n, M, tree_seed, path_seed, state_bits, batch_size, exact_k):
    result = run_cpu(n, M, tree_seed, path_seed, state_bits, batch_size, exact_k)
    return result.seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=1_000_000,
                    help='samples per n in the O(n) sweep')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=100_000)
    ap.add_argument('--tree-seed', type=lambda s: int(s, 0), default=0xC0FFEE)
    ap.add_argument('--path-seed', type=lambda s: int(s, 0), default=0xBAD5EED)
    ap.add_argument('--exact-k', type=int, default=4,
                    help='exact-log2k-threshold passed to split_ratio_cpu')
    ap.add_argument('--out-prefix', default='',
                    help='filename prefix, e.g. "cpu" -> cpu_on_timing.json '
                         '(default "" -> on_timing.json, matching plot_On_scaling.py)')
    args = ap.parse_args()

    prefix = f'{args.out_prefix}_' if args.out_prefix else ''
    seed, pseed = args.tree_seed, args.path_seed

    # ---- O(n) sweep: per-sample time vs n ----
    ns = [4, 6, 8, 12, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500, 750, 1000]
    on = {}
    # global warm-up (import / first-alloc effects)
    time_once(100, 10_000, seed, pseed, default_state_bits(100), args.batch_size, args.exact_k)
    for n in ns:
        state_bits = default_state_bits(n)
        # per-n warm-up, excluded from timing
        time_once(n, min(10_000, args.M), seed, pseed, state_bits, args.batch_size, args.exact_k)
        ts = [time_once(n, args.M, seed, pseed, state_bits, args.batch_size, args.exact_k)
              for _ in range(args.reps)]
        t = float(np.median(ts))
        on[n] = dict(total_s=t, per_sample_us=t / args.M * 1e6,
                     throughput=args.M / t)
        print(f'n={n:5d}  total={t:.4f}s  per-sample={t/args.M*1e6:.4f} us '
              f' thr={args.M/t:.3e}/s')
    json.dump(on, open(f'{prefix}on_timing.json', 'w'), indent=1)

    # ---- O(M) sweep: total time vs M at fixed n ----
    Ms = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    oM = {}
    for n in (100, 1000):
        state_bits = default_state_bits(n)
        time_once(n, 10_000, seed, pseed, state_bits, args.batch_size, args.exact_k)  # warm-up
        row = []
        for M in Ms:
            ts = [time_once(n, M, seed, pseed, state_bits, args.batch_size, args.exact_k)
                  for _ in range(max(1, args.reps // 2))]
            row.append(float(np.median(ts)))
            print(f'n={n} M={M} t={row[-1]:.4f}s')
        oM[n] = dict(M=Ms, t=row)
    json.dump(oM, open(f'{prefix}oM_timing.json', 'w'), indent=1)

    print('wrote', f'{prefix}on_timing.json', 'and', f'{prefix}oM_timing.json')


if __name__ == '__main__':
    main()
