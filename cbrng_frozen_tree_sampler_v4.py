#!/usr/bin/env python3
"""
cbrng_frozen_tree_sampler.py

Hash-free / SHA-free frozen-tree sampler using a stateless counter-style
random-access generator.  The sampler supports n up to 1000 on CPU and, if
CuPy/CUDA is available, on GPU.

Core idea
---------
A frozen split is generated on demand from a node state:

    R_{d,u} = g_K( CBRNG(tree_seed, n, d, node_state(u), stream) )

where K = 2**(n-d-1).  The node state is a wide, well-mixed state that is
updated deterministically when the sampler takes child bit 0 or 1.  Samples that
reach the same prefix have the same node state and therefore the same frozen
R_{d,u}.  No table of R_{d,u} is stored.

This file deliberately avoids SHA.  It uses a SplitMix64-style counter mixer for
portability in NumPy/CuPy.  If you want a standardized CBRNG, replace mix64 with
Philox/Threefry; the surrounding interface is the same.

Usage examples
--------------
CPU, n=20, M=1e6, plot Porter-Thomas diagnostic:
    python cbrng_frozen_tree_sampler.py --engine cpu --n 20 --M 1000000 --plot

GPU, n=20, M=1e6, plot Porter-Thomas diagnostic:
    python cbrng_frozen_tree_sampler.py --engine gpu --n 20 --M 1000000 --plot

CPU, n=1000, M=10000, no full histogram:
    python cbrng_frozen_tree_sampler.py --engine cpu --n 1000 --M 10000

Notes
-----
For n >= 30, the code does not build a 2**n histogram.  It only streams samples
and reports timing/checksums.  For n < 30, it counts leaves and can plot the
scaled empirical probabilities z = 2**n * p_hat(x) against exp(-z).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# uint64 overflow is intentional for counter-based mixing.
np.seterr(over="ignore")

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
C1 = np.uint64(0xBF58476D1CE4E5B9)
C2 = np.uint64(0x94D049BB133111EB)
GOLD = np.uint64(0x9E3779B97F4A7C15)
SALT0 = np.uint64(0xD1B54A32D192ED03)
SALT1 = np.uint64(0xABC98388FB8FAC03)
SALT2 = np.uint64(0x8CB92BA72F3D8DD7)


def _u64(x: int) -> np.uint64:
    return np.uint64(x & 0xFFFFFFFFFFFFFFFF)


def mix64_np(x: np.ndarray | np.uint64) -> np.ndarray | np.uint64:
    """SplitMix64 finalizer; vectorized over uint64 arrays."""
    x = np.asarray(x, dtype=np.uint64) if not isinstance(x, np.uint64) else x
    x = (x ^ (x >> np.uint64(30))) * C1
    x = (x ^ (x >> np.uint64(27))) * C2
    x = x ^ (x >> np.uint64(31))
    return x.astype(np.uint64, copy=False) if isinstance(x, np.ndarray) else np.uint64(x)


def uniform01_from_u64(x: np.ndarray) -> np.ndarray:
    """Convert uint64 words to float64 midpoint uniforms using top 53 bits."""
    j = x >> np.uint64(11)
    return (j.astype(np.float64) + 0.5) * (2.0 ** -53)


def probit_np(u: np.ndarray) -> np.ndarray:
    """Acklam inverse-normal approximation, vectorized."""
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    a = np.array([
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ])
    plow = 0.02425
    phigh = 1.0 - plow
    z = np.empty_like(u, dtype=np.float64)

    lo = u < plow
    hi = u > phigh
    mid = ~(lo | hi)

    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(u[lo]))
        z[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
        z[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        z[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return z


def default_state_bits(n: int, safety_bits: int = 128) -> int:
    """
    Default wide node state.

    For whole-depth collision safety, one wants roughly B >= 2d + safety.
    With deepest internal d=n-1, this is approximately 2n+safety.
    We also enforce a floor of 256 bits.
    """
    return max(256, 2 * n + safety_bits)


def state_words_from_bits(bits: int) -> int:
    return max(1, (bits + 63) // 64)


def init_node_state_cpu(batch: int, words: int, seed: int, n: int) -> np.ndarray:
    """Initial wide state for the root node, shared by all walkers."""
    base = _u64(seed) ^ (GOLD * _u64(n + 1)) ^ SALT0
    root = np.empty(words, dtype=np.uint64)
    for w in range(words):
        root[w] = mix64_np(base + GOLD * _u64(w + 1) + SALT1)
    return np.broadcast_to(root, (batch, words)).copy()


def cbrng_word_cpu(keys: np.ndarray, seed: int, n: int, depth: int, stream: int) -> np.ndarray:
    """
    Random-access word from wide node state and counter fields.

    keys: uint64 array of shape (batch, words).
    stream: independent word/substream index.
    """
    batch, words = keys.shape
    x = np.full(batch, _u64(seed), dtype=np.uint64)
    x ^= GOLD * _u64(n + 0x100000001)
    x ^= SALT0 * _u64(depth + 1)
    x ^= SALT1 * _u64(stream + 1)
    for w in range(words):
        x = mix64_np(x ^ keys[:, w] ^ (GOLD * _u64(w + 17)))
    return mix64_np(x ^ SALT2)


def path_coin_cpu(global_ids: np.ndarray, path_seed: int, n: int, depth: int) -> np.ndarray:
    """Independent Bernoulli uniforms for path sampling."""
    x = global_ids.astype(np.uint64)
    x ^= _u64(path_seed) + GOLD * _u64(depth + 1)
    x ^= SALT0 * _u64(n + 1)
    return uniform01_from_u64(mix64_np(x))


def child_update_cpu(keys: np.ndarray, seed: int, n: int, depth: int, bits: np.ndarray) -> np.ndarray:
    """Update wide node state after taking child bit 0/1.

    This is O(batch * words), not O(batch * words**2): we first fold the
    whole parent state into one 64-bit accumulator, then expand it into the
    requested number of child-state words.
    """
    batch, words = keys.shape
    new_keys = np.empty_like(keys)
    bit64 = bits.astype(np.uint64)

    fold = np.full(batch, _u64(seed), dtype=np.uint64)
    fold ^= SALT0 * _u64(n + 1)
    fold ^= SALT1 * _u64(depth + 1)
    fold ^= GOLD * (bit64 + np.uint64(1))
    for j in range(words):
        fold = mix64_np(fold ^ keys[:, j] ^ (SALT2 * _u64(j + 1)))

    for w in range(words):
        new_keys[:, w] = mix64_np(fold ^ (GOLD * _u64(w + 1)) ^ (SALT2 * _u64(0x100 + w)))
    return new_keys


def split_ratio_cpu(
    keys: np.ndarray,
    seed: int,
    n: int,
    depth: int,
    exact_log2k_threshold: int = 4,
) -> np.ndarray:
    """Generate vector of frozen split ratios R_{d,u}."""
    log2k = n - depth - 1
    batch = keys.shape[0]
    if log2k > 102:
        return np.full(batch, 0.5, dtype=np.float64)

    u = uniform01_from_u64(cbrng_word_cpu(keys, seed, n, depth, stream=0))

    if log2k == 0:
        return np.clip(u, 1e-15, 1.0 - 1e-15)

    if log2k <= exact_log2k_threshold:
        # Exact Beta(K,K) via Gamma(K,1)/(Gamma(K,1)+Gamma(K,1)) for integer K.
        k = 1 << log2k
        gx = np.zeros(batch, dtype=np.float64)
        gy = np.zeros(batch, dtype=np.float64)
        for j in range(k):
            ux = uniform01_from_u64(cbrng_word_cpu(keys, seed, n, depth, stream=1 + j))
            uy = uniform01_from_u64(cbrng_word_cpu(keys, seed, n, depth, stream=1 + k + j))
            gx += -np.log(np.maximum(ux, 1e-300))
            gy += -np.log(np.maximum(uy, 1e-300))
        return np.clip(gx / (gx + gy), 1e-15, 1.0 - 1e-15)

    z = probit_np(u)
    sigma = 1.0 / math.sqrt(4.0 * ((2.0 ** (log2k + 1)) + 1.0))
    return np.clip(0.5 + sigma * z, 1e-15, 1.0 - 1e-15)


@dataclass
class RunResult:
    n: int
    M: int
    engine: str
    seconds: float
    counts: Optional[np.ndarray]
    checksum: int
    state_bits: int
    state_words: int


def run_cpu(
    n: int,
    M: int,
    seed: int,
    path_seed: int,
    state_bits: int,
    batch_size: int,
    exact_log2k_threshold: int,
) -> RunResult:
    if not (1 <= n <= 1000):
        raise ValueError("n must satisfy 1 <= n <= 1000")
    words = state_words_from_bits(state_bits)
    do_counts = n < 30
    counts = np.zeros(1 << n, dtype=np.uint64) if do_counts else None
    checksum = np.uint64(0)

    t0 = time.perf_counter()
    start = 0
    while start < M:
        b = min(batch_size, M - start)
        keys = init_node_state_cpu(b, words, seed, n)
        ids = np.arange(start, start + b, dtype=np.uint64)
        leaves = np.zeros(b, dtype=np.uint32) if do_counts else None
        local_checksum = np.zeros(b, dtype=np.uint64)

        for depth in range(n):
            R = split_ratio_cpu(keys, seed, n, depth, exact_log2k_threshold)
            coin = path_coin_cpu(ids, path_seed, n, depth)
            bits = (coin >= R).astype(np.uint8)
            if do_counts:
                leaves = (leaves << np.uint32(1)) | bits.astype(np.uint32)
            local_checksum ^= cbrng_word_cpu(keys, seed ^ 0xA5A5A5A5A5A5A5A5, n, depth, stream=99)
            keys = child_update_cpu(keys, seed, n, depth, bits)

        checksum ^= np.bitwise_xor.reduce(local_checksum)
        if do_counts:
            counts += np.bincount(leaves, minlength=1 << n).astype(np.uint64)
        start += b

    t1 = time.perf_counter()
    return RunResult(n, M, "cpu", t1 - t0, counts, int(checksum), state_bits, words)


def sample_bitstrings_cpu(
    n: int,
    k: int,
    seed: int,
    path_seed: int,
    state_bits: int,
    exact_log2k_threshold: int,
    start_id: int = 0,
) -> list[str]:
    """Return the first k sampled bitstrings using the same CBRNG rules.

    This is intentionally a small-output helper for inspection/debugging.
    It does not store all M samples; main() calls it only for --show-samples K.
    For GPU runs, this CPU helper still reproduces the same bitstrings because
    the frozen tree and path coins are deterministic functions of the seeds.
    """
    if k <= 0:
        return []
    if not (1 <= n <= 1000):
        raise ValueError("n must satisfy 1 <= n <= 1000")

    words = state_words_from_bits(state_bits)
    keys = init_node_state_cpu(k, words, seed, n)
    ids = np.arange(start_id, start_id + k, dtype=np.uint64)
    bitmat = np.empty((k, n), dtype=np.uint8)

    for depth in range(n):
        R = split_ratio_cpu(keys, seed, n, depth, exact_log2k_threshold)
        coin = path_coin_cpu(ids, path_seed, n, depth)
        bits = (coin >= R).astype(np.uint8)
        bitmat[:, depth] = bits + ord("0")
        keys = child_update_cpu(keys, seed, n, depth, bits)

    return [row.tobytes().decode("ascii") for row in bitmat]


GPU_KERNEL = r'''
extern "C" {

#define MASK64 0xffffffffffffffffULL
#define C1 0xBF58476D1CE4E5B9ULL
#define C2 0x94D049BB133111EBULL
#define GOLD 0x9E3779B97F4A7C15ULL
#define SALT0 0xD1B54A32D192ED03ULL
#define SALT1 0xABC98388FB8FAC03ULL
#define SALT2 0x8CB92BA72F3D8DD7ULL

__device__ unsigned long long mix64(unsigned long long x) {
    x = (x ^ (x >> 30)) * C1;
    x = (x ^ (x >> 27)) * C2;
    x = x ^ (x >> 31);
    return x;
}

__device__ double u01_from_u64(unsigned long long x) {
    unsigned long long j = x >> 11;
    return ((double)j + 0.5) * 1.11022302462515654042e-16; // 2^-53
}

__device__ double probit(double u) {
    if (u < 1e-15) u = 1e-15;
    if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
    const double a0=-3.969683028665376e01, a1=2.209460984245205e02;
    const double a2=-2.759285104469687e02, a3=1.383577518672690e02;
    const double a4=-3.066479806614716e01, a5=2.506628277459239e00;
    const double b0=-5.447609879822406e01, b1=1.615858368580409e02;
    const double b2=-1.556989798598866e02, b3=6.680131188771972e01;
    const double b4=-1.328068155288572e01;
    const double c0=-7.784894002430293e-03, c1=-3.223964580411365e-01;
    const double c2=-2.400758277161838e00, c3=-2.549732539343734e00;
    const double c4=4.374664141464968e00, c5=2.938163982698783e00;
    const double d0=7.784695709041462e-03, d1=3.224671290700398e-01;
    const double d2=2.445134137142996e00, d3=3.754408661907416e00;
    const double plow=0.02425, phigh=0.97575;
    double q, r;
    if (u < plow) {
        q = sqrt(-2.0 * log(u));
        return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) /
               ((((d0*q+d1)*q+d2)*q+d3)*q+1.0);
    }
    if (u > phigh) {
        q = sqrt(-2.0 * log(1.0-u));
        return -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) /
                ((((d0*q+d1)*q+d2)*q+d3)*q+1.0);
    }
    q = u - 0.5; r = q*q;
    return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q /
           (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0);
}

__device__ unsigned long long cbrng_word(
    const unsigned long long key[MAX_WORDS],
    unsigned long long seed,
    int n,
    int depth,
    int stream,
    int words
) {
    unsigned long long x = seed;
    x ^= GOLD * ((unsigned long long)n + 0x100000001ULL);
    x ^= SALT0 * ((unsigned long long)depth + 1ULL);
    x ^= SALT1 * ((unsigned long long)stream + 1ULL);
    for (int w=0; w<words; ++w) {
        x = mix64(x ^ key[w] ^ (GOLD * ((unsigned long long)w + 17ULL)));
    }
    return mix64(x ^ SALT2);
}

__device__ double path_coin(unsigned long long sample_id, unsigned long long path_seed, int n, int depth) {
    unsigned long long x = sample_id;
    x ^= path_seed + GOLD * ((unsigned long long)depth + 1ULL);
    x ^= SALT0 * ((unsigned long long)n + 1ULL);
    return u01_from_u64(mix64(x));
}

__device__ void init_root(unsigned long long key[MAX_WORDS], unsigned long long seed, int n, int words) {
    unsigned long long base = seed ^ (GOLD * ((unsigned long long)n + 1ULL)) ^ SALT0;
    for (int w=0; w<words; ++w) {
        key[w] = mix64(base + GOLD * ((unsigned long long)w + 1ULL) + SALT1);
    }
}

__device__ void update_child(
    unsigned long long key[MAX_WORDS],
    unsigned long long seed,
    int n,
    int depth,
    int bit,
    int words
) {
    unsigned long long old[MAX_WORDS];
    for (int w=0; w<words; ++w) old[w] = key[w];

    unsigned long long fold = seed;
    fold ^= SALT0 * ((unsigned long long)n + 1ULL);
    fold ^= SALT1 * ((unsigned long long)depth + 1ULL);
    fold ^= GOLD * ((unsigned long long)bit + 1ULL);
    for (int j=0; j<words; ++j) {
        fold = mix64(fold ^ old[j] ^ (SALT2 * ((unsigned long long)j + 1ULL)));
    }
    for (int w=0; w<words; ++w) {
        key[w] = mix64(fold ^ (GOLD * ((unsigned long long)w + 1ULL)) ^
                       (SALT2 * (0x100ULL + (unsigned long long)w)));
    }
}

__device__ double split_ratio(
    const unsigned long long key[MAX_WORDS],
    unsigned long long seed,
    int n,
    int depth,
    int words,
    int exact_threshold
) {
    int log2k = n - depth - 1;
    if (log2k > 102) return 0.5;
    double u = u01_from_u64(cbrng_word(key, seed, n, depth, 0, words));
    if (log2k == 0) {
        if (u < 1e-15) u = 1e-15;
        if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
        return u;
    }
    if (log2k <= exact_threshold) {
        int k = 1 << log2k;
        double gx = 0.0, gy = 0.0;
        for (int j=0; j<k; ++j) {
            double ux = u01_from_u64(cbrng_word(key, seed, n, depth, 1+j, words));
            double uy = u01_from_u64(cbrng_word(key, seed, n, depth, 1+k+j, words));
            if (ux < 1e-300) ux = 1e-300;
            if (uy < 1e-300) uy = 1e-300;
            gx += -log(ux);
            gy += -log(uy);
        }
        double r = gx / (gx + gy);
        if (r < 1e-15) r = 1e-15;
        if (r > 1.0 - 1e-15) r = 1.0 - 1e-15;
        return r;
    }
    double z = probit(u);
    double sigma = 1.0 / sqrt(4.0 * (pow(2.0, (double)(log2k + 1)) + 1.0));
    double r = 0.5 + sigma * z;
    if (r < 1e-15) r = 1e-15;
    if (r > 1.0 - 1e-15) r = 1.0 - 1e-15;
    return r;
}

__global__ void sample_kernel(
    int n,
    unsigned long long M_batch,
    unsigned long long global_start,
    unsigned long long tree_seed,
    unsigned long long path_seed,
    int words,
    int exact_threshold,
    int do_counts,
    unsigned int* leaves,
    unsigned long long* checksums
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M_batch) return;
    unsigned long long sample_id = global_start + idx;

    unsigned long long key[MAX_WORDS];
    init_root(key, tree_seed, n, words);
    unsigned int leaf = 0U;
    unsigned long long checksum = 0ULL;

    for (int depth=0; depth<n; ++depth) {
        double R = split_ratio(key, tree_seed, n, depth, words, exact_threshold);
        double coin = path_coin(sample_id, path_seed, n, depth);
        int bit = (coin >= R) ? 1 : 0;
        if (do_counts) {
            leaf = (leaf << 1) | (unsigned int)bit;
        }
        checksum ^= cbrng_word(key, tree_seed ^ 0xA5A5A5A5A5A5A5A5ULL, n, depth, 99, words);
        update_child(key, tree_seed, n, depth, bit, words);
    }
    if (do_counts) leaves[idx] = leaf;
    checksums[idx] = checksum;
}

}
'''


def run_gpu(
    n: int,
    M: int,
    seed: int,
    path_seed: int,
    state_bits: int,
    batch_size: int,
    exact_log2k_threshold: int,
    threads: int = 128,
) -> RunResult:
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GPU engine requires CuPy with CUDA available") from e

    if not (1 <= n <= 1000):
        raise ValueError("n must satisfy 1 <= n <= 1000")
    words = state_words_from_bits(state_bits)
    if words > 64:
        raise ValueError("GPU kernel supports at most 64 uint64 state words; reduce --state-bits")

    code = GPU_KERNEL.replace("MAX_WORDS", str(words))
    kernel = cp.RawKernel(code, "sample_kernel", options=("--std=c++11",))
    do_counts = n < 30
    counts_gpu = cp.zeros(1 << n, dtype=cp.uint64) if do_counts else None
    checksum_host = np.uint64(0)

    # Warm-up: force kernel JIT compilation and CUDA context init BEFORE timing,
    # so the reported time measures sampling throughput, not one-time nvcc/driver
    # startup. CuPy compiles a RawKernel lazily on first launch; without this the
    # first timed batch would absorb several seconds of compilation overhead and
    # make small-M GPU runs look (misleadingly) slower than the CPU.
    _warm_leaves = cp.empty(1, dtype=cp.uint32)
    _warm_checks = cp.empty(1, dtype=cp.uint64)
    kernel(
        (1,),
        (1,),
        (
            np.int32(n),
            np.uint64(1),
            np.uint64(0),
            np.uint64(seed & 0xFFFFFFFFFFFFFFFF),
            np.uint64(path_seed & 0xFFFFFFFFFFFFFFFF),
            np.int32(words),
            np.int32(exact_log2k_threshold),
            np.int32(0),
            _warm_leaves,
            _warm_checks,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    start = 0
    while start < M:
        b = min(batch_size, M - start)
        leaves = cp.empty(b, dtype=cp.uint32) if do_counts else cp.empty(1, dtype=cp.uint32)
        checksums = cp.empty(b, dtype=cp.uint64)
        blocks = (b + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                np.int32(n),
                np.uint64(b),
                np.uint64(start),
                np.uint64(seed & 0xFFFFFFFFFFFFFFFF),
                np.uint64(path_seed & 0xFFFFFFFFFFFFFFFF),
                np.int32(words),
                np.int32(exact_log2k_threshold),
                np.int32(1 if do_counts else 0),
                leaves,
                checksums,
            ),
        )
        if do_counts:
            counts_gpu += cp.bincount(leaves, minlength=1 << n).astype(cp.uint64)
        # CuPy does not implement bitwise_xor.reduce; fold on the host instead.
        # The checksums array is small (one entry per sample in this batch) and
        # the transfer is negligible next to GPU sampling.
        checksum_host ^= np.uint64(np.bitwise_xor.reduce(cp.asnumpy(checksums)))
        start += b
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    counts = cp.asnumpy(counts_gpu) if do_counts else None
    return RunResult(n, M, "gpu", t1 - t0, counts, int(checksum_host), state_bits, words)


def plot_pt(counts: np.ndarray, n: int, M: int, output: str) -> None:
    import matplotlib.pyplot as plt

    D = 1 << n
    z = D * counts.astype(np.float64) / max(M, 1)
    bins = np.linspace(0, 8, 80)
    x = np.linspace(0, 8, 500)
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(z, bins=bins, density=True, alpha=0.65, label="empirical all leaves")
    ax.plot(x, np.exp(-x), linewidth=2.2, label=r"$e^{-z}$")
    ax.set_xlabel(r"$z = 2^n \hat p(x)$")
    ax.set_ylabel("density")
    ax.set_title(f"Frozen-tree CBRNG sampler, n={n}, M={M:,}")
    seen = int((counts > 0).sum())
    ax.text(0.98, 0.98, f"seen {seen:,}/{D:,}", transform=ax.transAxes,
            ha="right", va="top")
    ax.set_xlim(0, 8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def print_summary(result: RunResult) -> None:
    print(f"engine:      {result.engine}")
    print(f"n:           {result.n}")
    print(f"M:           {result.M:,}")
    print(f"state bits:  {result.state_bits} ({result.state_words} x uint64 words)")
    print(f"time:        {result.seconds:.3f} s")
    print(f"checksum:    0x{result.checksum:016x}")
    if result.counts is not None:
        counts = result.counts
        D = counts.size
        z = D * counts.astype(np.float64) / max(int(counts.sum()), 1)
        seen = int((counts > 0).sum())
        print(f"seen leaves: {seen:,}/{D:,} ({seen/D:.2%})")
        print(f"mean z:      {z.mean():.6f}")
        print(f"var z:       {z.var():.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBRNG frozen-tree sampler, CPU/GPU")
    p.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--n", type=int, required=True, help="bitstring length / tree depth, up to 1000")
    p.add_argument("--M", type=int, required=True, help="number of samples")
    p.add_argument("--tree-seed", type=lambda s: int(s, 0), default=0xC0FFEE)
    p.add_argument("--path-seed", type=lambda s: int(s, 0), default=0xBAD5EED)
    p.add_argument("--state-bits", default="auto",
                   help="wide node-state bits. 'auto' uses max(256, 2*n+128).")
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--exact-log2k-threshold", type=int, default=4,
                   help="use exact Gamma-ratio Beta(K,K) for log2(K)<=threshold")
    p.add_argument("--plot", action="store_true", help="plot PT distribution if n < 30")
    p.add_argument("--output", default="cbrng_frozen_tree_pt.png")
    p.add_argument("--gpu-threads", type=int, default=128)
    p.add_argument("--show-samples", type=int, default=0, metavar="K",
                   help="print the first K sampled bitstrings after the run")
    p.add_argument("--samples-output", default=None,
                   help="optional text file to save the shown bitstrings")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.n <= 1000):
        raise SystemExit("n must satisfy 1 <= n <= 1000")
    if args.M < 1:
        raise SystemExit("M must be at least 1")
    if args.show_samples < 0:
        raise SystemExit("--show-samples must be non-negative")
    if args.state_bits == "auto":
        state_bits = default_state_bits(args.n)
    else:
        state_bits = int(args.state_bits)
    if state_bits < 64:
        raise SystemExit("--state-bits must be at least 64")
    if args.engine == "gpu" and state_words_from_bits(state_bits) > 64:
        raise SystemExit("GPU version supports at most 64 uint64 state words")

    if args.engine == "cpu":
        result = run_cpu(
            args.n, args.M, args.tree_seed, args.path_seed,
            state_bits, args.batch_size, args.exact_log2k_threshold,
        )
    else:
        result = run_gpu(
            args.n, args.M, args.tree_seed, args.path_seed,
            state_bits, args.batch_size, args.exact_log2k_threshold,
            threads=args.gpu_threads,
        )

    print_summary(result)

    if args.show_samples:
        k = min(args.show_samples, args.M)
        samples = sample_bitstrings_cpu(
            args.n, k, args.tree_seed, args.path_seed,
            state_bits, args.exact_log2k_threshold, start_id=0,
        )
        print("sampled bitstrings:")
        for i, bits in enumerate(samples):
            print(f"  x[{i}] = {bits}")
        if args.samples_output:
            outdir = os.path.dirname(os.path.abspath(args.samples_output))
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            with open(args.samples_output, "w", encoding="utf-8") as f:
                for i, bits in enumerate(samples):
                    f.write(f"x[{i}] = {bits}\n")
            print(f"saved sampled bitstrings: {args.samples_output}")
    if args.plot:
        if result.counts is None:
            print("PT plot skipped: n >= 30 would require a 2**n count array.")
        else:
            plot_pt(result.counts, args.n, args.M, args.output)
            print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()
