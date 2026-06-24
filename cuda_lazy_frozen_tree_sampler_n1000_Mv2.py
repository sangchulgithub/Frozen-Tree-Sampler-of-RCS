#!/usr/bin/env python3
"""
cuda_lazy_frozen_tree_sampler_n1000.py

CUDA lazy Hurwitz / frozen-tree sampler for large bit length, designed for

    n <= 1000, M ~= 1_000_000

The original uint64-prefix CUDA sampler can only represent n <= 63 because the
prefix itself is stored as a uint64 integer.  This version replaces the explicit
prefix by a deterministic 128-bit rolling prefix hash.  Equal prefixes follow
the same rolling hash, while different prefixes collide with probability about
2^-128.  For M=10^6 and n=1000, this is negligible for sampled-path diagnostics.

Each CUDA thread generates one sample path and its log probability under the
same frozen tree:

    R_u ~ Beta(K,K), K = 2^(n-|u|-1)

Numerical split rule
--------------------
For a node at depth d:

  * If K <= exact_gamma_k_max, draw Beta(K,K) exactly by Gamma(K,1) sums.
  * Else if log2(K) <= max_normal_log2k, use the Gaussian approximation
        R = 1/2 + Normal(0, 1/[4(2K+1)]).
  * Else set R = 1/2 exactly.

The last case is intentional. In float64, for very large K the Haar fluctuation
around 1/2 is below machine precision and cannot affect a double-precision path
probability. This makes n=1000 practical: most shallow levels are exactly fair,
while the deeper levels carry the Porter-Thomas / Dirichlet fluctuations.

Outputs
-------
By default the script stores sampled bitstrings in packed uint64 words:
    packed.shape = (M, ceil(n/64))

For n=1000 and M=1_000_000, this is 128 MB for packed bits, plus about 16 MB for
logp and logZ arrays.

Install
-------
    pip install numba numpy

Examples
--------
Check CUDA:

    python cuda_lazy_frozen_tree_sampler_n1000.py --check

Sample one million 1000-bit strings and save packed output:

    python cuda_lazy_frozen_tree_sampler_n1000.py --n 1000 --M 1000000 \
        --seed 2026 --save-npz samples_n1000_M1e6.npz

Summary only, without copying packed bitstrings back to host:

    python cuda_lazy_frozen_tree_sampler_n1000.py --n 1000 --M 1000000 \
        --no-bits

Preview first 5 sampled bitstrings:

    python cuda_lazy_frozen_tree_sampler_n1000.py --n 1000 --M 1000000 \
        --preview 5
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from numba import cuda, uint64, float64, int32, int64


# ---------------------------------------------------------------------------
# CUDA device RNG: SplitMix64
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def sm64_step(state):
    state = uint64(state + uint64(0x9E3779B97F4A7C15))
    z = state
    z = uint64((z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9))
    z = uint64((z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB))
    z = uint64(z ^ (z >> uint64(31)))
    return state, z


@cuda.jit(device=True)
def u01_from_u64(x):
    y = x >> uint64(11)
    return (float64(y) + 0.5) * 1.1102230246251565e-16


@cuda.jit(device=True)
def rng_u01(state):
    state, x = sm64_step(state)
    return state, u01_from_u64(x)


@cuda.jit(device=True)
def rng_normal(state):
    state, u1 = rng_u01(state)
    state, u2 = rng_u01(state)

    if u1 < 1e-300:
        u1 = 1e-300

    rad = math.sqrt(-2.0 * math.log(u1))
    ang = 6.283185307179586476925286766559 * u2
    return state, rad * math.cos(ang)


# ---------------------------------------------------------------------------
# 128-bit rolling prefix hash
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def empty_prefix_hash(global_seed, n):
    h0 = uint64(global_seed) ^ (uint64(n) * uint64(0xD6E8FEB86659FD93))
    h1 = uint64(global_seed) ^ (uint64(n) * uint64(0xA5A3564E27F886E7))
    h0, z0 = sm64_step(h0)
    h1, z1 = sm64_step(h1 ^ z0)
    return z0, z1


@cuda.jit(device=True)
def child_prefix_hash(h0, h1, depth, bit):
    """
    Deterministically update the prefix hash after appending `bit`.

    This is not a cryptographic proof of injectivity, but as a 128-bit rolling
    hash it makes prefix collisions negligible for M*n sampled node visits.
    """
    b = uint64(bit + 1)
    d = uint64(depth + 1)

    s0 = h0 ^ (h1 + uint64(0x9E3779B97F4A7C15))
    s0 = s0 ^ (d * uint64(0xBF58476D1CE4E5B9))
    s0 = s0 ^ (b * uint64(0x94D049BB133111EB))
    s0, z0 = sm64_step(s0)

    s1 = h1 ^ (z0 + uint64(0xD1B54A32D192ED03))
    s1 = s1 ^ (d * uint64(0xA24BAED4963EE407))
    s1 = s1 ^ (b * uint64(0x9FB21C651E98DF25))
    s1, z1 = sm64_step(s1)

    return z0, z1


@cuda.jit(device=True)
def node_hash(global_seed, n, depth, h0, h1):
    """
    Stateless node RNG seed from global seed, n, depth, and 128-bit prefix hash.
    """
    s = uint64(global_seed)
    s = s ^ (uint64(n) * uint64(0xD6E8FEB86659FD93))
    s = s ^ (uint64(depth + 1) * uint64(0xA5A3564E27F886E7))
    s = s ^ (h0 * uint64(0x9E3779B97F4A7C15))
    s = s ^ (h1 * uint64(0xBF58476D1CE4E5B9))

    s, z = sm64_step(s)
    s, z = sm64_step(z ^ h0)
    s, z = sm64_step(z ^ h1)
    return z


# ---------------------------------------------------------------------------
# Node split: R_u ~ Beta(K,K)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def gamma_integer_k(state, K):
    acc = 0.0
    for _ in range(K):
        state, u = rng_u01(state)
        if u < 1e-300:
            u = 1e-300
        acc += -math.log(u)
    return state, acc


@cuda.jit(device=True)
def beta_symmetric_node(global_seed, n, depth, h0, h1,
                        exact_gamma_k_max, max_normal_log2k):
    """
    Compute deterministic frozen split R_u for node (depth, prefix_hash).

    K = 2^(n-depth-1). Avoid uint64 shifts when log2K is large.
    """
    log2K = n - depth - 1
    state = node_hash(global_seed, n, depth, h0, h1)

    # Exact Gamma-ratio only for small integer K.
    if log2K <= 62:
        K64 = uint64(1) << uint64(log2K)
        if K64 <= uint64(exact_gamma_k_max):
            Ki = int32(K64)
            state, gx = gamma_integer_k(state, Ki)
            state, gy = gamma_integer_k(state, Ki)
            r_exact = gx / (gx + gy)

            if r_exact < 1e-15:
                r_exact = 1e-15
            if r_exact > 1.0 - 1e-15:
                r_exact = 1.0 - 1e-15
            return r_exact

    # For very large K, the fluctuation around 1/2 is below useful float64
    # precision. Skipping the normal RNG is the main n=1000 speedup.
    if log2K > max_normal_log2k:
        return 0.5

    # Normal approximation. Var(Beta(K,K)) = 1/[4(2K+1)] ~ 1/(8K).
    # sigma ~= 2^[-(log2K+3)/2].
    sigma = math.pow(2.0, -0.5 * (float64(log2K) + 3.0))
    state, z = rng_normal(state)
    r = 0.5 + sigma * z

    if r < 1e-15:
        r = 1e-15
    if r > 1.0 - 1e-15:
        r = 1.0 - 1e-15

    return r


# ---------------------------------------------------------------------------
# Main CUDA kernel: one sample per thread
# ---------------------------------------------------------------------------

@cuda.jit
def frozen_tree_n1000_kernel(n, M, n_words, global_seed,
                             exact_gamma_k_max, max_normal_log2k,
                             store_bits, packed, logps, logzs):
    tid = cuda.grid(1)
    if tid >= M:
        return

    # Initialize packed row if requested.
    if store_bits != 0:
        for w in range(n_words):
            packed[tid, w] = uint64(0)

    # Independent path RNG.
    state = uint64(global_seed) ^ (uint64(tid + 1) * uint64(0xD1B54A32D192ED03))
    state, junk = sm64_step(state)
    state = state ^ junk

    h0, h1 = empty_prefix_hash(uint64(global_seed), n)

    lp = 0.0

    for depth in range(n):
        r = beta_symmetric_node(uint64(global_seed), n, depth, h0, h1,
                                exact_gamma_k_max, max_normal_log2k)

        state, u = rng_u01(state)

        bit = 0
        if u < r:
            lp += math.log(r)
            bit = 0
        else:
            lp += math.log(1.0 - r)
            bit = 1

            if store_bits != 0:
                word = depth >> 6
                off = 63 - (depth & 63)
                packed[tid, word] = packed[tid, word] | (uint64(1) << uint64(off))

        h0, h1 = child_prefix_hash(h0, h1, depth, bit)

    logps[tid] = lp
    logzs[tid] = lp + float64(n) * 0.693147180559945309417232121458


# ---------------------------------------------------------------------------
# Host utilities
# ---------------------------------------------------------------------------

def sample_cuda_n1000(
    n: int,
    M: int,
    seed: int = 2026,
    exact_gamma_k_max: int = 64,
    max_normal_log2k: int = 106,
    threads_per_block: int = 128,
    store_bits: bool = True,
):
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your NVIDIA driver and Numba CUDA install.")

    if not (1 <= n <= 1000):
        raise ValueError("n must satisfy 1 <= n <= 1000.")
    if M < 1:
        raise ValueError("M must be positive.")
    if exact_gamma_k_max < 1:
        raise ValueError("exact_gamma_k_max must be positive.")
    if max_normal_log2k < 0:
        raise ValueError("max_normal_log2k must be non-negative.")

    n_words = (n + 63) // 64 if store_bits else 1
    packed_d = cuda.device_array((M, n_words), dtype=np.uint64)
    logps_d = cuda.device_array(M, dtype=np.float64)
    logzs_d = cuda.device_array(M, dtype=np.float64)

    blocks = (M + threads_per_block - 1) // threads_per_block

    frozen_tree_n1000_kernel[blocks, threads_per_block](
        np.int32(n),
        np.int64(M),
        np.int32(n_words),
        np.uint64(seed),
        np.int32(exact_gamma_k_max),
        np.int32(max_normal_log2k),
        np.int32(1 if store_bits else 0),
        packed_d,
        logps_d,
        logzs_d,
    )
    cuda.synchronize()

    logps = logps_d.copy_to_host()
    logzs = logzs_d.copy_to_host()

    if store_bits:
        packed = packed_d.copy_to_host()
    else:
        packed = None

    return packed, logps, logzs


def summarize(logps, logzs, n, M, elapsed):
    z = np.exp(logzs)
    return {
        "n": int(n),
        "M": int(M),
        "entropy_nats_estimate=-mean_logp": float(-np.mean(logps)),
        "entropy_bits_estimate": float(-np.mean(logps) / math.log(2.0)),
        "mean_Z=E[2^n p(X)]": float(np.mean(z)),
        "XEB_like=mean_Z-1": float(np.mean(z) - 1.0),
        "mean_Z2": float(np.mean(z * z)),
        "mean_Z3": float(np.mean(z * z * z)),
        "min_logp": float(np.min(logps)),
        "max_logp": float(np.max(logps)),
        "min_logZ": float(np.min(logzs)),
        "max_logZ": float(np.max(logzs)),
        "wall_time_seconds": float(elapsed),
        "samples_per_second": float(M / max(elapsed, 1e-12)),
    }


def unpack_bitstring(row, n):
    bits = []
    for depth in range(n):
        word = depth >> 6
        off = 63 - (depth & 63)
        bit = (int(row[word]) >> off) & 1
        bits.append("1" if bit else "0")
    return "".join(bits)


def save_npz(path, packed, logps, logzs, n, seed, exact_gamma_k_max, max_normal_log2k):
    payload = {
        "logp": logps,
        "logZ": logzs,
        "n": np.array([n], dtype=np.int32),
        "seed": np.array([seed], dtype=np.uint64),
        "exact_gamma_k_max": np.array([exact_gamma_k_max], dtype=np.int32),
        "max_normal_log2k": np.array([max_normal_log2k], dtype=np.int32),
    }
    if packed is not None:
        payload["packed_bits_u64"] = packed
    np.savez(path, **payload)


def save_preview_txt(path, packed, logps, logzs, n, k):
    if packed is None:
        raise ValueError("Cannot save bitstring preview because --no-bits was used.")
    k = min(k, len(logps))
    with open(path, "w") as f:
        f.write("index bitstring logp logZ Z_2n_p\n")
        for i in range(k):
            b = unpack_bitstring(packed[i], n)
            f.write(f"{i} {b} {float(logps[i])} {float(logzs[i])} {math.exp(float(logzs[i]))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--M", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exact-gamma-k-max", type=int, default=64)
    parser.add_argument(
        "--max-normal-log2k",
        type=int,
        default=106,
        help="Use Gaussian split approximation only for log2(K) <= this value; above it R=0.5 in float64.",
    )
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--no-bits", action="store_true", help="Do not store/copy packed bitstrings; only logp/logZ summaries.")
    parser.add_argument("--save-npz", type=str, default="", help="Save packed_bits_u64, logp, and logZ as an NPZ file.")
    parser.add_argument("--preview", type=int, default=0, help="Print the first K sampled bitstrings.")
    parser.add_argument("--save-preview", type=str, default="", help="Save first --preview bitstrings to a text file.")
    parser.add_argument("--check", action="store_true", help="Print CUDA device info and exit.")
    args = parser.parse_args()

    if args.check:
        print("cuda.is_available():", cuda.is_available())
        if cuda.is_available():
            cuda.detect()
        return

    store_bits = not args.no_bits or bool(args.save_npz) or args.preview > 0 or bool(args.save_preview)

    print("CUDA lazy frozen-tree sampler for n<=1000")
    print("-----------------------------------------")
    print(f"n={args.n}, M={args.M}, seed={args.seed}")
    print(f"exact_gamma_k_max={args.exact_gamma_k_max}")
    print(f"max_normal_log2k={args.max_normal_log2k}")
    print(f"threads={args.threads}")
    print(f"store_bits={store_bits}")
    if store_bits:
        n_words = (args.n + 63) // 64
        print(f"packed_words_per_sample={n_words}")
        print(f"packed_bits_host_bytes≈{args.M * n_words * 8:,}")

    t0 = time.perf_counter()
    packed, logps, logzs = sample_cuda_n1000(
        n=args.n,
        M=args.M,
        seed=args.seed,
        exact_gamma_k_max=args.exact_gamma_k_max,
        max_normal_log2k=args.max_normal_log2k,
        threads_per_block=args.threads,
        store_bits=store_bits,
    )
    t1 = time.perf_counter()

    stats = summarize(logps, logzs, args.n, args.M, t1 - t0)
    for k, v in stats.items():
        print(f"{k}: {v}")

    if args.preview > 0:
        if packed is None:
            print("No bitstrings stored; cannot preview.")
        else:
            print()
            print(f"First {min(args.preview, args.M)} bitstrings:")
            for i in range(min(args.preview, args.M)):
                print(i, unpack_bitstring(packed[i], args.n), float(logps[i]), float(logzs[i]), math.exp(float(logzs[i])))

    if args.save_preview:
        save_preview_txt(args.save_preview, packed, logps, logzs, args.n, args.preview if args.preview > 0 else 10)
        print(f"saved_preview: {args.save_preview}")

    if args.save_npz:
        s0 = time.perf_counter()
        save_npz(args.save_npz, packed, logps, logzs, args.n, args.seed,
                 args.exact_gamma_k_max, args.max_normal_log2k)
        s1 = time.perf_counter()
        print(f"saved_npz: {args.save_npz}")
        print(f"npz_write_seconds: {s1 - s0:.6f}")


if __name__ == "__main__":
    main()
