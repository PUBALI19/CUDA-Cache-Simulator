# CUDA Cache Simulator

A cycle-accurate, multi-level cache simulator (LRU replacement, hardware
prefetching via stream buffers) ported from a class-based C++ implementation
to flat, GPU-compatible data structures — and parallelized in CUDA to sweep
hundreds of cache configurations concurrently.

**Result: ~3x speedup** over a sequential CPU sweep, achieved not by adding
threads blindly, but by tracking down and fixing several real GPU-specific
bugs and performance bottlenecks along the way.

---

## What this simulates

A configurable L1 (optionally L2) set-associative cache with:
- Configurable block size, cache size, associativity
- LRU replacement policy
- Hardware stream-buffer prefetching (configurable depth `PREF_N` / `PREF_M`)
- Full hit/miss/write-back/prefetch statistics, matching a standard
  cache-simulator assignment spec

Given a memory access trace (`r`/`w` + address per line), it reports miss
rates, write-backs, prefetch effectiveness, and memory traffic for a given
configuration.

## Why parallelize this

Cache architects routinely need to answer questions like *"what's the best
associativity/size/block-size/prefetch-depth combination for this
workload?"* — which means running the same trace against many candidate
configurations. Each configuration is completely independent of every other
one, which makes a **design-space sweep** (not the trace replay itself) a
natural candidate for GPU parallelism: one GPU thread per configuration,
each thread replaying the full trace against its own cache state.

## Project structure

```
cache_sim_flat.h    Core simulator logic (plain C++, used by CPU sweep)
cache_sim_flat.cc   CPU sequential sweep across cache configurations
cache_sim_gpu.h     Same simulator logic, tagged __host__ __device__
cache_sim_gpu.cu    CUDA kernel + GPU sweep (cudaMallocManaged, multi-block launch)
```

Both `.h` files contain identical simulation logic — the CUDA version is
purely the original logic with `__host__ __device__` annotations added, so
correctness can be directly diffed between CPU and GPU runs on every config.

## How to run

```bash
# CPU baseline
g++ cache_sim_flat.cc -o cache_sim_flat
./cache_sim_flat <trace_file>

# GPU version
nvcc --expt-relaxed-constexpr cache_sim_gpu.cu -o cache_sim_gpu_cuda
./cache_sim_gpu_cuda <trace_file>
```

Note: the sweep itself iterates over internally-defined arrays of
`assoc`/`size`/`blocksize`/`pref_n`/`pref_m` values, not the CLI arguments
(only the trace file argument is actually used by the sweep — the rest are
vestigial from an earlier single-config version and kept only to satisfy the
argument count check).

## Methodology

Every GPU result was validated against the CPU sweep **line-by-line, for
every configuration**, before any timing number was trusted. This caught
several real bugs before they could silently corrupt results (see below).

Kernel timing was measured with `cudaEvent`s bracketing just the kernel
launch (excludes CUDA context init, file I/O, and host-side result
printing). CPU timing was measured with `std::chrono::high_resolution_clock`
around just the sweep loop. Both were run in the same Colab GPU-runtime
session for a fair comparison.

## Results

| Threads (configs) | Sweep dimensions | CPU time | GPU kernel time | Ratio |
|---|---|---|---|---|
| 6 | assoc | 50.1 ms | 761.4 ms | 15.2x slower |
| 48 | assoc × size | 341.4 ms | 1673.9 ms | 4.9x slower |
| 192 | assoc × size × blocksize | 1773.3 ms | 3109.1 ms | 1.75x slower |
| 648 | assoc × size × blocksize × prefetch (N, M) | 8736.4 ms | 2898.7 ms | **3.0x faster** |

Correctness: verified identical miss rates (to 4 decimal places) between CPU
and GPU across all 648 configurations, including prefetch-enabled cases
exercising the stream-buffer hit/miss/eviction logic.

## Bugs found and fixed along the way

**1. Floating-point `log2` precision divergence (CPU vs GPU)**
`set_tag_bits_cal()` computed the number of set-index bits via
`(int)std::log2(no_of_sets)`. On the GPU, `log2(8.0)` returned something
fractionally under `3.0` (fast-math device intrinsics don't guarantee
bit-exact results vs. host `libm`), which truncated to `2` instead of `3`,
silently corrupting every set index for that configuration — while CPU and
GPU agreed on 5 out of 6 tested configs, giving a false sense of
correctness. **Fix:** `(int)std::round(std::log2(...))` instead of relying
on truncation.

**2. Single-block kernel launch (`<<<1, N>>>`)**
Left the GPU using only 1 of the T4's ~40 streaming multiprocessors,
regardless of thread count — directly the issue the CUDA tutorial's
"Out of the Blocks" section addresses. Fixed by computing block count from
thread count (`numBlocks = ceil(N / threadsPerBlock)`), matching the
general `blockIdx.x * blockDim.x + threadIdx.x` indexing formula.

**3. Unified memory oversubscription**
An early sweep configuration (`max_assoc=128`, wide size range) requested
~21.5 GB of `cudaMallocManaged` memory on a 16 GB T4. Rather than failing
cleanly, this caused severe page-thrashing that looked identical to a
hang. Fixed by keeping per-thread memory footprint (`max_sets * max_assoc`)
within realistic bounds for the sweep grid.

**4. Per-thread memory sizing mismatch**
When adding a new sweep dimension, the `min_blocksize` constant used to
compute the shared array size wasn't updated to match the actual minimum
value in the new array — a silent source of potential cross-thread memory
corruption (each GPU thread's private "slice" of a shared array is
computed via pointer offset, so an undersized slice bleeds into the next
thread's data). Caught via manual worst-case verification before running,
not after.

## What the scaling trend shows

Speedup only appeared at 648 threads, and only after combining **more
threads** with **more work per thread** (prefetch logic exercises
additional code paths — `sb_hit`, `sb_pos_finder`, `sb_hit_continue`) and
**correctly-bounded per-thread memory**. Thread count alone (192 → 448 at
similar per-thread footprint) did not reliably improve the ratio — in one
case, widening the associativity/size range (192 → 448 configs) *increased*
per-thread memory footprint 8x and made the ratio worse, despite more
threads.

This points to the underlying reason GPU acceleration is hard here: this
simulator has an inherently **sequential per-thread dependency chain** —
each memory access mutates LRU/tag/valid state that the next access reads —
so each GPU thread does a long, branchy, dependent computation rather than
GPUs' ideal case of many independent, uniform, short operations (e.g. dense
vector addition). The speedup that does show up comes from having enough
parallel *and* sufficiently substantial work to outweigh GPU thread
overhead and unified-memory costs — not from parallelism alone.

## Hardware

Tested on Google Colab, NVIDIA T4 GPU (16 GB), CUDA via `nvcc`.

## Acknowledgments

CUDA fundamentals (kernels, unified memory, prefetching, grid-stride loops)
based on NVIDIA's [*An Even Easier Introduction to CUDA*](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).
