# CUDA-Accelerated Cache Simulator Design-Space Sweep

A cycle-accurate, multi-level cache simulator (LRU replacement, hardware
prefetching via stream buffers) ported from a class-based C++ implementation
to flat, GPU-compatible data structures, and parallelized in CUDA to sweep
hundreds of cache configurations concurrently.

**Result: ~3.6x speedup** over a sequential CPU sweep, achieved not by
adding threads blindly, but by tracking down and fixing several real
GPU-specific bugs and performance bottlenecks along the way.

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

Cache architects routinely need to answer questions like "what's the best
associativity/size/block-size/prefetch-depth combination for this
workload?", which means running the same trace against many candidate
configurations. Each configuration is completely independent of every other
one, which makes a design-space sweep (not the trace replay itself) a
natural candidate for GPU parallelism: one GPU thread per configuration,
each thread replaying the full trace against its own private cache state.

## Project structure

```
cache_sim_flat.h    Core simulator logic (plain C++, used by CPU sweep)
cache_sim_flat.cc   CPU sequential sweep across cache configurations
cache_sim_gpu.h     Same simulator logic, tagged __host__ __device__
cache_sim_gpu.cu    CUDA kernel + GPU sweep (cudaMallocManaged, multi-block launch)
gcc_trace.txt       Sample memory access trace used for all experiments
```

Both `.h` files contain identical simulation logic. The CUDA version is
purely the original logic with `__host__ __device__` annotations added, so
correctness can be directly diffed between CPU and GPU runs on every
configuration, with zero logic duplication.

## How to run

```bash
# CPU baseline
g++ cache_sim_flat.cc -o cache_sim_flat_sweep
./cache_sim_flat_sweep gcc_trace.txt

# GPU version
nvcc --expt-relaxed-constexpr cache_sim_gpu.cu -o cache_sim_gpu_cuda
./cache_sim_gpu_cuda gcc_trace.txt
```

Both programs take a single argument: the trace file. The sweep grid
itself (which associativity, size, block-size, and prefetch-depth values to
test) is defined as arrays inside `main()`, not passed on the command line.
This reflects what the program actually does (run a fixed, predetermined
design-space sweep), rather than simulating one configuration per
invocation.

## Why L2 is not included in the sweep

The underlying simulator (`l1_cache()` / `l2_cache()` in the `.h` files)
fully supports an optional L2 cache with its own associativity, size, and
prefetch configuration. Every `l2_cache(...)` call in `l1_cache()` is
already implemented and guarded by `if (l2_size != 0)`.

The sweep, however, always passes `l2_size = 0`, disabling L2 entirely.
This was a deliberate simplification made early on to keep the first
working sweep tractable, isolating "does the parallel-sweep mechanism work
correctly" from "does every simulator feature work correctly in a
parallel sweep." It was kept for the rest of the experiments because the
L1-only results already demonstrated the project's core findings
(correctness verification methodology, and the GPU-vs-CPU scaling
behavior). It was not dropped for memory or performance reasons; the
sweep's actual memory usage never approached the GPU's capacity limit at
any scale tested (see Methodology below).

Extending the sweep to include L2 would follow exactly the same pattern
already used for L1 and for prefetching: add `l2_size_values[]` and
`l2_assoc_values[]` as additional sweep dimensions, give each GPU thread
its own private L2 arrays (`l2_arr_valid`, `l2_tag_storage`, `l2_arr_dirty`,
`l2_arr_lru`) via the same pointer-offset slicing used for L1, and pass real
`l2_size`/`l2_assoc` values into `l1_cache(...)` instead of hardcoded zeros.
This is a natural next step, not a limitation of the underlying design.

## Methodology

### Correctness-first development

At every stage of this project (flattening the data structures, porting to
CUDA, adding each new sweep dimension) the GPU output was compared
line-by-line against the CPU sweep, for every configuration in the grid,
before any timing number was trusted. This surfaced several real bugs (see
below) that would otherwise have silently produced plausible-looking but
wrong results.

### Two-pass trace loading

The memory trace is read once to count the total number of references
(`total_no_of_ref`), the file pointer is rewound, and a second pass loads
every `(rw, address)` pair into a flat array. This is necessary because GPU
device code cannot call `fscanf()`. The entire trace must exist in memory
(host or, for the GPU version, unified memory) before the kernel launches,
and every thread replays the identical trace sequence against its own
configuration.

### Per-thread private memory via pointer-offset slicing

Rather than allocating one small array per GPU thread (impractical with
hundreds of threads), every per-thread data structure (`l1_arr_valid`,
`l1_tag_storage`, `l1_arr_dirty`, `l1_arr_dirty_addr`, `l1_arr_lru`) is
allocated once, as a single large `cudaMallocManaged` block sized
`num_configs * max_sets * max_assoc`. Each thread computes a
`base = tid * max_sets * max_assoc` offset and works within its own slice
via ordinary pointer arithmetic. The underlying simulator functions have no
awareness that they're operating on a slice rather than a dedicated array.
`max_sets` and `max_assoc` are computed from the largest set count and
associativity that occur anywhere in the sweep grid, so every thread's
slice is large enough for its own configuration's actual needs (smaller
configurations simply leave part of their slice unused; since every access
checks a valid bit before trusting stored data, unused slots never cause
incorrect hits).

### Timing methodology

GPU kernel time was measured with `cudaEvent`s bracketing only the kernel
launch itself (`cudaEventRecord` immediately before and after
`sweep_kernel<<<...>>>`), which excludes CUDA context initialization, file
I/O, and host-side result printing. CPU sweep time was measured with
`std::chrono::high_resolution_clock` around the equivalent sweep loop only.
Both were measured within the same Colab GPU-runtime session, back-to-back,
to avoid comparing across different underlying hardware allocations.

### Memory budget verification

Before running any new, larger sweep grid, the true worst-case per-thread
memory requirement (maximized over every valid combination in the grid, not
just the global `max_sets * max_assoc`) was checked against the actual
per-thread budget, and the total `num_configs * per_thread_size` across all
arrays was checked against the GPU's total memory (16 GB on the T4 used
here). This step was added after an early, oversized sweep grid attempted
to allocate roughly 21.5 GB of unified memory, which did not fail cleanly;
it caused severe page-thrashing that was initially indistinguishable from a
hang.

## Results

| Threads (configs) | Sweep dimensions | CPU time | GPU kernel time | Ratio |
|---|---|---|---|---|
| 6 | assoc | 50.2 ms | 744.8 ms | 14.8x slower |
| 48 | assoc x size | 349.4 ms | 1619.3 ms | 4.6x slower |
| 192 | assoc x size x blocksize | 1455.7 ms | 3180.0 ms | 2.2x slower |
| 448 | assoc x size x blocksize (wider ranges) | 3911.4 ms | 12801.2 ms | 3.3x slower |
| 648 | assoc x size x blocksize x prefetch (N, M) | 9106.8 ms | 2559.3 ms | 3.6x faster |

Correctness: verified identical miss rates (to 4 decimal places) between CPU
and GPU across all configurations at every scale, including prefetch-enabled
cases exercising the stream-buffer hit/miss/eviction logic for the first
time at 648 threads.

## Why 192 to 448 threads made the GPU slower, and why prefetch was the fix

Going from 192 to 448 threads did not come from adding a new kind of work;
it came from widening the existing associativity, size, and block-size
ranges (for example, allowing associativity up to 128, which no real cache
implementation uses). This increased `max_assoc` and, with it, the
per-thread memory footprint (`max_sets * max_assoc`) by roughly 8x, since
every thread's allocated slice has to be large enough for the single
largest configuration in the entire grid, even though most threads never
use most of that space. The result was more total unified memory in play
and a worse GPU/CPU ratio, despite running more threads.

This showed that simply adding more values to the same three dimensions
was the wrong lever to pull: it inflated per-thread memory without adding
any new, meaningful computation per thread. The fix was to add a genuinely
different dimension instead: prefetch depth (`PREF_N`, `PREF_M`). Enabling
prefetching exercises additional code paths inside the simulator
(`sb_hit`, `sb_pos_finder`, `sb_hit_continue`, `sb_miss`) that are skipped
entirely when prefetching is off, so each thread does substantially more
real computation per configuration rather than simply allocating a larger,
mostly-unused block of memory. Combined with keeping associativity and
size within realistic bounds (so per-thread memory stayed small), this is
what produced the first genuine speedup, at 648 threads.

## What the scaling trend shows

Speedup only appeared once more threads were combined with more work per
thread and correctly-bounded per-thread memory. Thread count alone did not
reliably improve the ratio, as the 192 to 448 regression shows.

This points to the underlying reason GPU acceleration is hard for this
particular workload: the simulator has an inherently sequential per-thread
dependency chain. Each memory access mutates LRU/tag/valid state that the
very next access reads, so each GPU thread performs a long, branchy,
dependent computation, rather than the many-independent-short-operations
shape GPUs are built for (dense vector addition, for example). The speedup
that does appear comes from having enough parallel work, of sufficient
substance per thread, to outweigh GPU thread overhead and unified-memory
costs, not from parallelism alone.

## Bugs found and fixed along the way

**1. Floating-point log2 precision divergence (CPU vs GPU)**
`set_tag_bits_cal()` computed the number of set-index bits via
`(int)std::log2(no_of_sets)`. On the GPU, `log2(8.0)` returned something
fractionally under `3.0` (fast-math device intrinsics don't guarantee
bit-exact results vs. host libm), which truncated to `2` instead of `3`,
silently corrupting every set index for that configuration, while CPU and
GPU agreed on 5 out of 6 tested configs at the time, giving a false sense
of correctness from partial verification. Fix:
`(int)std::round(std::log2(...))` instead of relying on truncation.

**2. Single-block kernel launch (`<<<1, N>>>`)**
Left the GPU using only 1 of the T4's roughly 40 streaming
multiprocessors, regardless of thread count. This is directly the issue the
CUDA tutorial's "Out of the Blocks" section addresses. Fixed by computing
block count from thread count (`numBlocks = ceil(N / threadsPerBlock)`),
matching the general `blockIdx.x * blockDim.x + threadIdx.x` indexing
formula already in use.

**3. Unified memory oversubscription**
An early sweep configuration (`max_assoc=128`, wide size range, at high
thread count) requested roughly 21.5 GB of `cudaMallocManaged` memory on a
16 GB T4. Rather than failing cleanly, this caused severe page-thrashing
that looked identical to a hang. Fixed by explicitly computing and checking
per-thread and total memory footprint before launching any new sweep grid,
and keeping the grid within realistic bounds.

**4. Per-thread memory sizing mismatch**
When a new sweep dimension's minimum value changed (adding a smaller block
size to the grid, for example), the constant used to compute the shared
array size (`min_blocksize`) wasn't always updated to match; a silent
source of potential cross-thread memory corruption, since each GPU thread's
private slice of a shared array is computed via pointer offset, and an
undersized slice bleeds into the next thread's data. Caught via manual
worst-case verification before running, not after a bad result appeared.

## Hardware

Tested on Google Colab, NVIDIA T4 GPU (16 GB), CUDA via `nvcc`.

## Acknowledgments

CUDA fundamentals (kernels, unified memory, prefetching, grid-stride loops)
based on NVIDIA's An Even Easier Introduction to CUDA:
https://developer.nvidia.com/blog/even-easier-introduction-cuda/
