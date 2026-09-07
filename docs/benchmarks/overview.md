# Benchmarks

This section covers running `kvcompress`'s benchmarks and interpreting the
output.

## Overview

The benchmarks fall into four groups:

- **Memory** (`benchmarks/memory.py`) — bytes occupied by the compressed
  cache at different ratios.
- **Throughput** (`benchmarks/throughput.py`) — compress / decompress
  wall-time per call.
- **Reconstruction** (`benchmarks/reconstruction.py`) — relative Frobenius
  error vs ratio, mirroring paper Table 2.
- **Long-context** (`scripts/run_needle_haystack.py`) — needle retrieval
  with compression enabled.

## Running

```bash
# Everything at once.
kvcompress benchmark --suite all --output-dir benchmarks/output/

# Just memory.
kvcompress benchmark --suite memory --output-dir benchmarks/output/

# Standalone scripts.
python -m scripts.run_table2_reconstruction --T 1024 --ratio 2.0 \
    --output benchmarks/output/table2.json

python -m scripts.run_memory_benchmark --T 1024 --dh 128 \
    --ratios 2 3 4 5 8 --methods jolt flashjolt lowrank \
    --output benchmarks/output/memory.json

python -m scripts.run_speed_benchmark --T 1024 --dh 128 --ratio 3 \
    --output benchmarks/output/speed.json
```

Outputs are JSON files. Plots (PNG) are generated when matplotlib is
installed (`pip install "kvcompress[bench]"`).

## Interpreting results

### Memory

| Column | Meaning |
|---|---|
| `target` | The compression ratio you asked for. |
| `orig(B)` | Bytes of the uncompressed K/V cache. |
| `compressed(B)` | Bytes occupied by the compressed payload. |
| `actual` | Achieved ratio (orig / compressed). |

For `flashjolt` at 3× you should see ~3× actual. For low-rank, the
actual depends on the rank you chose — the script uses
`rank = max(1, int(min(m·T, dh) / target))` which is approximate.

### Speed

| Column | Meaning |
|---|---|
| `compress_ms` | Wall time for one `comp.compress(K, V)` call. |
| `decompress_ms` | Wall time for one `comp.decompress(kp, vp)` call. |

On CPU, JoLT and FlashJoLT are roughly comparable at small `T` (≤ 256);
FlashJoLT pulls ahead at `T ≥ 1024`. On GPU, FlashJoLT is 5-13× faster
than exact JoLT at matched quality (paper Section 5).

### Reconstruction

| Column | Meaning |
|---|---|
| `K error` / `V error` | Relative Frobenius error of the reconstructed tensor. |
| `ratio` | Achieved compression ratio. |

JoLT should have an order-of-magnitude lower error than int4 at the
same nominal ratio. The synthetic KV generator mimics Mistral-7B's
spectra, so the numbers are qualitatively comparable to the paper's
Table 2.

## Adding a new benchmark

1. Add a module under `benchmarks/` that runs the benchmark and emits a
   list of dicts.
2. Add a `main()` function with `argparse` for CLI invocation.
3. Wire it into `cli.py`'s `benchmark` command.
4. Optionally add a plotter in `benchmarks/plot.py`.
5. Document it here.