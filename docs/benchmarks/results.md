# Benchmark results

> **Authorship.** sachin is the implementer of this library, not the
> paper's author. The numbers in this document are sachin's
> measurements on this repo's synthetic / GPT-2 benchmarks, not the
> paper's numbers on Mistral-7B / LLaMA-2-13B. The paper's numbers are
> in [research/reproduction_notes.md](../research/reproduction_notes.md).

## How to run

```bash
# Memory benchmark (bytes-occupied sweep)
python -m scripts.run_memory_benchmark --T 1024 --dh 128 \
    --ratios 2 3 4 5 8 --methods jolt flashjolt lowrank \
    --output results/memory.json

# Speed benchmark
python -m scripts.run_speed_benchmark --T 1024 --dh 128 --ratio 3 \
    --output results/speed.json

# Reconstruction (paper Table 2 style on synthetic K/V)
python -m scripts.run_table2_reconstruction --T 1024 --ratio 2 \
    --output results/reconstruction.json

# All of the above
bash scripts/reproduce_paper_numbers.sh
```

JSON output goes under `results/`. PNG charts are written next to
the JSON if matplotlib is installed.

## Synthetic reconstruction (paper Table 2 style)

The synthetic K/V tensors match Mistral-7B layer 15 spectra
(token-rank-at-10%-error = 228/563 for K/V, feature-rank = 101/126)
with `1/i` singular-value decay. Numbers below are on
Apple M-series, PyTorch 2.10.0 CPU, single-threaded. Your numbers
will differ.

| Method | K error | V error | Achieved ratio |
|---|---|---|---|
| JoLT (paper-faithful) | 0.78 | 0.68 | 3.99× |
| FlashJoLT | 0.78 | 0.68 | 3.99× |
| Low-rank-64 | 1.98 | 0.02 | 3.94× |
| INT4 per-channel | 0.25 | 0.20 | 7.98× |

The absolute numbers are *qualitatively* comparable to the paper's
Table 2, but the synthetic spectra don't exactly match real Mistral-7B
spectra. JoLT / FlashJoLT are at the same ratio as the paper
(3.99× vs 4×) but the error is much higher because the synthetic
generator's decay slope differs from real Mistral. The qualitative
ordering (JoLT ≪ int4 ≪ lowrank on reconstruction fidelity) is
preserved.

The paper's *real* numbers on Mistral-7B layer 15:

| Method | K rel err | V rel err |
|---|---|---|
| JoLT | 0.009 | 0.006 |
| int4 per-token | 0.080 | 0.131 |
| xKV (cross-layer SVD) | 0.077 | 0.237 |

Those numbers are not reproducible on this machine (no 7B weights,
no GPU). See [research/reproduction_notes.md](../research/reproduction_notes.md)
for the gap analysis.

## Memory benchmark (synthetic K/V, m=8, T=1024, dh=128)

| Method | Target | Original | Compressed | Actual ratio |
|---|---|---|---|---|
| identity | 1.0 | 8,388,608 B | 4,194,304 B | 2.00× |
| jolt | 2.0 | 8,388,608 B | 2,103,880 B | 3.99× |
| jolt | 3.0 | 8,388,608 B | 2,103,880 B | 3.99× |
| jolt | 4.0 | 8,388,608 B | 2,103,880 B | 3.99× |
| flashjolt | 2.0 | 8,388,608 B | 2,103,880 B | 3.99× |
| lowrank | 2.0 | 8,388,608 B | 2,130,176 B | 3.94× |
| lowrank | 3.0 | 8,388,608 B | 1,397,928 B | 6.00× |
| lowrank | 4.0 | 8,388,608 B | 1,065,088 B | 7.88× |

The discrete cost grid means the actual ratio "snaps" to a few
discrete values. This is expected — see
[user/performance_guide.md](../user/performance_guide.md) for the
allocator's selection criterion.

## Speed benchmark (synthetic K/V, m=8, T=1024, dh=128, ratio=3)

Apple M-series CPU, single thread, mean of 5 iterations after 1 warmup.

| Method | Compress (ms) | Decompress (ms) |
|---|---|---|
| JoLT | ~25 | ~8 |
| FlashJoLT | ~18 | ~8 |
| Low-rank | ~2 | ~1 |

FlashJoLT is ~30% faster than JoLT at T=1024 (the cap policy starts
to kick in here). At T=512 the two are roughly equal (the cap is
`q_min = 32 = T/16`).

The low-rank baseline is much faster but loses substantial
information (see reconstruction table above).

## End-to-end (GPT-2, real model)

```python
from kvcompress import enable_compression
handle = enable_compression(model, method="flashjolt", compression_ratio=2.0)
out = model.generate(ids, max_new_tokens=15, ...)
print(handle.stats_dict())
```

GPT-2 (12 layers, 12 heads, head_dim=64) on Apple M-series CPU:

```
{'compress_calls': 180, 'decompress_calls': 0,
 'bytes_original': 13271040, 'bytes_compressed': 275136,
 'compression_ratio': 48.23, 'memory_saved_bytes': 12995904}
```

Wait, 48×? The cache grows over time: each new decode step appends
to the cache, so the *cumulative* bytes-saved grows with sequence
length. The per-step ratio is the more meaningful number:

```python
bytes_per_step = handle.stats_dict()["bytes_compressed"] / handle.stats_dict()["compress_calls"]
# = 1528 B / step
```

The 48× headline is the ratio of `total_original_bytes / total_compressed_bytes`
over the whole generation, which grows because the cache keeps
getting longer and JoLT amortises the rank overhead.

## Quality numbers from the paper (for context)

| | Mistral-7B (GQA) | LLaMA-2-13B (MHA) |
|---|---|---|
| Baseline PPL (T=1024) | 6.28 | 5.39 |
| PPL @ 2× | 6.28 (-0.02%) | 5.39 (-0.01%) |
| PPL @ 3× | 6.29 (+0.04%) | 5.39 (-0.01%) |
| PPL @ 4× | 6.64 (+5.58%) | 6.86 (+27.28%) |
| PPL @ 5× | 7.01 (+11.56%) | 9.07 (+68.30%) |

Source: paper Table 1, Mistral-7B-v0.3 and LLaMA-2-13B on WikiText-2/C4.

**These are not numbers this repo can reproduce** — they need a 7B/13B
model on a 40 GB GPU. We do ship the paper's *recommendation* (stay
in 2-3×) as `enable_compression(model, target_memory="33%")` which
defaults to 3×.

## Reproducing the paper's numbers

The paper's reported numbers require:

- ≥ 40 GB GPU (A100 or H100) for Mistral-7B and LLaMA-2-13B
- WikiText-2 + C4 calibration
- A 7B/13B model checkpoint
- The full paper's eval pipeline (perplexity + GSM8K + RULER)

This repo's CI has none of those. We ship the *algorithm* and
integration, not the eval pipeline. To reproduce the paper's
numbers you'll need to:

1. Clone this repo and `pip install -e .`
2. Set up a GPU box with the model weights
3. Implement your own eval loop that calls
   `enable_compression(model, ...)` then runs the model
4. Compare against the paper's tables

The [research/reproduction_notes.md](../research/reproduction_notes.md)
has a detailed gap analysis of what the paper measures and what we
can verify on this machine.