# Running benchmarks

## Prerequisites

```bash
pip install "kvfold[bench]"
```

This installs `matplotlib`, `pandas`, and `datasets` for plotting and
loading real text data.

## Memory benchmark

```bash
python -m scripts.run_memory_benchmark \
    --T 1024 --dh 128 \
    --ratios 2 3 4 5 8 \
    --methods jolt flash low \
    --output results/memory.json
```

Output:

```
method           target     orig(B)  compressed(B)     actual
----------------------------------------------------------------------
identity          1.00     8388608        4194304      2.00x
jolt              2.00     8388608        2103880      3.99x
jolt              3.00     8388608        2103880      3.99x
...
```

## Speed benchmark

```bash
python -m scripts.run_speed_benchmark \
    --T 1024 --dh 128 --ratio 3 \
    --warmup 1 --iterations 5 \
    --output results/speed.json
```

Output:

```
method              compress_ms   decompress_ms     ratio
------------------------------------------------------------
jolt                    23.45          8.12      3.99x
flash                    18.32          7.95      3.99x
low                      2.15          1.42      3.94x
```

## Reconstruction benchmark

```bash
python -m scripts.run_table2_reconstruction \
    --T 1024 --ratio 2.0 \
    --output results/table2.json
```

Output:

```
method                      K error    V error      ratio
------------------------------------------------------------
jolt                         0.7617     0.6840      3.99x
flash                        0.7617     0.6840      3.99x
low-64                       1.9809     0.0157      3.94x
int4-per-channel             0.2468     0.2032      7.98x
```

Note: numbers are qualitative because the synthetic K/V doesn't match
real Mistral's exact spectrum.

JSON artifacts contain `metadata`, `parameters`, and `results`. Metadata records
Python, PyTorch, platform, device, seed, warmup count, and timed iteration count;
speed rows also include standard deviation and min/max timings. Do not compare
numbers across machines without retaining that metadata.

## Long-context (needle)

```bash
python -m scripts.run_needle_haystack \
    --model gpt2 \
    --context-length 512 \
    --needle "The secret code is 12345."
```

## Profile

```bash
python -m scripts.profile_model --model gpt2 --ratio 3.0
```

## Sweep

```bash
python -m scripts.sweep_compression_ratio --model gpt2 --ratios 1.5 2 3 4
```

## Reproduce paper numbers

```bash
bash scripts/reproduce_paper_numbers.sh
```

Writes JSON outputs under `results/` and prints human-readable tables
to stdout.
