# Benchmark results

Reference numbers captured on the developer's CPU-only macOS box.
These are **not** paper-reproduction numbers; they show the relative
behaviour of each compressor on synthetic K/V tensors with shapes
similar to a small model's layer.

Shapes used:

- `m=8` heads, `T=256` tokens, `dh=64` (memory + speed sweeps)
- `m=8` heads, `T=256` tokens, `dh=128` (reconstruction)

Run them again with `scripts/reproduce_paper_numbers.sh` (memory +
speed + reconstruction only; the perplexity sweep is stubbed).

## Memory

| method   | target_ratio | bytes_original | bytes_compressed | achieved_ratio |
|----------|-------------:|---------------:|-----------------:|---------------:|
| identity | 1.00         | 1048576        | 524288           | 2.00x          |
| jolt     | 2.00         | 1048576        | 264520           | 3.96x          |
| jolt     | 3.00         | 1048576        | 264520           | 3.96x          |
| jolt     | 4.00         | 1048576        | 264520           | 3.96x          |
| flashjolt| 2.00         | 1048576        | 264520           | 3.96x          |
| flashjolt| 3.00         | 1048576        | 264520           | 3.96x          |
| flashjolt| 4.00         | 1048576        | 264520           | 3.96x          |
| lowrank  | 2.00         | 1048576        | 270464           | 3.88x          |
| lowrank  | 3.00         | 1048576        | 177492           | 5.91x          |
| lowrank  | 4.00         | 1048576        | 135232           | 7.75x          |

## Speed (target_ratio=3.0)

| method   | compress_ms | decompress_ms | achieved_ratio |
|----------|------------:|--------------:|---------------:|
| jolt     | 21.53       | 0.44          | 3.96x          |
| flashjolt| 17.57       | 0.43          | 3.96x          |
| lowrank  | 3.97        | 0.21          | 5.91x          |

## Reconstruction (target_ratio=2.0, m=8 T=256 dh=128)

| method           | K error | V error | achieved_ratio |
|------------------|--------:|--------:|---------------:|
| jolt             | 0.2349  | 0.2153  | 3.81x          |
| flashjolt        | 0.2349  | 0.2156  | 3.81x          |
| lowrank-64       | 0.0127  | 0.0137  | 3.76x          |
| int4-per-channel | 0.2001  | 0.1863  | 7.94x          |