#!/usr/bin/env bash
# Run every paper-table script in order. Writes JSON outputs under results/.
set -euo pipefail

mkdir -p results

echo "== table 2 reconstruction =="
python -m scripts.run_table2_reconstruction \
    --T 1024 --ratio 2.0 \
    --output results/table2_ratio2.json
python -m scripts.run_table2_reconstruction \
    --T 1024 --ratio 3.0 \
    --output results/table2_ratio3.json

echo "== memory benchmark =="
python -m scripts.run_memory_benchmark \
    --T 1024 --dh 128 \
    --ratios 2 3 4 5 8 \
    --methods jolt flashjolt lowrank \
    --output results/memory.json

echo "== speed benchmark =="
python -m scripts.run_speed_benchmark \
    --T 1024 --dh 128 --ratio 3 \
    --output results/speed.json

echo "== perplexity grid (skipped: run_table1_perplexity module not yet written) =="
# python -m scripts.run_table1_perplexity --T 1024 --ratio 3 --output results/perplexity.json || true
echo "    (table 1 perplexity sweeps belong in benchmarks/, not scripts/)"

echo "== done. results in ./results/ =="