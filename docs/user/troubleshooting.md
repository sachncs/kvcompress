# Troubleshooting

Common error messages and what to do about them.

## "Exactly one of `target_memory` or `compression_ratio` must be provided."

You passed both or neither. Pick one:

```python
enable_compression(model, method="flashjolt", target_memory="33%")
# OR
enable_compression(model, method="flashjolt", compression_ratio=3.0)
```

## "Invalid `cache_implementation` (kvcompress)."

`kvcompress` automatically maps whatever you pass to `cache_implementation`
to `"dynamic"` under the hood, since Hugging Face's strict allow-list does
not include `"kvcompress"`. This message means the mapping failed for some
reason; please open an issue.

## "not implemented" — unknown method

You passed a method that hasn't been implemented in this milestone.
Currently supported: `jolt`, `flashjolt`, `lowrank`, `int2`, `int4`,
`int8`, `fp8`, `fp16`, `bf16`, `identity`.

## Output quality drops after enabling compression

1. Check that you're in the free zone (2-3×). At higher ratios, GQA
   degrades gracefully but MHA degrades sharply.
2. Try `method="jolt"` instead of `flashjolt` to rule out the randomized
   SVD.
3. Inspect `handle.stats_dict()` to confirm the allocator actually met
   the target ratio.

## Compression is slow

1. Use `flashjolt` instead of `jolt`.
2. Install the `triton` extra: `pip install "kvcompress[triton]"`. (Note:
   the Triton path is only faster on NVIDIA GPUs.)
3. Check whether the cache is on CPU (slow) — see
   [Performance guide](performance_guide.md).

## `handle.disable()` doesn't seem to restore behaviour

Make sure you called `disable()` on the *same handle* returned by
`enable_compression`. Patches are per-handle and you can have multiple
in flight.