# Performance guide

This page is about squeezing the most out of `kvcompress` for your
specific workload.

## Compression ratio

The free zone (paper Section 6) is 2-3× for both GQA and MHA models.
Beyond that:

- **GQA models (Mistral, Qwen, Gemma2):** degrade gracefully. 4-8× is
  acceptable for many workloads.
- **MHA models (LLaMA, Phi-2):** degrades sharply at 4-5×. Stay in the
  2-3× range unless you have measurement showing otherwise.

## Layer groups

By default, the allocator treats the entire model as a single group.
You can split it into multiple groups with `layer_groups=G`:

```python
enable_compression(model, method="flashjolt", target_memory="33%", layer_groups=4)
```

This lets the allocator give early layers different ranks/bits than late
layers. The paper's ablations show per-group allocation beats uniform
allocation by up to 1 PPL at 4×, but in the free zone the two are within
noise.

## Residual bit-widths

`bits=(0, 2, 4, 8)` lets the allocator pick per-cell. In the free zone the
allocator typically chooses `b=4`; at high compression it goes to `b=8`
(values) or `b=2` (keys).

If you want a fixed residual bit-width across the board (e.g. for
hardware reasons), restrict the tuple:

```python
enable_compression(model, method="flashjolt", target_memory="33%", bits=(0, 4))
```

## RoPE handling

Keys carry RoPE rotation. JoLT compresses keys **before** RoPE and
re-applies the rotation at decode — this is the paper's recommendation
(Appendix C), which finds +22-44% lower key-error on LLaMA and +29-70%
on Mistral compared to post-RoPE compression.

## Device handling

The compressor runs on whatever device the cache tensors live on. If you
move the model to CUDA after `enable_compression`, the cache follows it
via `DynamicCache.update`'s standard device-tracking.

For multi-GPU serving, keep the cache on the same device as the model's
parameters. The HF adapter does not currently support tensor-parallel
cache sharding — use the vLLM adapter instead (see `docs/research/paper_notes.md`).

## Decode-time cost

Compression happens once at every cache update. Decompression happens
once at every cache read (i.e., every attention call). On long contexts,
decompression is the dominant cost; on short contexts (T < 512),
compression dominates.

The cache manager keeps the last appended slice in *uncompressed* form
when possible, so the first attention call after a write skips the
decompression step. This is most helpful for short contexts.

## Avoiding common pitfalls

1. **Don't enable compression during training.** JoLT is inference-only;
   gradients through the Tucker / JL path are not supported.
2. **Don't change `model.dtype` after `enable_compression`.** The cache
   uses the dtype at enable time.
3. **Don't disable mid-generation.** The patched DynamicCache will route
   the in-flight reads to nowhere.

See [Troubleshooting](troubleshooting.md) for error message decoding.