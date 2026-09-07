# FAQ

### Does JoLT work with encoder-decoder models?

The DynamicCache interception in `enable_compression` patches
encoder-decoder caches too, but the allocator and reconstruction have
been validated only on decoder-only models. Encoder-decoder cross
attention has a different shape pattern; use `method="identity"` if you
hit issues.

### Does JoLT work with sliding-window attention (Mistral)?

Yes. The DynamicCache subclass doesn't care about the window logic —
HF's cache layer handles eviction; we only compress what survives.

### Can I quantize at a different precision than the model?

The compressor stores factors in fp16 by default. Override with
`factor_dtype=torch.bfloat16` (or `torch.float32`) on the compressor
constructor if you want different precision. (Currently exposed via the
direct API; pass through `enable_compression(..., factor_dtype=...)`.)

### How does JoLT compare to H2O / SnapKV / StreamingLLM?

Those are *eviction* methods (drop low-importance tokens). JoLT is a
*compression* method (represent every token more cheaply). The two are
complementary; you could run eviction first and then compress what
remains.

### Where's the paper's perplexity reproduction?

See `docs/research/reproduction_notes.md`. The numbers in the paper's
Table 1 (Mistral-7B, LLaMA-2-13B) require a GPU box with ≥40 GB and the
respective model weights, neither of which is reproducible in this repo's
CI. We do provide:

- Reconstruction fidelity at 2× on synthetic K/V that mimics Mistral's
  spectrum (`scripts/run_table2_reconstruction.py`).
- Algorithm correctness tests (`tests/unit/`).
- End-to-end integration tests on GPT-2 (`tests/integration/test_gpt2.py`).

### Does JoLT work with FlashAttention?

Yes, transparently. The HF adapter patches `DynamicCache`; attention
implementation is independent.

### How do I extend JoLT to a new model family?

See [Adding an adapter](../dev/adding_an_adapter.md). Most families
need only a no-op shim because `DynamicCache` interception covers them.