# Spectral motivation

The paper's Section 3 establishes empirically that **head and layer axes
are essentially incompressible**, while **token and feature axes carry
almost all the redundancy**. This is the empirical finding that motivates
partial Tucker with identity bases on the head and layer modes.

## Measurements from the paper

| Tensor | Mode | σ₁/σ_min | rank for ≤10% error |
|---|---|---|---|
| K | heads (8) | 1.5 | 8/8 |
| K | tokens (1024) | 412K | 228/1024 |
| K | features (128) | 27.4 | 101/128 |
| V | heads (8) | 1.4 | 8/8 |
| V | tokens (1024) | 957K | 563/1024 |
| V | features (128) | 2.7 | 126/128 |

The first row (heads) tells us every KV head carries roughly the same
energy — none can be dropped cheaply. This is expected for a GQA model:
GQA already merged redundant heads upstream. The same observation holds
for MHA in the paper's Appendix B.

The token row shows that for keys, 22% of components capture 90% of
energy (rank 228 / 1024). For values, 55% of components are needed
(rank 563 / 1024). The feature row shows that for keys, 79% of components
matter (rank 101 / 128); for values, the spectrum is nearly flat.

## Consequences for algorithm design

1. **Don't compress the head or layer axis.** A full four-mode allocator
   reproduces the partial method byte-for-byte by driving head and layer
   ranks back to full size (paper Appendix B.2).
2. **Allocate ranks and bits separately for K and V.** Values are 2-3×
   harder to compress than keys; a single shared budget would
   under-compress values and over-compress keys.
3. **Use a residual.** Pure low-rank truncation cannot reach
   near-lossless fidelity on the flat value spectrum — even at full rank
   the residual carries real energy.
4. **Cap the token rank in randomized SVD by context, not by ratio.**
   The cap policy `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)` ensures the
   cap grows sublinearly with `T`; at long contexts the actual token
   rank grows sublinearly too (paper Appendix D).

## Our implementation

- `compressor/tucker.py:partial_tucker_st_hosvd` keeps identity bases on
  mode 0 (head/layer) by construction; only modes 1 and 2 are truncated.
- `compressor/allocator.py:Cell` carries `kind="key"` or `kind="value"`
  so the allocator picks `(rT, rd, b)` per K/V cell.
- `compressor/flashjolt.py:flashjolt_cap` implements the cap policy.

You can verify the spectral claims on your own model by running:

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("...")
# Capture one layer's K after a forward pass.
K = capture_kv(model, "Hello", layer=15)  # shape (B, n_kv, T, dh)
u = mode_n_unfold(K.view(K.shape[0] * K.shape[1], K.shape[2], K.shape[3]), 1)
s = torch.linalg.svdvals(u)
print("σ₁/σ_min:", s[0] / s[-1])
print("rank for ≤10% error:", (torch.cumsum(s**2, dim=0) / (s**2).sum() > 0.9).nonzero()[0].item())
```