# Algorithm walkthrough

End-to-end description of JoLT and FlashJoLT, with code references.

## JoLT compress

```
K, V ∈ R^{m × T × dh}
```

1. **Build the cell description.**

   ```python
   cells = [
       Cell(shape=(m, T, dh), kind="key", layer_group=0),
       Cell(shape=(m, T, dh), kind="value", layer_group=0),
   ]
   ```

   See `compressor/allocator.py:Cell`.

2. **Run the joint allocator.**

   ```python
   allocator = JointAllocator(target_ratio=3.0, bits_grid=(0, 2, 4, 8))
   result = allocator.optimize(cells)
   ```

   For each cell, the allocator enumerates `(rT, rd, b)` candidates,
   builds the Lagrangian objective `e + λ·s`, and finds `λ` by bisection
   to hit the byte budget.

3. **Partial Tucker ST-HOSVD.**

   ```python
   factors = partial_tucker_st_hosvd(K, r_token=rt, r_feature=rd, svd=svd)
   ```

   Returns `core ∈ R^{m × rT × rd}`, `u_token ∈ R^{T × rT}`,
   `u_feature ∈ R^{dh × rd}`. The head/layer axis is pinned to identity.

4. **Compute the residual.**

   ```python
   recon = reconstruct_partial_tucker(factors, K.shape)
   residual = K - recon
   ```

5. **JL-rotate and quantize the residual.**

   ```python
   residual_payload = encode_residual(
       residual,
       bits=b,
       seed=seed,
       distribution="gaussian",
   )
   ```

   Returns a `ResidualPayload` carrying the projection seed, packed codes,
   scale, zero-point, and original shape.

6. **Serialize into a `CompressedPayload`.**

   ```python
   payload = CompressedPayload(
       method="jolt",
       shape=tuple(K.shape),
       dtype=K.dtype,
       metadata={
           "r_token": rt,
           "r_feature": rd,
           "bits": b,
           "residual_seed": residual_payload.projection_seed,
           ...
       },
       data={
           "core": core.to(torch.float16),
           "u_token": u_token.to(torch.float16),
           "u_feature": u_feature.to(torch.float16),
           "residual_packed": residual_payload.packed,
           "residual_scale": residual_payload.scale,
           ...
       },
   )
   ```

## JoLT decompress

```python
core = payload.data["core"].to(torch.float32)
u_token = payload.data["u_token"].to(torch.float32)
u_feature = payload.data["u_feature"].to(torch.float32)
x = reconstruct_partial_tucker(TuckerFactors(core, u_token, u_feature, ...), payload.shape)

if "residual_packed" in payload.data:
    residual = decode_residual(ResidualPayload.from_dict(...))
    x = x + residual

return x.to(payload.dtype)
```

## FlashJoLT

Identical to JoLT except:

1. The SVD is `randomised` with `oversampling=10, n_power=2`.
2. The token-mode SVD is capped at `q_cap = min(max(q_min(R), ⌈T/32⌉), 512)`.
3. The allocator sees the *true* tail mass via `SVDResult.tail_mass`; this
   corrects the under-truncation caused by the cap.

## Allocation grid

The allocator's candidate grid per cell is `(candidate_rt, candidate_rd, b)`:

- `candidate_rt`: 1..32 dense, then {48, 64, 96, 128, 192, 256, 384, 512, T}.
- `candidate_rd`: 1..32 dense for small `dh`, then {4, 8, 16, 32, 64, 96, 128, 192, 256, dh}.
- `b ∈ bits_grid`: default `(0, 2, 4, 8)`.

For one cell of `(m=8, T=256, dh=64)`, the grid has roughly 50 × 12 × 4 = 2400 candidates. Two cells = 4800 candidates. The Lagrangian bisection evaluates the grid ~30 times, so ~140k evals per layer. In our tests this is sub-millisecond on CPU.

## End-to-end HF integration

```
model.generate(ids) ──► for each layer:
                          ├─ attention compute
                          ├─ DynamicCache.update(K, V, layer_idx)
                          │    └─ super().update (concatenate)
                          │    └─ compressor.compress(K, V)
                          │    └─ cache_manager.store(payload)
                          ▼
                       next attention step:
                          ├─ past_key_values[layer_idx]
                          │    └─ cache_manager.retrieve()
                          │    └─ compressor.decompress()
                          │    └─ paste into layer.keys / layer.values
                          ▼
                       attention continues normally
```