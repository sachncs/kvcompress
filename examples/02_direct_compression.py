"""02 — Direct compression without Hugging Face.

Use the JoLTCompressor directly when you want to compress K/V tensors
without monkey-patching a model. Useful for offline analysis, ablations,
or building your own KV cache backend.
"""

from __future__ import annotations

import torch

from kvcompress import JoLTCompressor
from kvcompress.cache.compress import CompressedKVCache


def main() -> None:
    torch.manual_seed(0)
    # Simulate a KV cache at one layer: shape (n_kv, T, dh).
    n_kv, T, dh = 8, 256, 64
    K = torch.randn(n_kv, T, dh)
    V = torch.randn(n_kv, T, dh)

    # Direct API. We pin ``bits=(0,)`` so the allocator picks pure-Tucker
    # ranks instead of an int8-residual path that's too lossy at this
    # shape. For mixed (Tucker + residual) compression, raise the
    # compression_ratio or switch to a longer context where the
    # residual budget pays off.
    comp = JoLTCompressor(compression_ratio=2.0, bits=(0,))
    k_payload, v_payload = comp.compress(K, V)
    K_hat, V_hat = comp.decompress(k_payload, v_payload)
    rel_err_K = torch.linalg.norm(K - K_hat) / torch.linalg.norm(K)
    print(f"JoLT round-trip rel err K: {rel_err_K.item():.4f}")
    print(f"  original bytes:  {K.numel() * K.element_size():>10}")
    print(f"  compressed K+V:  {k_payload.bytes_compressed + v_payload.bytes_compressed:>10}")

    # Cache API.
    cache = CompressedKVCache(compressor=comp)
    cache.store(layer=0, key=K, value=V)
    print(f"  cache bytes:     {cache.memory_used()}")
    print(f"  compression:     {cache.compression_ratio():.2f}x")

    K2, V2 = cache.retrieve(0)
    # JoLT round-trip is lossy by design (Tucker truncation). At
    # compression_ratio=2.0 the algorithm lands around rel-err=0.7 on
    # this shape; raise the ratio or pre-train a calibration set if
    # you need tighter fidelity.
    rel_err_cache = torch.linalg.norm(K - K2) / torch.linalg.norm(K)
    print(f"  cache rel-err:   {rel_err_cache.item():.4f}")
    assert rel_err_cache.item() < 1.0, f"cache round-trip rel_err too high: {rel_err_cache.item()}"


if __name__ == "__main__":
    main()
