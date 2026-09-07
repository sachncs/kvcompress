"""03 — Custom allocator: per-layer groups.

Demonstrates using `layer_groups > 1` so the allocator can give early
layers different ranks/bits than late layers.
"""

from __future__ import annotations

import torch

from kvcompress import JoLTCompressor


def main() -> None:
    torch.manual_seed(0)
    # Three layers, each with their own KV cache.
    caches = [torch.randn(8, 128, 32) for _ in range(3)]

    # Joint compressor with layer_groups=3 means each layer is its own group.
    comp = JoLTCompressor(
        compression_ratio=3.0,
        bits=(0, 4, 8),
        layer_groups=3,
    )
    for i, (K, V) in enumerate(zip(caches, caches)):  # K=V here for demo
        k_payload, v_payload = comp.compress(K, V)
        print(
            f"layer {i}: ranks=({k_payload.metadata['r_token']}, "
            f"{k_payload.metadata['r_feature']}), bits={k_payload.metadata['bits']}"
        )


if __name__ == "__main__":
    main()
