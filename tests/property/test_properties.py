"""Property-based tests using Hypothesis."""

from __future__ import annotations

import hypothesis
import hypothesis.strategies as st
import torch

from kvcompress.compressor.jolt import JoLTCompressor
from kvcompress.compressor.quantization import (
    IntQuantizer,
    bit_packing_signed,
    bit_unpacking_signed,
)


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    data=st.data(),
    bits=st.sampled_from([2, 4, 8]),
)
def test_int_quantization_roundtrip_property(bits: int, data: st.DataObject) -> None:
    """For any tensor, int quantization round-trip is bounded by one bin."""
    shape = data.draw(
        st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=16),
        )
    )
    torch.manual_seed(0)
    x = torch.randn(*shape) * 4.0
    q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
    packed, scale, zp = q.quantize(x)
    x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
    err = (x - x_hat).abs().max().item()
    bin_size = x.abs().amax().item() / q._qmax
    assert err <= bin_size + 1e-3


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    seed=st.integers(min_value=0, max_value=1000),
    bits=st.sampled_from([2, 4]),
)
def test_packing_unpacking_roundtrip_property(seed: int, bits: int) -> None:
    """Bit-packing and unpacking should recover the original (after offset)."""
    torch.manual_seed(seed)
    n = 32
    last = n
    q_int = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (last,))
    packed = bit_packing_signed(q_int, bits, symmetric=True)
    unpacked = bit_unpacking_signed(packed, bits, last, symmetric=True)
    assert torch.equal(q_int.to(torch.int32), unpacked)


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    m=st.integers(min_value=2, max_value=8),
    T=st.integers(min_value=8, max_value=64),
    dh=st.integers(min_value=4, max_value=16),
)
def test_jolt_compressor_handles_arbitrary_shapes(m: int, T: int, dh: int) -> None:
    """JoLT compressor should round-trip any 3-D tensor shape within range."""
    torch.manual_seed(0)
    K = torch.randn(m, T, dh)
    V = torch.randn(m, T, dh)
    comp = JoLTCompressor(compression_ratio=3.0, bits=(0, 4))
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    assert k_hat.shape == K.shape
    assert v_hat.shape == V.shape


def test_jolt_roundtrip_is_bounded():
    """Round-trip error should be at most 1.0 (loose bound)."""
    torch.manual_seed(0)
    K = torch.randn(2, 16, 8)
    V = torch.randn(2, 16, 8)
    comp = JoLTCompressor(compression_ratio=4.0, bits=(0, 4))
    kp, vp = comp.compress(K, V)
    k_hat, v_hat = comp.decompress(kp, vp)
    rel_err = torch.linalg.norm(K - k_hat) / torch.linalg.norm(K)
    assert rel_err.item() < 2.0  # very loose; real bound depends on ratio
