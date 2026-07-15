"""Algorithmic tests for the JL projection and quantization path.

These tests verify the actual math: norm preservation (JL lemma) and
quantization-error bounds (half a bin).
"""

from __future__ import annotations


import pytest
import torch

from kvcompress.compressor.jl import (
    cached_projection,
    clear_projection_cache,
    gaussian_projection,
    rademacher_projection,
)
from kvcompress.compressor.quantization import (
    IntQuantizer,
    bit_packing_signed,
    bit_unpacking_signed,
    dequantize_tensor,
    get_quantizer,
    quantize_tensor,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_projection_cache()


def test_jl_gaussian_norm_preservation() -> None:
    """The Johnson-Lindenstrauss lemma: ``E[||Πx||²] ≈ ||x||²`` for a
    Gaussian projection Π. Verify empirically with many samples.
    """
    torch.manual_seed(0)
    dim = 64
    proj = gaussian_projection(dim, dim, seed=0)
    # Average squared-norm ratio over many vectors.
    x = torch.randn(2048, dim)
    y = proj.apply(x)
    ratios = (y * y).sum(dim=-1) / (x * x).sum(dim=-1)
    mean_ratio = float(ratios.mean().item())
    # The mean ratio is exactly 1.0 in expectation; allow 10% slack on a
    # 2048-sample Monte Carlo.
    assert abs(mean_ratio - 1.0) < 0.1, f"E[||Πx||²/||x||²] = {mean_ratio:.4f}, expected ~1.0"


def test_jl_rademacher_norm_preservation() -> None:
    """Same as above for the Rademacher (sign) distribution."""
    torch.manual_seed(0)
    dim = 64
    proj = rademacher_projection(dim, dim, seed=0)
    x = torch.randn(2048, dim)
    y = proj.apply(x)
    ratios = (y * y).sum(dim=-1) / (x * x).sum(dim=-1)
    mean_ratio = float(ratios.mean().item())
    assert abs(mean_ratio - 1.0) < 0.15, f"E[||Πx||²/||x||²] = {mean_ratio:.4f}, expected ~1.0"


def test_jl_apply_shape_preservation() -> None:
    """``apply`` preserves shape on arbitrary input ranks."""
    proj = gaussian_projection(8, 8, seed=0)
    for shape in [(8,), (4, 8), (2, 3, 8), (1, 1, 4, 8)]:
        x = torch.randn(*shape)
        y = proj.apply(x)
        assert y.shape == x.shape, f"shape mismatch: {shape} -> {y.shape}"


def test_jl_projection_is_orthonormal_at_limit() -> None:
    """For a square Gaussian projection, the columns of Π are
    approximately orthonormal (this is the JL lemma at the matrix level).
    """
    dim = 128
    proj = gaussian_projection(dim, dim, seed=0)
    gram = proj.matrix.t() @ proj.matrix / dim
    # Off-diagonal elements should be small relative to the diagonal.
    off_diag = (gram - torch.diag(torch.diagonal(gram))).abs().mean()
    diag = torch.diagonal(gram).mean()
    assert (
        off_diag < 0.1 * diag
    ), f"off-diagonal mean {off_diag:.4f} too large vs diagonal {diag:.4f}"


def test_jl_cache_returns_same_object() -> None:
    proj1 = cached_projection(8, 16, distribution="gaussian", seed=0)
    proj2 = cached_projection(8, 16, distribution="gaussian", seed=0)
    assert proj1 is proj2


def test_jl_cache_different_seeds_give_different_matrices() -> None:
    a = cached_projection(8, 16, distribution="gaussian", seed=0)
    b = cached_projection(8, 16, distribution="gaussian", seed=1)
    assert not torch.allclose(a.matrix, b.matrix)


def test_jl_distribution_switch_yields_different_matrices() -> None:
    g = cached_projection(8, 16, distribution="gaussian", seed=0)
    r = cached_projection(8, 16, distribution="rademacher", seed=0)
    assert not torch.allclose(g.matrix, r.matrix)


def test_quant_error_bounded_by_one_bin() -> None:
    """For symmetric int-N quantisation with per-channel scale, the
    error on each element is at most half a bin: ``|x - x_hat| ≤ scale / 2``.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 16) * 4.0
    for bits in (2, 4, 8):
        q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
        packed, scale, zp = q.quantize(x)
        x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
        # Per-channel bin size.
        bin_size = (x.abs().amax(dim=0) / q._qmax).clamp(min=1e-9)
        per_channel_err = (x - x_hat).abs().amax(dim=0)
        # Each channel's max error is at most half a bin (rounding) plus
        # fp32 noise.
        assert (
            per_channel_err <= bin_size / 2 + 1e-3
        ).all(), f"bits={bits}: per-channel error {per_channel_err} exceeds bin/2 = {bin_size / 2}"


def test_quant_error_bounded_by_one_bin_asymmetric() -> None:
    """Asymmetric quantisation: error ≤ full bin (one bin wider on each
    side than symmetric)."""
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 4.0
    for bits in (2, 4, 8):
        q = IntQuantizer(bits=bits, symmetric=False, per_channel=True)
        packed, scale, zp = q.quantize(x)
        x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
        # Bin size = (xmax - xmin) / (qmax - qmin).
        rng = (x.amax(dim=0) - x.amin(dim=0)).clamp(min=1e-9)
        bin_size = rng / (q._qmax - q._qmin)
        per_channel_err = (x - x_hat).abs().amax(dim=0)
        assert (
            per_channel_err <= bin_size + 1e-3
        ).all(), f"bits={bits}: per-channel error {per_channel_err} exceeds bin = {bin_size}"


def test_packing_unpacking_roundtrip_per_bit_width() -> None:
    """Bit-packing then unpacking must recover the original signed values
    for every supported bit-width (2, 4, 8).
    """
    torch.manual_seed(0)
    for bits in (2, 4, 8):
        for last in (4, 8, 16, 32):
            qmin = -(1 << (bits - 1))
            qmax = (1 << (bits - 1)) - 1
            q_int = torch.randint(qmin, qmax + 1, (last,))
            packed = bit_packing_signed(q_int, bits, symmetric=True)
            unpacked = bit_unpacking_signed(packed, bits, last, symmetric=True)
            assert torch.equal(
                unpacked, q_int.to(torch.int32)
            ), f"packing roundtrip failed for bits={bits}, last={last}"


def test_quantizer_dispatch_returns_int_quantizer() -> None:
    q = get_quantizer("int4")
    assert isinstance(q, IntQuantizer)
    assert q.bits == 4
    q2 = get_quantizer("int4", symmetric=False, per_channel=False, group_size=4)
    assert q2.symmetric is False
    assert q2.per_channel is False
    assert q2.group_size == 4


def test_quantize_tensor_round_trip_preserves_shape() -> None:
    """``quantize_tensor`` and ``dequantize_tensor`` preserve shape."""
    torch.manual_seed(0)
    x = torch.randn(4, 16) * 2.0
    for bits in (2, 4, 8):
        payload = quantize_tensor(x, dtype=f"int{bits}")
        x_hat = dequantize_tensor(payload, dtype=f"int{bits}")
        assert x_hat.shape == x.shape, f"shape mismatch: {x.shape} -> {x_hat.shape}"


def test_quant_round_trip_converges_for_smooth_signals() -> None:
    """A smooth signal (e.g. ``sin(x)``) is well-approximated by 4-bit
    quantisation because the per-channel scale adapts to the local range.
    """
    x = torch.linspace(-3, 3, 100).unsqueeze(0)
    for bits in (4, 8):
        q = IntQuantizer(bits=bits, symmetric=True, per_channel=True)
        packed, scale, zp = q.quantize(x)
        x_hat = q.dequantize(packed, scale, zp, output_dtype=torch.float32)
        rel_err = (x - x_hat).norm() / x.norm()
        assert rel_err < 0.1, f"bits={bits}: rel_err={rel_err.item()}"


def test_quant_per_group_smaller_error_than_per_tensor() -> None:
    """Per-group scales should give *better* approximation than a single
    per-tensor scale (assuming the data has local variation).
    """
    # A signal with very different ranges in different regions.
    x = torch.cat(
        [
            torch.linspace(0, 1, 50) * 0.001,  # tiny range
            torch.linspace(0, 1, 50) * 10.0,  # large range
        ]
    ).unsqueeze(0)
    q_per_tensor = IntQuantizer(bits=4, symmetric=True, per_channel=False)
    q_per_group = IntQuantizer(bits=4, symmetric=True, per_channel=False, group_size=20)
    p1 = q_per_tensor.quantize(x)
    p2 = q_per_group.quantize(x)
    err_t = (x - q_per_tensor.dequantize(*p1)).norm() / x.norm()
    err_g = (x - q_per_group.dequantize(*p2)).norm() / x.norm()
    # Per-group should be at least as good as per-tensor (often better).
    assert err_g <= err_t + 1e-9, f"per-group {err_g} > per-tensor {err_t}"
