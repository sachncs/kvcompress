"""Parity tests: PyTorch einsum vs Triton kernel.

These tests run the Triton-fused Tucker reconstruction kernel against
the PyTorch einsum reference and assert they match to within
``1e-5`` relative Frobenius error.

Skip on CPU (the kernel falls back to einsum and the test is
meaningless). The ``@pytest.mark.gpu`` marker is declared in
``pyproject.toml``; tests are excluded from the default CI lane.
"""

from __future__ import annotations

import pytest
import torch

from kvfold.core.tucker import Tucker, reconstruct_partial_tucker


pytestmark = pytest.mark.gpu


def _has_cuda_and_triton() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cuda_and_triton(), reason="CUDA + Triton required")
def test_triton_matches_torch_within_1e_minus_5() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    core = torch.randn(8, 16, 16, device=device, dtype=torch.float32)
    u_token = torch.randn(128, 16, device=device, dtype=torch.float32)
    u_feature = torch.randn(64, 16, device=device, dtype=torch.float32)

    factors = Tucker(
        core=core,
        u_token=u_token,
        u_feature=u_feature,
        token_sv=None,
        feature_sv=None,
        token_tail_mass=0.0,
        feature_tail_mass=0.0,
    )
    shape = (8, 128, 64)

    torch_out = reconstruct_partial_tucker(factors, shape, backend="torch")
    triton_out = reconstruct_partial_tucker(factors, shape, backend="triton")

    diff = float(torch.linalg.norm(torch_out - triton_out) / torch.linalg.norm(torch_out))
    assert diff < 1e-5, f"triton-vs-torch rel error: {diff}"


@pytest.mark.skipif(not _has_cuda_and_triton(), reason="CUDA + Triton required")
def test_triton_kernel_returns_right_shape() -> None:
    from kvfold.kernel.triton.tucker import triton_tucker_reconstruct

    torch.manual_seed(0)
    device = torch.device("cuda")
    core = torch.randn(4, 8, 8, device=device, dtype=torch.float32)
    u_token = torch.randn(32, 8, device=device)
    u_feature = torch.randn(16, 8, device=device)
    out = triton_tucker_reconstruct(core, u_token, u_feature)
    assert out.shape == (4, 32, 16)
