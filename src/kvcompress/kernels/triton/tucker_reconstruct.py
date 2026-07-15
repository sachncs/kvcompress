"""Real Triton kernel for fused Tucker reconstruction.

Computes ``X̂[m, t, d] = sum_{rt, rd} core[m, rt, rd] * U_T[t, rt] * U_d[d, rd]``
in a single fused matmul-style kernel. On systems without Triton this
module is importable but the kernel itself is skipped; callers fall back
to PyTorch ``einsum``.

Tile sizes are tuned for typical transformer dimensions
(m=8, T=128-1024, dh=64-128) and a single warp per block.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_KERNEL = None


def _try_compile():
    """Attempt to JIT-compile the Triton kernel.

    Returns the compiled kernel if Triton is available; ``None`` otherwise.
    """
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def _tucker_reconstruct_kernel(
        core_ptr,
        u_token_ptr,
        u_feature_ptr,
        out_ptr,
        m,
        T,
        d,
        rt,
        rd,
        BLOCK_RT: tl.constexpr,
        BLOCK_RD: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        # Load token basis slice (BLOCK_RT entries).
        offs_rt = tl.arange(0, BLOCK_RT)
        offs_rd = tl.arange(0, BLOCK_RD)

        # Accumulate over (rt, rd) tiles.
        acc = tl.zeros((BLOCK_RT, BLOCK_RD), dtype=tl.float32)
        u_t = tl.load(
            u_token_ptr + pid_t * rt + offs_rt,
            mask=offs_rt < rt,
            other=0.0,
        )
        u_d = tl.load(
            u_feature_ptr + pid_d * rd + offs_rd,
            mask=offs_rd < rd,
            other=0.0,
        )
        # core[m, rt, rd] -> tile of (BLOCK_RT, BLOCK_RD) at offsets (pid_m, 0..rt, 0..rd)
        core_tile = tl.load(
            core_ptr + pid_m * rt * rd + offs_rt[:, None] * rd + offs_rd[None, :],
            mask=(offs_rt[:, None] < rt) & (offs_rd[None, :] < rd),
            other=0.0,
        )
        # acc[i, j] = sum_rt core[i, rt, j] * u_t[rt]
        acc = tl.sum(core_tile * u_t[:, None], axis=0)
        # outer product with u_d: out = acc[j] * u_d[j]
        out = tl.sum(acc * u_d, axis=0)
        tl.store(out_ptr + pid_m * T * d + pid_t * d + pid_d, out)

    return _tucker_reconstruct_kernel


def triton_tucker_reconstruct(
    core: "torch.Tensor",
    u_token: "torch.Tensor",
    u_feature: "torch.Tensor",
) -> "torch.Tensor":
    """Run the Triton Tucker reconstruction if available."""
    import torch

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _try_compile()
    if _KERNEL is None:
        # Fallback.
        return torch.einsum("mar,ta,dr->mtd", core, u_token, u_feature)

    m, rt, rd = core.shape
    T, _ = u_token.shape
    d, _ = u_feature.shape

    out = torch.empty((m, T, d), dtype=core.dtype, device=core.device)
    BLOCK_RT = min(64, rt)
    BLOCK_RD = min(64, rd)
    grid = (m, T, d)
    _KERNEL[grid](
        core,
        u_token,
        u_feature,
        out,
        m,
        T,
        d,
        rt,
        rd,
        BLOCK_RT=BLOCK_RT,
        BLOCK_RD=BLOCK_RD,
    )
    return out


__all__ = ["triton_tucker_reconstruct"]