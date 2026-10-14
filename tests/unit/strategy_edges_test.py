from __future__ import annotations

import pytest
import torch

from kvfold.core.svd import DecomposerRegistry, Exact, Randomized
from kvfold.core.tucker import TuckerBackendRegistry, mode_n_fold, mode_n_unfold


def test_exact_rejects_non_matrix_input() -> None:
    with pytest.raises(Exception, match="2-D"):
        Exact().decompose(torch.randn(2, 2, 2), rank=1)


def test_randomized_rejects_non_matrix_input() -> None:
    with pytest.raises(Exception, match="2-D"):
        Randomized(seed=0).decompose(torch.randn(2, 2, 2), rank=1)


def test_randomized_caps_rank_and_preserves_shape() -> None:
    result = Randomized(seed=0, cap=3).decompose(torch.randn(8, 5), rank=2)
    assert result.u.shape[0] == 8
    assert result.vh.shape[1] == 5
    assert result.rank == 2


def test_decomposer_registry_supports_exact_and_randomized() -> None:
    registry = DecomposerRegistry()
    assert isinstance(registry.default("exact"), Exact)
    assert isinstance(registry.default("randomised", seed=4, cap=3), Randomized)
    with pytest.raises(ValueError, match="unknown decomposer"):
        registry.default("unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_unfold_fold_are_inverses(mode: int) -> None:
    tensor = torch.randn(2, 3, 4)
    unfolded = mode_n_unfold(tensor, mode)
    assert torch.allclose(mode_n_fold(unfolded, mode, tuple(tensor.shape)), tensor)


def test_tucker_backend_registry_has_torch_backend() -> None:
    registry = TuckerBackendRegistry()
    assert registry.resolve("torch").backend == "torch"
    with pytest.raises(KeyError, match="unknown tucker backend"):
        registry.resolve("missing")
