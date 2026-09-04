"""Tests for kvfold.config: MethodConfig subclasses and the registry."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import pytest
import torch

from kvfold.config import (
    REGISTRY,
    Bf16Config,
    FlashConfig,
    Float8Config,
    FloatConfig,
    Fp16Config,
    Int2Config,
    Int4Config,
    Int8Config,
    JoltConfig,
    LowRankConfig,
    MethodConfig,
    MethodConfigRegistry,
    PassConfig,
)
from kvfold.errors import MethodConfigError


def test_registry_has_all_methods() -> None:
    names = REGISTRY.names()
    assert set(names) == {"jolt", "flash", "low", "int2", "int4", "int8", "fp8", "fp16", "bf16", "pass"}


def test_registry_resolve_known() -> None:
    assert REGISTRY.resolve("jolt") is JoltConfig
    assert REGISTRY.resolve("flash") is FlashConfig
    assert REGISTRY.resolve("pass") is PassConfig


def test_registry_resolve_unknown_raises() -> None:
    with pytest.raises(KeyError):
        REGISTRY.resolve("nope")


def test_registry_build_validates() -> None:
    config = REGISTRY.build("jolt", ratio=4.0, bits=(0, 4, 8))
    assert isinstance(config, JoltConfig)
    assert config.ratio == 4.0
    assert config.bits == (0, 4, 8)


def test_registry_build_rejects_unknown_kwarg() -> None:
    with pytest.raises(MethodConfigError) as ei:
        REGISTRY.build("jolt", ratio=4.0, unknown_field=1)
    assert "unknown_field" in ei.value.field


def test_registry_register_duplicate_raises() -> None:
    reg = MethodConfigRegistry()
    reg.register("jolt", JoltConfig)
    with pytest.raises(ValueError):
        reg.register("jolt", JoltConfig)


def test_registry_register_wrong_type_raises() -> None:
    reg = MethodConfigRegistry()
    with pytest.raises(TypeError):
        reg.register("xx", object)


def test_jolt_config_validates_ratio() -> None:
    with pytest.raises(MethodConfigError) as ei:
        JoltConfig(ratio=0.5).validate()
    assert ei.value.field == "ratio"


def test_jolt_config_validates_bits() -> None:
    with pytest.raises(MethodConfigError) as ei:
        JoltConfig(bits=(3, 7)).validate()
    assert ei.value.field == "bits"


def test_jolt_config_validates_distribution() -> None:
    with pytest.raises(MethodConfigError) as ei:
        JoltConfig(distribution="laplace").validate()
    assert ei.value.field == "distribution"


def test_jolt_config_validates_layer_groups() -> None:
    with pytest.raises(MethodConfigError):
        JoltConfig(layer_groups=0).validate()


def test_jolt_config_validates_group_size() -> None:
    with pytest.raises(MethodConfigError):
        JoltConfig(group_size=0).validate()


def test_flash_config_validates_ratio() -> None:
    with pytest.raises(MethodConfigError):
        FlashConfig(ratio=0.5).validate()


def test_flash_config_validates_cap() -> None:
    with pytest.raises(MethodConfigError):
        FlashConfig(cap=0).validate()


def test_low_rank_config_validates_rank() -> None:
    with pytest.raises(MethodConfigError):
        LowRankConfig(rank=0).validate()


@pytest.mark.parametrize("cls,bits", [(Int2Config, 2), (Int4Config, 4), (Int8Config, 8)])
def test_int_config_valid_bits(cls: type[MethodConfig], bits: int) -> None:
    cls(bits=bits).validate()


@pytest.mark.parametrize("bits", [1, 3, 7, 16])
def test_int_config_invalid_bits(bits: int) -> None:
    with pytest.raises(MethodConfigError):
        Int8Config(bits=bits).validate()


def test_int_config_validates_group_size() -> None:
    with pytest.raises(MethodConfigError):
        Int8Config(group_size=0).validate()


def test_float_config_validates_dtype_when_constructed() -> None:
    """FloatConfig.validate() rejects non-16-bit dtypes; exercised via a non-frozen subclass path."""
    # The public surface only exposes Fp16Config / Bf16Config with frozen dtypes;
    # we still verify the validate logic by subclassing FloatConfig in-test.
    @dataclasses.dataclass(frozen=True)
    class BadConfig(FloatConfig):
        method: ClassVar[str] = "fp16"
        dtype: torch.dtype = torch.float32

    bad = BadConfig()
    with pytest.raises(MethodConfigError):
        bad.validate()


def test_fp16_and_bf16_have_correct_dtypes() -> None:
    assert Fp16Config().dtype == torch.float16
    assert Bf16Config().dtype == torch.bfloat16


def test_pass_config_rejects_non_float() -> None:
    with pytest.raises(MethodConfigError):
        PassConfig(dtype=torch.int32).validate()


def test_float8_config_validates_variant() -> None:
    from kvfold.errors import MethodConfigError
    with pytest.raises(MethodConfigError):
        Float8Config(variant="e3m2").validate()


def test_float8_config_validates_group_size() -> None:
    with pytest.raises(MethodConfigError):
        Float8Config(group_size=0).validate()


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (JoltConfig, {"ratio": 4.0, "bits": (0, 4)}),
        (FlashConfig, {"ratio": 2.0}),
        (LowRankConfig, {"rank": 32}),
        (Int8Config, {"bits": 8}),
        (PassConfig, {}),
        (Float8Config, {"variant": "e4m3"}),
    ],
)
def test_round_trip_through_dict(cls: type[MethodConfig], kwargs: dict) -> None:
    original = cls(**kwargs)
    blob = original.to_dict()
    restored = cls.from_dict(blob)
    assert original == restored


def test_frozen_configs_cannot_be_mutated() -> None:
    config = JoltConfig(ratio=3.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ratio = 4.0


def test_each_config_has_method_attribute() -> None:
    assert JoltConfig.method == "jolt"
    assert FlashConfig.method == "flash"
    assert LowRankConfig.method == "low"
    assert Int2Config.method == "int2"
    assert Int4Config.method == "int4"
    assert Int8Config.method == "int8"
    assert Float8Config.method == "fp8"
    assert Fp16Config.method == "fp16"
    assert Bf16Config.method == "bf16"
    assert PassConfig.method == "pass"


def test_registry_supported_alias() -> None:
    assert REGISTRY.supported() == REGISTRY.names()
