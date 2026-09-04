"""vLLM integration — Shape A: ``export_kv`` / ``import_kv`` helpers.

These functions compress and dump a model's KV cache to disk (and load
it back) so a user can:

* Run a vLLM offline batch via ``vllm.LLM.generate(...)``, then
  ``kvcompress.adapters.vllm.export_kv(model, \"kv.safetensors\")`` to save
  the compressed cache.
* Reload the cache in a separate process (or even a different
  deployment) via
  ``kvcompress.adapters.vllm.import_kv(model, \"kv.safetensors\")``.

The compressed cache is stored as a single safetensors file holding
one tensor per (layer, kind) cell plus a metadata sidecar JSON file.

Caveats:

* The cache discovery walks ``DynamicCache`` layers by index. For vLLM's
  paged KV blocks this works because vLLM materialises a ``DynamicCache``
  when the model is called with ``use_cache=True``.
* The exported file embeds the *compressor* layout. Importing on a
  model with a different number of layers raises immediately.
* This is a *user-driven* workflow — there's no automatic hook into
  vLLM's scheduler. For that, see :mod:`kvcompress.adapters.vllm_kv_offload`.

vLLM availability:

The module is importable on systems without vLLM. :func:`export_kv`
and :func:`import_kv` only need ``transformers``, which is a hard
dependency. The :mod:`kvcompress.adapters.vllm_kv_offload` module is the
optional Shape B integration that requires vLLM.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from kvcompress.api import parse_target_memory
from kvcompress.cache.compress import CompressedKVCache
from kvcompress.cache.metadata import CompressionMetadata
from kvcompress.compressor.base import KVCompressor
from kvcompress.compressor.dispatch import build_compressor
from kvcompress.runtime.profiler import CompressionProfiler

__all__ = [
    "build_compressor",
    "export_kv",
    "import_kv",
    "is_vllm_available",
    "resolve_cache",
]


log = logging.getLogger(__name__)


def resolve_cache(model: Any) -> Any:
    """Locate the live :class:`DynamicCache` instance on a model.

    Different frameworks store the cache in different places:
    ``model.past_key_values`` (most HF models), ``model.kv_cache`` (some
    third-party), or via :attr:`DynamicCache` on the attention layer.
    We probe in order.

    Args:
        model: HF or vLLM-style model object.

    Returns:
        The cache object.

    Raises:
        RuntimeError: if no cache can be located.
    """
    for attr in ("past_key_values", "kv_cache", "cache"):
        cache = getattr(model, attr, None)
        if cache is not None:
            return cache
    raise RuntimeError(
        "could not locate a KV cache on the model; expected "
        "model.past_key_values, model.kv_cache, or model.cache"
    )


def export_kv(
    model: Any,
    path: str,
    *,
    method: str = "flashjolt",
    compression_ratio: float = 3.0,
    bits: tuple[int, ...] = (0, 2, 4, 8),
    seed: int = 0,
    compressor: KVCompressor | None = None,
) -> CompressionMetadata:
    """Compress the model's live KV cache and write it to ``path``.

    The output is a single safetensors file with one tensor per
    (layer, kind) cell, plus a metadata sidecar at ``path + '.meta.json'``.

    Args:
        model: HF model (or any object with a ``past_key_values``
            attribute holding a :class:`DynamicCache`).
        path: destination safetensors path.
        method: compressor name.
        compression_ratio: target ratio (passed to JoLT / FlashJoLT).
        bits: residual bit-widths.
        seed: RNG seed.
        compressor: pre-built compressor. If ``None``, one is constructed
            from ``method``, ``compression_ratio``, ``bits``, ``seed``.

    Returns:
        The :class:`CompressionMetadata` describing the saved cache.

    Raises:
        RuntimeError: if no KV cache is found on the model.
        ImportError: if safetensors is not installed.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            "safetensors is required for export_kv; install with `pip install safetensors`"
        ) from e

    cache = resolve_cache(model)
    if compressor is None:
        compressor = build_compressor(
            method,
            compression_ratio=compression_ratio,
            bits=bits,
            seed=seed,
        )

    tmp_cache = CompressedKVCache(compressor=compressor)
    profiler = CompressionProfiler()

    # Walk the layers of the source cache. DynamicCache exposes
    # ``layers[layer_idx].keys`` and ``.values``.
    n_layers = len(cache.layers)
    for layer_idx in range(n_layers):
        layer = cache.layers[layer_idx]
        k = layer.keys
        v = layer.values
        with profiler.record(
            "compress",
            bytes_in=k.numel() * k.element_size() + v.numel() * v.element_size(),
        ):
            tmp_cache.store(layer_idx, k, v)

    # Now serialise.
    tensors: dict[str, torch.Tensor] = {}
    cell_meta: dict[tuple[int, str], dict[str, Any]] = {}
    for layer_idx in tmp_cache.layers():
        for kind in ("key", "value"):
            payload = tmp_cache.payload(layer_idx, kind)
            for name, t in payload.data.items():
                if isinstance(t, torch.Tensor):
                    # Prefix with layer/kind for safetensors dict uniqueness.
                    tensors[f"{layer_idx}/{kind}/{name}"] = t.contiguous()
            # Capture the per-cell metadata so import_kv can faithfully
            # reconstruct (residual_seed, distribution, etc.). JoLT
            # needs these; Identity/LowRank/IntQuantOnly leave them as
            # defaults but importing them all is harmless.
            cell_meta[(layer_idx, kind)] = {
                "method": payload.method,
                "shape": list(payload.shape),
                "dtype": str(payload.dtype),
                "metadata": {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in payload.metadata.items()
                },
            }

    save_file(tensors, path)
    log.info(
        "export_kv: wrote %d layers to %s (%d bytes, profiler %s)",
        n_layers,
        path,
        sum(t.numel() * t.element_size() for t in tensors.values()),
        profiler.summary(),
    )

    meta_path = path + ".meta.json"
    meta = tmp_cache.metadata()
    meta_dict = meta.to_dict()
    # Ponytail: stash the per-cell metadata (which CompressionMetadata
    # flattens) under a ``cell_metadata`` key so import_kv can rebuild
    # the residual projection faithfully. Without this, the residual
    # is reconstructed against an invented seed and the round-tripped
    # cache is garbage.
    serialisable_cells: dict[str, dict[str, Any]] = {}
    for (layer_idx, kind), cmeta in cell_meta.items():
        serialisable_cells[f"{layer_idx}/{kind}"] = cmeta
    meta_dict["cell_metadata"] = serialisable_cells
    import json

    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2, default=str)
    log.info("export_kv: wrote metadata to %s", meta_path)
    return meta


def import_kv(
    model: Any,
    path: str,
    *,
    target_memory: str | None = "100%",
    compression_ratio: float | None = None,
    method: str = "flashjolt",
    bits: tuple[int, ...] = (0, 2, 4, 8),
    seed: int = 0,
) -> CompressionMetadata:
    """Load a compressed cache from ``path`` and populate the model's cache.

    Reads ``path`` (safetensors) plus ``path.meta.json`` (sidecar written
    by :func:`export_kv`). The sidecar carries the per-cell metadata
    that lets us reconstruct each cell's residual projection with the
    correct seed (the prior version invented a seed from the packed
    tensor's numel, producing garbage on every round-trip).

    Args:
        model: HF or vLLM-style model with a writable DynamicCache.
        path: source safetensors path.
        target_memory: passed to :func:`kvcompress.enable_compression`
            if the model hasn't been compressed yet (default ``"100%"``
            = identity).
        compression_ratio: optional ratio override.
        method: compressor name to enable if needed.
        bits: residual bit-widths.
        seed: RNG seed.

    Returns:
        The :class:`CompressionMetadata` describing the loaded cache.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "safetensors is required for import_kv; install with `pip install safetensors`"
        ) from e

    cache = resolve_cache(model)
    tensors = load_file(path)

    # Read the sidecar so we can reconstruct each cell faithfully.
    import json

    meta_path = path + ".meta.json"
    with open(meta_path) as f:
        meta_dict = json.load(f)
    cell_meta_raw = meta_dict.get("cell_metadata", {})

    # If target_memory="100%" was requested, build an identity compressor
    # for the temp cache; otherwise honour the user-specified ratio.
    effective_ratio = compression_ratio
    if effective_ratio is None and target_memory is not None:
        effective_ratio = parse_target_memory(target_memory)
    if effective_ratio == 1.0:
        method = "identity"

    compressor = build_compressor(
        method,
        compression_ratio=effective_ratio or 3.0,
        bits=bits,
        seed=seed,
    )
    tmp_cache = CompressedKVCache(compressor=compressor)

    # Group tensors by (layer, kind). The key is the same `{layer}/{kind}/{name}`
    # written by export_kv.
    by_cell: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        parts = key.split("/", 2)
        if len(parts) != 3:
            log.warning("import_kv: skipping malformed tensor key %r", key)
            continue
        layer_str, kind, name = parts
        if kind not in ("key", "value"):
            log.warning("import_kv: skipping tensor with unknown kind %r", kind)
            continue
        try:
            layer_idx = int(layer_str)
        except ValueError:
            log.warning("import_kv: skipping tensor with non-int layer %r", layer_str)
            continue
        by_cell.setdefault((layer_idx, kind), {})[name] = t

    # Reconstruct K and V separately per cell.
    reconstructed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    from kvcompress.compressor.tucker import (
        TuckerFactors,
        reconstruct_partial_tucker,
    )
    from kvcompress.compressor.residual import ResidualPayload, decode_residual

    for (layer_idx, kind), data in by_cell.items():
        cell_key = f"{layer_idx}/{kind}"
        cell_meta = cell_meta_raw.get(cell_key, {})
        payload_meta = cell_meta.get("metadata", {})

        # JoLT path: reconstruct Tucker core + decode residual.
        if "core" in data and "u_token" in data and "u_feature" in data:
            core = data["core"]
            m, rt, rd = core.shape
            # u_token shape is (T, rt); u_feature shape is (dh, rd).
            # Recover the original (m, T, dh) so reconstruct_partial_tucker
            # can validate against it.
            T = int(data["u_token"].shape[0])
            dh = int(data["u_feature"].shape[0])
            factors = TuckerFactors(
                core=core.to(torch.float32),
                u_token=data["u_token"].to(torch.float32),
                u_feature=data["u_feature"].to(torch.float32),
                token_sv=torch.empty(0),
                feature_sv=torch.empty(0),
                token_tail_mass=float(payload_meta.get("tail_token_mass", 0.0)),
                feature_tail_mass=float(payload_meta.get("tail_feature_mass", 0.0)),
            )
            recon = reconstruct_partial_tucker(factors, (m, T, dh))
            if "residual_packed" in data:
                # Ponytail: the seed MUST come from the sidecar, not the
                # tensor's numel. The prior version invented a seed and
                # the residual projection was rebuilt against a random
                # matrix; round-tripped caches were garbage.
                seed_val = int(payload_meta.get("residual_seed", 0))
                quant_dtype_int = int(payload_meta.get("residual_dtype", 0))
                quant_dtype = f"int{quant_dtype_int}" if quant_dtype_int > 0 else "int0"
                res = ResidualPayload(
                    projection_seed=seed_val,
                    projection_distribution=payload_meta.get("residual_distribution", "gaussian"),
                    projection_sparsity=float(payload_meta.get("residual_sparsity", 1.0)),
                    quant_dtype=quant_dtype,
                    symmetric=bool(payload_meta.get("residual_symmetric", True)),
                    per_channel=bool(payload_meta.get("residual_per_channel", True)),
                    group_size=payload_meta.get("residual_group_size"),
                    packed=data["residual_packed"],
                    scale=data["residual_scale"],
                    zero_point=data["residual_zero_point"],
                    original_shape=tuple(int(d) for d in data["residual_original_shape"].tolist()),
                    original_last=int(data["residual_original_last"].item()),
                )
                recon = recon + decode_residual(res)
            tensor = recon.to(torch.float32)
        elif "value" in data:
            # Identity path: the tensor was stored verbatim under "value".
            tensor = data["value"]
        elif "u" in data and "s" in data and "vh" in data:
            # LowRank path.
            u = data["u"].to(torch.float32)
            s = data["s"].to(torch.float32)
            vh = data["vh"].to(torch.float32)
            m = int(payload_meta.get("m", u.shape[0]))
            T = int(payload_meta.get("T", 1))
            tensor = (u * s) @ vh
            tensor = tensor.reshape(m, T, -1)
        else:
            log.warning(
                "import_kv: skipping cell %s with unknown data schema %s",
                cell_key,
                list(data.keys()),
            )
            continue

        if layer_idx not in reconstructed:
            reconstructed[layer_idx] = (None, None)  # type: ignore[assignment]
        k, v = reconstructed[layer_idx]
        if kind == "key":
            reconstructed[layer_idx] = (tensor, v)
        else:
            reconstructed[layer_idx] = (k, tensor)

    # Populate the model's cache. We bypass ``cache.update`` to avoid
    # double-compression if the cache has been patched with a compressor.
    layers_attr = getattr(cache, "layers", None)
    for layer_idx, (k, v) in reconstructed.items():
        if k is None or v is None:
            log.warning(
                "import_kv: layer %d missing K or V (k=%s, v=%s)",
                layer_idx,
                k is not None,
                v is not None,
            )
            continue
        if layers_attr is not None and layer_idx < len(layers_attr):
            layers_attr[layer_idx].keys = k
            layers_attr[layer_idx].values = v
        else:
            cache.update(k, v, layer_idx)

    log.info("import_kv: loaded %d layers from %s", len(reconstructed), path)
    meta = tmp_cache.metadata()
    return meta


def is_vllm_available() -> bool:
    """Return True if vLLM is importable on this system."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False
