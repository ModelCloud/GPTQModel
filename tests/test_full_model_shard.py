# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Full-model per-layer sharding integration tests.

These tests exercise both the standalone ``reshard()`` API and the
``model.save()`` per-layer path against the local model cache under
``/monster/data/model``.  They are skipped automatically when the referenced
checkpoints are not present, so they can live in the repo and run in
environments where the cache is mounted.
"""

import json
import os
import tempfile
from typing import List, Optional

import pytest
from safetensors import safe_open
from transformers import AutoConfig

from gptqmodel import BACKEND, GPTQModel, ShardStrategy, get_best_device
from gptqmodel.quantization.config import (
    ExpertsRoutingOverride,
    MoEConfig,
    QuantizeConfig,
)
from gptqmodel.utils.reshard import reshard
from gptqmodel.utils.torch import torch_empty_cache


LLAMA_MODEL_DIR = "/monster/data/model/Llama-3.2-1B-Instruct"
MODEL_ROOT = "/monster/data/model"


def _model_type(path: str) -> Optional[str]:
    """Read ``model_type`` from config.json without importing model code."""
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type")
    except Exception:
        return None


def _weight_file_size(path: str) -> int:
    """Sum bytes of safetensors/bin/pt weight files under ``path``."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                total += os.path.getsize(os.path.join(root, f))
    return total


def _is_qwen3_moe_dir(path: str) -> bool:
    """Return ``True`` if ``path`` is a Qwen3-MoE checkpoint.

    Requires a readable ``config.json`` with ``model_type == "qwen3_moe"``;
    no directory-name heuristic is used so non-Qwen3-MoE or incomplete
    config-only folders are not selected.
    """
    return _model_type(path) == "qwen3_moe"


def _find_smallest_qwen3_moe() -> Optional[str]:
    """Return the smallest Qwen3-MoE checkpoint under ``/monster/data/model``."""
    if not os.path.isdir(MODEL_ROOT):
        return None

    candidates = [
        os.path.join(MODEL_ROOT, entry)
        for entry in os.listdir(MODEL_ROOT)
        if os.path.isdir(os.path.join(MODEL_ROOT, entry))
        and _is_qwen3_moe_dir(os.path.join(MODEL_ROOT, entry))
    ]

    # Only consider checkpoints that actually contain weight files.
    sized = [(p, _weight_file_size(p)) for p in candidates]
    sized = [(p, s) for p, s in sized if s > 0]
    if not sized:
        return None

    return min(sized, key=lambda item: item[1])[0]


def _skip_if_missing(path: str) -> None:
    if not os.path.isdir(path):
        pytest.skip(f"Checkpoint not available: {path}")


def _has_safetensors(path: str) -> bool:
    return any(
        f.endswith(".safetensors")
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )


def _has_weight_files(path: str) -> bool:
    return any(
        f.endswith((".safetensors", ".bin", ".pt"))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )


def _is_quantized_checkpoint(path: str) -> bool:
    return any(
        os.path.isfile(os.path.join(path, name))
        for name in ("quantize_config.json", "quant_config.json")
    )


def _output_safetensors(target: str) -> List[str]:
    return sorted(
        f
        for f in os.listdir(target)
        if f.endswith(".safetensors") and not f.startswith(".")
    )


def _assert_per_layer_layout(target: str) -> None:
    """Verify that ``target`` is a valid per-layer sharded checkpoint."""
    index_path = os.path.join(target, "model.safetensors.index.json")
    assert os.path.isfile(index_path), f"Missing safetensors index in {target}"

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    assert weight_map, "Empty weight_map in index"

    shards = _output_safetensors(target)
    assert shards, f"No output safetensors files in {target}"
    assert any("model-" in s and "-of-" in s for s in shards), (
        f"No model-XXXXX-of-YYYYY shards in {shards}"
    )

    for name, shard in weight_map.items():
        assert os.path.isfile(os.path.join(target, shard)), (
            f"{name} points to missing shard {shard}"
        )


def _assert_format_metadata(target: str) -> None:
    """Spot-check that output shards carry the expected safetensors metadata."""
    shards = _output_safetensors(target)
    assert shards
    with safe_open(os.path.join(target, shards[0]), framework="pt", device="cpu") as handler:
        metadata = handler.metadata()
    assert metadata.get("format") == "pt"


def _synthetic_calibration(size: int = 8) -> List[str]:
    """Return a small list of text strings usable for quantization."""
    base = "The quick brown fox jumps over the lazy dog. "
    return [base * (i + 2) for i in range(size)]


def _is_moe_model(path: str, trust_remote_code: bool = False) -> bool:
    try:
        config = AutoConfig.from_pretrained(
            path, trust_remote_code=trust_remote_code, local_files_only=True
        )
    except Exception:
        return False
    return (
        getattr(config, "num_experts", 0) > 0
        or getattr(config, "n_routed_experts", 0) > 0
        or "moe" in getattr(config, "model_type", "").lower()
    )


def _build_quantize_config(path: str, trust_remote_code: bool = False) -> QuantizeConfig:
    if _is_moe_model(path, trust_remote_code=trust_remote_code):
        return QuantizeConfig(
            bits=4,
            group_size=128,
            moe=MoEConfig(routing=ExpertsRoutingOverride()),
        )
    return QuantizeConfig(bits=4, group_size=128)


def _load_and_quantize_if_needed(
    path: str,
    trust_remote_code: bool = False,
):
    """Load a checkpoint, quantizing dense checkpoints first if necessary."""
    device = get_best_device(BACKEND.GPTQ_TORCH)

    if _is_quantized_checkpoint(path):
        model = GPTQModel.load(
            path,
            backend=BACKEND.GPTQ_TORCH,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    else:
        qcfg = _build_quantize_config(path, trust_remote_code=trust_remote_code)
        model = GPTQModel.load(
            path,
            quantize_config=qcfg,
            backend=BACKEND.GPTQ_TORCH,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        model.quantize(_synthetic_calibration(), batch_size=1)

    return model


@pytest.fixture(scope="module")
def qwen3_moe_dir() -> Optional[str]:
    path = _find_smallest_qwen3_moe()
    if path is None:
        pytest.skip("No Qwen3-MoE checkpoint found in /monster/data/model")
    return path


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_reshard_llama_3_2_1b_instruct():
    _skip_if_missing(LLAMA_MODEL_DIR)
    if not _has_safetensors(LLAMA_MODEL_DIR):
        pytest.skip("Llama-3.2-1B-Instruct has no .safetensors files")

    with tempfile.TemporaryDirectory() as target:
        reshard(
            LLAMA_MODEL_DIR,
            target,
            strategy=ShardStrategy.PER_LAYER,
            overwrite=True,
            progress=False,
        )
        _assert_per_layer_layout(target)
        _assert_format_metadata(target)


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_reshard_smallest_qwen3_moe(qwen3_moe_dir: str):
    if not _has_safetensors(qwen3_moe_dir):
        pytest.skip("Smallest Qwen3-MoE checkpoint has no .safetensors files")

    with tempfile.TemporaryDirectory() as target:
        reshard(
            qwen3_moe_dir,
            target,
            strategy=ShardStrategy.PER_LAYER,
            overwrite=True,
            progress=False,
            trust_remote_code=True,
        )
        _assert_per_layer_layout(target)
        _assert_format_metadata(target)


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_save_per_layer_llama_3_2_1b_instruct():
    _skip_if_missing(LLAMA_MODEL_DIR)
    if not _has_weight_files(LLAMA_MODEL_DIR):
        pytest.skip("Llama-3.2-1B-Instruct has no weight files")

    model = None
    try:
        model = _load_and_quantize_if_needed(LLAMA_MODEL_DIR, trust_remote_code=False)
        kernel = model.qlinear_kernel
        if not getattr(kernel, "SUPPORTS_SHARDS", False):
            kernel_name = getattr(kernel, "__name__", repr(kernel))
            pytest.skip(f"{kernel_name} does not support sharded checkpoints")

        with tempfile.TemporaryDirectory() as target:
            model.save(target, shard_strategy=ShardStrategy.PER_LAYER)
            _assert_per_layer_layout(target)
            _assert_format_metadata(target)

            # Verify the saved per-layer checkpoint can be reloaded.
            loaded = GPTQModel.load(
                target,
                backend=BACKEND.GPTQ_TORCH,
                device=get_best_device(BACKEND.GPTQ_TORCH),
            )
            assert loaded.quantized
            del loaded
    finally:
        if model is not None:
            del model
        torch_empty_cache()


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_save_per_layer_smallest_qwen3_moe(qwen3_moe_dir: str):
    _skip_if_missing(qwen3_moe_dir)
    if not _has_weight_files(qwen3_moe_dir):
        pytest.skip("Smallest Qwen3-MoE checkpoint has no weight files")

    model = None
    try:
        model = _load_and_quantize_if_needed(qwen3_moe_dir, trust_remote_code=True)
        kernel = model.qlinear_kernel
        if not getattr(kernel, "SUPPORTS_SHARDS", False):
            kernel_name = getattr(kernel, "__name__", repr(kernel))
            pytest.skip(f"{kernel_name} does not support sharded checkpoints")

        with tempfile.TemporaryDirectory() as target:
            model.save(target, shard_strategy=ShardStrategy.PER_LAYER)
            _assert_per_layer_layout(target)
            _assert_format_metadata(target)

            loaded = GPTQModel.load(
                target,
                backend=BACKEND.GPTQ_TORCH,
                device=get_best_device(BACKEND.GPTQ_TORCH),
                trust_remote_code=True,
            )
            assert loaded.quantized
            del loaded
    finally:
        if model is not None:
            del model
        torch_empty_cache()
