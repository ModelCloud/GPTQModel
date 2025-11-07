# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Safetensor-level dequantization helpers for common quant formats."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..quantization.dtype import dequantize_f4_e2m1, dequantize_f8_e4m3
from ..utils.logger import setup_logger


LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from compressed_tensors.compressors.base import BaseCompressor
    from compressed_tensors.quantization.quant_scheme import QuantizationScheme


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def list_safetensor_files(model_path: Path) -> Tuple[list, Optional[dict]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        files = sorted(set(index.get("weight_map", {}).values()))
        return files, index

    files = sorted([p.name for p in model_path.glob("*.safetensors")])
    return files, None


def finalize_for_save(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast to ``target_dtype`` when floating point and move to CPU with optimal layout."""

    if torch.is_floating_point(tensor):
        tensor = tensor.to(target_dtype)

    tensor_cpu = tensor.to("cpu")
    if tensor_cpu.ndim >= 4:
        tensor_cpu = tensor_cpu.contiguous(memory_format=torch.channels_last)
    return tensor_cpu


def normalize_device(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    device = device.strip()
    if device.lower() == "cpu":
        return None

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Unsupported device type: {device}")

    if dev.index is None:
        return "cuda:0"
    return f"cuda:{dev.index}"


def _get_compressed_tensors_dependencies() -> dict:
    try:
        from compressed_tensors.compressors.base import BaseCompressor
        from compressed_tensors.compressors.model_compressors.model_compressor import (
            map_module_to_scheme,
        )
        from compressed_tensors.quantization import QuantizationConfig
        from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'compressed-tensors' package. Install it with 'pip install compressed-tensors'."
        ) from exc

    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'accelerate' package. Install it with 'pip install accelerate'."
        ) from exc

    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Support for compressed-tensors quantized models requires the "
            "'transformers' package."
        ) from exc

    return {
        "QuantizationConfig": QuantizationConfig,
        "apply_quantization_config": apply_quantization_config,
        "BaseCompressor": BaseCompressor,
        "map_module_to_scheme": map_module_to_scheme,
        "init_empty_weights": init_empty_weights,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    }


def _discover_compressed_tensors_module_schemes(
    model_path: Path,
    quant_config,
    *,
    deps: dict,
) -> Dict[str, "QuantizationScheme"]:
    AutoConfig = deps["AutoConfig"]
    init_empty_weights = deps["init_empty_weights"]
    apply_quantization_config = deps["apply_quantization_config"]
    map_module_to_scheme = deps["map_module_to_scheme"]

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    loader_candidates = (
        deps["AutoModelForCausalLM"],
        deps["AutoModelForSeq2SeqLM"],
        deps["AutoModel"],
    )

    loader_errors: list[tuple[str, Exception]] = []
    model = None
    for loader in loader_candidates:
        if loader is None:
            continue

        with init_empty_weights(include_buffers=True):
            try:
                model = loader.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
            except Exception as exc:  # pragma: no cover - depends on available loaders
                loader_errors.append((loader.__name__, exc))
                continue
            else:
                break

    if model is None:
        if loader_errors:
            names = ", ".join(name for name, _ in loader_errors)
            last_error = loader_errors[-1][1]
            raise RuntimeError(
                f"Failed to instantiate model from '{model_path}' while inspecting "
                f"compressed-tensors modules. Loaders attempted: {names}."
            ) from last_error
        raise RuntimeError(
            f"Failed to instantiate model from '{model_path}' while inspecting "
            "compressed-tensors modules."
        )

    try:
        apply_quantization_config(model, quant_config, run_compressed=False)
        module_to_scheme = map_module_to_scheme(model)
    finally:
        del model

    return dict(module_to_scheme)


def _prepare_compressed_tensors_context(
    model_path: Path, quant_cfg: dict
) -> tuple["QuantizationConfig", Dict[str, "QuantizationScheme"], "BaseCompressor"]: # noqa
    deps = _get_compressed_tensors_dependencies()

    QuantizationConfig = deps["QuantizationConfig"]
    quant_config = QuantizationConfig.model_validate(quant_cfg)
    quant_format = (quant_config.format or "").lower()
    if quant_format != "pack-quantized":
        raise ValueError(
            f"Unsupported compressed-tensors format '{quant_config.format}'. "
            "Only 'pack-quantized' is currently supported."
        )

    module_to_scheme = _discover_compressed_tensors_module_schemes(
        model_path, quant_config, deps=deps
    )

    BaseCompressor = deps["BaseCompressor"]
    compressor = BaseCompressor.load_from_registry(quant_format, config=quant_config)
    return quant_config, module_to_scheme, compressor


def resolve_block_size(config: dict) -> Optional[Tuple[int, int]]:
    quant_cfg = config.get("quantization_config", {}) or {}
    block_size = quant_cfg.get("weight_block_size")
    if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])
    return None


def infer_block_shape(weight_shape: Tuple[int, int], scale_tensor: torch.Tensor) -> Tuple[int, int]:
    rows, cols = weight_shape
    shape = tuple(scale_tensor.shape)

    if scale_tensor.ndim == 0:
        LOG.debug(
            "Inferred block size (%d, %d) from scalar scale tensor for weight shape %s",
            rows,
            cols,
            weight_shape,
        )
        return rows, cols

    if shape == weight_shape:
        LOG.debug(
            "Inferred element-wise scaling (block size 1x1) for weight shape %s",
            weight_shape,
        )
        return 1, 1

    if scale_tensor.ndim == 2:
        block_rows = shape[0]
        block_cols = shape[1]
        if block_rows == 0 or block_cols == 0:
            raise ValueError("scale tensor has zero-sized dimension")
        if rows % block_rows != 0 or cols % block_cols != 0:
            raise ValueError("scale tensor shape incompatible with weight dimensions")
        inferred = (rows // block_rows, cols // block_cols)
        LOG.debug(
            "Inferred block size %s from 2D scale tensor shape %s and weight shape %s",
            inferred,
            shape,
            weight_shape,
        )
        return inferred

    if scale_tensor.ndim == 1:
        count = shape[0]
        if count == 0:
            raise ValueError("scale tensor is empty")

        candidates: list[Tuple[int, int]] = []
        for row_blocks in range(1, count + 1):
            if count % row_blocks != 0:
                continue
            if rows % row_blocks != 0:
                continue
            col_blocks = count // row_blocks
            if cols % col_blocks != 0:
                continue
            block_rows = rows // row_blocks
            block_cols = cols // col_blocks
            candidates.append((block_rows, block_cols))

        if candidates:
            candidates.sort(
                key=lambda bc: (
                    abs(bc[0] - bc[1]),
                    -min(bc),
                    -max(bc),
                    bc[0],
                )
            )
            inferred = candidates[0]
            LOG.debug(
                "Inferred block size %s from 1D scale tensor (count=%d) and weight shape %s",
                inferred,
                count,
                weight_shape,
            )
            return inferred

        if rows % count == 0:
            inferred = (rows // count, cols)
            LOG.debug(
                "Inferred row-only scaling block size %s from 1D scale tensor (count=%d)",
                inferred,
                count,
            )
            return inferred
        if cols % count == 0:
            inferred = (rows, cols // count)
            LOG.debug(
                "Inferred column-only scaling block size %s from 1D scale tensor (count=%d)",
                inferred,
                count,
            )
            return inferred

        raise ValueError("unable to infer block size from 1D scale tensor")

        raise ValueError("unsupported scale tensor rank for block size inference")


def detect_format(model_path: Path, config: dict) -> str:
    quant_cfg = config.get("quantization_config", {}) or {}
    method = (quant_cfg.get("quant_method") or "").lower()
    fmt = (quant_cfg.get("fmt") or "").lower()

    files, _ = list_safetensor_files(model_path)
    if not files:
        raise FileNotFoundError("No .safetensors files found in model directory")

    with safe_open(model_path / files[0], framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        # Prefer dtype-based detection
        for key in keys:
            if key.endswith(".weight"):
                tensor = reader.get_tensor(key)
                if tensor.dtype == torch.float8_e4m3fn:
                    LOG.debug("Detected FP8 weights via dtype on tensor '%s'", key)
                    return "fp8"
                if tensor.dtype == torch.uint8 and (key + "_scale") in keys:
                    LOG.debug("Detected NVFP4 weights via dtype on tensor '%s'", key)
                    return "nvfp4"
        if any(k.endswith(".weight_packed") for k in keys):
            LOG.debug(
                "Detected compressed-tensors pack-quantized format via '.weight_packed' "
                "metadata in shard '%s'",
                files[0],
            )
            return "compressed-pack"
        if any(k.endswith(".weight_scale") for k in keys):
            LOG.debug("Detected NVFP4 format via '.weight_scale' metadata in shard '%s'", files[0])
            return "nvfp4"
        if any(k.endswith(".weight_scale_inv") for k in keys):
            LOG.debug("Detected FP8 format via '.weight_scale_inv' metadata in shard '%s'", files[0])
            return "fp8"
        if any(k.endswith(".qweight") for k in keys):
            has_g = any(k.endswith(".g_idx") for k in keys)
            LOG.debug(
                "Detected %s format via qweight tensors in shard '%s'",
                "gptq" if has_g else "awq",
                files[0],
            )
            return "gptq" if has_g else "awq"

    if fmt == "float8_e4m3fn":
        LOG.debug("Detected FP8 format via config fmt=%s", fmt)
        return "fp8"
    if method in ("gptq", "gptqmodel"):
        LOG.debug("Detected GPTQ format via quant_method=%s", method)
        return "gptq"
    if method == "awq":
        LOG.debug("Detected AWQ format via quant_method=%s", method)
        return "awq"
    if method == "compressed-tensors":
        fmt_name = (quant_cfg.get("format") or "").lower()
        if fmt_name == "pack-quantized":
            LOG.debug(
                "Detected compressed-tensors format via quant_method=%s and format=%s",
                method,
                fmt_name,
            )
            return "compressed-pack"

    raise ValueError("Unable to detect quantization format for model")


def unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows, cols * pack_factor, dtype=torch.int32)
    for i in range(pack_factor):
        result[:, i::pack_factor] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows * pack_factor, cols, dtype=torch.int32)
    for i in range(pack_factor):
        result[i::pack_factor, :] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def convert_fp8_shard(
    reader,
    target_dtype: torch.dtype,
    *,
    block_shape: Optional[Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        tensor = reader.get_tensor(key)
        if key.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            if scale_key not in reader.keys():
                raise KeyError(f"Missing scale inverse tensor for {key}")
            scale_inv = reader.get_tensor(scale_key)
            LOG.debug("Using scale_inv tensor '%s' for FP8 weight '%s'", scale_key, key)

            rows, cols = tensor.shape
            effective_block = block_shape
            if effective_block is None:
                try:
                    effective_block = infer_block_shape((rows, cols), scale_inv)
                    LOG.debug("Inferred block size %s for weight '%s'", effective_block, key)
                except ValueError as exc:
                    LOG.debug(
                        "Falling back to full-tensor block size for weight '%s' (%s)",
                        key,
                        exc,
                    )
                    effective_block = (rows, cols)
            else:
                LOG.debug("Using configured block size %s for weight '%s'", effective_block, key)

            block_rows, block_cols = effective_block
            if block_rows <= 0 or block_cols <= 0:
                raise ValueError(f"Inferred invalid block size {effective_block} for {key}")
            if rows % block_rows != 0 or cols % block_cols != 0:
                raise ValueError(
                    f"Tensor {key} shape {tensor.shape} incompatible with block size {effective_block}"
                )

            deq = dequantize_f8_e4m3(
                tensor,
                scale_inv=scale_inv,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = finalize_for_save(deq, target_dtype)
        elif key.endswith("_scale_inv"):
            LOG.debug("Dropping auxiliary FP8 tensor '%s' after dequantization", key)
            continue
        else:
            tensors[key] = finalize_for_save(tensor, target_dtype)
    return tensors


def convert_nvfp4_shard(reader, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in reader.keys():
        tensor = reader.get_tensor(key)
        if key.endswith(".weight") and tensor.dtype == torch.uint8:
            scale_key = key + "_scale"
            if scale_key not in reader.keys():
                raise KeyError(f"Missing scale tensor for {key}")
            scale = reader.get_tensor(scale_key)
            LOG.debug("Using scale tensor '%s' for NVFP4 weight '%s'", scale_key, key)
            deq = dequantize_f4_e2m1(
                tensor,
                scale=scale,
                axis=None,
                target_dtype=target_dtype,
            )
            tensors[key] = finalize_for_save(deq, target_dtype)
        elif key.endswith("_weight_scale"):
            LOG.debug("Dropping auxiliary NVFP4 tensor '%s' after dequantization", key)
            continue
        else:
            tensors[key] = finalize_for_save(tensor, target_dtype)
    return tensors


def convert_awq_file(path: Path, target_dtype: torch.dtype, device: str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
                LOG.debug("Collected AWQ qweight tensor '%s'", key)
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
                LOG.debug("Collected AWQ qzeros tensor '%s'", key)
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
                LOG.debug("Collected AWQ scale tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete AWQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        bits = 4

        unpacked_weight = unpack_cols(qweight, bits).to(torch.float32)
        unpacked_zeros = unpack_cols(qzeros, bits).to(torch.float32)

        num_groups = scales.shape[0]
        group_size = unpacked_weight.shape[0] // num_groups
        scales_full = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
        zeros_full = unpacked_zeros.repeat_interleave(group_size, dim=0)

        weight = (unpacked_weight - zeros_full) * scales_full
        tensors[prefix + ".weight"] = finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        LOG.debug("Dequantized AWQ module '%s' to dtype %s", prefix, target_dtype)
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def convert_gptq_file(path: Path, target_dtype: torch.dtype, config: dict, device: str) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if key.endswith(".qweight"):
                prefix = key[:-len(".qweight")]
                module_buffers[prefix]["qweight"] = tensor
                LOG.debug("Collected GPTQ qweight tensor '%s'", key)
            elif key.endswith(".qzeros"):
                prefix = key[:-len(".qzeros")]
                module_buffers[prefix]["qzeros"] = tensor
                LOG.debug("Collected GPTQ qzeros tensor '%s'", key)
            elif key.endswith(".scales"):
                prefix = key[:-len(".scales")]
                module_buffers[prefix]["scales"] = tensor
                LOG.debug("Collected GPTQ scale tensor '%s'", key)
            elif key.endswith(".g_idx"):
                prefix = key[:-len(".g_idx")]
                module_buffers[prefix]["g_idx"] = tensor
                LOG.debug("Collected GPTQ g_idx tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        missing = {k for k in ("qweight", "qzeros", "scales", "g_idx") if k not in buf}
        if missing:
            raise KeyError(f"Incomplete GPTQ buffers for module {prefix}: missing {missing}")

        qweight = buf["qweight"]
        qzeros = buf["qzeros"]
        scales = buf["scales"]
        g_idx = buf["g_idx"].to(torch.long)

        bits = config.get("bits", 4)
        weight_int = unpack_rows(qweight, bits)
        zeros = unpack_cols(qzeros, bits)

        scales_full = scales.to(torch.float32)[g_idx]
        zeros_full = zeros.to(torch.float32)[g_idx]
        weight = (weight_int.to(torch.float32) - zeros_full) * scales_full
        tensors[prefix + ".weight"] = finalize_for_save(
            weight.to(target_dtype).t().contiguous(),
            target_dtype,
        )
        LOG.debug(
            "Dequantized GPTQ module '%s' with %d-bit groups to dtype %s",
            prefix,
            bits,
            target_dtype,
        )
        if prefix + ".bias" in tensors:
            tensors[prefix + ".bias"] = finalize_for_save(tensors[prefix + ".bias"], target_dtype)

    return tensors


def convert_compressed_pack_file(
    path: Path,
    target_dtype: torch.dtype,
    *,
    device: str,
    module_to_scheme: Dict[str, "QuantizationScheme"],
    compressor: "BaseCompressor",
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    module_buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    with safe_open(path, framework="pt", device=device) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if key.endswith(".weight_packed"):
                prefix = key[: -len(".weight_packed")]
                module_buffers[prefix]["weight_packed"] = tensor
                LOG.debug("Collected compressed weight tensor '%s'", key)
            elif key.endswith(".weight_scale"):
                prefix = key[: -len(".weight_scale")]
                module_buffers[prefix]["weight_scale"] = tensor
                LOG.debug("Collected compressed scale tensor '%s'", key)
            elif key.endswith(".weight_zero_point"):
                prefix = key[: -len(".weight_zero_point")]
                module_buffers[prefix]["weight_zero_point"] = tensor
                LOG.debug("Collected compressed zero-point tensor '%s'", key)
            elif key.endswith(".weight_g_idx"):
                prefix = key[: -len(".weight_g_idx")]
                module_buffers[prefix]["weight_g_idx"] = tensor
                LOG.debug("Collected compressed group-index tensor '%s'", key)
            elif key.endswith(".weight_shape"):
                prefix = key[: -len(".weight_shape")]
                module_buffers[prefix]["weight_shape"] = tensor
                LOG.debug("Collected compressed shape tensor '%s'", key)
            else:
                tensors[key] = finalize_for_save(tensor, target_dtype)

    for prefix, buf in module_buffers.items():
        scheme = module_to_scheme.get(prefix)
        if scheme is None:
            raise KeyError(
                f"No quantization scheme registered for compressed module '{prefix}'."
            )

        if scheme.weights is None:
            raise ValueError(
                f"Module '{prefix}' does not define weight quantization parameters."
            )

        required_fields = {"weight_packed", "weight_scale", "weight_shape"}
        missing = required_fields.difference(buf)
        if missing:
            raise KeyError(
                f"Compressed tensors for module '{prefix}' in shard '{path.name}' "
                f"are missing required fields: {sorted(missing)}"
            )

        weight = compressor.decompress_weight(
            compressed_data=buf,
            quantization_args=scheme.weights,
        )
        tensors[prefix + ".weight"] = finalize_for_save(weight, target_dtype)

    return tensors


def copy_aux_files(model_path: Path, output_path: Path, skip: Iterable[str]) -> None:
    for item in model_path.iterdir():
        if item.name in skip:
            continue
        target = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def dequantize_model(
    model_path: Path | str,
    output_path: Path | str,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> None:
    model_path = Path(model_path)
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"Output path {output_path} already exists")

    output_path.mkdir(parents=True)

    config = load_json(model_path / "config.json")
    quant_cfg = config.get("quantization_config", {}) or {}
    fmt = detect_format(model_path, config)

    files, index = list_safetensor_files(model_path)
    if not files:
        raise RuntimeError("No safetensor files to convert")

    device_str = normalize_device(device)
    if device_str is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU dequantization")
        torch.cuda.set_device(torch.device(device_str))
    open_device = device_str or "cpu"

    block_shape = resolve_block_size(config) if fmt == "fp8" else None

    if block_shape is not None:
        LOG.debug("Configured FP8 block size %s found in quantization_config", block_shape)
    else:
        LOG.debug("No explicit FP8 block size found; will infer from scale tensors if needed")

    compressed_module_to_scheme: Dict[str, "QuantizationScheme"] = {}
    compressed_compressor: Optional["BaseCompressor"] = None
    if fmt == "compressed-pack":
        if not quant_cfg:
            raise ValueError(
                "compressed-tensors model requires a populated 'quantization_config' entry."
            )
        _, compressed_module_to_scheme, compressed_compressor = _prepare_compressed_tensors_context(
            model_path, quant_cfg
        )
        LOG.debug(
            "Prepared compressed-tensors context with %d modules",
            len(compressed_module_to_scheme),
        )

    log = setup_logger()
    LOG.debug(
        "Starting dequantization for model '%s' (format=%s, target_dtype=%s, device=%s)",
        model_path,
        fmt,
        target_dtype,
        open_device,
    )
    pb = log.pb(range(len(files))).manual().set(show_left_steps=False).title(f"Dequantizing ({fmt})")
    pb.draw()

    weight_map: Dict[str, str] = {}
    total_size = 0

    try:
        for idx, filename in enumerate(files):
            path = model_path / filename
            LOG.debug("Processing shard '%s' for format %s on device %s", filename, fmt, open_device)
            if fmt == "fp8":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = convert_fp8_shard(reader, target_dtype, block_shape=block_shape)
            elif fmt == "nvfp4":
                with safe_open(path, framework="pt", device=open_device) as reader:
                    tensors = convert_nvfp4_shard(reader, target_dtype)
            elif fmt == "awq":
                tensors = convert_awq_file(path, target_dtype, open_device)
            elif fmt == "gptq":
                tensors = convert_gptq_file(path, target_dtype, quant_cfg, open_device)
            elif fmt == "compressed-pack":
                if compressed_compressor is None:
                    raise RuntimeError("Compressed-tensors compressor was not initialized")
                tensors = convert_compressed_pack_file(
                    path,
                    target_dtype,
                    device=open_device,
                    module_to_scheme=compressed_module_to_scheme,
                    compressor=compressed_compressor,
                )
            else:
                raise ValueError(f"Unsupported format {fmt}")

            save_file(tensors, str(output_path / filename))
            weight_map.update({str(name): filename for name in tensors})
            total_size += sum(t.element_size() * t.numel() for t in tensors.values())
            pb.subtitle(filename).next().draw()
    finally:
        pb.close()

    if index is not None:
        new_index = dict(index)
    else:
        new_index = {}
    metadata = dict(new_index.get("metadata", {}))
    metadata["total_size"] = total_size
    new_index["metadata"] = metadata
    new_index["weight_map"] = weight_map
    write_json(output_path / "model.safetensors.index.json", new_index)

    new_config = dict(config)
    new_config.pop("quantization_config", None)
    new_config["torch_dtype"] = str(target_dtype).split(".")[-1]
    write_json(output_path / "config.json", new_config)

    skip_files = set(files) | {"config.json", "model.safetensors.index.json"}
    copy_aux_files(model_path, output_path, skip_files)
