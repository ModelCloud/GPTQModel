# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import pytest
import torch

import gptqmodel
import gptqmodel.exllamav3.ext as exllamav3_ext
import gptqmodel.extension as extension_api
import gptqmodel.nn_modules.triton_utils.planar as planar_utils
import gptqmodel.utils.adjacent_exact as adjacent_exact_utils
import gptqmodel.utils.amplin as amplin_utils
import gptqmodel.utils.awq as awq_utils
import gptqmodel.utils.cannoe as cannoe_utils
import gptqmodel.utils.cpp as cpp_utils
import gptqmodel.utils.diagnostic_metrics as diagnostic_metrics_utils
import gptqmodel.utils.exllamav2 as exllamav2_utils
import gptqmodel.utils.gptq_block as gptq_block_utils
import gptqmodel.utils.grasshopper as grasshopper_utils
import gptqmodel.utils.hadamard as hadamard_utils
import gptqmodel.utils.machete as machete_utils
import gptqmodel.utils.marlin as marlin_utils
import gptqmodel.utils.marlin_lora as marlin_lora_utils
import gptqmodel.utils.marlin_moe as marlin_moe_utils
import gptqmodel.utils.pangolin as pangolin_utils
import gptqmodel.utils.paroquant as paroquant_utils
import gptqmodel.utils.qqq as qqq_utils
import gptqmodel.utils.qvq_cuda as qvq_cuda_utils
import gptqmodel.utils.swordfish as swordfish_utils
import gptqmodel.utils.trilin as trilin_utils
import gptqmodel_ext.planar as planar_api


class _FakeExtension:
    def __init__(self, name: str, *, ok: bool = True, error: str = "", already_loaded: bool = False):
        self.display_name = name
        self.ok = ok
        self.error = error
        self.already_loaded = already_loaded
        self.load_calls = 0
        self.clear_cache_calls = 0
        self.max_parallel_loads = 0
        self._active_loads = 0
        self._active_lock = threading.Lock()
        self._ops = {"test_op": object()}

    def _ops_available(self) -> bool:
        return self.already_loaded

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1

    def load(self) -> bool:
        self.load_calls += 1
        with self._active_lock:
            self._active_loads += 1
            self.max_parallel_loads = max(self.max_parallel_loads, self._active_loads)
        time.sleep(0.02)
        with self._active_lock:
            self._active_loads -= 1
        return self.ok

    def last_error_message(self) -> str:
        return self.error

    def namespace_object(self) -> object:
        return self

    def op(self, op_name: str):
        return self._ops[op_name]


def _install_fake_extensions(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fakes = {
        "adjacent_exact": _FakeExtension("AdjacentExact CUDA"),
        "pack_block_cpu": _FakeExtension("pack_block_cpu"),
        "gptq_block": _FakeExtension("GPTQ CUDA block quantization"),
        "qvq_cuda": _FakeExtension("QVQ planar CUDA GEMV"),
        "floatx_cpu": _FakeExtension("floatx_cpu"),
        "diagnostic_metrics_cpu": _FakeExtension("diagnostic_metrics_cpu"),
        "diagnostic_metrics_cuda": _FakeExtension("diagnostic_metrics_cuda"),
        "awq": _FakeExtension("AWQ"),
        "qqq": _FakeExtension("QQQ"),
        "exllamav2": _FakeExtension("ExLlamaV2 GPTQ"),
        "exllamav2_awq": _FakeExtension("ExLlamaV2 AWQ"),
        "exllamav3": _FakeExtension("ExLlamaV3"),
        "machete": _FakeExtension("Machete"),
        "marlin_fp16": _FakeExtension("Marlin fp16"),
        "marlin_bf16": _FakeExtension("Marlin bf16"),
        "marlin_lora": _FakeExtension("Marlin fused LoRA"),
        "marlin_moe": _FakeExtension("Marlin MoE"),
        "trilin": _FakeExtension("Trilin native 3-bit WMMA"),
        "amplin": _FakeExtension("Amplin Ampere GPTQ W4A16 GEMV"),
        "pangolin": _FakeExtension("Pangolin planar GPTQ GEMV"),
        "pangolin_cpu": _FakeExtension("Pangolin planar GPTQ GEMV CPU"),
        "grasshopper": _FakeExtension("GrassHopper GPTQ grouped GEMV/GEMM"),
        "swordfish": _FakeExtension("Swordfish"),
        "paroquant": _FakeExtension("ParoQuant rotation"),
        "hadamard": _FakeExtension("Fast Hadamard transform"),
        "cannoe": _FakeExtension("Cannoe V3"),
        "cannoe_ascendc": _FakeExtension("Cannoe Ascend C"),
    }

    monkeypatch.setattr(
        adjacent_exact_utils,
        "_ADJACENT_EXACT_TORCH_OPS_EXTENSION",
        fakes["adjacent_exact"],
    )
    monkeypatch.setattr(
        adjacent_exact_utils, "adjacent_exact_cuda_supported", lambda: True
    )
    monkeypatch.setattr(cpp_utils, "_pack_block_extension", lambda: fakes["pack_block_cpu"])
    monkeypatch.setattr(
        gptq_block_utils,
        "_GPTQ_BLOCK_TORCH_OPS_EXTENSION",
        fakes["gptq_block"],
    )
    monkeypatch.setattr(gptq_block_utils, "gptq_block_cuda_supported", lambda: True)
    monkeypatch.setattr(qvq_cuda_utils, "_QVQ_CUDA_TORCH_OPS_EXTENSION", fakes["qvq_cuda"])
    monkeypatch.setattr(qvq_cuda_utils, "qvq_cuda_supported", lambda: True)
    monkeypatch.setattr(cpp_utils, "_floatx_cpu_extension", lambda: fakes["floatx_cpu"])
    monkeypatch.setattr(
        diagnostic_metrics_utils,
        "_DIAGNOSTIC_METRICS_CPU_EXTENSION",
        fakes["diagnostic_metrics_cpu"],
    )
    monkeypatch.setattr(
        diagnostic_metrics_utils,
        "_DIAGNOSTIC_METRICS_CUDA_EXTENSION",
        fakes["diagnostic_metrics_cuda"],
    )
    monkeypatch.setattr(awq_utils, "_AWQ_TORCH_OPS_EXTENSION", fakes["awq"])
    monkeypatch.setattr(qqq_utils, "_QQQ_TORCH_OPS_EXTENSION", fakes["qqq"])
    monkeypatch.setattr(exllamav2_utils, "_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION", fakes["exllamav2"])
    monkeypatch.setattr(exllamav2_utils, "_EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION", fakes["exllamav2_awq"])
    monkeypatch.setattr(exllamav3_ext, "_EXLLAMAV3_TORCH_OPS_EXTENSION", fakes["exllamav3"])
    monkeypatch.setattr(machete_utils, "_MACHETE_TORCH_OPS_EXTENSION", fakes["machete"])
    monkeypatch.setattr(machete_utils, "_validate_machete_device_support", lambda: True)
    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fakes["marlin_fp16"])
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", fakes["marlin_bf16"])
    monkeypatch.setattr(marlin_lora_utils, "_MARLIN_LORA_TORCH_OPS_EXTENSION", fakes["marlin_lora"])
    monkeypatch.setattr(marlin_lora_utils, "marlin_lora_supported", lambda: True)
    monkeypatch.setattr(marlin_moe_utils, "_MARLIN_MOE_TORCH_OPS_EXTENSION", fakes["marlin_moe"])
    monkeypatch.setattr(trilin_utils, "_TRILIN_TORCH_OPS_EXTENSION", fakes["trilin"])
    monkeypatch.setattr(amplin_utils, "_AMPLIN_TORCH_OPS_EXTENSION", fakes["amplin"])
    monkeypatch.setattr(amplin_utils, "amplin_supported", lambda: True)
    monkeypatch.setattr(pangolin_utils, "_PANGOLIN_TORCH_OPS_EXTENSION", fakes["pangolin"])
    monkeypatch.setattr(pangolin_utils, "_PANGOLIN_CPU_TORCH_OPS_EXTENSION", fakes["pangolin_cpu"])
    monkeypatch.setattr(pangolin_utils, "pangolin_supported", lambda: True)
    monkeypatch.setattr(pangolin_utils, "pangolin_cpu_supported", lambda: True)
    monkeypatch.setattr(
        grasshopper_utils,
        "_GRASSHOPPER_TORCH_OPS_EXTENSION",
        fakes["grasshopper"],
    )
    monkeypatch.setattr(grasshopper_utils, "grasshopper_supported", lambda: True)
    monkeypatch.setattr(swordfish_utils, "_SWORDFISH_TORCH_OPS_EXTENSION", fakes["swordfish"])
    monkeypatch.setattr(swordfish_utils, "_validate_swordfish_device_support", lambda: True)
    monkeypatch.setattr(paroquant_utils, "_PAROQUANT_ROTATION_EXTENSION", fakes["paroquant"])
    monkeypatch.setattr(hadamard_utils, "_HADAMARD_TORCH_OPS_EXTENSION", fakes["hadamard"])
    monkeypatch.setattr(hadamard_utils, "hadamard_supported", lambda: True)
    monkeypatch.setattr(cannoe_utils, "_CANNOE_V3_TORCH_OPS_EXTENSION", fakes["cannoe"])
    monkeypatch.setattr(cannoe_utils, "_CANNOE_ASCENDC_TORCH_OPS_EXTENSION", fakes["cannoe_ascendc"])
    monkeypatch.setattr(cannoe_utils, "_cannoe_v3_supported", lambda: True)
    monkeypatch.setattr(cannoe_utils, "_cannoe_ascendc_supported", lambda: True)

    return fakes


def test_package_root_exports_extension_module():
    assert gptqmodel.extension is extension_api


def test_extension_api_locks_are_initialized_eagerly():
    assert set(extension_api._EXTENSION_API_LOCKS) == set(extension_api.available_extensions())


def test_planar_external_api_proxies_shared_implementation():
    api = planar_api.PangolinAPI()

    assert planar_api.__all__ == ["PangolinAPI"]
    assert api.bits == pangolin_utils.PANGOLIN_BITS
    assert api.max_m == pangolin_utils.PANGOLIN_MAX_M
    assert api.supported_m == pangolin_utils.PANGOLIN_SUPPORTED_M
    assert api.planar_fused_max_m == planar_utils.PLANAR_FUSED_MAX_M
    assert api.planar_gemv_max_m == planar_utils.PLANAR_GEMV_MAX_M
    assert api.supported is pangolin_utils.pangolin_supported
    assert api.runtime_available is pangolin_utils.pangolin_runtime_available
    assert api.ensure_runtime_available is pangolin_utils.ensure_pangolin_runtime_available
    assert api.runtime_error is pangolin_utils.pangolin_runtime_error
    assert api.cpu_supported is pangolin_utils.pangolin_cpu_supported
    assert api.cpu_runtime_available is pangolin_utils.pangolin_cpu_runtime_available
    assert api.ensure_cpu_runtime_available is pangolin_utils.ensure_pangolin_cpu_runtime_available
    assert api.cpu_runtime_error is pangolin_utils.pangolin_cpu_runtime_error
    assert api.g_idx_block_uniform is pangolin_utils.g_idx_block_uniform
    assert api.gemv is pangolin_utils.pangolin_gemv
    assert api.dequant is planar_utils.planar_dequant
    assert api.planar_gemv is planar_utils.planar_gemv
    assert api.planar_matmul is planar_utils.planar_matmul


def test_load_adjacent_exact_cuda_alias_builds_solver_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="adjacent_exact_cuda")

    assert result == {"adjacent_exact": True}
    assert fakes["adjacent_exact"].load_calls == 1


def test_load_gptq_block_cuda_alias_builds_native_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="gptq_block_cuda")

    assert result == {"gptq_block": True}
    assert fakes["gptq_block"].load_calls == 1


def test_load_qvq_gemv_alias_builds_native_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="qvq_gemv")

    assert result == {"qvq_cuda": True}
    assert fakes["qvq_cuda"].load_calls == 1


def test_qvq_extension_exposes_no_legacy_alias():
    legacy_name = "q" + "tip"

    assert legacy_name not in extension_api.available_extensions()
    with pytest.raises(ValueError, match="Unknown extension"):
        extension_api.load(name=f"{legacy_name}_cuda")


def test_load_defaults_to_all_extensions(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load()

    assert result == {
        "adjacent_exact": True,
        "pack_block_cpu": True,
        "gptq_block": True,
        "qvq_cuda": True,
        "floatx_cpu": True,
        "diagnostic_metrics_cpu": True,
        "diagnostic_metrics_cuda": True,
        "awq": True,
        "qqq": True,
        "exllamav2": True,
        "exllamav2_awq": True,
        "exllamav3": True,
        "machete": True,
        "marlin_fp16": True,
        "marlin_bf16": True,
        "trilin": True,
        "amplin": True,
        "pangolin": True,
        "pangolin_cpu": True,
        "marlin_lora": True,
        "marlin_moe": True,
        "grasshopper": True,
        "swordfish": True,
        "paroquant": True,
        "hadamard": True,
        "cannoe": True,
        "cannoe_ascendc": True,
    }
    assert all(fake.load_calls == 1 for fake in fakes.values())


def test_load_all_skips_extensions_unsupported_on_this_host(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(machete_utils, "_validate_machete_device_support", lambda: False)

    result = extension_api.load()

    assert "machete" not in result
    assert fakes["machete"].load_calls == 0


def test_load_specific_unsupported_extension_raises_without_building(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(machete_utils, "_validate_machete_device_support", lambda: False)
    monkeypatch.setattr(machete_utils, "machete_runtime_error", lambda: "Machete unsupported on this device.")

    with pytest.raises(RuntimeError, match="Machete unsupported on this device."):
        extension_api.load(name="machete")

    assert fakes["machete"].load_calls == 0


def test_load_marlin_alias_builds_both_variants(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="marlin")

    assert result == {"marlin_fp16": True, "marlin_bf16": True}
    assert fakes["marlin_fp16"].load_calls == 1
    assert fakes["marlin_bf16"].load_calls == 1
    assert fakes["awq"].load_calls == 0


def test_load_marlin_lora_alias_builds_fused_adapter_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="lora-marlin")

    assert result == {"marlin_lora": True}
    assert fakes["marlin_lora"].load_calls == 1
    assert fakes["marlin_fp16"].load_calls == 0


def test_load_trilin_alias_builds_native_3bit_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="trilin-3bit")

    assert result == {"trilin": True}
    assert fakes["trilin"].load_calls == 1


def test_load_amplin_alias_builds_ampere_w4_extension(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="gptq-amplin")

    assert result == {"amplin": True}
    assert fakes["amplin"].load_calls == 1


def test_load_specific_extension_honors_use_cache_false(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load(name="exllama-v2-awq", use_cache=False)

    assert result == {"exllamav2_awq": True}
    assert fakes["exllamav2_awq"].clear_cache_calls == 1
    assert fakes["exllamav2_awq"].load_calls == 1


def test_load_raises_for_unknown_extension(monkeypatch):
    _install_fake_extensions(monkeypatch)

    with pytest.raises(ValueError, match="Unknown extension"):
        extension_api.load(name="missing_extension")


def test_load_aggregates_extension_failures(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    fakes["awq"].ok = False
    fakes["awq"].error = "AWQ toolchain failure"

    with pytest.raises(RuntimeError, match="AWQ toolchain failure"):
        extension_api.load(name="awq")


def test_use_cache_false_requires_fresh_process_for_loaded_extensions(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    fakes["qqq"].already_loaded = True

    with pytest.raises(RuntimeError, match="Restart Python to force recompilation"):
        extension_api.load(name="qqq", use_cache=False)

    assert fakes["qqq"].clear_cache_calls == 0
    assert fakes["qqq"].load_calls == 0


def test_op_routes_through_extension_api(monkeypatch):
    _install_fake_extensions(monkeypatch)

    op = extension_api.op("awq", "test_op")

    assert op is awq_utils._AWQ_TORCH_OPS_EXTENSION._ops["test_op"]


def test_load_serializes_same_extension_across_threads(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    errors: list[Exception] = []

    def runner():
        try:
            extension_api.load(name="awq")
        except Exception as exc:  # pragma: no cover - assertion path below
            errors.append(exc)

    threads = [threading.Thread(target=runner) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert fakes["awq"].max_parallel_loads == 1
