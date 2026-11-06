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
import gptqmodel.utils.awq as awq_utils
import gptqmodel.utils.cpp as cpp_utils
import gptqmodel.utils.cpu_features as cpu_features
import gptqmodel.utils.exllamav2 as exllamav2_utils
import gptqmodel.utils.hadamard as hadamard_utils
import gptqmodel.utils.machete as machete_utils
import gptqmodel.utils.marlin as marlin_utils
import gptqmodel.utils.paroquant as paroquant_utils
import gptqmodel.utils.qqq as qqq_utils
import gptqmodel.utils.rocm as rocm
import gptqmodel.utils.swordfish as swordfish_utils
import gptqmodel.utils.vptq_cuda as vptq_utils


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
    # These tests cover the API mechanics (aliases, use_cache, locking), not
    # host support. Pin the host family so the ROCm-gated ExLlamaV2 entries stay
    # resolvable on a HIP host and the assertions stay deterministic.
    monkeypatch.setattr(exllamav2_utils, "IS_ROCM", False)

    fakes = {
        "pack_block_cpu": _FakeExtension("pack_block_cpu"),
        "floatx_cpu": _FakeExtension("floatx_cpu"),
        "int4_qzero_group_scaled_cpu": _FakeExtension("int4_qzero_group_scaled_cpu"),
        "awq": _FakeExtension("AWQ"),
        "qqq": _FakeExtension("QQQ"),
        "vptq": _FakeExtension("VPTQ CUDA"),
        "exllamav2": _FakeExtension("ExLlamaV2 GPTQ"),
        "exllamav2_awq": _FakeExtension("ExLlamaV2 AWQ"),
        "exllamav3": _FakeExtension("ExLlamaV3"),
        "machete": _FakeExtension("Machete"),
        "marlin_fp16": _FakeExtension("Marlin fp16"),
        "marlin_bf16": _FakeExtension("Marlin bf16"),
        "paroquant": _FakeExtension("ParoQuant rotation"),
        "hadamard": _FakeExtension("Fast Hadamard transform"),
        "swordfish": _FakeExtension("Swordfish"),
    }

    monkeypatch.setattr(cpp_utils, "_pack_block_extension", lambda: fakes["pack_block_cpu"])
    monkeypatch.setattr(cpp_utils, "_floatx_cpu_extension", lambda: fakes["floatx_cpu"])
    monkeypatch.setattr(
        cpp_utils,
        "_int4_qzero_group_scaled_cpu_extension",
        lambda: fakes["int4_qzero_group_scaled_cpu"],
    )
    # Host CPU features must not decide whether "all" includes this extension.
    monkeypatch.setattr(cpu_features, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(cpu_features, "avx512_bf16_error", lambda: "")
    monkeypatch.setattr(awq_utils, "_AWQ_TORCH_OPS_EXTENSION", fakes["awq"])
    monkeypatch.setattr(qqq_utils, "_QQQ_TORCH_OPS_EXTENSION", fakes["qqq"])
    monkeypatch.setattr(vptq_utils, "_VPTQ_TORCH_OPS_EXTENSION", fakes["vptq"])
    # VPTQ is CUDA-only; pin support so the test does not depend on the host GPU.
    monkeypatch.setattr(vptq_utils, "_validate_vptq_build_support", lambda: True)
    monkeypatch.setattr(exllamav2_utils, "_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION", fakes["exllamav2"])
    monkeypatch.setattr(exllamav2_utils, "_EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION", fakes["exllamav2_awq"])
    monkeypatch.setattr(exllamav3_ext, "_EXLLAMAV3_TORCH_OPS_EXTENSION", fakes["exllamav3"])
    monkeypatch.setattr(machete_utils, "_MACHETE_TORCH_OPS_EXTENSION", fakes["machete"])
    monkeypatch.setattr(machete_utils, "_validate_machete_build_support", lambda: True)
    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fakes["marlin_fp16"])
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", fakes["marlin_bf16"])
    monkeypatch.setattr(paroquant_utils, "_PAROQUANT_ROTATION_EXTENSION", fakes["paroquant"])
    monkeypatch.setattr(hadamard_utils, "_HADAMARD_TORCH_OPS_EXTENSION", fakes["hadamard"])
    monkeypatch.setattr(hadamard_utils, "hadamard_supported", lambda: True)
    monkeypatch.setattr(swordfish_utils, "_SWORDFISH_TORCH_OPS_EXTENSION", fakes["swordfish"])
    monkeypatch.setattr(swordfish_utils, "_validate_swordfish_build_support", lambda: True)

    # Keep the fake set in lockstep with the registry: a new extension must be
    # added here instead of silently escaping into a real JIT build.
    assert set(fakes) == set(extension_api.available_extensions()), (
        "Add a fake for every extension in `available_extensions()` so this test "
        "never builds a real torch.ops extension."
    )

    return fakes


def test_package_root_exports_extension_module():
    assert gptqmodel.extension is extension_api


def test_extension_api_locks_are_initialized_eagerly():
    assert set(extension_api._EXTENSION_API_LOCKS) == set(extension_api.available_extensions())


def test_load_defaults_to_all_extensions(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)

    result = extension_api.load()

    assert result == dict.fromkeys(fakes, True)
    assert all(fake.load_calls == 1 for fake in fakes.values())


def test_load_all_skips_extensions_unsupported_on_this_host(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(machete_utils, "_validate_machete_build_support", lambda: False)

    result = extension_api.load()

    assert "machete" not in result
    assert fakes["machete"].load_calls == 0


def test_load_all_skips_vptq_without_an_nvidia_cuda_device(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(vptq_utils, "_validate_vptq_build_support", lambda: False)

    result = extension_api.load()

    assert "vptq" not in result
    assert fakes["vptq"].load_calls == 0


def test_load_specific_vptq_reports_the_build_requirement(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(vptq_utils, "_validate_vptq_build_support", lambda: False)

    with pytest.raises(RuntimeError, match="NVIDIA CUDA device"):
        extension_api.load(name="vptq")

    assert fakes["vptq"].load_calls == 0


@pytest.mark.parametrize(
    ("cuda_available", "is_rocm", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_vptq_build_support_requires_nvidia_cuda(
    monkeypatch, cuda_available, is_rocm, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(rocm, "IS_ROCM", is_rocm)

    assert vptq_utils._validate_vptq_build_support() is expected


def test_vptq_runtime_helpers_fail_fast_without_nvidia_cuda(monkeypatch):
    monkeypatch.setattr(vptq_utils, "_validate_vptq_build_support", lambda: False)

    assert vptq_utils.prewarm_vptq_extension() is False
    assert vptq_utils.vptq_cuda_runtime_available() is False
    assert "NVIDIA CUDA device" in vptq_utils.vptq_cuda_runtime_error()


def test_load_specific_unsupported_extension_raises_without_building(monkeypatch):
    fakes = _install_fake_extensions(monkeypatch)
    monkeypatch.setattr(machete_utils, "_validate_machete_build_support", lambda: False)
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
