# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Defaults GPT-QModel installs on ROCm, and their cross-process contract.

Two process-wide defaults are covered because both silently change results for
every ROCm user:

* ``PYTORCH_ALLOC_CONF`` must not enable ``expandable_segments`` on ROCm; the
  VMM segment release/remap in ``empty_cache()`` corrupts ordinary Torch
  results on ROCm 10.1 / gfx1201.
* ``ROCBLAS_USE_HIPBLASLT`` must default to ``0`` on ROCm; rocBLAS otherwise
  forwards its GEMMs to hipBLASLt, which selects a different kernel per process
  and makes identical quantization runs save different checkpoints.

Both rules defer to an explicit caller value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from gptqmodel import _alloc_conf, _rocm_env


IS_ROCM = torch.version.hip is not None
REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_ALLOC_ENV_VARS = (
    *_alloc_conf.ALLOC_CONF_ENV_VARS,
    _alloc_conf.HIP_ALLOC_CONF_ENV_VAR,
)
BLAS_ENV_VAR = _rocm_env.ROCBLAS_USE_HIPBLASLT_ENV


def _child_env(**overrides):
    """Environment for a fresh interpreter with no caller-provided defaults."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ALL_ALLOC_ENV_VARS and key != BLAS_ENV_VAR
    }
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    # Never let a test reach the network for a model.
    env["HF_HUB_OFFLINE"] = "1"
    env.update(overrides)
    return env


def _run_child(script: str, *args, env=None, timeout=1800):
    proc = subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        env=env or _child_env(),
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Allocator configuration
# ---------------------------------------------------------------------------


def test_cuda_default_keeps_expandable_segments(monkeypatch):
    monkeypatch.setattr(_alloc_conf, "running_on_rocm", lambda: False)

    assert _alloc_conf.default_alloc_conf() == _alloc_conf.DEFAULT_ALLOC_CONF
    assert "expandable_segments:True" in _alloc_conf.default_alloc_conf()


def test_rocm_default_drops_expandable_segments(monkeypatch):
    monkeypatch.setattr(_alloc_conf, "running_on_rocm", lambda: True)

    conf = _alloc_conf.default_alloc_conf()

    assert conf == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert "expandable_segments" not in conf
    # The fragmentation knobs that do not trigger the VMM bug are kept.
    assert "max_split_size_mb:256" in conf
    assert "garbage_collection_threshold:0.7" in conf


def test_rocm_default_is_installed_without_expandable_segments(monkeypatch):
    for name in ALL_ALLOC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_alloc_conf, "running_on_rocm", lambda: True)
    monkeypatch.setattr(_alloc_conf, "_apply_runtime_alloc_conf", lambda conf: None)

    installed = _alloc_conf.apply_default_alloc_conf()

    assert installed == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert "expandable_segments" not in installed
    for name in _alloc_conf.ALLOC_CONF_ENV_VARS:
        assert os.environ[name] == installed
    # The HIP-only alias is a caller input, not something we write.
    assert _alloc_conf.HIP_ALLOC_CONF_ENV_VAR not in os.environ


@pytest.mark.parametrize("name", ALL_ALLOC_ENV_VARS)
def test_explicit_user_alloc_conf_is_never_overridden(monkeypatch, name):
    for other in ALL_ALLOC_ENV_VARS:
        monkeypatch.delenv(other, raising=False)
    user_value = "expandable_segments:False,max_split_size_mb:128"
    monkeypatch.setenv(name, user_value)
    monkeypatch.setattr(_alloc_conf, "running_on_rocm", lambda: True)

    assert _alloc_conf.alloc_conf_configured()
    assert _alloc_conf.apply_default_alloc_conf() is None
    assert os.environ[name] == user_value


# ---------------------------------------------------------------------------
# rocBLAS selection
# ---------------------------------------------------------------------------


def test_rocm_blas_default_is_installed(monkeypatch):
    monkeypatch.delenv(BLAS_ENV_VAR, raising=False)
    monkeypatch.setattr(_rocm_env, "running_on_rocm", lambda: True)

    assert _rocm_env.apply_default_rocm_blas_env() == "0"
    assert os.environ[BLAS_ENV_VAR] == "0"


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_explicit_rocm_blas_value_is_preserved(monkeypatch, value):
    monkeypatch.setenv(BLAS_ENV_VAR, value)
    monkeypatch.setattr(_rocm_env, "running_on_rocm", lambda: True)

    assert _rocm_env.apply_default_rocm_blas_env() is None
    assert os.environ[BLAS_ENV_VAR] == value


def test_non_rocm_hosts_are_untouched(monkeypatch):
    monkeypatch.delenv(BLAS_ENV_VAR, raising=False)
    monkeypatch.setattr(_rocm_env, "running_on_rocm", lambda: False)

    assert _rocm_env.apply_default_rocm_blas_env() is None
    assert BLAS_ENV_VAR not in os.environ


# ---------------------------------------------------------------------------
# Real ROCm process checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IS_ROCM, reason="ROCm-only defaults")
def test_import_installs_both_rocm_defaults():
    script = """
        import json, os
        import gptqmodel  # noqa: F401
        print(json.dumps({
            "blas": os.environ.get("ROCBLAS_USE_HIPBLASLT"),
            "alloc": os.environ.get("PYTORCH_ALLOC_CONF"),
            "cuda_alloc": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }))
    """
    payload = _last_json(_run_child(script, timeout=600).stdout)

    assert payload["blas"] == "0"
    assert payload["alloc"] == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert payload["cuda_alloc"] == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert "expandable_segments" not in payload["alloc"]


@pytest.mark.skipif(not IS_ROCM, reason="ROCm-only allocator bug")
def test_default_alloc_conf_does_not_corrupt_a_matmul():
    """The issue's pure-Torch repro: 0 bad rows under the installed default."""

    script = """
        import json, os
        import torch
        import gptqmodel  # noqa: F401  installs the default before allocator init

        torch.manual_seed(0)
        rows, k, n = 2097152, 16, 16
        a = torch.randn(rows, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)
        reference = a @ b
        torch.cuda.synchronize()
        bad = 0
        for _ in range(3):
            torch.cuda.empty_cache()
            out = a @ b
            torch.cuda.synchronize()
            bad += (out != reference).any(dim=1).sum().item()
        print(json.dumps({
            "bad_rows": bad,
            "expandable": "expandable_segments" in (os.environ.get("PYTORCH_ALLOC_CONF") or ""),
        }))
    """
    payload = _last_json(_run_child(script, timeout=900).stdout)

    assert payload["expandable"] is False
    assert payload["bad_rows"] == 0


def _cached_quantization_model():
    """Local snapshot of a cached model that reproduces the per-process GEMM choice."""

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
    except Exception:
        return None
    if isinstance(cached, str) and os.path.exists(cached):
        # A local directory skips the hub API call that a repo id triggers.
        return str(Path(cached).parent)
    return None


@pytest.mark.skipif(not IS_ROCM, reason="ROCm-only cross-process bug")
@pytest.mark.parametrize("method", ["gptq", "qqq"])
def test_quantization_is_reproducible_across_processes(tmp_path, method):
    """Two separate quantization processes must save identical tensors.

    Both methods accumulate a calibration Hessian, so both must run it on
    rocBLAS: GPTQ's fp32 ``X^T X`` and QQQ's ``add_batch`` GEMM (issue #4123).
    """

    model_id = _cached_quantization_model()
    if model_id is None:
        pytest.skip("Qwen/Qwen2.5-0.5B-Instruct is not in the local HF cache")

    script = """
        import hashlib, json, os, sys
        import torch
        import gptqmodel  # noqa: F401  installs the deterministic rocBLAS default
        from gptqmodel import CalibrationConfig, GPTQModel, QuantizeConfig

        out_path = sys.argv[1]
        method = sys.argv[2]
        samples = [
            f"Calibration sample {i}: GPT-QModel quantizes large language models with GPTQ, "
            "AWQ and Paro while keeping the checkpoint in its packed representation."
            for i in range(32)
        ]
        kwargs = dict(bits=4, group_size=128, sym=True, offload_to_disk=False)
        if method != "gptq":
            kwargs["method"] = method
        model = GPTQModel.load(
            os.environ["REPRO_MODEL_ID"], QuantizeConfig(**kwargs)
        )
        model.quantize(CalibrationConfig(dataset=samples, batch_size=1))

        hashes = {}
        for name, tensor in model.model.state_dict().items():
            if not torch.is_tensor(tensor) or tensor.device.type == "meta":
                continue
            # reshape(-1) handles 0-dim scalars, which some checkpoint layouts
            # now carry (view() rejects a 0-dim tensor of a different dtype).
            flat = tensor.detach().to("cpu").contiguous().reshape(-1)
            payload = flat.view(torch.uint8).numpy().tobytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
        with open(out_path, "w") as handle:
            json.dump(hashes, handle, sort_keys=True)
        print(json.dumps({
            "blas": os.environ.get("ROCBLAS_USE_HIPBLASLT"),
            "method": method,
            "tensors": len(hashes),
        }))
    """

    runs = []
    for index in range(2):
        out_path = tmp_path / f"hashes_{index}.json"
        proc = _run_child(
            script,
            str(out_path),
            method,
            env=_child_env(REPRO_MODEL_ID=model_id),
            timeout=3600,
        )
        summary = _last_json(proc.stdout)
        with open(out_path) as handle:
            summary["hashes"] = json.load(handle)
        runs.append(summary)

    # Deterministic regression for the default itself: the child inherited no
    # caller value, so the installed default is what rocBLAS saw.
    assert [run["blas"] for run in runs] == ["0", "0"]

    assert [run["method"] for run in runs] == [method, method]

    first, second = runs[0]["hashes"], runs[1]["hashes"]
    assert first and len(first) == len(second)
    assert len(first) > 300, f"only {len(first)} materialized tensors; expected the full model"
    differing = [name for name in first if first[name] != second.get(name)]
    assert not differing, (
        f"{len(differing)}/{len(first)} saved tensors differ across processes: {differing[:8]}"
    )
