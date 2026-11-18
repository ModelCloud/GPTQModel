# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 routed-expert Hessians need a concrete owner device (#5093).

With ``auto_forward_data_parallel=True`` every device worker forwards its own
batch, so a routed expert's shared Hessian accumulator receives a partial from
more than one GPU. EXL3 resolved the owner from the bare ``qcfg.device``
family, which ``as_torch_device`` keeps index-less on purpose. ``GPTQ`` then
moved each partial to ``torch.device("cuda")`` -- the calling worker's current
GPU -- while keying it under ``"cuda"``, so the second worker's
``existing.add_(xtx)`` mixed ``cuda:0`` and ``cuda:1`` and quantization crashed
in layer 0.

The regression drives the tiny Qwen3-MoE fixture through EXL3 twice on two
GPUs: the default data-parallel forward and the serial forward. It asserts the
owners are concrete, every routed projection captured the whole calibration
set, and both runs agree on the finalized Hessian (the split changes the
accumulation order only, not the sum).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


_NUM_LAYERS = 2
_NUM_EXPERTS = 4
_ROWS = 4
_TOKENS_PER_ROW = 32
_EXPECTED_NSAMPLES = _ROWS * _TOKENS_PER_ROW
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# EXL3 requires 128-aligned input features on every projection it quantizes.
_CONFIG_ARGS = {
    "num_hidden_layers": _NUM_LAYERS,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}
_HESSIAN_TOLERANCE = 1e-6


def _build_fixture(model_dir: Path) -> None:
    """Persist the two-layer Qwen3-MoE fixture with EXL3-aligned widths."""

    import torch

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    try:
        # The installed HF helper wins over tests/kernels where it exists; the
        # local test package is the only ``kernels`` module in the GPU image.
        import kernels  # noqa: F401
    except ModuleNotFoundError:
        pass
    from test_checkpoint_quantization import build_native_model
    from test_tiny_moe_quant_smoke import _build_local_tokenizer

    torch.manual_seed(1234)
    model = build_native_model(
        "qwen3_moe", dict(_CONFIG_ARGS), moe_intermediate_size=128
    )
    model.save_pretrained(model_dir)
    _build_local_tokenizer(model_dir)


def _calibration():
    """Four deterministic rows of 32 tokens."""

    import torch

    torch.manual_seed(6789)
    return [
        {
            "input_ids": torch.randint(3, 100, (1, _TOKENS_PER_ROW)),
            "attention_mask": torch.ones(1, _TOKENS_PER_ROW, dtype=torch.long),
        }
        for _ in range(_ROWS)
    ]


def _quantize(model_dir: Path, *, auto_forward_data_parallel: bool):
    """Quantize the fixture and return owners, sample counts and Hessians."""

    import threading

    import torch

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import (
        EXL3Config,
        ExpertsRoutingOverride,
        MoEConfig,
    )
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(6789)
    # A bare family is the documented default for a quantize load; the loader
    # strips the ordinal. Passing "cuda:0" here would hide the defect.
    config = EXL3Config(
        bits=4,
        device="cuda",
        auto_forward_data_parallel=auto_forward_data_parallel,
        offload_to_disk=False,
        # Route every token to every expert: a projection below the full count
        # lost a batch instead of being skipped by the router.
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )
    model = GPTQModel.load(str(model_dir), quantize_config=config, backend=BACKEND.AUTO)

    lock = threading.Lock()
    owners = set()
    hessians = {}
    nsamples = {}

    original_set_owner = GPTQ.set_hessian_accumulator_device
    original_finalize = GPTQ.finalize_hessian

    def record_owner(task, device):
        with lock:
            owners.add(str(torch.device(device)))
        return original_set_owner(task, device)

    def record_finalize(task, *args, **kwargs):
        hessian = original_finalize(task, *args, **kwargs)
        name = task._named_module.full_name
        with lock:
            nsamples[name] = task.nsamples
            hessians[name] = hessian.detach().to("cpu", torch.float64).clone()
        return hessian

    GPTQ.set_hessian_accumulator_device = record_owner
    GPTQ.finalize_hessian = record_finalize
    try:
        model.quantize(
            CalibrationConfig(
                dataset=_calibration(),
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.AUTO,
        )
    finally:
        GPTQ.set_hessian_accumulator_device = original_set_owner
        GPTQ.finalize_hessian = original_finalize
        del model
        torch.cuda.empty_cache()

    return {"owners": sorted(owners), "nsamples": nsamples, "hessians": hessians}


def _run_in_subprocess(
    model_dir: Path,
    *,
    mode: str = "run",
    auto_forward_data_parallel: bool = True,
    out_path: Path | None = None,
    timeout: int = 1800,
):
    env = os.environ.copy()
    # Inherit the caller's visibility settings (CUDA_VISIBLE_DEVICES or
    # ROCR_VISIBLE_DEVICES). The fixture already checked that the parent sees
    # at least two devices, and forcing "0,1" would address different GPUs
    # than the caller selected.
    env["HF_HUB_OFFLINE"] = "1"
    env["LOGBAR_ANIMATION"] = "0"
    env["GPTQ_BYPASS_SAFETY_CHECKS"] = "1"
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH")) if part
    )
    command = [
        sys.executable,
        # Run the driver as a module, not as a script: a forkserver child
        # re-imports the parent's ``__main__``, which would nest pytest.
        "-m",
        f"{Path(__file__).resolve().parent.name}.{Path(__file__).resolve().stem}",
        "--mode",
        mode,
        "--model-dir",
        str(model_dir),
        "--auto-dp",
        "1" if auto_forward_data_parallel else "0",
    ]
    if out_path is not None:
        command.extend(["--out", str(out_path)])
    result = subprocess.run(
        command,
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{mode} subprocess failed "
        f"(auto_forward_data_parallel={auto_forward_data_parallel}):\n"
        + result.stdout[-8000:]
        + result.stderr[-8000:]
    )
    if mode == "build":
        assert (model_dir / "config.json").is_file(), (
            "build subprocess did not persist the fixture:\n"
            + result.stdout[-4000:]
            + result.stderr[-4000:]
        )
        return None
    import torch

    record = json.loads(out_path.read_text())
    record["hessians"] = torch.load(
        str(out_path.with_suffix(".pt")), map_location="cpu", weights_only=True
    )
    return record


def _expert_projections(record):
    return {
        name: value
        for name, value in record["nsamples"].items()
        if ".mlp.experts." in name and name.endswith(_PROJECTIONS)
    }


def _assert_concrete_owners(owners):
    import torch

    assert owners, "no routed expert ever registered a Hessian owner"
    for owner in owners:
        device = torch.device(owner)
        assert device.index is not None, (
            "routed expert Hessian owner is index-less; data-parallel workers "
            f"would resolve it to different GPUs: {owners}"
        )


def test_bare_device_family_expands_to_visible_ordinals(monkeypatch):
    """The default ``qcfg.device`` family must become concrete owners (#5093)."""

    from types import SimpleNamespace

    import torch

    from gptqmodel.looper.exllamav3_processor import routed_expert_owner_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    qcfg = SimpleNamespace(device="cuda", moe_vram_strategy_devices=None)

    assert routed_expert_owner_devices(qcfg) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]


def test_resolved_pool_is_preferred_over_the_bare_family(monkeypatch):
    """Capture follows the looper's filtered pool, not every visible GPU."""

    from types import SimpleNamespace

    import torch

    from gptqmodel.looper.exllamav3_processor import routed_expert_owner_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    qcfg = SimpleNamespace(device="cuda", moe_vram_strategy_devices=None)

    assert routed_expert_owner_devices(qcfg, [torch.device("cuda:1")]) == [
        torch.device("cuda:1")
    ]


def test_explicit_ordinal_is_kept_as_is(monkeypatch):
    from types import SimpleNamespace

    import torch

    from gptqmodel.looper.exllamav3_processor import routed_expert_owner_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    qcfg = SimpleNamespace(device="cuda:1", moe_vram_strategy_devices=None)

    assert routed_expert_owner_devices(qcfg) == [torch.device("cuda:1")]


def test_indexless_accumulator_device_is_rejected():
    """A thread-relative device cannot key a shared accumulator (#5093)."""

    import torch
    from torch import nn

    from gptqmodel.quantization.gptq import GPTQ

    capture = GPTQ(nn.Linear(4, 4, bias=False))

    with pytest.raises(ValueError, match="concrete ordinal"):
        capture.set_hessian_accumulator_device(torch.device("cuda"))

    # CPU and concrete ordinals stay valid.
    cpu_capture = GPTQ(nn.Linear(4, 4, bias=False))
    cpu_capture.set_hessian_accumulator_device(torch.device("cpu"))
    assert cpu_capture.hessian_accumulator_device == torch.device("cpu")
    cuda_capture = GPTQ(nn.Linear(4, 4, bias=False))
    cuda_capture.set_hessian_accumulator_device("cuda:1")
    assert cuda_capture.hessian_accumulator_device == torch.device("cuda:1")


@pytest.fixture(scope="module")
def native_model_dir():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for the data-parallel forward")
    work_dir = Path(tempfile.mkdtemp(prefix="exl3-moe-owner-5093-"))
    model_dir = work_dir / "native"
    _run_in_subprocess(model_dir, mode="build")
    try:
        yield model_dir
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.gpu
@pytest.mark.slow
def test_default_data_parallel_owns_routed_hessians_on_concrete_devices(
    native_model_dir, tmp_path
):
    """The default DP run must not crash and must name real owner devices."""

    out_path = tmp_path / "dp.json"
    record = _run_in_subprocess(
        native_model_dir, auto_forward_data_parallel=True, out_path=out_path
    )

    import torch

    _assert_concrete_owners(record["owners"])
    expected_owners = {
        f"cuda:{index}" for index in range(torch.cuda.device_count())
    }
    assert set(record["owners"]) == expected_owners, (
        "routed experts were not spread over every data-parallel device: "
        f"{record['owners']} != {sorted(expected_owners)}"
    )

    experts = _expert_projections(record)
    expected_names = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(_NUM_LAYERS)
        for expert in range(_NUM_EXPERTS)
        for projection in _PROJECTIONS
    }
    assert set(experts) == expected_names
    incomplete = {
        name: value for name, value in experts.items()
        if value != _EXPECTED_NSAMPLES
    }
    assert not incomplete, f"routed experts lost calibration tokens: {incomplete}"


@pytest.mark.gpu
@pytest.mark.slow
def test_data_parallel_hessian_matches_the_serial_capture(native_model_dir, tmp_path):
    """Splitting the capture across owners must not change the Hessian.

    Only layer 0 is compared: its inputs come from the dense embeddings, so
    the capture is independent of quantization. Later layers replay the
    already-quantized layer 0, and data-parallel quantization is not
    bit-identical to the serial path (see ``docs/multi_gpu_quantization.md``),
    so their inputs legitimately differ. Every layer still has to agree on the
    per-module sample counts.
    """

    import torch

    dp_path = tmp_path / "dp.json"
    serial_path = tmp_path / "serial.json"
    dp = _run_in_subprocess(
        native_model_dir, auto_forward_data_parallel=True, out_path=dp_path
    )
    serial = _run_in_subprocess(
        native_model_dir, auto_forward_data_parallel=False, out_path=serial_path
    )

    assert dp["nsamples"] == serial["nsamples"], (
        "data-parallel and serial captures disagree on per-module sample counts"
    )

    worst_name = None
    worst_error = 0.0
    compared = 0
    for name, reference in serial["hessians"].items():
        if not name.startswith("model.layers.0."):
            continue
        actual = dp["hessians"][name]
        assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
        error = (actual - reference).norm() / reference.norm().clamp_min(1e-30)
        compared += 1
        if error > worst_error:
            worst_error = float(error)
            worst_name = name
    assert compared > 0, "no layer-0 Hessian was compared"
    assert worst_error <= _HESSIAN_TOLERANCE, (
        f"data-parallel Hessian differs from the serial capture at {worst_name}: "
        f"normalized error {worst_error:.3e} > {_HESSIAN_TOLERANCE:.0e}"
    )


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "run"), default="run")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--auto-dp", choices=("0", "1"), default="1")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    if args.mode == "build":
        _build_fixture(model_dir)
        print("BUILD OK", flush=True)
        return 0

    record = _quantize(
        model_dir, auto_forward_data_parallel=args.auto_dp == "1"
    )
    payload = {
        "owners": record["owners"],
        "nsamples": record["nsamples"],
    }
    import torch

    torch.save(record["hessians"], str(Path(args.out).with_suffix(".pt")))
    Path(args.out).write_text(json.dumps(payload))
    print("RUN OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
