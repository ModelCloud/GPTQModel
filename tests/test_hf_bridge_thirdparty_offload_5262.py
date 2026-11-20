# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5262: third-party checkpoints must re-select offload-safe kernels.

#5247 re-selected kernels after Transformers resolved ``device_map="auto"`` +
``max_memory``, but only for checkpoints the bridge recognizes as native
GPT-QModel ones. AutoAWQ / AutoGPTQ checkpoints have no ``meta.quantizer``, so
the bridge returned ``None``, the AWQ quantizer kept ``_gptqmodel_model`` unset,
and ``_reselect_kernels_for_resolved_device_map`` returned early. AUTO then kept
``MarlinLinear`` / ``AwqMarlinLinear`` for a map that spills to CPU/disk and the
load crashed in ``post_init``:

* ``ValueError: AWQ Marlin requires CUDA tensors, got `cpu`.``
* ``RuntimeError: CPU-offloaded Marlin requires a writable in-memory weights map``

The fixture quantizes a tiny Qwen2 with GPTQModel, then makes a second copy of
each checkpoint with ``meta`` removed from both manifests. That copy is
indistinguishable from an AutoAWQ/AutoGPTQ checkpoint at the provenance check,
which is the path under test.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys

import pytest
import torch


OFFLOAD_SAFE_KERNELS = {"TritonV2Linear", "AwqGEMMTritonLinear"}
AUTO_MAX_MEMORY = {
    "cpu": {0: "512KiB", "cpu": "10GiB"},
    "disk": {0: "512KiB", "cpu": "64KiB"},
}


@pytest.fixture(scope="module", autouse=True)
def hf_bridge_environment(tmp_path_factory):
    """Publish a metadata-only distribution for the Transformers gates."""

    import gptqmodel

    monkey = pytest.MonkeyPatch()
    dist_root = tmp_path_factory.mktemp("dist")
    dist_info = dist_root / f"gptqmodel-{gptqmodel.__version__}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: gptqmodel\nVersion: {gptqmodel.__version__}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("gptqmodel: \n", encoding="utf-8")

    monkey.syspath_prepend(str(dist_root))
    try:
        from optimum.utils import import_utils as optimum_import_utils
    except ImportError:
        optimum_import_utils = None
    if optimum_import_utils is not None:
        monkey.setattr(optimum_import_utils, "_gptqmodel_available", True, raising=False)
        quantizer = sys.modules.get("optimum.gptq.quantizer")
        if quantizer is not None and not hasattr(quantizer, "QuantizeConfig"):
            importlib.reload(quantizer)
    from transformers.utils import import_utils as transformers_import_utils

    transformers_import_utils.is_gptqmodel_available.cache_clear()
    try:
        yield str(dist_root)
    finally:
        monkey.undo()
        transformers_import_utils.is_gptqmodel_available.cache_clear()


def _strip_third_party_meta(destination):
    """Remove GPT-QModel provenance from both checkpoint manifests."""

    for filename in ("config.json", "quantize_config.json"):
        path = destination / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        quantize_config = payload.get("quantization_config", payload)
        if isinstance(quantize_config, dict):
            quantize_config.pop("meta", None)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Tiny native AWQ/GPTQ checkpoints plus meta-free third-party copies."""

    if not torch.cuda.is_available():
        pytest.skip("an offloading map needs an accelerator for the resident layers")
    pytest.importorskip("optimum", reason="Transformers' GPTQ quantizer needs optimum")

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import FORMAT, AWQConfig, GPTQConfig
    from gptqmodel.quantization.config import CalibrationConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    root = tmp_path_factory.mktemp("hf-bridge-5262")
    base = root / "base"
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(base)
    tokenizer = build_wordlevel_tokenizer(vocab_size=1024)
    tokenizer.save_pretrained(base)

    texts = [
        "the quick brown fox jumps over the lazy dog and keeps running through the field",
        "a tiny calibration sample with enough tokens to survive length filtering",
        "another synthetic calibration example deliberately verbose for the tokenizer",
        "gravity attracts objects with mass and measurements help test predictions",
        "the baker mixed flour with water and yeast before leaving the dough to rise",
        "a function transforms inputs into results and tests compare expected values",
        "water evaporates from lakes, clouds form when vapor cools, and rain returns it",
        "a young child watched a small bird spread its wings and fly back to its nest",
    ]
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in texts]

    formats = {
        # V1 is required: `_is_native_gptqmodel_config` treats GPTQ_V2 as native
        # even without `meta`, so a V2 fixture would not exercise the fallback.
        "gptq4": GPTQConfig(bits=4, group_size=128, device="cpu", offload_to_disk=False),
        "awq_gemm": AWQConfig(
            bits=4, group_size=128, device="cpu", offload_to_disk=False, format=FORMAT.GEMM
        ),
    }

    result = {}
    for name, quantize_config in formats.items():
        model = GPTQModel.load(
            str(base),
            quantize_config=quantize_config,
            backend=BACKEND.TORCH,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        model.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
        native = root / f"{name}-native"
        model.save(str(native))
        del model

        third_party = root / f"{name}-thirdparty"
        shutil.copytree(native, third_party)
        _strip_third_party_meta(third_party)
        result[name] = {"native": str(native), "third_party": str(third_party)}

    # Guard the fixture: a provenance regression here would silently turn the
    # third-party nodes back into native ones.
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.utils.hf_load import _is_native_gptqmodel_config

    for name, paths in result.items():
        assert _is_native_gptqmodel_config(QuantizeConfig.from_pretrained(paths["native"]))
        assert not _is_native_gptqmodel_config(
            QuantizeConfig.from_pretrained(paths["third_party"])
        ), f"{name}: the third-party copy still looks native"
    return result


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("an offloading map needs an accelerator for the resident layers")


def _require_hf_quantizers():
    pytest.importorskip("optimum", reason="Transformers' GPTQ quantizer needs optimum")


def _kernel_name(model):
    return type(model.model.layers[0].self_attn.q_proj).__name__


def _offloaded_kernel_name(model):
    """Kernel of a quantized projection the resolved map actually offloads."""

    from gptqmodel.utils.hf_load import _resolved_placement
    from gptqmodel.utils.importer import is_offload_placement

    resolved = dict(model.hf_device_map)
    for name, module in model.named_modules():
        if name.endswith(".self_attn.q_proj") and is_offload_placement(
            _resolved_placement(resolved, name)
        ):
            return type(module).__name__
    raise AssertionError(f"no offloaded quantized projection in {resolved}")


def _materialized_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _logits(model):
    input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids.to(_materialized_device(model))).logits.float().cpu()


def _auto_kwargs(spill, tmp_path):
    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
        "max_memory": dict(AUTO_MAX_MEMORY[spill]),
        "attn_implementation": "eager",
    }
    if spill == "disk":
        kwargs["offload_folder"] = str(tmp_path / "offload")
        kwargs["offload_state_dict"] = True
    return kwargs


@pytest.mark.parametrize("spill", ["cpu", "disk"])
@pytest.mark.parametrize("name", ["awq_gemm", "gptq4"])
@pytest.mark.parametrize("source", ["third_party", "native"])
def test_auto_max_memory_reselects_offload_safe_kernel(
    checkpoints, name, source, spill, tmp_path
):
    """The resolved map must re-select a kernel that survives CPU/disk offload."""

    _require_cuda()
    _require_hf_quantizers()
    from transformers import AutoModelForCausalLM

    checkpoint = checkpoints[name][source]
    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float16, device_map={"": 0}
    )
    expected = _logits(reference)
    reference_kernel = _kernel_name(reference)
    del reference
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **_auto_kwargs(spill, tmp_path))
    try:
        resolved = dict(model.hf_device_map)
        assert spill in {str(value) for value in resolved.values()}, (
            f"{source}/{name}/{spill}: `max_memory` did not produce a {spill} "
            f"placement; resolved map was {model.hf_device_map}"
        )
        kernel = _offloaded_kernel_name(model)
        assert kernel in OFFLOAD_SAFE_KERNELS, (
            f"{source}/{name}/{spill}: AUTO selected `{kernel}` for a map that "
            f"spills to {spill}; the resident load used `{reference_kernel}`"
        )
        logits = _logits(model)
        assert torch.isfinite(logits).all(), f"{source}/{name}/{spill}: non-finite logits"
        tolerance = 8 * torch.finfo(torch.float16).eps * max(
            float(expected.abs().max()), 1.0
        )
        delta = (logits - expected).abs().max().item()
        assert delta <= tolerance, (
            f"{source}/{name}/{spill}: offloaded logits differ by {delta:.3e} "
            f"from the single-GPU load (tolerance {tolerance:.3e})"
        )
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.parametrize("name", ["awq_gemm", "gptq4"])
def test_third_party_no_offload_keeps_the_auto_kernel(checkpoints, name):
    """A resident map must not be re-routed: the fallback is offload-only."""

    _require_cuda()
    _require_hf_quantizers()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoints[name]["third_party"], dtype=torch.float16, device_map={"": 0}
    )
    try:
        assert _kernel_name(model) in {"MarlinLinear", "AwqMarlinLinear"}
        assert torch.isfinite(_logits(model)).all()
    finally:
        del model
        torch.cuda.empty_cache()


def test_reselect_context_is_built_for_a_third_party_config():
    """The fallback context is exactly the state the reselect helper reads."""

    from transformers import Qwen2Config, Qwen2ForCausalLM

    from gptqmodel.utils.hf_load import hf_gptqmodel_reselect_context

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    model.config.quantization_config = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    }
    context = hf_gptqmodel_reselect_context(model)
    assert context is not None
    assert context.quantize_config.method.name == "AWQ"
    assert context.quant_linear is None
    assert context.lm_head_name == "lm_head"

    model.config.quantization_config = None
    assert hf_gptqmodel_reselect_context(model) is None


_LATE_IMPORT_LOAD = r"""
import sys

import torch
from transformers import AutoModelForCausalLM

result_path = sys.argv[1]
checkpoint = sys.argv[2]
spill = sys.argv[3]
offload_folder = sys.argv[4]

# The regression is about the load that imports gptqmodel from inside
# transformers' `replace_with_awq_linear`; a pre-imported package would install
# the prepare hook before this load and hide it.
assert "gptqmodel" not in sys.modules, "gptqmodel was imported before the load"

kwargs = {
    "dtype": torch.float16,
    "device_map": "auto",
    "max_memory": {0: "512KiB", "cpu": "10GiB" if spill == "cpu" else "64KiB"},
    "attn_implementation": "eager",
}
if spill == "disk":
    kwargs["offload_folder"] = offload_folder
    kwargs["offload_state_dict"] = True

model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)

from gptqmodel.utils.hf_load import _resolved_placement
from gptqmodel.utils.importer import is_offload_placement

resolved = dict(model.hf_device_map)
offloaded = []
for name, module in model.named_modules():
    if name.endswith(".self_attn.q_proj") and is_offload_placement(
        _resolved_placement(resolved, name)
    ):
        offloaded.append(type(module).__name__)

device = next(
    parameter.device
    for parameter in model.parameters()
    if parameter.device.type != "meta"
)
input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long, device=device)
with torch.inference_mode():
    logits = model(input_ids).logits.float().cpu()

torch.save(
    {
        "logits": logits,
        "offloaded_kernels": offloaded,
        "placements": sorted({str(value) for value in resolved.values()}),
    },
    result_path,
)
print("LOADED", flush=True)
"""


@pytest.mark.parametrize("spill", ["cpu", "disk"])
def test_late_import_load_reselects_offload_safe_kernel(
    checkpoints, hf_bridge_environment, spill, tmp_path
):
    """The load that imports gptqmodel mid-prepare must re-select kernels (#5395).

    Transformers imports ``gptqmodel`` from inside
    ``replace_with_awq_linear``, so the bridge's prepare wrapper never runs for
    that first load and ``validate_environment`` had no model to re-select
    against. A third-party AWQ checkpoint with ``device_map="auto"`` and a
    CPU/disk spill therefore kept ``AwqMarlinLinear`` and crashed in
    ``post_init``. Each load must run in a fresh process with no prior
    ``gptqmodel`` import, so this is a subprocess test.
    """

    _require_cuda()
    _require_hf_quantizers()
    import os
    import subprocess
    import sys
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    checkpoint = checkpoints["awq_gemm"]["third_party"]
    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    expected = _logits(reference)
    del reference
    torch.cuda.empty_cache()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(repo_root),
            hf_bridge_environment,
            env.get("PYTHONPATH"),
        )
        if part
    )
    result_path = tmp_path / "late-import-load.pt"
    offload_folder = tmp_path / "offload"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _LATE_IMPORT_LOAD,
            str(result_path),
            checkpoint,
            spill,
            str(offload_folder),
        ],
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        text=True,
    )

    assert completed.returncode == 0, (
        f"a plain-transformers AWQ load with a {spill} spill failed:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    payload = torch.load(result_path, weights_only=True)
    assert spill in payload["placements"], payload["placements"]
    assert payload["offloaded_kernels"], payload["placements"]
    assert set(payload["offloaded_kernels"]) <= OFFLOAD_SAFE_KERNELS, (
        f"AUTO kept a CUDA-only kernel for a {spill}-offloaded layer: "
        f"{payload['offloaded_kernels']}"
    )
    tolerance = 8 * torch.finfo(torch.float16).eps * max(
        float(expected.abs().max()), 1.0
    )
    delta = (payload["logits"] - expected).abs().max().item()
    assert delta <= tolerance, (
        f"the {spill}-spilled load differs by {delta:.3e} from the single-GPU "
        f"load (tolerance {tolerance:.3e})"
    )
