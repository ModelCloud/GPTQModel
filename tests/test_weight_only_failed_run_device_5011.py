# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A failed weight-only ``quantize()`` must return the in-flight layer (#5011).

``WeightOnlyLooper`` moves the layer it is staging to the quantization device
(``BaseQModel.pre_quantize``), and the success path moves it back only in
``post_quantize``. The failure branch rehomed kernels (#4947) and restored base
modules (#4546) but never returned the in-flight layer, so after a cancelled or
failed RTN/GGUF/FP8 run that layer kept accelerator tensors while the rest of
the model was on CPU. Every later forward then raised
``Expected all tensors to be on the same device``.

``ModuleLooper`` solved this for #4546 with ``_track_inflight_layer`` /
``_restore_inflight_layer``; both loopers now share that tracker
(``gptqmodel/looper/inflight_layer.py``).
"""

from __future__ import annotations

from typing import Callable, Dict

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.looper import weight_only_looper as weight_only_looper_module
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.quantization import config as C


_INJECTED = "injected weight-only failure"


def _tiny_model_dir(tmp_path_factory) -> str:
    """A four-layer Qwen2 model small enough to quantize on CPU in seconds."""

    from transformers import Qwen2Config, Qwen2ForCausalLM

    from tests.tiny_remote_models import build_wordlevel_tokenizer

    destination = tmp_path_factory.mktemp("weight-only-5011") / "model-4l"
    config = Qwen2Config(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(destination)
    build_wordlevel_tokenizer().save_pretrained(destination)
    return str(destination)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> str:
    return _tiny_model_dir(tmp_path_factory)


def _accelerator_devices(model: torch.nn.Module) -> Dict[str, list[str]]:
    """Every parameter/buffer that is not on CPU or a documented meta shell."""

    offenders: Dict[str, list[str]] = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        device_type = tensor.device.type
        if device_type in {"cpu", "meta"}:
            continue
        offenders.setdefault(str(tensor.device), []).append(name)
    return offenders


def _run_failing_quantize(
    model_dir: str,
    config_factory: Callable[[], object],
    monkeypatch,
) -> GPTQModel:
    """Quantize until the injected failure in layer 2 and return the model."""

    import gptqmodel.looper.weight_only_processor as weight_only_processor

    model = GPTQModel.load(model_dir, config_factory(), dtype=torch.float16)
    original = weight_only_processor.WeightOnlyProcessor.quantize_module

    def _fail_in_layer_two(self, module, *args, **kwargs):
        if ".layers.2." in getattr(module, "full_name", ""):
            raise RuntimeError(_INJECTED)
        return original(self, module, *args, **kwargs)

    monkeypatch.setattr(
        weight_only_processor.WeightOnlyProcessor,
        "quantize_module",
        _fail_in_layer_two,
    )
    with pytest.raises(RuntimeError, match=_INJECTED):
        model.quantize(None)
    return model


CONFIG_FACTORIES: Dict[str, Callable[[], object]] = {
    "rtn": lambda: C.RTNConfig(bits=4, group_size=32, device="cuda"),
    "rtn-no-offload": lambda: C.RTNConfig(
        bits=4, group_size=32, device="cuda", offload_to_disk=False
    ),
    "gguf": lambda: C.GGUFConfig(bits=4, device="cuda"),
    "gguf-q-k-m": lambda: C.GGUFConfig(bits=4, format="q_k_m", device="cuda"),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
@pytest.mark.parametrize("name", sorted(CONFIG_FACTORIES))
def test_failed_weight_only_run_leaves_no_layer_on_the_accelerator(
    tiny_model_dir, monkeypatch, name
):
    model = _run_failing_quantize(
        tiny_model_dir, CONFIG_FACTORIES[name], monkeypatch
    )

    offenders = _accelerator_devices(model.model)
    assert offenders == {}, (
        "a failed weight-only run left tensors on the quantization device; "
        f"the next forward raises a device mismatch: {offenders}"
    )

    # A disk-offloaded run keeps the finished layers as meta shells (#4807), so
    # this only asserts the #5011 symptom: no device mismatch. The
    # inference-tensor version-counter repair is a separate contract (#5012).
    input_ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    try:
        with torch.inference_mode():
            model.model(input_ids)
    except RuntimeError as exc:
        assert "Expected all tensors to be on the same device" not in str(exc), str(exc)


def test_failure_branch_restores_the_inflight_layer(monkeypatch):
    """The cleanup must restore the in-flight layer before anything else."""

    calls: list[str] = []
    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = None

    def _raise(**_kwargs):
        raise RuntimeError("layer failure")

    looper._loop_impl = _raise
    looper._restore_inflight_layer = lambda: calls.append("inflight")
    looper._restore_offloaded_base_modules = lambda strict: calls.append(f"base:{strict}")
    looper._materialize_unreached_meta_modules = lambda: calls.append("unreached")
    looper._materialize_meta_base_modules = lambda strict: calls.append(f"meta:{strict}")
    looper._restore_disk_offloaded_modules = lambda: calls.append("disk")
    looper._materialize_remaining_meta_modules = lambda: calls.append("remaining")
    looper.processor = type(
        "_Processor",
        (),
        {"release_failed_run": staticmethod(lambda: calls.append("release"))},
    )()
    monkeypatch.setattr(
        weight_only_looper_module,
        "rehome_failed_run_kernels",
        lambda _model: calls.append("kernels"),
    )
    monkeypatch.setattr(
        weight_only_looper_module, "release_private_offload_root", lambda _model: None
    )

    with pytest.raises(RuntimeError, match="layer failure"):
        looper.loop()

    assert calls[0] == "inflight", calls
    assert "release" in calls and "kernels" in calls, calls


def test_success_path_clears_the_tracker():
    """A completed layer must not be restored again by a later failure."""

    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = None
    module = torch.nn.Linear(2, 2)
    looper._track_inflight_layer(module, torch.device("cpu"))
    assert looper._inflight_layer is not None

    looper._clear_inflight_layer()
    assert looper._inflight_layer is None
    # A no-op restore must not raise for a tracker that was never set.
    looper._restore_inflight_layer()


def test_meta_home_device_falls_back_to_cpu():
    """A lazy shell has no runtime placement, so cleanup returns it to CPU."""

    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = None
    module = torch.nn.Linear(2, 2, device="meta")

    looper._track_inflight_layer(module)

    assert looper._inflight_layer is not None
    assert looper._inflight_layer[1].type == "cpu"
