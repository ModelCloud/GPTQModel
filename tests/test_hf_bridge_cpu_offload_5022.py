# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-offload placement and post-init regressions for GPT-QModel (#5022, #5194)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn


# A map that offloads the second decoder layer plus `lm_head`, and its mirror
# that offloads the embeddings instead. Both are the shapes the issue reports.
OFFLOAD_MAPS = {
    "late-layer": {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": "cpu",
        "model.norm": 0,
        "lm_head": "cpu",
    },
    "embeddings": {
        "model.embed_tokens": "cpu",
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.norm": 0,
        "lm_head": 0,
    },
}

OFFLOAD_SAFE_KERNELS = {"TritonV2Linear", "AwqGEMMTritonLinear"}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Quantize a tiny base model to each packed format exercised by the loader."""
    _require_cuda()

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import FORMAT, AWQConfig, GPTQConfig, RTNConfig
    from gptqmodel.quantization.config import CalibrationConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    root = tmp_path_factory.mktemp("cpu-offload-5022")
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
        "gptq4": GPTQConfig(bits=4, group_size=128, device="cpu", offload_to_disk=False),
        "gptq_v2": GPTQConfig(
            bits=4, group_size=128, device="cpu", offload_to_disk=False, format=FORMAT.GPTQ_V2
        ),
        "rtn4": RTNConfig(bits=4, group_size=128, device="cpu", offload_to_disk=False),
        "awq_gemm": AWQConfig(
            bits=4, group_size=128, device="cpu", offload_to_disk=False, format=FORMAT.GEMM
        ),
    }

    paths = {}
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
        destination = root / name
        model.save(str(destination))
        del model
        paths[name] = str(destination)
    return paths


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("a CPU-offload device map needs an accelerator for the resident layers")


def _materialized_device(model):
    """The device of the first non-meta parameter (an offloaded layer has none)."""

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _logits(model):
    input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
    device = _materialized_device(model)
    with torch.inference_mode():
        return model(input_ids.to(device)).logits.float().cpu()


@pytest.mark.parametrize("map_name", OFFLOAD_MAPS)
def test_native_load_uses_an_offload_safe_kernel(checkpoints, map_name):
    """Explicit CPU offload uses a compatible kernel and preserves logits."""
    from gptqmodel import GPTQModel

    for name, checkpoint in checkpoints.items():
        reference = GPTQModel.load(checkpoint, device="cuda:0", dtype=torch.float16, attn_implementation="eager")
        expected = _logits(reference.model)
        del reference
        model = GPTQModel.load(
            checkpoint,
            device_map=OFFLOAD_MAPS[map_name],
            dtype=torch.float16,
            attn_implementation="eager",
        )
        try:
            kernel = type(model.model.model.layers[0].self_attn.q_proj).__name__
            assert kernel in OFFLOAD_SAFE_KERNELS, f"{name}: native load picked `{kernel}`"
            tolerance = 8 * torch.finfo(torch.float16).eps * max(float(expected.abs().max()), 1.0)
            torch.testing.assert_close(_logits(model.model), expected, rtol=0, atol=tolerance)
        finally:
            del model
            torch.cuda.empty_cache()


def test_auto_cpu_offload_only_applies_to_auto():
    """An explicit backend keeps the caller's choice; AUTO gets the contract."""

    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import auto_cpu_offload, device_map_uses_cpu

    offloaded = {"a": 0, "b": "cpu"}
    resident = {"a": 0, "b": 1}

    assert device_map_uses_cpu(offloaded)
    assert not device_map_uses_cpu(resident)
    # An int is an ordinal, and `disk` is an accelerate offload spelling, not a
    # CPU-resident placement.
    assert not device_map_uses_cpu({"a": 0, "b": 1})
    assert not device_map_uses_cpu({"a": "disk"})
    assert not device_map_uses_cpu(None)
    assert not device_map_uses_cpu("auto")

    assert auto_cpu_offload(None, offloaded)
    assert auto_cpu_offload(BACKEND.AUTO, offloaded)
    assert auto_cpu_offload(BACKEND.AUTO_TRAINABLE, offloaded)
    assert not auto_cpu_offload(BACKEND.AUTO, resident)
    # An explicit backend is the caller's choice: `_post_init_cpu_offloaded_marlin`
    # repacks from the in-memory offload map.
    assert not auto_cpu_offload(BACKEND.MARLIN, offloaded)


def test_prefixed_dataset_over_offloaded_weights_loader_is_writable():
    """The map Transformers actually attaches must be accepted (#5022)."""

    from accelerate.hooks import AlignDevicesHook
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset

    from gptqmodel.utils.model import _cpu_offload_writable_storage

    state_dict = {"block.qweight": torch.zeros(1)}
    index = {"block.scales": {"safetensors_file": "x.safetensors"}}
    loader = OffloadedWeightsLoader(state_dict=state_dict, index=index)
    weights_map = PrefixedDataset(loader, prefix="prefix.")

    storage, prefix, disk_index = _cpu_offload_writable_storage(weights_map)

    assert storage is state_dict
    assert prefix == "prefix."
    assert disk_index is index
    # A plain dict map keeps working, and an unsupported map still fails closed.
    plain, plain_prefix, plain_disk = _cpu_offload_writable_storage({"a": 1})
    assert (plain, plain_prefix, plain_disk) == ({"a": 1}, "", None)
    with pytest.raises(RuntimeError, match="writable in-memory weights map"):
        _cpu_offload_writable_storage(object())
    # The hook type is what the post-init inspects; keep the import honest.
    assert AlignDevicesHook is not None


def _marlin_layer(desc_act=False):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    layer = MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=desc_act,
        sym=True,
        in_features=256,
        out_features=128,
        bias=True,
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(5022)
    layer.qweight.copy_(
        torch.randint(
            -(2**31),
            2**31 - 1,
            layer.qweight.shape,
            dtype=torch.int32,
            generator=generator,
        )
    )
    layer.scales.copy_(torch.randn(layer.scales.shape, generator=generator))
    layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator))
    layer.qzeros.zero_()
    layer.g_idx.copy_(torch.arange(layer.in_features, dtype=torch.int32) // layer.group_size)
    return layer


def _stub_marlin_runtime(monkeypatch):
    import gptqmodel.nn_modules.qlinear.marlin as marlin_module

    monkeypatch.setattr(marlin_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_module, "marlin_validate_runtime_device", lambda *args, **kwargs: (8, 0)
    )
    monkeypatch.setattr(marlin_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(
        marlin_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_module,
        "gptq_marlin_repack",
        lambda packed, perm, size_k, size_n, num_bits, dtype=None: packed.reshape(
            size_k // 16, size_n * 16 // (32 // num_bits)
        ).contiguous(),
    )
    monkeypatch.setattr(marlin_module, "marlin_permute_scales", lambda scales, **kwargs: scales)
    monkeypatch.setattr(marlin_module, "marlin_permute_bias", lambda bias: bias)


def _marlin_layer_or_skip():
    """Build the fixture layer, or skip when the Marlin kernel is absent.

    A CPU-only torch build has no Marlin extension, so the class raises
    `ImportError: Marlin kernel requires CUDA` at construction. These nodes
    exercise the offload map, not the kernel, so skipping is the honest result.
    """

    try:
        return _marlin_layer()
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"the Marlin kernel cannot be constructed in this runtime: {exc}")


@pytest.mark.parametrize("disk_backed", [False, True])
def test_offloaded_weights_loader_map_is_repacked_in_place(monkeypatch, disk_backed):
    """A real `OffloadedWeightsLoader` map is refreshed, a disk entry is refused."""

    import accelerate
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset

    from gptqmodel.utils.model import gptqmodel_post_init

    _stub_marlin_runtime(monkeypatch)

    resident = _marlin_layer_or_skip()
    expected = copy.deepcopy(resident)
    expected.post_init()
    expected_state = expected.state_dict()

    layer = copy.deepcopy(resident)
    model = nn.Sequential(layer)
    # `cpu_offload` builds the same hook Transformers gets from
    # `dispatch_model`; swap its plain-dict map for the
    # `PrefixedDataset(OffloadedWeightsLoader)` Accelerate actually attaches.
    accelerate.cpu_offload(model, execution_device=torch.device("cpu"))
    hook = layer._hf_hook
    assert layer.qweight.is_meta
    state_dict = {name: tensor.clone() for name, tensor in hook.weights_map.dataset.items()}
    index = {}
    if disk_backed:
        # Move one tensor into the disk index: that entry cannot be rewritten.
        moved = next(iter(state_dict))
        index[moved] = {"safetensors_file": "unused.safetensors", "weight_name": moved}
        state_dict.pop(moved)
    loader = OffloadedWeightsLoader(state_dict=state_dict, index=index)
    hook.weights_map = PrefixedDataset(loader, prefix=hook.weights_map.prefix)
    if disk_backed:
        with pytest.raises(RuntimeError, match="disk-backed entries"):
            gptqmodel_post_init(model, use_act_order=False)
        assert layer.qweight.is_meta
        return

    gptqmodel_post_init(model, use_act_order=False)

    # The layer is back on meta and the loader's in-memory dict now holds the
    # packed Marlin layout the hook will replay.
    assert layer.qweight.is_meta
    # Every packed tensor the hook replays must be the repacked one.
    # `qzeros_format` is excluded: it is a mirror that `state_dict()` derives
    # from the module's own `_qzeros_format` (`_emitted_qzeros_format`), so the
    # map's copy is never the value a replay uses.
    non_persistent = set(layer._non_persistent_buffers_set) | {"qzeros_format"}
    rewritten = 0
    for name, want in expected_state.items():
        if name in non_persistent:
            continue
        key = f"{hook.weights_map.prefix}{name}"
        torch.testing.assert_close(loader.state_dict[key], want, rtol=0, atol=0)
        rewritten += 1
    assert rewritten >= 5, f"only {rewritten} tensors were checked"

    # Two replays catch a stale map after the hook returns the layer to meta.
    for _ in range(2):
        hook.pre_forward(layer)
        try:
            for name, want in expected_state.items():
                torch.testing.assert_close(layer.state_dict()[name], want, rtol=0, atol=0)
        finally:
            hook.post_forward(layer, None)
        assert all(parameter.is_meta for parameter in layer.parameters(recurse=False))


def test_awq_marlin_uses_the_shared_offload_post_init(monkeypatch):
    """AWQ Marlin had no offload path at all; it must share the GPTQ one."""

    # Constructing the module validates the Marlin runtime, which needs CUDA.
    _require_cuda()
    from gptqmodel.nn_modules.qlinear import marlin_awq
    from gptqmodel.utils import model as model_utils

    calls = []
    original = model_utils._post_init_cpu_offloaded_marlin
    monkeypatch.setattr(
        model_utils,
        "_post_init_cpu_offloaded_marlin",
        lambda module: calls.append(type(module).__name__),
    )
    try:
        layer = marlin_awq.AwqMarlinLinear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=256,
            out_features=128,
            bias=False,
        )
        layer.to("meta")
        model_utils.gptqmodel_post_init(nn.Sequential(layer), use_act_order=False)
    finally:
        monkeypatch.setattr(model_utils, "_post_init_cpu_offloaded_marlin", original)

    assert calls == ["AwqMarlinLinear"]


def test_offload_predicates_cover_cpu_and_disk():
    """AUTO must treat `disk` like `cpu`: neither leaves resident tensors."""

    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import (
        auto_cpu_offload,
        device_map_offloads_layers,
        device_map_uses_cpu,
        is_offload_placement,
    )

    assert is_offload_placement("cpu")
    assert is_offload_placement(torch.device("cpu"))
    assert is_offload_placement("disk")
    assert is_offload_placement("DISK")
    assert not is_offload_placement(None)
    assert not is_offload_placement(0)
    assert not is_offload_placement("cuda:0")
    assert not is_offload_placement(torch.device("cuda", 0))

    assert device_map_offloads_layers({"a": 0, "b": "cpu"})
    assert device_map_offloads_layers({"a": 0, "b": "disk"})
    assert not device_map_offloads_layers({"a": 0, "b": 1})
    assert not device_map_offloads_layers(None)
    # An unresolved keyword does not describe an offload placement.
    assert not device_map_offloads_layers("auto")

    # The wider predicate is what AUTO selection uses now (#5194).
    assert auto_cpu_offload(BACKEND.AUTO, {"a": "disk"})
    assert auto_cpu_offload(None, {"a": "disk"})
    assert not auto_cpu_offload(BACKEND.MARLIN, {"a": "disk"})
    # `device_map_uses_cpu` keeps its narrower "is CPU-resident" meaning.
    assert not device_map_uses_cpu({"a": "disk"})


class _OffloadableLayer(nn.Module):
    """The contract `_post_init_offloaded` relies on, without a real kernel."""

    SUPPORTS_CPU_OFFLOAD = True

    def __init__(self):
        super().__init__()
        self.qweight = nn.Parameter(
            torch.zeros(4, dtype=torch.int32), requires_grad=False
        )
        self.calls = 0

    def checkpoint_payload_tensors(self):
        return (self.qweight,)

    def checkpoint_payload_is_meta(self):
        return any(tensor.is_meta for tensor in self.checkpoint_payload_tensors())

    def post_init(self):
        # A disk-offloaded layer is materialized for the call, then returned.
        assert not self.qweight.is_meta
        self.calls += 1


class _NonOffloadableLayer(_OffloadableLayer):
    """A CUDA-only kernel: its `post_init` needs resident tensors."""

    SUPPORTS_CPU_OFFLOAD = False


def test_post_init_offloaded_materializes_a_disk_layer_and_restores_meta():
    """A disk-mapped layer must run `post_init` and go back to meta (#5194)."""

    import accelerate

    from gptqmodel.utils.model import _post_init_offloaded

    layer = _OffloadableLayer()
    model = nn.Sequential(layer)
    accelerate.cpu_offload(model, execution_device=torch.device("cpu"))
    assert layer.qweight.is_meta

    _post_init_offloaded(layer)

    assert layer.calls == 1
    # The offload map stays authoritative: the hook re-materializes on forward.
    assert layer.qweight.is_meta
    hook = layer._hf_hook
    hook.pre_forward(layer)
    try:
        assert not layer.qweight.is_meta
    finally:
        hook.post_forward(layer, None)


def test_post_init_offloaded_rejects_a_kernel_without_the_declaration():
    """Fail closed instead of running a CUDA-only `post_init` on meta."""

    import accelerate

    from gptqmodel.utils.model import _post_init_offloaded

    layer = _NonOffloadableLayer()
    model = nn.Sequential(layer)
    accelerate.cpu_offload(model, execution_device=torch.device("cpu"))
    assert layer.qweight.is_meta

    with pytest.raises(RuntimeError, match="SUPPORTS_CPU_OFFLOAD"):
        _post_init_offloaded(layer)
    assert layer.calls == 0
    assert layer.qweight.is_meta


def test_post_init_offloaded_keeps_the_plain_path_for_resident_layers():
    """A resident layer needs no hook: `post_init` runs in place."""

    from gptqmodel.utils.model import _post_init_offloaded

    layer = _OffloadableLayer()
    _post_init_offloaded(layer)
    assert layer.calls == 1
    assert not layer.qweight.is_meta
