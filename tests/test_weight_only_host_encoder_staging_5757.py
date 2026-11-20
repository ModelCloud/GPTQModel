# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5757: CPU-backed direct packing must not stage the dense weight on GPU.

GGUF and BitsAndBytes encode on the host, and NF12 stages an eligible host
source inside its own native encoder. Staging the layer on the quantization
device first therefore copied the whole dense weight down and back for
nothing: a 4096 x 4096 BF16 projection paid 64 MiB of pure round-trip traffic,
and a lazy shell held VRAM for a layer that never encodes there.

The public ``quantize()`` path now keeps those sources on their host placement.
The requested device still selects the inference backend and owns the packed
checkpoint handoff, and RTN keeps its real accelerator quantization.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.quantization.config import BitsAndBytesConfig, GGUFConfig, NF12Config, RTNConfig
from gptqmodel.utils.backend import BACKEND


_SIZE = 128

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the accelerator leg needs a CUDA/ROCm device",
)


class _OneProjection(BaseQModel):
    module_tree = ["model", "layers", "#", {"": ("proj:0",)}]


def _native(size: int = _SIZE) -> nn.Module:
    torch.manual_seed(123)
    native = nn.Module()
    native.model = nn.Module()
    native.model.layers = nn.ModuleList([nn.Module()])
    native.model.layers[0].proj = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
    native.config = SimpleNamespace(
        use_cache=False,
        tie_word_embeddings=False,
        model_type="host_encoder_staging_fixture",
    )
    return native


def _model(native: nn.Module, config) -> _OneProjection:
    return _OneProjection(
        model=native.eval(),
        quantized=False,
        quantize_config=config,
        tokenizer=None,
    )


def _gguf_config(device: str) -> GGUFConfig:
    return GGUFConfig(bits="q4_0", smoother=None, offload_to_disk=False, device=device)


def _run(model: _OneProjection, *, pack_failure: bool = False):
    """Run the public quantize() path and record every dense-source placement."""

    placements = []
    pre_quantize = BaseQModel.pre_quantize
    quantize_module = WeightOnlyProcessor.quantize_module
    pack_original = GGUFTorchLinear.pack_original

    def observe_staging(self, layer, *args, **kwargs):
        placements.append(("enter layer staging", layer.proj.weight.device.type))
        staged = pre_quantize(self, layer, *args, **kwargs)
        placements.append(("exit layer staging", staged.proj.weight.device.type))
        return staged

    def observe_prepack(self, named, *args, **kwargs):
        placements.append(("enter direct-pack processor", named.weight.device.type))
        return quantize_module(self, named, *args, **kwargs)

    def observe_pack(self, linear, *args, **kwargs):
        placements.append(("enter pack_original", linear.weight.device.type))
        if pack_failure:
            raise RuntimeError("pack failed for the #5757 recovery check")
        return pack_original(self, linear, *args, **kwargs)

    with (
        patch.object(BaseQModel, "pre_quantize", observe_staging),
        patch.object(WeightOnlyProcessor, "quantize_module", observe_prepack),
        patch.object(GGUFTorchLinear, "pack_original", observe_pack),
    ):
        model.quantize(backend=BACKEND.TORCH)
    return [placement for _, placement in placements]


def _packed_bytes(layer) -> bytes:
    for name in ("qweight", "nf12_packed"):
        tensor = getattr(layer, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    raise AssertionError(f"no packed weight buffer on {type(layer).__name__}")


@CUDA_REQUIRED
def test_gguf_accelerator_run_keeps_the_host_source_on_the_host():
    """A CUDA-configured GGUF run must not round-trip the dense weight."""

    host_native = _native()
    _run(_model(host_native, _gguf_config("cpu")))
    cpu_bytes = _packed_bytes(host_native.model.layers[0].proj)

    accelerator_native = _native()
    placements = _run(_model(accelerator_native, _gguf_config("cuda:0")))

    assert placements == ["cpu", "cpu", "cpu", "cpu"], placements
    # The host path must produce the same checkpoint bytes as before.
    assert _packed_bytes(accelerator_native.model.layers[0].proj) == cpu_bytes


@CUDA_REQUIRED
def test_gguf_lazy_shell_materializes_on_the_host():
    """A lazy layer must not reserve accelerator VRAM for a host-side encoder."""

    native = _native()
    dense_weight = native.model.layers[0].proj.weight.detach().clone()
    native.model.layers[0].proj = native.model.layers[0].proj.to("meta")
    model = _model(native, _gguf_config("cuda:0"))

    materialized_on = []
    shell_module_materialize = BaseQModel.shell_module_materialize

    def observe_materialize(self, target_submodule, device, *args, **kwargs):
        materialized_on.append(torch.device(device).type)
        # Materialize from the host copy that a lazy checkpoint would stream in.
        staged = target_submodule.to_empty(device=torch.device(device))
        with torch.no_grad():
            staged.proj.weight.copy_(dense_weight.to(torch.device(device)))
        return shell_module_materialize(
            self, staged, torch.device(device), *args, **kwargs
        )

    with patch.object(BaseQModel, "shell_module_materialize", observe_materialize):
        placements = _run(model)

    assert materialized_on == ["cpu"], materialized_on
    # The lazy layer enters on `meta` and must be materialized on the host.
    assert placements == ["meta", "cpu", "cpu", "cpu"], placements


@CUDA_REQUIRED
def test_bitsandbytes_accelerator_run_keeps_the_host_source_on_the_host():
    pytest.importorskip("bitsandbytes")

    native = _native()
    placements = _run(_model(native, BitsAndBytesConfig(bits=4, device="cuda:0")))

    assert placements == ["cpu", "cpu", "cpu"], placements


@CUDA_REQUIRED
def test_nf12_accelerator_run_leaves_staging_to_its_native_encoder():
    """NF12 stages an eligible host source itself, so the looper must not."""

    host_native = _native()
    _run(_model(host_native, NF12Config(device="cpu")))
    cpu_bytes = _packed_bytes(host_native.model.layers[0].proj)

    accelerator_native = _native()
    placements = _run(_model(accelerator_native, NF12Config(device="cuda:0")))

    assert placements == ["cpu", "cpu", "cpu"], placements
    assert _packed_bytes(accelerator_native.model.layers[0].proj) == cpu_bytes


@CUDA_REQUIRED
def test_rtn_accelerator_run_still_quantizes_on_the_device():
    """The host-encoder exemption must not leak into RTN's device quantization."""

    native = _native()
    placements = _run(
        _model(native, RTNConfig(device="cuda:0", bits=4, group_size=32, format="gptq"))
    )

    assert placements == ["cpu", "cuda", "cuda"], placements


@CUDA_REQUIRED
def test_gguf_pack_failure_leaves_the_source_on_the_host():
    """A failed host-side pack must not strand accelerator tensors on the layer."""

    native = _native()
    model = _model(native, _gguf_config("cuda:0"))

    with pytest.raises(RuntimeError, match="pack failed for the #5757 recovery check"):
        _run(model, pack_failure=True)

    layer = native.model.layers[0].proj
    for tensor in layer.parameters():
        assert tensor.device.type == "cpu", tensor.device
    for tensor in layer.buffers():
        assert tensor.device.type == "cpu", tensor.device

    # The public path still works once the packer recovers.
    recovered_native = _native()
    assert _run(_model(recovered_native, _gguf_config("cuda:0"))) == [
        "cpu",
        "cpu",
        "cpu",
        "cpu",
    ]
