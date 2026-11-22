# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Install inference-only NF12 layers in a loaded Inkling 2 Mini model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from ....nn_modules.forward_dtype_lock import ForwardDtypeLockMixin
from ....utils.nf12_tenstorrent import NF12Descriptor, prepare_nf12, read_nf12, tensor_sha256
from .modeling import Inkling2LoopForCausalLM, Inkling2LoopForConditionalGeneration
from .vision import Inkling2LoopVisionModel


class _NF12Linear(ForwardDtypeLockMixin, nn.Module):
    weight_dtype = torch.bfloat16

    def __init__(self, source, layer, xla):
        super().__init__()
        self.layer = layer
        self.xla = xla
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_parameter("bias", source.bias)
        self.name = layer.descriptor.tensor_name
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        if self.xla:
            device = self.layer.device
            bias = self.bias.to(device) if self.bias is not None else None
            return self.layer.linear(x.to(device), bias).to(x.device)
        return self.layer.linear(x, self.bias)


class _NF12Embedding(nn.Module):
    def __init__(self, source, layer, *, xla):
        super().__init__()
        self.layer = layer
        self.xla = xla
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.embed_norm = getattr(source, "embed_norm", nn.Identity())
        self.eval()

    @torch.no_grad()
    def forward(self, ids):
        # As with nn.Embedding, the model reads the stored padding row. The
        # low-level raw gather also supports an explicitly zeroed padding ID.
        if self.xla and ids.device.type == "cpu":
            return self.embed_norm(self.layer.embedding_to_host(ids))
        return self.embed_norm(self.layer.embedding(ids).to(ids.device))


def _targets(model, vision, text):
    is_vision = isinstance(model, Inkling2LoopVisionModel)
    if text and not is_vision:
        prefix = "model.language_model" if isinstance(model, Inkling2LoopForConditionalGeneration) else "model"
        yield prefix + ".embed_tokens"
        yield "lm_head"
    if not vision:
        return
    if not is_vision and not isinstance(model, Inkling2LoopForConditionalGeneration):
        raise ValueError("Inkling NF12 vision requires a vision tower")
    prefix = "" if is_vision else "model.vision_tower."
    tower = model if is_vision else model.model.vision_tower
    yield prefix + "embeddings.patch_embedding"
    yield prefix + "embeddings.position_embedding"
    for idx in range(len(tower.layers)):
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            yield f"{prefix}layers.{idx}.self_attn.{projection}"
        yield f"{prefix}layers.{idx}.mlp.fc1"
        yield f"{prefix}layers.{idx}.mlp.fc2"
    yield prefix + "merger.linear_1"
    yield prefix + "merger.linear_2"


def install_inkling2_nf12(
    model,
    *,
    backend: str,
    device,
    vision: bool = True,
    text: bool = True,
    checkpoint: Mapping[str, tuple[NF12Descriptor, Path]] | None = None,
    native_lm_head: str = "canonical",
    ttxla_lm_head: str = "canonical",
    embedding_layout: str = "canonical",
) -> dict[str, NF12Descriptor]:
    """Replace the requested BF16 inference leaves after normal checkpoint load.

    Pass an already-open single Blackhole TTNN device or a TT-XLA torch device.
    The optional mapping is keyed by the model's runtime weight names and binds
    each canonical safetensors payload to a descriptor with a BF16 source hash.
    On native TTNN, ``native_lm_head="sfpu"`` selects the single-copy SFPU
    layout for the bias-free head and requires the source Blackhole runtime.
    On TT-XLA, ``ttxla_lm_head="sfpu"`` selects its single-copy head layout
    and requires the bundled Blackhole TT-XLA kernel interface.
    ``embedding_layout="row_major"`` derives the CUDA/HIP row format for token
    and position tables from the same verified canonical checkpoint.
    The caller owns device lifetime and should install after model placement.
    """
    if not isinstance(model, (Inkling2LoopForConditionalGeneration, Inkling2LoopForCausalLM, Inkling2LoopVisionModel)):
        raise TypeError("NF12 installation expects an Inkling 2 model")
    if model.training:
        raise ValueError("Load Inkling 2 for inference with eval() before installing NF12")
    if native_lm_head not in {"canonical", "sfpu"}:
        raise ValueError(f"Unknown native Inkling NF12 LM-head layout {native_lm_head!r}")
    if ttxla_lm_head not in {"canonical", "sfpu"}:
        raise ValueError(f"Unknown TT-XLA Inkling NF12 LM-head layout {ttxla_lm_head!r}")
    if embedding_layout not in {"canonical", "row_major"}:
        raise ValueError(f"Unknown Inkling NF12 embedding layout {embedding_layout!r}")
    if native_lm_head == "sfpu" and (backend != "ttnn" or not text or isinstance(model, Inkling2LoopVisionModel)):
        raise ValueError("The SFPU LM head requires native TTNN and text installation")
    if ttxla_lm_head == "sfpu" and (backend != "ttxla" or not text or isinstance(model, Inkling2LoopVisionModel)):
        raise ValueError("The TT-XLA SFPU LM head requires the TT-XLA backend and text installation")
    if backend == "ttnn":
        from ....utils.nf12_ttnn import TTNNNF12 as executor
    elif backend == "ttxla":
        from ....utils.nf12_ttxla import TTXLANF12 as executor
    else:
        raise ValueError(f"Unknown Inkling NF12 backend {backend!r}")

    replacements, descriptors = {}, {}
    for name in _targets(model, vision, text):
        source = model.get_submodule(name)
        if not isinstance(source, (nn.Linear, nn.Embedding)) or source.weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected an Inkling 2 BF16 checkpoint leaf at {name}")
        weight_name = name + ".weight"
        original = source.weight.detach().cpu()
        if checkpoint is None:
            descriptor, weight = prepare_nf12(original, weight_name)
        else:
            descriptor, path = checkpoint[weight_name]
            if (descriptor.logical_shape != tuple(original.shape)
                    or descriptor.source_sha256 != tensor_sha256(original)):
                raise ValueError(f"NF12 checkpoint source mismatch for {weight_name}")
            weight = read_nf12(path, descriptor)

        if name == "lm_head" and "sfpu" in (native_lm_head, ttxla_lm_head):
            if not isinstance(source, nn.Linear) or source.bias is not None:
                raise ValueError("The SFPU NF12 LM head must be a bias-free linear")
            if backend == "ttxla":
                from ....utils.nf12_ttxla_sfpu import TTXLASFPUNF12Head

                layer = TTXLASFPUNF12Head(device, descriptor, weight)
            else:
                from ....utils.nf12_ttnn_sfpu import TTNNSFPUNF12Head

                layer = TTNNSFPUNF12Head(device, descriptor, weight)
        elif isinstance(source, nn.Embedding) and embedding_layout == "row_major":
            if backend == "ttnn":
                from ....utils.nf12_ttnn_rows import TTNNNF12RowEmbedding

                layer = TTNNNF12RowEmbedding(device, descriptor, weight)
            else:
                layer = executor(device, descriptor, weight, row_major=True)
        else:
            layer = executor(device, descriptor, weight)
        if isinstance(source, nn.Linear):
            replacements[name] = _NF12Linear(source, layer, backend == "ttxla")
        else:
            replacements[name] = _NF12Embedding(source, layer, xla=backend == "ttxla")
        descriptors[weight_name] = descriptor
    for name, module in replacements.items():
        model.set_submodule(name, module)
    return descriptors
