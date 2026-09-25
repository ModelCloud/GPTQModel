# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounded output-channel partitions for offline QVQ head calibration.

The wrapper keeps the full vocabulary distribution for the real-Fisher loss.
Each child linear exposes one principal output block to the YAQA collector.
This is a preparation operator; serving and checkpoint formats are separate.
"""

from __future__ import annotations

import torch
from torch import nn


class VocabBlockLinear(nn.Module):
    """Present a dense vocabulary head as contiguous independent row blocks."""

    def __init__(self, source: nn.Linear, block_rows: int = 2048):
        super().__init__()
        if not isinstance(source, nn.Linear) or source.bias is not None:
            raise TypeError("QVQ vocabulary blocking requires a bias-free nn.Linear head")
        if block_rows < 16 or block_rows % 16:
            raise ValueError("QVQ vocabulary block rows must be a positive multiple of 16")
        if source.out_features % 16 or source.in_features % 16:
            raise ValueError("QVQ vocabulary dimensions must be multiples of 16")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.block_rows = block_rows
        self.blocks = nn.ModuleList()
        for start in range(0, source.out_features, block_rows):
            stop = min(start + block_rows, source.out_features)
            block = nn.Linear(source.in_features, stop - start, bias=False,
                              device=source.weight.device, dtype=source.weight.dtype)
            with torch.no_grad():
                block.weight.copy_(source.weight[start:stop])
            self.blocks.append(block)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(hidden) for block in self.blocks], dim=-1)

    def yaqa_targets(self, prefix: str = "lm_head") -> dict[str, nn.Linear]:
        return {f"{prefix}.blocks.{index}": block for index, block in enumerate(self.blocks)}


class QVQVocabHead(nn.Module):
    """Concatenate independently quantized vocabulary blocks in token order."""

    def __init__(self, blocks: list[nn.Module], in_features: int, out_features: int):
        super().__init__()
        if not blocks or in_features < 1 or out_features < 1:
            raise ValueError("QVQ vocabulary head requires nonempty blocks and dimensions")
        self.blocks = nn.ModuleList(blocks)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(hidden) for block in self.blocks], dim=-1)


def install_qvq_vocab_head_delta(
    model: nn.Module,
    artifact_dir: str,
    *,
    expected_model_path: str,
    arm: str = "candidate",
    allow_partial: bool = False,
) -> QVQVocabHead:
    """Install an offline block-head delta on a GPTQModel/HF causal model.

    This loader supports PyTorch validation. It does not register the delta
    as a standard QVQ checkpoint or provide a ZML serving implementation.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear

    if arm not in {"baseline", "candidate", "guarded"}:
        raise ValueError("vocabulary-head arm must be baseline, candidate, or guarded")
    artifact = Path(artifact_dir)
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest.get("schema") != "qvq.vocab-head-delta.v1":
        raise ValueError("unsupported QVQ vocabulary-head artifact schema")
    if str(Path(expected_model_path).resolve()) != manifest.get("model_path"):
        raise ValueError("QVQ vocabulary-head artifact was built for a different model")
    if not allow_partial and manifest.get("complete_head") != "true":
        raise ValueError("partial QVQ vocabulary head cannot replace a model endpoint")
    source = model if hasattr(model, "set_output_embeddings") else model.model
    head = source.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise TypeError("QVQ vocabulary-head delta requires a dense nn.Linear endpoint")
    if (int(manifest["head_columns"]) != head.in_features
            or int(manifest["head_rows"]) != head.out_features):
        raise ValueError("QVQ vocabulary-head delta does not match model endpoint dimensions")
    bits = float(manifest["bits"])
    block_rows = int(manifest["block_rows"])
    count = int(manifest["block_count"])
    blocks = []
    with safe_open(str(artifact / f"{arm}.safetensors"), framework="pt", device="cpu") as packed:
        metadata = packed.metadata()
        if any(metadata.get(key) != manifest[key]
               for key in ("schema", "model_path", "bits", "block_rows", "block_count")):
            raise ValueError("QVQ vocabulary-head tensor metadata differs from manifest")
        for index in range(count):
            start = index * block_rows
            stop = min(start + block_rows, head.out_features)
            prefix = f"lm_head.blocks.{index}."
            tensors = {
                key.removeprefix(prefix): packed.get_tensor(key).to(device=head.weight.device)
                for key in packed.keys() if key.startswith(prefix)  # noqa: SIM118 - safetensors reader API
            }
            if not {"trellis", "SU", "SV"}.issubset(tensors):
                raise ValueError(f"QVQ vocabulary block {index} is missing required tensors")
            block = QVQLinear(
                bits=bits, in_features=head.in_features, out_features=stop - start,
                bias=False, backend=BACKEND.QVQ,
                name=f"lm_head.blocks.{index}", dtype=head.weight.dtype,
                tensors=tensors, v2b2_p32=bits < 4,
                bank_count=2 if bits < 4 else 1,
            ).eval()
            block.post_init()
            blocks.append(block)
    replacement = QVQVocabHead(blocks, head.in_features, head.out_features)
    source.config.tie_word_embeddings = False
    source.set_output_embeddings(replacement)
    return replacement


def factored_head_fisher_loss(
    errors: list[torch.Tensor],
    output_factors: list[torch.Tensor],
    input_hessian: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate the complete two-sided head Fisher without an output Gram.

    For output Fisher ``S @ S.T`` and concatenated weight error ``E``, the
    objective is ``tr((E @ S).T @ H @ (E @ S))``. Keeping ``E @ S`` small
    retains the interactions between different vocabulary blocks.
    """
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Fisher oracle dtype must be FP32 or FP64")
    if not errors or len(errors) != len(output_factors):
        raise ValueError("Fisher oracle requires matching nonempty error/factor blocks")
    width = input_hessian.shape[0]
    rank = output_factors[0].shape[1]
    if input_hessian.shape != (width, width):
        raise ValueError("Fisher input Hessian must be square")
    if any(error.shape != (width, factor.shape[0]) or factor.shape[1] != rank
           for error, factor in zip(errors, output_factors)):
        raise ValueError("Fisher blocks have mismatched input, output, or factor dimensions")
    if any(error.device != input_hessian.device or factor.device != input_hessian.device
           for error, factor in zip(errors, output_factors)):
        raise ValueError("Fisher blocks and input Hessian must share a device")
    projected = sum(error.to(dtype) @ factor.to(dtype) for error, factor in zip(errors, output_factors))
    return torch.einsum("ir,ij,jr->", projected, input_hessian.to(dtype), projected)
