# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Issue #4661: plain Transformers must load a GPT-QModel AWQ MoE checkpoint.

``quant_method: "awq"`` never reaches Optimum, so #4084's Optimum-only bridge
left Transformers' own ``AwqQuantizer`` path in place. That path replaces only
``nn.Linear`` modules, so a checkpoint that stores per-expert
``experts.<n>.*.qweight`` tensors against fused 3-D expert parameters loaded
with every routed expert randomly initialized and no error.

The regression quantizes a tiny two-layer OLMoE model with AWQ, saves it, and
loads it through ``AutoModelForCausalLM.from_pretrained`` in a process that
imported ``gptqmodel`` first. It asserts that the expert tensors are consumed
(no MISSING/UNEXPECTED expert keys), that every expert projection is a packed
kernel, and that the loaded packed tensors are bit-identical to
``GPTQModel.load`` before comparing logits.
"""

import json

import pytest
import torch
from transformers import AutoModelForCausalLM, OlmoeConfig, OlmoeForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import AWQConfig


_HIDDEN = 64
_EXPERTS = 4
_VOCAB = 128


def _tiny_olmoe_config() -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_local_experts=_EXPERTS,
        max_position_embeddings=64,
    )


def _calibration():
    torch.manual_seed(0)
    return [
        {"input_ids": torch.randint(0, _VOCAB, (1, 16))}
        for _ in range(4)
    ]


@pytest.fixture(scope="module")
def awq_checkpoint(tmp_path_factory):
    source = tmp_path_factory.mktemp("olmoe_native")
    destination = tmp_path_factory.mktemp("olmoe_awq")

    torch.manual_seed(0)
    OlmoeForCausalLM(_tiny_olmoe_config()).save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        quantize_config=AWQConfig(
            bits=4, group_size=32, device="cpu", offload_to_disk=False
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration(),
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )
    model.save(str(destination))
    assert json.loads((destination / "config.json").read_text())[
        "quantization_config"
    ]["quant_method"] == "awq"
    return str(destination)


def _expert_suffixes(info, field):
    return sorted(
        key
        for key in info[field]
        if ".mlp.experts" in key and ".qzeros_format" not in key
    )


def test_plain_transformers_loads_every_awq_expert(awq_checkpoint):
    hf_model, info = AutoModelForCausalLM.from_pretrained(
        awq_checkpoint,
        dtype=torch.float16,
        device_map="cpu",
        output_loading_info=True,
    )

    # The bug: the fused 3-D expert parameters stayed dense and every packed
    # expert tensor was dropped, silently random-initializing the experts.
    assert _expert_suffixes(info, "missing_keys") == []
    assert _expert_suffixes(info, "unexpected_keys") == []

    for layer in hf_model.model.layers:
        experts = layer.mlp.experts
        assert not hasattr(experts, "gate_up_proj")
        for expert in experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                module = getattr(expert, projection)
                assert isinstance(module, BaseQuantLinear), (
                    f"{projection} is {type(module).__name__}, not a packed kernel"
                )


def test_plain_transformers_load_decodes_the_same_expert_words(awq_checkpoint):
    reference = GPTQModel.load(awq_checkpoint, backend=BACKEND.AWQ_TORCH, device="cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(
        awq_checkpoint, dtype=torch.float16, device_map="cpu"
    )

    reference_tensors = dict(reference.model.named_buffers())
    for name, tensor in hf_model.named_buffers():
        if ".mlp.experts" not in name or not name.endswith(("qweight", "qzeros", "scales")):
            continue
        assert name in reference_tensors, name
        assert torch.equal(tensor, reference_tensors[name]), name

    torch.manual_seed(1)
    input_ids = torch.randint(0, _VOCAB, (1, 16))
    with torch.inference_mode():
        expected = reference.model(input_ids, use_cache=False).logits.float()
        actual = hf_model(input_ids, use_cache=False).logits.float()
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape
    # Both sides run packed 4-bit experts; the kernels differ, so compare at
    # FP16 logit scale rather than bit-for-bit.
    assert (actual - expected).abs().max().item() < 0.05


def test_plain_transformers_load_without_gptqmodel_refuses_dropped_experts(awq_checkpoint):
    """The plain flow cannot recover experts dropped before the bridge existed.

    ``gptqmodel`` is imported from inside ``replace_with_awq_linear``, so the
    bridge's prepare step cannot run for that load and the packed expert tensors
    were already reported UNEXPECTED. The post hook must raise an actionable
    error instead of returning randomly initialized experts (#4795).
    """

    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import torch\n"
        "from transformers import AutoModelForCausalLM\n"
        "AutoModelForCausalLM.from_pretrained("
        f"{awq_checkpoint!r}, device_map='cpu', dtype=torch.float16)\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH")) if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=awq_checkpoint,
        env=env,
        text=True,
    )

    assert completed.returncode != 0, completed.stdout
    assert "did not consume" in completed.stderr
    assert "experts.0." in completed.stderr
    assert "Import `gptqmodel`" in completed.stderr
