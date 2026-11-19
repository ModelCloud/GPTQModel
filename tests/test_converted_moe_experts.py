# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Defuser-converted vendored MoE experts are deterministic and self-contained.

* Families Defuser replaces at construction time with a routed expert loop
  (``run_routed_experts``) feed each expert its token rows ordered by top-k
  slot, then token. GPT-QModel converts the vendored block after construction,
  and its converted experts must keep that row order: AWQ scale search is
  sensitive to it, so a different order moves expert scales.
* A converted model must unpickle and run in a fresh process, which never ran
  the conversion and so never registered Defuser's ``linear_loop`` experts
  implementation in Transformers' experts registry.
"""

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from gptqmodel.models.loader import _convert_model_with_defuser
from tests.test_defuser_vendored_classes import _build, _Definition


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTED_LOOP_FAMILIES = ("mixtral", "qwen3_moe", "qwen2_moe", "qwen3_next")


@pytest.mark.parametrize("family", ROUTED_LOOP_FAMILIES)
def test_converted_experts_visit_tokens_in_routed_loop_order(family):
    model = _build(family, "gptqmodel.models.ports.native")
    assert _convert_model_with_defuser(_Definition, model, cleanup_original=True)

    calls, rows = [], {}
    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".experts") and "0" in module._modules:
            hooks.append(module.register_forward_pre_hook(
                lambda m, args, name=name: calls.append((name, m.num_experts, args[0].detach(), args[1].detach()))
            ))
            for expert in range(module.num_experts):
                proj = module.get_submodule(f"{expert}.up_proj")
                hooks.append(proj.register_forward_pre_hook(
                    lambda m, args, key=(name, expert): rows.setdefault(key, []).append(args[0].detach())
                ))
    assert hooks

    ids = torch.randint(1, 64, (2, 9), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        model(input_ids=ids)

    assert calls
    for name, num_experts, hidden, top_k_index in calls:
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        for expert in range(num_experts):
            _, token_idx = torch.where(mask[expert])
            if token_idx.numel() == 0:
                assert (name, expert) not in rows
                continue
            cached = rows[(name, expert)].pop(0)
            assert torch.equal(cached, hidden[token_idx]), (name, expert)


_FRESH_PROCESS = textwrap.dedent(
    """
    import sys, torch
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    assert "linear_loop" not in ALL_EXPERTS_FUNCTIONS
    archive = torch.load(sys.argv[1], weights_only=False)
    with torch.no_grad():
        logits = archive["model"](input_ids=archive["ids"]).logits
    assert torch.equal(logits, archive["logits"]), (logits - archive["logits"]).abs().max()
    # Self-contained: running the model did not need the registry entry.
    assert "linear_loop" not in ALL_EXPERTS_FUNCTIONS
    print("FRESH_PROCESS_OK")
    """
)


@pytest.mark.parametrize("family", ["mixtral", "qwen3_moe", "glm4_moe"])
def test_converted_moe_model_runs_after_unpickling_in_fresh_process(family, tmp_path):
    model = _build(family, "gptqmodel.models.ports.native")
    assert _convert_model_with_defuser(_Definition, model, cleanup_original=True)
    ids = torch.randint(1, 64, (2, 7), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    archive = tmp_path / "model.pt"
    torch.save({"model": model, "ids": ids, "logits": logits}, archive)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    ))
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, str(archive)],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=600,
    )
    assert result.returncode == 0 and "FRESH_PROCESS_OK" in result.stdout, result.stderr[-4000:]


def _expert_quant_digest(checkpoint: Path) -> str:
    from safetensors.torch import load_file

    tensors = load_file(str(checkpoint / "model.safetensors"))
    digest = hashlib.sha256()
    for key in sorted(tensors):
        if ".experts." in key and key.rsplit(".", 1)[-1] in ("qweight", "qzeros", "scales"):
            digest.update(key.encode())
            digest.update(tensors[key].contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# sha256 of every expert qweight/qzeros/scales tensor produced by GPT-QModel
# main (556f4b22d; Transformers' own classes with Defuser's construction-time
# ``Linear*SparseMoeBlock``, i.e. the ``run_routed_experts`` loop) for the tiny
# checkpoints below on CPU torch. The pre-fix token-major ``linear_loop`` order
# produced different scales. These track main's AWQ output: when main
# deliberately changes AWQ numerics, regenerate them by running this test's
# quantization on a main checkout (the vendored classes must still match).
_MAIN_AWQ_EXPERT_DIGESTS = {
    "mixtral": "b9a9ecfd674b9e42f666491dbfbe16141062a1b7511b0484ffaad2ccc6da66a0",
    "qwen3_moe": "23b14c84e921198df0b3c9535c4a12c9ef39588d6077049da4cfaa8fa4a51d0d",
}


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_MAIN_AWQ_EXPERT_DIGESTS))
def test_tiny_moe_awq_expert_scales_match_routed_loop_reference(family, tmp_path):
    # Transformers classes only write the reference checkpoint (test oracle).
    from transformers import MixtralConfig, MixtralForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
    from gptqmodel.quantization import FORMAT, METHOD
    from tests.test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer

    common = {
        "num_hidden_layers": 1, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 4,
        "num_experts_per_tok": 2, "vocab_size": 128, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
    }
    torch.manual_seed(0)
    if family == "qwen3_moe":
        native = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            intermediate_size=128, moe_intermediate_size=32, num_experts=4, **common,
        ))
    else:
        native = MixtralForCausalLM(MixtralConfig(intermediate_size=64, num_local_experts=4, **common))
    source, quantized = tmp_path / "native", tmp_path / "quant"
    native.save_pretrained(source)
    tokenizer = _build_local_tokenizer(source)

    model = GPTQModel.load(
        str(source),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device="cpu", method=METHOD.AWQ, format=FORMAT.GEMM,
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(dataset=_build_calibration_dataset(tokenizer), batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(quantized)

    assert _expert_quant_digest(quantized) == _MAIN_AWQ_EXPERT_DIGESTS[family]
