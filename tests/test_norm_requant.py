# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fast CPU coverage for norm tuning through ``requantize(norm=...)``."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from gptqmodel import NormConfig, NormType, QuantizeEmbed
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.writer import _save_module_replacement_safetensors
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.quantization.norm_tuning import tune_norm_params
from gptqmodel.utils.model import pack_module


def _tiny_qwen3(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


def _bare(cls, hf_model=None, *, quantized: bool = True):
    model = BaseQModel.__new__(cls)
    nn.Module.__init__(model)
    model.quantized = quantized
    model.load_quantized_model = False
    if hf_model is not None:
        model.model = hf_model
    return model


def _kl(student, teacher, input_ids) -> float:
    with torch.no_grad():
        t = F.log_softmax(teacher(input_ids=input_ids).logits.float(), -1)
        s = F.log_softmax(student(input_ids=input_ids).logits.float(), -1)
    return float((t.exp() * (t - s)).sum(-1).mean())


def test_norm_modules_come_from_module_tree_tags():
    assert LlamaQModel.get_norm_modules() == ["input_layernorm", "post_attention_layernorm"]
    assert Qwen3QModel.get_norm_modules() == [
        "input_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "post_attention_layernorm",
    ]
    # The tag is metadata only: the quantization layer walk is unchanged.
    flat = [name for block in Qwen3QModel.build_layer_modules(Qwen3QModel.module_tree) for name in block]
    assert "input_layernorm:!" in flat
    assert "self_attn.q_norm:!" in flat
    assert not any(":norm" in name for name in flat)


def test_norm_module_names_expand_layers_and_final_norm():
    model = _bare(Qwen3QModel, _tiny_qwen3())
    names = model.norm_module_names()
    assert names[:4] == [
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_norm",
        "model.layers.0.self_attn.k_norm",
        "model.layers.0.post_attention_layernorm",
    ]
    assert names[-1] == "model.norm"
    assert len(names) == 2 * 4 + 1


def test_tune_norm_params_trains_only_norm_gains():
    teacher = _tiny_qwen3()
    student = copy.deepcopy(teacher)
    names = _bare(Qwen3QModel, student).norm_module_names()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name in names:
            gain = student.get_submodule(name).weight
            gain.mul_(1 + 0.3 * torch.randn(gain.shape, generator=generator))
    before = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()}
    grad_flags = {name: param.requires_grad for name, param in student.named_parameters()}

    input_ids = torch.randint(0, 64, (8, 16), generator=generator)
    batches = [{"input_ids": input_ids[i : i + 2], "attention_mask": torch.ones(2, 16, dtype=torch.long)} for i in range(0, 8, 2)]
    kl_before = _kl(student, teacher, input_ids)

    stats = tune_norm_params(student, teacher, batches, names, NormConfig(teacher=teacher, steps=60, lr=1e-2, reg=0.0))

    assert stats["modules"] == names
    assert _kl(student, teacher, input_ids) < 0.5 * kl_before
    changed = {name for name, tensor in student.state_dict().items() if not torch.equal(tensor, before[name])}
    assert changed == {f"{name}.weight" for name in names}
    assert {name: param.requires_grad for name, param in student.named_parameters()} == grad_flags


def test_tune_norm_params_honors_skip():
    teacher = _tiny_qwen3()
    student = copy.deepcopy(teacher)
    names = _bare(Qwen3QModel, student).norm_module_names()
    batches = [{"input_ids": torch.randint(0, 64, (2, 16))}]
    config = NormConfig(teacher=teacher, steps=2, skip=("q_norm", "k_norm"))
    stats = tune_norm_params(student, teacher, batches, names, config)
    assert len(stats["modules"]) == 2 * 2 + 1
    assert not any(name.endswith(("q_norm", "k_norm")) for name in stats["modules"])


def test_tune_norm_params_escapes_inference_scope_and_clears_stale_gradients():
    teacher = _tiny_qwen3()
    student = copy.deepcopy(teacher)
    names = _bare(Qwen3QModel, student).norm_module_names()
    for name in names:
        student.get_submodule(name).weight.grad = torch.ones_like(student.get_submodule(name).weight)

    with torch.inference_mode():
        batches = [{"input_ids": torch.randint(0, 64, (2, 8))}]
        tune_norm_params(
            student,
            teacher,
            batches,
            names,
            NormConfig(teacher=teacher, steps=1, lr=1e-2, reg=0.0),
        )

    assert all(student.get_submodule(name).weight.grad is None for name in names)


def test_requantize_dispatches_norm_without_embedding_target():
    model = _bare(BaseQModel, nn.Module())
    captured = {}

    def requantize_norm(**kwargs):
        captured.update(kwargs)
        return {"modules": ["model.norm"]}

    model._requantize_norm = requantize_norm
    model._requantize_embeddings = lambda **kwargs: pytest.fail("embedding requant must not run")
    config = NormConfig(teacher="teacher-path", steps=3)

    result = model.requantize(calibration=["sample"], norm=config, batch_size=4)

    assert result == {"norm": {"modules": ["model.norm"]}}
    assert captured["norm"] is config
    assert captured["batch_size"] == 4


def test_requantize_rejects_removed_rmsnorm_argument():
    model = _bare(BaseQModel, nn.Module())
    config = NormConfig(teacher="teacher-path", steps=3)
    with pytest.raises(TypeError, match="unexpected keyword argument 'rmsnorm'"):
        model.requantize(calibration=["sample"], rmsnorm=config)


def test_norm_validates_arguments():
    with pytest.raises(ValueError, match="teacher"):
        NormConfig()
    with pytest.raises(ValueError, match="steps"):
        NormConfig(teacher="t", steps=0)
    with pytest.raises(TypeError, match="NormConfig"):
        _bare(BaseQModel).requantize(calibration=["sample"], norm={"teacher": "t"})
    assert NormConfig(teacher="t", types="layernorm").types == (NormType.LAYERNORM,)
    assert NormConfig(teacher="t", types=(NormType.LAYERNORM, "rmsnorm", "rmsnorm")).types == (
        NormType.LAYERNORM,
        NormType.RMSNORM,
    )
    with pytest.raises(ValueError, match="Unsupported norm type"):
        NormConfig(teacher="t", types=("groupnorm",))
    with pytest.raises(ValueError, match="at least one"):
        NormConfig(teacher="t", types=())


@pytest.mark.parametrize("state", ["meta_parameter", "meta_buffer", "inference_parameter", "v1_zeros"])
def test_unsafe_runtime_rejected_before_tuning_mutates_state(state):
    student = _tiny_qwen3()
    names = _bare(Qwen3QModel, student).norm_module_names()
    if state == "meta_parameter":
        student.model.norm.weight = nn.Parameter(torch.empty(32, device="meta"))
    elif state == "meta_buffer":
        student.register_buffer("offloaded", torch.empty(1, device="meta"))
    elif state == "inference_parameter":
        with torch.inference_mode():
            student.model.norm.weight = nn.Parameter(student.model.norm.weight.clone())
    else:
        student.lm_head = TorchLinear(bits=4, group_size=16, sym=True, desc_act=False,
            in_features=32, out_features=64, bias=False, format=FORMAT.GPTQ)
        student.lm_head.qzero_format(format=1)
    before = {name: (param, param.requires_grad, param.grad) for name, param in student.named_parameters()}
    modes = {name: module.training for name, module in student.named_modules()}
    with pytest.raises(ValueError, match="reload.*BACKEND.TORCH"):
        tune_norm_params(student, _tiny_qwen3(), [{"input_ids": torch.ones(1, 8, dtype=torch.long)}],
            names, NormConfig(teacher="unused", steps=1))
    assert modes == {name: module.training for name, module in student.named_modules()}
    for name, param in student.named_parameters():
        old, flag, grad = before[name]
        assert param is old and param.requires_grad == flag and param.grad is grad


def test_combined_v1_call_fails_before_embedding_or_teacher_work():
    model = _bare(Qwen3QModel, _tiny_qwen3())
    model.quantize_config = QuantizeConfig(bits=4, group_size=16, format=FORMAT.GPTQ, offload_to_disk=False)
    model._requantize_embeddings = lambda **kwargs: pytest.fail("must reject before embedding packing")
    model._requantize_norm = lambda **kwargs: pytest.fail("must reject before teacher loading")
    with pytest.raises(ValueError, match="Requantize embeddings separately"):
        model.requantize(calibration=["sample"], embed_quant_mode=QuantizeEmbed.OUTPUT,
            norm=NormConfig(teacher="unused", steps=1))


def test_packed_rmsnorm_tuning_save_reload_preserves_quantized_tensors(tmp_path, monkeypatch):
    import gptqmodel.nn_modules.qlinear.torch as torch_linear_module

    monkeypatch.setattr(torch_linear_module, "torch_compile", lambda fn, **_kwargs: fn)

    class DenseLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.norm = nn.RMSNorm(16)
            self.proj = nn.Linear(16, 32, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            return SimpleNamespace(logits=self.proj(self.norm(self.embed(input_ids))))

    class PackedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.norm = nn.RMSNorm(16)
            self.proj = TorchLinear(
                bits=4,
                group_size=16,
                sym=False,
                desc_act=False,
                in_features=16,
                out_features=32,
                bias=False,
                format=FORMAT.GPTQ_V2,
            )

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            return SimpleNamespace(logits=self.proj(self.norm(self.embed(input_ids))))

    torch.manual_seed(4)
    teacher = DenseLM().eval()
    student = PackedLM().eval()
    student.embed.load_state_dict(teacher.embed.state_dict())
    student.norm.load_state_dict(teacher.norm.state_dict())
    scales = torch.full((1, 32), 0.1)
    zeros = torch.zeros((1, 32))
    codes = torch.clamp(torch.round(teacher.proj.weight.detach().T / scales), 0, 15)
    packed_linear = nn.Linear(16, 32, bias=False)
    packed_linear.weight.data.copy_((codes * scales).T)
    pack_module(
        "proj",
        {"proj": student.proj},
        scales.T.contiguous(),
        zeros.T.contiguous(),
        student.proj.g_idx,
        {"proj": packed_linear},
        TorchLinear,
        None,
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=16,
            sym=False,
            format=FORMAT.GPTQ_V2,
            offload_to_disk=False,
        ),
    )
    student.norm.weight.data.mul_(1.2)
    packed_before = {name: value.detach().clone() for name, value in student.proj.state_dict().items()}
    batches = [{"input_ids": torch.randint(0, 32, (2, 8))}]
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    source_file = source / "model.safetensors"
    save_file(student.state_dict(), str(source_file))
    turtle = SimpleNamespace(
        model_local_path=str(source),
        _weight_map=dict.fromkeys(student.state_dict(), "model.safetensors"),
    )

    tune_norm_params(
        student,
        teacher,
        batches,
        ["norm"],
        NormConfig(teacher=teacher, steps=2, lr=1e-2, reg=0.0),
    )

    _save_module_replacement_safetensors(
        student,
        turtle,
        ["norm"],
        save_dir=str(saved),
        metadata={"format": "pt"},
    )

    source_tensors = load_file(str(source_file))
    saved_tensors = load_file(str(saved / "model.safetensors"))
    assert set(saved_tensors) == set(source_tensors)
    assert saved_tensors["norm.weight"].dtype == source_tensors["norm.weight"].dtype
    assert not torch.equal(saved_tensors["norm.weight"], source_tensors["norm.weight"])
    for name, value in packed_before.items():
        assert torch.equal(saved_tensors[f"proj.{name}"], value)

    reloaded = PackedLM()
    reloaded.load_state_dict(saved_tensors)
    for name, value in saved_tensors.items():
        assert torch.equal(reloaded.state_dict()[name], value)
