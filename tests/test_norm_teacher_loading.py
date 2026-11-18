# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real norm-tuning consumers must select teacher code independently of the student."""

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from gptqmodel import CalibrationConfig, GPTQModel, NormConfig
from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
from gptqmodel.models.ports.native.gpt2.modeling_gpt2 import GPT2LMHeadModel as VendoredGPT2LMHeadModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.security import BYPASS_ENV, RemoteCodeSecurityError


CALIBRATION = [{"input_ids": [4, 7, 9, 12, 25, 6, 31, 11], "attention_mask": [1] * 8},
               {"input_ids": [6, 12, 5, 18, 29, 7, 13, 22], "attention_mask": [1] * 8}]


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def native_torch_runtime(monkeypatch):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    # This suite qualifies eager Torch autograd and packed storage, not compile.
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch.torch_compile", lambda fn, **kwargs: fn)


@pytest.fixture(scope="module")
def packed_student(tmp_path_factory):
    root = tmp_path_factory.mktemp("norm-teacher-student")
    source, packed = root / "source", root / "packed"
    torch.manual_seed(987)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         tie_word_embeddings=False)
    Qwen3ForCausalLM(config).half().eval().save_pretrained(source)
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4},
                                            unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                           eos_token="[EOS]", unk_token="[UNK]").save_pretrained(source)
    model = GPTQModel.load(str(source), quantize_config=QuantizeConfig(bits=4, group_size=16, desc_act=False,
                          device="cpu", format=FORMAT.GPTQ_V2, offload_to_disk=False, pack_impl="original"),
                          backend=BACKEND.TORCH, dtype=torch.float16)
    model.quantize(
        CalibrationConfig(dataset=CALIBRATION, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(packed))
    return packed


def student(path, *, trust=False, dtype=torch.float16):
    loaded = GPTQModel.load(str(path), backend=BACKEND.TORCH, device="cpu", dtype=dtype)
    # Exercise the existing consumer policy flag, not a new per-teacher switch.
    loaded.trust_remote_code = trust
    assert sum(isinstance(module, TorchLinear) for module in loaded.model.modules()) == 7
    return loaded


def dense_teacher(path, family):
    torch.manual_seed(230)
    if family == "owned":
        config = InternLMConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
                                rotary={"type": "origin", "base": 10000}, bias=True)
        model = InternLMForCausalLM(config)
    else:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=2,
                                          n_inner=64, n_positions=64, bos_token_id=1, eos_token_id=2))
    model.eval().save_pretrained(path)
    # The gpt2 definition loads the vendored Transformers 5.23 class.
    teacher_class = type(model) if family == "owned" else VendoredGPT2LMHeadModel
    config_path = path / "config.json"
    payload = json.loads(config_path.read_text())
    payload["auto_map"] = {"AutoConfig": "hostile.Forbidden", "AutoModelForCausalLM": "hostile.Forbidden"}
    config_path.write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
                                    "raise RuntimeError('checkpoint teacher Python executed')\n")
    return teacher_class, marker


def tune(model, teacher):
    return model.requantize(
        CalibrationConfig(dataset=CALIBRATION, min_length=1),
        norm=NormConfig(teacher=teacher, steps=2, lr=0.02, reg=0.0),
    )["norm"]


def state(model):
    return {name: value.detach().clone() for name, value in model.model.state_dict().items()}


@pytest.mark.parametrize("family", ["owned", "native"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_norm_teacher_uses_own_config_and_matches_supplied_dense_teacher(
    tmp_path, monkeypatch, packed_student, family, trust, bypass,
):
    if bypass is not None:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    teacher_class, marker = dense_teacher(tmp_path, family)
    oracle = teacher_class.from_pretrained(tmp_path, dtype=torch.float16)
    model = student(packed_student, trust=trust)
    reference = student(packed_student, trust=trust)
    before = state(model)
    observed = []
    original = teacher_class.forward

    def forward(self, *args, **kwargs):
        observed.append((type(self), type(self.config), self.dtype, self.get_input_embeddings().weight.device))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(teacher_class, "forward", forward)
    stats = tune(model, str(tmp_path))
    assert len(observed) == 2
    assert all(entry == (teacher_class, type(oracle.config), torch.float16, torch.device("cpu")) for entry in observed)
    expected = tune(reference, oracle)
    assert stats["modules"] == expected["modules"]
    assert stats["steps"] == 2 and stats["skipped_steps"] == 0
    changed = {name for name, value in state(model).items() if not torch.equal(value, before[name])}
    assert changed and changed.issubset({f"{name}.weight" for name in stats["modules"]})
    for name, value in state(reference).items():
        torch.testing.assert_close(state(model)[name], value, atol=0, rtol=0, msg=name)
    assert not marker.exists()


@pytest.mark.parametrize("family", ["owned", "native"])
def test_teacher_requested_bfloat16_device_and_supplied_module_control(tmp_path, monkeypatch, packed_student, family):
    teacher_class, marker = dense_teacher(tmp_path, family)
    model = student(packed_student, dtype=torch.bfloat16)
    observed = []
    original = teacher_class.forward

    def forward(self, *args, **kwargs):
        observed.append((self.dtype, self.get_input_embeddings().weight.device))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(teacher_class, "forward", forward)
    assert tune(model, str(tmp_path))["skipped_steps"] == 0
    assert observed == [(torch.bfloat16, torch.device("cpu"))] * 2
    teacher = teacher_class.from_pretrained(tmp_path, dtype=torch.float32)
    original_flags = {name: parameter.requires_grad for name, parameter in teacher.named_parameters()}
    teacher.train()
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    assert tune(model, teacher)["skipped_steps"] == 0
    assert observed[-2:] == [(torch.float32, torch.device("cpu"))] * 2
    assert teacher.training
    assert {name: parameter.requires_grad for name, parameter in teacher.named_parameters()} == original_flags
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert not marker.exists()


def test_teacher_missing_payload_failure_leaves_student_unchanged_and_retries(tmp_path, packed_student):
    _, marker = dense_teacher(tmp_path, "owned")
    model = student(packed_student)
    before = state(model)
    source = tmp_path / "model.safetensors"
    source.rename(tmp_path / "weights.unavailable")
    with pytest.raises(OSError):
        tune(model, str(tmp_path))
    for name, value in state(model).items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert not getattr(model, "_module_replacement_prefixes", set())
    (tmp_path / "weights.unavailable").rename(source)
    assert tune(model, str(tmp_path))["skipped_steps"] == 0
    assert not marker.exists()


def remote_teacher(path):
    dense_teacher(path, "native")
    marker = path / "REMOTE_EXECUTED"
    source = ("from pathlib import Path\nfrom transformers import GPT2Config, GPT2LMHeadModel\n"
              f"Path({str(marker)!r}).touch()\n"
              "class RemoteConfig(GPT2Config):\n    model_type = 'norm_teacher_unported'\n"
              "class RemoteModel(GPT2LMHeadModel):\n    config_class = RemoteConfig\n")
    (path / "remote.py").write_text(source)
    config = path / "config.json"
    payload = json.loads(config.read_text())
    payload.update(model_type="norm_teacher_unported", auto_map={"AutoConfig": "remote.RemoteConfig",
                   "AutoModelForCausalLM": "remote.RemoteModel"})
    config.write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_unknown_teacher_remote_code_requires_both_consents(tmp_path, monkeypatch, packed_student, trust, bypass):
    if bypass is not None:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    marker = remote_teacher(tmp_path)
    model = student(packed_student, trust=trust)
    before = state(model)
    if trust and bypass == "1":
        assert tune(model, str(tmp_path))["skipped_steps"] == 0
        assert marker.exists()
    else:
        error = RemoteCodeSecurityError if trust else ValueError
        with pytest.raises(error):
            tune(model, str(tmp_path))
        assert not marker.exists()
        for name, value in state(model).items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        assert not getattr(model, "_module_replacement_prefixes", set())


def test_teacher_permission_is_rechecked_after_failed_and_successful_attempts(tmp_path, monkeypatch, packed_student):
    marker = remote_teacher(tmp_path)
    model = student(packed_student, trust=True)
    with pytest.raises(RemoteCodeSecurityError):
        tune(model, str(tmp_path))
    assert not marker.exists()
    monkeypatch.setenv(BYPASS_ENV, "1")
    assert tune(model, str(tmp_path))["skipped_steps"] == 0
    marker.unlink()
    before = state(model)
    monkeypatch.delenv(BYPASS_ENV)
    with pytest.raises(RemoteCodeSecurityError):
        tune(model, str(tmp_path))
    assert not marker.exists()
    for name, value in state(model).items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
