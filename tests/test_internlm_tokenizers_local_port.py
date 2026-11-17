# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned release DATA, never checkpoint Python, through the public loader."""

import hashlib
import json
import shutil
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.models.definitions.internlm2 import InternLM2QModel
from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
from gptqmodel.models.ports.internlm.tokenization_internlm import InternLMTokenizer
from gptqmodel.models.ports.internlm2 import InternLM2Config, InternLM2ForCausalLM
from gptqmodel.models.ports.internlm2.tokenization_internlm2 import InternLM2Tokenizer
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config
from gptqmodel.utils.security import BYPASS_ENV


RELEASES = [
    ("internlm", "internlm-chat-7b", "4e2d2e185058a4cbaa2377ba1bb993eb599ce492"),
    ("internlm2", "internlm2_5-7b-chat", "eb72b541689b0432dade0435feb339b89fdd39ff"),
    ("internlm2", "internlm2_5-1_8b-chat", "57dcbc5455bf050d27a80f0ebd7808387caa714d"),
    ("internlm2", "internlm2-chat-7b", "c2ba64483dc50b3f8eb2d8271c4b9877a79ed2e2"),
]
HASHES = {
    "internlm": {
        "tokenizer.model": "aab622d98c98677a1a51f969e25765154487bf3e85c7819db105db2fcacba83f",
        "tokenizer_config.json": "f447aaf3b5e4868df7eef513b7a638c5684cfabf2922b6eda7c9cbd01805638f",
        "special_tokens_map.json": "833c7a70faf348f64bc7e62aa070b575205b5acc47201e109ec37a3516e0eba8",
    },
    "internlm2": {
        "tokenizer.model": "f868398fc4e05ee1e8aeba95ddf18ddcc45b8bce55d5093bead5bbf80429b48b",
        "tokenizer_config.json": "3ad60a8ee8a0a63a1f1d51884787d557eb6a312e5689148b4d599639ac42801b",
        "special_tokens_map.json": "e636e880c43a263dc00724686c9ad90e2d23bd1467fe9ed1b95740cd9f387202",
    },
}
CLASSES = {
    "internlm": (InternLMConfig, InternLMForCausalLM, InternLMTokenizer, InternLMQModel),
    "internlm2": (InternLM2Config, InternLM2ForCausalLM, InternLM2Tokenizer, InternLM2QModel),
}


@lru_cache
def release_assets(family, repo, revision):
    files = {}
    for name, digest in HASHES[family].items():
        path = Path(hf_hub_download("internlm/" + repo, name, revision=revision))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        files[name] = path
    return files


def copy_release_assets(path, release=RELEASES[1]):
    path.mkdir(parents=True, exist_ok=True)
    for name, source in release_assets(*release).items():
        shutil.copyfile(source, path / name)


def poison(path):
    marker = path / "CHECKPOINT_PYTHON_EXECUTED"
    for name, mapping in [
        ("config.json", {"AutoConfig": "trap.EvilConfig", "AutoModelForCausalLM": "trap.EvilModel"}),
        ("tokenizer_config.json", {"AutoTokenizer": ["trap.EvilTokenizer", "trap.EvilTokenizer"]}),
    ]:
        payload = json.loads((path / name).read_text())
        payload["auto_map"] = mapping
        (path / name).write_text(json.dumps(payload))
    (path / "trap.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python ran')\n",
    )
    return marker


def tiny_model(family, vocab_size=103168):
    config, model, _, _ = CLASSES[family]
    return model(config(vocab_size=vocab_size, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        max_position_embeddings=128, attn_implementation="eager")).eval()


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=RELEASES, ids=lambda item: item[1])
def release(request, tmp_path):
    copy_release_assets(tmp_path, request.param)
    return request.param[0], tmp_path


@pytest.mark.parametrize("text,expected", [
    ("Hello world!", [1, 9843, 2028, 346]),
    ("def f(x):\n    return x + 1", [1, 882, 413, 2226, 1122, 388, 821, 991, 619, 262, 312]),
    ("<s>hi</s>", [1, 1, 6176, 2]),
])
def test_release_golden_ids(release, text, expected):
    family, path = release
    tokenizer = CLASSES[family][2].from_pretrained(path, local_files_only=True)
    assert tokenizer.encode(text) == expected
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text.replace("<s>", "").replace("</s>", "")
    assert tokenizer.decode(expected) == (" <s><s> hi</s>" if text == "<s>hi</s>" else " <s>" + text)


def test_release_distinct_vocab_chat_and_specials(release):
    family, path = release
    tokenizer = CLASSES[family][2].from_pretrained(path, local_files_only=True)
    expected = ([1, 387, 20858, 12772, 364, 68242, 68316, 98739] if family == "internlm" else
                [1, 387, 20858, 12772, 364, 69093, 69167, 243, 162, 156, 133])
    text = "  leading spaces\n中文测试🙂"
    assert tokenizer.encode(text) == expected
    assert tokenizer.decode(expected, skip_special_tokens=True) == text
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id, tokenizer.pad_token_id) == (1, 2, 0, 2)
    if family == "internlm":
        prompt = "<|User|>:你好<eoh>\n<|Bot|>:"
        assert tokenizer.encode(prompt) == [1, 333, 352, 1621, 352, 27232, 76379, 103027, 364, 333, 352, 23845, 352, 27232]
        assert tokenizer.chat_template is None  # Release has no HF chat template.
    else:
        messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "你好!"}]
        prompt = "<s><|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n你好!<|im_end|>\n<|im_start|>assistant\n"
        assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == prompt
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        assert ids == [1, 92543, 9081, 364, 2770, 657, 11100, 281, 92542, 364, 92543, 1008, 364, 77230, 346, 92542, 364, 92543, 525, 11353, 364]
        assert tokenizer.encode("<|plugin|><|interpreter|><|action_end|><|action_start|><|im_end|><|im_start|>", add_special_tokens=False) == list(range(92538, 92544))


@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_assets_public_tokenizer_security_and_reload(release, tmp_path, monkeypatch, trust, bypass):
    family, path = release
    config_cls, _, tokenizer_cls, _ = CLASSES[family]
    config_cls(hidden_size=8, num_attention_heads=2, num_hidden_layers=1, intermediate_size=16).save_pretrained(path)
    marker = poison(path)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    config = load_model_config(path, trust_remote_code=trust)
    tokenizer = load_hf_tokenizer(path, model_config=config, trust_remote_code=trust,
                                  code_revision="never-import", local_files_only=True, padding_side="left")
    assert type(tokenizer) is tokenizer_cls
    assert tokenizer.encode("Hello world!") == [1, 9843, 2028, 346]
    assert tokenizer.pad_token_id == 0 and tokenizer.padding_side == "left"  # Tokenicer policy.
    saved = tmp_path / "saved"
    tokenizer.save_pretrained(saved)
    config.save_pretrained(saved)
    assert not list(saved.glob("*.py"))
    assert hashlib.sha256((saved / "tokenizer.model").read_bytes()).hexdigest() == HASHES[family]["tokenizer.model"]
    restored = load_hf_tokenizer(saved, model_config=config, trust_remote_code=trust, local_files_only=True)
    assert type(restored) is tokenizer_cls
    assert restored.encode("  leading spaces\n中文测试🙂") == tokenizer.encode("  leading spaces\n中文测试🙂")
    assert restored.chat_template == tokenizer.chat_template and restored.padding_side == "left"
    assert not marker.exists()


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_assets_full_public_dense_load_and_generate(tmp_path, monkeypatch, family, trust, bypass):
    release = RELEASES[0 if family == "internlm" else 1]
    original = tiny_model(family)
    original.save_pretrained(tmp_path)
    copy_release_assets(tmp_path, release)
    marker = poison(tmp_path)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    wrapper = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(wrapper) is CLASSES[family][3]
    assert type(wrapper.tokenizer.tokenizer) is CLASSES[family][2]
    batch = wrapper.tokenizer("Hello world!", return_tensors="pt")
    assert batch["input_ids"].tolist() == [[1, 9843, 2028, 346]]
    with torch.inference_mode():
        torch.testing.assert_close(wrapper.model(**batch).logits, original(**batch).logits)
        generated = wrapper.generate(**batch, max_new_tokens=1, do_sample=False)
    assert generated.shape == (1, 5) and torch.equal(generated[:, :4], batch["input_ids"])
    assert not marker.exists()


@pytest.mark.parametrize("family", CLASSES)
def test_old_binding_reproduces_release_failure_then_retry(tmp_path, monkeypatch, family):
    copy_release_assets(tmp_path, RELEASES[0 if family == "internlm" else 1])
    config = tiny_model(family, vocab_size=32).config
    config.save_pretrained(tmp_path)
    marker = poison(tmp_path)
    definition = CLASSES[family][3]
    original = definition.loader
    with monkeypatch.context() as broken:
        broken.setattr(definition, "loader", replace(original, tokenizer_class=None))
        with pytest.raises(ValueError, match="not available natively"):
            load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=False, local_files_only=True)
    assert type(load_hf_tokenizer(tmp_path, model_config=config, local_files_only=True)) is CLASSES[family][2]
    assert not marker.exists()


@pytest.mark.parametrize("family", CLASSES)
def test_missing_and_corrupt_asset_repairs_same_path(tmp_path, family):
    copy_release_assets(tmp_path, RELEASES[0 if family == "internlm" else 1])
    tokenizer_class = CLASSES[family][2]
    data = (tmp_path / "tokenizer.model").read_bytes()
    (tmp_path / "tokenizer.model").unlink()
    with pytest.raises((OSError, TypeError, ValueError)):
        tokenizer_class.from_pretrained(tmp_path, local_files_only=True)
    (tmp_path / "tokenizer.model").write_bytes(b"invalid sentencepiece data")
    with pytest.raises(RuntimeError, match="[Pp]arse"):
        tokenizer_class.from_pretrained(tmp_path, local_files_only=True)
    (tmp_path / "tokenizer.model").write_bytes(data)
    tokenizer = tokenizer_class.from_pretrained(tmp_path, local_files_only=True)
    assert tokenizer.encode("Hello world!") == [1, 9843, 2028, 346]
    (tmp_path / "tokenizer.model").unlink()
    rescued = tmp_path / "rescued"
    tokenizer.save_pretrained(rescued)
    assert (rescued / "tokenizer.model").read_bytes() == data
    assert tokenizer_class.from_pretrained(rescued, local_files_only=True).encode("Hello world!") == [1, 9843, 2028, 346]


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_asset_failure_repair_and_retry(tmp_path, monkeypatch, family, trust, bypass):
    copy_release_assets(tmp_path, RELEASES[0 if family == "internlm" else 1])
    config = tiny_model(family, vocab_size=32).config
    config.save_pretrained(tmp_path)
    marker = poison(tmp_path)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    data = (tmp_path / "tokenizer.model").read_bytes()
    (tmp_path / "tokenizer.model").unlink()
    with pytest.raises(ValueError, match="requires tokenizer.model"):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    (tmp_path / "tokenizer.model").write_bytes(b"not a SentencePiece model")
    with pytest.raises(RuntimeError, match="[Pp]arse"):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    (tmp_path / "tokenizer.model").write_bytes(data)
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=trust, local_files_only=True)
    assert type(tokenizer) is CLASSES[family][2]
    assert tokenizer.encode("Hello world!") == [1, 9843, 2028, 346]
    assert not marker.exists()


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("bos,eos", [(False, False), (True, False), (False, True), (True, True)])
def test_flags_pair_masks_and_save_reload(tmp_path, family, bos, eos):
    copy_release_assets(tmp_path, RELEASES[0 if family == "internlm" else 1])
    tokenizer_cls = CLASSES[family][2]
    tokenizer = tokenizer_cls.from_pretrained(tmp_path, add_bos_token=bos, add_eos_token=eos, local_files_only=True)
    actual = tokenizer("hello", "world", return_special_tokens_mask=True, return_token_type_ids=True)
    core = tokenizer.encode("hello", add_special_tokens=False) + tokenizer.encode("world", add_special_tokens=False)
    assert actual["input_ids"] == ([1] if bos else []) + core + ([2] if eos else [])
    assert actual["special_tokens_mask"] == ([1] if bos else []) + [0] * len(core) + ([1] if eos else [])
    assert actual["token_type_ids"] == [0] * len(actual["input_ids"])
    assert tokenizer.get_special_tokens_mask(actual["input_ids"], already_has_special_tokens=True) == actual["special_tokens_mask"]
    saved = tmp_path / "flags"
    tokenizer.save_pretrained(saved)
    restored = tokenizer_cls.from_pretrained(saved, local_files_only=True)
    assert restored("hello", "world", return_special_tokens_mask=True, return_token_type_ids=True) == actual
    prefixed = tokenizer.save_vocabulary(saved, filename_prefix="prefixed")
    assert Path(prefixed[0]).name == "prefixed-tokenizer.model" and Path(prefixed[0]).is_file()


@pytest.mark.parametrize("family", CLASSES)
def test_native_auto_cannot_load_release_without_checkpoint_code(tmp_path, family):
    copy_release_assets(tmp_path, RELEASES[0 if family == "internlm" else 1])
    with pytest.raises(ValueError, match="custom code|not exist|not.*imported"):
        AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False, local_files_only=True)
