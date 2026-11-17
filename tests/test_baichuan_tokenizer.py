# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real pinned Baichuan SentencePiece assets; never synthetic native replacements."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AddedToken, AutoConfig, AutoTokenizer

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.baichuan import BaiChuanQModel
from gptqmodel.models.ports.baichuan.configuration_baichuan import BaichuanConfig
from gptqmodel.models.ports.baichuan.modeling_baichuan import BaichuanForCausalLM
from gptqmodel.utils.hf import load_hf_tokenizer
from gptqmodel.utils.security import BYPASS_ENV


CONTRACTS = json.loads((Path(__file__).parents[1] / 'gptqmodel/models/ports/baichuan/tokenizer_contracts.json').read_text())['cases']


@pytest.fixture(scope='module', params=CONTRACTS, ids=lambda c: c['repo_id'].split('/')[-1])
def real_assets(request, tmp_path_factory):
    case = request.param
    path = tmp_path_factory.mktemp('baichuan-real-tokenizer')
    # Data only: no checkpoint Python is downloaded or imported by this suite.
    for filename in ('tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json', 'generation_config.json'):
        source = Path(hf_hub_download(case['repo_id'], filename, revision=case['revision']))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case['assets'][filename]
        shutil.copyfile(source, path / filename)
    return path, case


def _tiny_checkpoint(path, case, hidden_size=8):
    config = BaichuanConfig(
        vocab_size=case['vocab_size'], hidden_size=hidden_size, intermediate_size=hidden_size * 2,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=128,
        position_embedding_type='alibi' if '13B' in case['repo_id'] else 'rope',
        normalize_lm_head='Baichuan2' in case['repo_id'],
    )
    torch.manual_seed(1807)
    model = BaichuanForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def test_public_load_real_tokenizer(real_assets, tmp_path):
    assets, case = real_assets
    shutil.copytree(assets, tmp_path, dirs_exist_ok=True)
    reference = _tiny_checkpoint(tmp_path, case)
    loaded = GPTQModel.load(str(tmp_path), device='cpu', dtype=torch.float32, trust_remote_code=False)
    sample = case['samples'][0]
    ids = loaded.tokenizer(sample['text'], return_tensors='pt').input_ids
    assert ids.tolist() == [sample['ids']]
    assert loaded.tokenizer.decode(ids[0]) == sample['decoded']
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=0, atol=0)


def test_real_source_oracle_ids_and_special_tokens(real_assets):
    assets, case = real_assets
    tokenizer = load_hf_tokenizer(assets, model_config=BaichuanConfig.from_pretrained(assets), local_files_only=True)
    for sample in case['samples']:
        assert tokenizer.encode(sample['text']) == sample['ids']
        assert tokenizer.decode(sample['ids']) == sample['decoded']
    for key, value in case['special_tokens'].items():
        # Runtime padding repair belongs to Tokenicer; the base-7B source has
        # no pad token and GPT-QModel's config explicitly supplies unk/pad ID0.
        if key != 'pad_token_id':
            assert getattr(tokenizer, key) == value
    assert tokenizer.pad_token_id == 0
    assert tokenizer.chat_template is None
    for index, fields in case['added_special_flags'].items():
        token = tokenizer.added_tokens_decoder[int(index)]
        for key, value in fields.items():
            assert getattr(token, key) == value


def test_direct_constructor_keeps_native_added_token_flags(real_assets):
    from gptqmodel.models.ports.baichuan.tokenization_baichuan import BaichuanTokenizer

    assets, case = real_assets
    # This constructor accepts native objects only; legacy JSON conversion
    # belongs to Tokenicer. The port must retain the original HF4 semantics.
    flags = case['added_special_flags']
    tokenizer = BaichuanTokenizer(
        vocab_file=str(assets / 'tokenizer.model'), add_bos_token=False,
        bos_token=AddedToken(**flags['1']), eos_token=AddedToken(**flags['2']),
        unk_token=AddedToken(**flags['0']), pad_token=AddedToken(**flags['0']),
    )
    for sample in case['samples']:
        assert tokenizer.encode(sample['text']) == sample['ids']
    for index, fields in flags.items():
        for key, value in fields.items():
            assert getattr(tokenizer.added_tokens_decoder[int(index)], key) == value


def _forbid_discovery(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError('Package tokenizer re-entered checkpoint Auto discovery')
    monkeypatch.setattr(AutoConfig, 'from_pretrained', reject)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', reject)


def _hostile(path):
    marker = path / 'CHECKPOINT_PYTHON_EXECUTED'
    (path / 'checkpoint_python.py').write_text(
        'from pathlib import Path\n'
        f'Path({str(marker)!r}).touch()\n'
        'raise AssertionError("checkpoint Python executed")\n'
    )
    mapping = {'AutoConfig': 'checkpoint_python.RemoteConfig',
               'AutoModelForCausalLM': 'checkpoint_python.RemoteModel',
               'AutoTokenizer': ['checkpoint_python.RemoteTokenizer', None]}
    for filename in ('config.json', 'tokenizer_config.json'):
        target = path / filename
        payload = json.loads(target.read_text())
        payload.update(auto_map=mapping, tokenizer_class='RemoteTokenizer')
        target.write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize('trust', [None, False, True])
@pytest.mark.parametrize('bypass', [None, '0', '1'])
def test_public_real_hostile_trust_combinations(real_assets, tmp_path, monkeypatch, trust, bypass):
    from gptqmodel.models.ports.baichuan.tokenization_baichuan import BaichuanTokenizer

    assets, case = real_assets
    shutil.copytree(assets, tmp_path, dirs_exist_ok=True)
    reference = _tiny_checkpoint(tmp_path, case)
    marker = _hostile(tmp_path)
    if bypass is not None:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    _forbid_discovery(monkeypatch)
    loaded = GPTQModel.load(str(tmp_path), device='cpu', dtype=torch.float32,
                            trust_remote_code=trust, code_revision='untrusted')
    assert isinstance(loaded.tokenizer.tokenizer, BaichuanTokenizer)
    sample = case['samples'][0]
    ids = loaded.tokenizer(sample['text'], return_tensors='pt').input_ids
    assert ids.tolist() == [sample['ids']]
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=0, atol=0)
        expected = reference.generate(ids, do_sample=False, max_new_tokens=2)
        assert torch.equal(loaded.generate(ids, do_sample=False, max_new_tokens=2), expected)
    assert not marker.exists()


def test_real_pair_bos_eos_masks_and_overrides(real_assets, tmp_path):
    from gptqmodel.models.ports.baichuan.tokenization_baichuan import BaiChuanTokenizer, BaichuanTokenizer

    assert BaiChuanTokenizer is BaichuanTokenizer
    assets, case = real_assets
    for pair in case['pairs']:
        tokenizer = load_hf_tokenizer(
            assets, model_config=BaichuanConfig.from_pretrained(assets),
            add_bos_token=pair['add_bos_token'], add_eos_token=pair['add_eos_token'],
            model_max_length=81, padding_side='left', sp_model_kwargs={'enable_sampling': False},
            local_files_only=True,
        )
        actual = tokenizer('你好', text_pair='Hello', return_special_tokens_mask=True, return_token_type_ids=True)
        assert dict(actual) == pair['encoded']
        assert tokenizer.model_max_length == 81 and tokenizer.padding_side == 'left'
        assert tokenizer.sp_model_kwargs == {'enable_sampling': False}
        saved = tmp_path / f"{pair['add_bos_token']}-{pair['add_eos_token']}"
        tokenizer.save_pretrained(saved)
        restored = load_hf_tokenizer(saved, model_config=BaichuanConfig.from_pretrained(assets), local_files_only=True)
        assert dict(restored('你好', text_pair='Hello', return_special_tokens_mask=True, return_token_type_ids=True)) == pair['encoded']
        assert restored.model_max_length == 81 and restored.padding_side == 'left'


def test_real_live_flags_and_missing_source_vocabulary_save(real_assets, tmp_path):
    assets, case = real_assets
    source = tmp_path / 'source'
    shutil.copytree(assets, source)
    tokenizer = load_hf_tokenizer(source, model_config=BaichuanConfig.from_pretrained(source), local_files_only=True)
    # Persist the live state after initialization, not stale init kwargs.
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    (source / 'tokenizer.model').unlink()
    saved = tmp_path / 'saved'
    tokenizer.save_pretrained(saved)
    assert hashlib.sha256((saved / 'tokenizer.model').read_bytes()).hexdigest() == case['assets']['tokenizer.model']
    restored = load_hf_tokenizer(saved, model_config=BaichuanConfig.from_pretrained(assets), local_files_only=True)
    assert restored.encode(case['samples'][0]['text']) == [1] + case['samples'][0]['ids'] + [2]
    prefixed = tmp_path / 'prefixed'
    tokenizer.save_pretrained(prefixed, filename_prefix='audit')
    flags = json.loads((prefixed / 'audit-tokenizer_config.json').read_text())
    assert flags['add_bos_token'] is True and flags['add_eos_token'] is True


@pytest.mark.parametrize('padding_side', ['left', 'right'])
def test_real_batch_padding_and_attention_masks(real_assets, padding_side):
    assets, case = real_assets
    tokenizer = load_hf_tokenizer(assets, model_config=BaichuanConfig.from_pretrained(assets),
                                  padding_side=padding_side, local_files_only=True)
    samples = case['samples'][:2]
    batch = tokenizer([sample['text'] for sample in samples], padding=True, return_tensors='pt')
    for row, sample in enumerate(samples):
        mask = batch.attention_mask[row].bool()
        assert batch.input_ids[row][mask].tolist() == sample['ids']
        assert torch.all(batch.input_ids[row][~mask] == 0)
        assert batch.attention_mask[row].sum().item() == len(sample['ids'])


def test_source_chat_assembly_and_explicit_template(real_assets):
    assets, case = real_assets
    tokenizer = load_hf_tokenizer(assets, model_config=BaichuanConfig.from_pretrained(assets), local_files_only=True)
    assert tokenizer.chat_template is None
    if 'chat' not in case:
        assert 'user_token_id' not in json.loads((assets / 'generation_config.json').read_text())
        return  # Base 7B has no source chat format; do not invent one.
    chat = case['chat']
    config = json.loads((assets / 'generation_config.json').read_text())
    assert config['user_token_id'] == chat['user_token_id']
    assert config['assistant_token_id'] == chat['assistant_token_id']
    assert tokenizer.convert_ids_to_tokens(chat['user_token_id']) == chat['user_token']
    assert tokenizer.convert_ids_to_tokens(chat['assistant_token_id']) == chat['assistant_token']
    ids = []
    for message in chat['messages']:
        if message['role'] != 'system':
            ids.append(chat['user_token_id'] if message['role'] == 'user' else chat['assistant_token_id'])
        ids.extend(tokenizer.encode(message['content']))
    ids.append(chat['assistant_token_id'])
    assert ids == chat['ids']  # Independently captured source/HF4 oracle.
    # An explicit caller-supplied template is honored, never silently installed.
    # Delimiters are real SentencePiece vocabulary entries rather than strings
    # guessed from a similarly shaped model's chat convention.
    template = (
        '{% for message in messages %}'
        "{% if message['role'] == 'user' %}" + chat['user_token'] +
        "{% elif message['role'] == 'assistant' %}" + chat['assistant_token'] +
        "{% endif %}{{ message['content'] }}{% endfor %}"
        '{% if add_generation_prompt %}' + chat['assistant_token'] + '{% endif %}'
    )
    # Role delimiters must be explicitly registered as special tokens by the
    # caller for rendering+tokenization; upstream build_chat_input inserts IDs.
    tokenizer.add_special_tokens({'additional_special_tokens': [
        AddedToken(chat['user_token'], normalized=False),
        AddedToken(chat['assistant_token'], normalized=False),
    ]})
    encoded = tokenizer.apply_chat_template(chat['messages'], chat_template=template,
                                            add_generation_prompt=True, tokenize=True, return_dict=False)
    assert encoded == chat['ids']
    assert tokenizer.chat_template is None


@pytest.mark.parametrize('broken', ['missing_model', 'malformed_model', 'malformed_config'])
def test_broken_real_asset_public_repair_retry(real_assets, tmp_path, monkeypatch, broken):
    assets, case = real_assets
    shutil.copytree(assets, tmp_path, dirs_exist_ok=True)
    _tiny_checkpoint(tmp_path, case)
    marker = _hostile(tmp_path)
    _forbid_discovery(monkeypatch)
    monkeypatch.setenv(BYPASS_ENV, '1')
    target = tmp_path / ('tokenizer_config.json' if broken == 'malformed_config' else 'tokenizer.model')
    original = target.read_bytes()
    if broken == 'missing_model':
        target.unlink()
    else:
        target.write_bytes(b'invalid asset {')
    with pytest.raises((ValueError, OSError, RuntimeError, TypeError)):
        GPTQModel.load(str(tmp_path), device='cpu', dtype=torch.float32, trust_remote_code=True)
    assert not marker.exists()
    target.write_bytes(original)
    loaded = GPTQModel.load(str(tmp_path), device='cpu', dtype=torch.float32, trust_remote_code=True)
    sample = case['samples'][0]
    assert loaded.tokenizer(sample['text']).input_ids == sample['ids']
    assert loaded.tokenizer.decode(sample['ids']) == sample['decoded']
    assert not marker.exists()


def test_public_real_save_reload_resave(real_assets, tmp_path, monkeypatch):
    assets, case = real_assets
    source = tmp_path / 'source'
    shutil.copytree(assets, source)
    _tiny_checkpoint(source, case)
    marker = _hostile(source)
    monkeypatch.setenv(BYPASS_ENV, '1')
    _forbid_discovery(monkeypatch)
    model = GPTQModel.load(str(source), device='cpu', dtype=torch.float32, trust_remote_code=True)
    ids = torch.tensor([case['samples'][0]['ids']])
    with torch.inference_mode():
        expected = model.model(ids).logits
        expected_generated = model.generate(ids, do_sample=False, max_new_tokens=2)
    for name in ('saved', 'resaved'):
        target = tmp_path / name
        # Dense GPT-QModel.save is intentionally a quantized-only writer in
        # this repository. Exercise the actual native safetensors/tokenizer
        # writers and reload them through the public GPTQModel API.
        model.model.save_pretrained(target, safe_serialization=True)
        model.tokenizer.save_pretrained(target)
        assert (target / 'model.safetensors').is_file()
        assert not list(target.glob('*.bin'))
        assert not list(target.glob('*.py'))
        assert hashlib.sha256((target / 'tokenizer.model').read_bytes()).hexdigest() == case['assets']['tokenizer.model']
        model = GPTQModel.load(str(target), device='cpu', dtype=torch.float32, trust_remote_code=True)
        for sample in case['samples']:
            assert model.tokenizer(sample['text']).input_ids == sample['ids']
            assert model.tokenizer.decode(sample['ids']) == sample['decoded']
        with torch.inference_mode():
            torch.testing.assert_close(model.model(ids).logits, expected, rtol=0, atol=0)
            assert torch.equal(model.generate(ids, do_sample=False, max_new_tokens=2), expected_generated)
    assert not marker.exists()


def test_real_tokenizer_public_awq_pack_save_reload(real_assets, tmp_path, monkeypatch):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    assets, case = real_assets
    source = tmp_path / 'source'
    shutil.copytree(assets, source)
    _tiny_checkpoint(source, case, hidden_size=16)
    marker = _hostile(source)
    _forbid_discovery(monkeypatch)
    model = GPTQModel.load(
        str(source), quantize_config=AWQConfig(device='cpu', group_size=16, fallback=None, offload_to_disk=False),
        backend=BACKEND.TORCH, dtype=torch.float16, trust_remote_code=False,
    )
    expected = {f'model.layers.0.{name}' for name in (
        'self_attn.W_pack', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    )}
    dense = {key: value.clone() for key, value in model.model.state_dict().items()
             if key.removesuffix('.weight') not in expected}
    assert model.awq_get_modules_for_scaling(model.model.model.layers[0], {}, {}) == []
    # Public string calibration must pass through the real owned tokenizer.
    model.quantize([sample['text'] for sample in case['samples'][:2]], batch_size=1,
                   backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert {name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected
    for key, value in dense.items():
        torch.testing.assert_close(model.model.state_dict()[key], value, rtol=0, atol=0)
    # Source Baichuan1 permits narrowing its checkpoint-owned inv_freq through
    # Module.half(). Make the saved state explicitly match the requested
    # reload dtype; quantized-loading dtype conversion is a separate concern.
    model.model.half()
    ids = torch.tensor([case['samples'][0]['ids']])
    with torch.inference_mode():
        logits = model.model(ids).logits
        generated = model.generate(ids, max_new_tokens=2, do_sample=False)
    assert torch.isfinite(logits).all()
    for attempt in range(2):
        target = tmp_path / f'awq-{attempt}'
        model.save(target)
        assert list(target.glob('*.safetensors')) and not list(target.glob('*.bin'))
        assert not list(target.glob('*.py'))
        restored = GPTQModel.load(str(target), device='cpu', backend=BACKEND.TORCH, dtype=torch.float16)
        assert hashlib.sha256((target / 'tokenizer.model').read_bytes()).hexdigest() == case['assets']['tokenizer.model']
        for sample in case['samples']:
            assert restored.tokenizer.encode(sample['text']) == sample['ids']
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0, msg=key)
        with torch.inference_mode():
            torch.testing.assert_close(restored.model(ids).logits, logits, rtol=0, atol=0)
            assert torch.equal(restored.generate(ids, max_new_tokens=2, do_sample=False), generated)
        model = restored
    assert not marker.exists()


def test_definition_owns_lazy_tokenizer_binding():
    assert BaiChuanQModel.loader.tokenizer_class == 'gptqmodel.models.ports.baichuan.tokenization_baichuan.BaichuanTokenizer'
