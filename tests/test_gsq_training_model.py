import json

import pytest
import torch

from gptqmodel.quantization import GSQTrainingConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def fixture():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=2)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).eval(), [{'input_ids': [1, 2, 3, 4]}]


def test_model_installs_packed_prefix_before_next_capture(monkeypatch):
    from gptqmodel.looper import gsq_training_model as bridge

    model, documents = fixture()
    embedding = model.model.embed_tokens.weight.detach().clone()
    capture = bridge.capture_llama_gsq_inputs
    observed = []

    def check_prefix(model, documents, *, layer_index):
        if layer_index:
            assert isinstance(model.model.layers[layer_index-1].self_attn.q_proj, TorchLinear)
        observed.append(layer_index)
        return capture(model, documents, layer_index=layer_index)
    monkeypatch.setattr(bridge, 'capture_llama_gsq_inputs', check_prefix)
    result = bridge.quantize_llama_gsq_model(model, documents, bits=2, group_size=32,
                                            gsq=GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2))
    assert result['state'] == 'complete' and observed == [0, 1]
    assert all(isinstance(layer.mlp.down_proj, TorchLinear) for layer in model.model.layers)
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]]), use_cache=False).logits).all()
    assert json.loads(json.dumps(result))['blocks'][1]['layer_index'] == 1


def test_model_failure_retains_completed_blocks_and_records(monkeypatch):
    from gptqmodel.looper import gsq_training_model as bridge

    model, documents = fixture()
    quantize = bridge.quantize_llama_gsq_capture
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('injected training failure')
        return quantize(*args, **kwargs)
    monkeypatch.setattr(bridge, 'quantize_llama_gsq_capture', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        bridge.quantize_llama_gsq_model(model, documents, bits=2, group_size=32)
    assert isinstance(model.model.layers[0].self_attn.q_proj, TorchLinear)
    assert type(model.model.layers[1].self_attn.q_proj) is torch.nn.Linear
    record = model.gsq_training_runs[-1]
    assert record['state'] == 'failed' and record['current_layer'] == 1
    assert len(record['blocks']) == 1


def test_model_disk_capture_is_cleaned_after_each_block(tmp_path):
    from gptqmodel.looper import gsq_training_model as bridge

    model, documents = fixture()
    result = bridge.quantize_llama_gsq_model(
        model,
        documents,
        bits=2,
        group_size=32,
        gsq=GSQTrainingConfig(enabled=False),
        offload_capture=True,
        capture_directory=tmp_path/'capture',
    )
    assert result['state'] == 'complete'
    assert result['capture_storage'] == 'disk'
    assert (tmp_path/'capture').is_dir()
    assert not list((tmp_path/'capture').iterdir())


def test_model_rejects_invalid_selection_before_mutation():
    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model

    model, documents = fixture()
    with pytest.raises(ValueError, match='ordered'):
        quantize_llama_gsq_model(model, documents, bits=2, group_size=32, layer_indices=[1, 0])
    assert not hasattr(model, 'gsq_training_runs')


@pytest.mark.parametrize('initializer', ['gptq', 'gptq_signed'])
def test_staged_model_uses_public_checkpoint_writer_and_loader(tmp_path, initializer):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model, save_llama_gsq_model

    model, documents = fixture()
    model.save_pretrained(tmp_path/'dense')
    result = quantize_llama_gsq_model(model, documents, bits=2, group_size=32,
                                      gsq=GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2,
                                                            initializer=initializer))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(128)}, unk_token='0')), unk_token='0', pad_token='0')
    inv_freq = model.model.rotary_emb.inv_freq.clone()
    save_llama_gsq_model(model, result, tmp_path/'checkpoint', tokenizer=tokenizer, source_model=tmp_path/'dense')
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, inv_freq, rtol=0, atol=0)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    tokens = torch.arange(1, 65)[None]
    with torch.no_grad():
        before = model(tokens, use_cache=False).logits
    restored = GPTQModel.load(str(tmp_path/'checkpoint'), backend=BACKEND.TORCH,
                              device='cpu', dtype=torch.float16, attn_implementation='eager')
    assert restored.quantize_config.meta['gsq_training']['enabled']
    assert restored.quantize_config.meta['gsq_training']['initializer'] == initializer
    with torch.no_grad():
        after = restored.model(tokens, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_staged_finalization_preserves_existing_wrapper(tmp_path):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig, FORMAT
    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model, finalize_llama_gsq_wrapper

    model, documents = fixture()
    wrapper = LlamaQModel(model=model, quantized=False,
                          quantize_config=GPTQConfig(bits=4, group_size=32),
                          model_local_path=str(tmp_path))
    run = quantize_llama_gsq_model(model, documents, bits=4, group_size=32)
    assert finalize_llama_gsq_wrapper(wrapper, run) is wrapper
    assert wrapper.model is model and wrapper.model_local_path == str(tmp_path)
    assert wrapper.quantized and wrapper.qlinear_kernel is TorchLinear
    assert wrapper.quantize_config.bits == 4 and wrapper.quantize_config.format == FORMAT.GPTQ_V2
    assert wrapper.quantize_config.meta['gsq_training']['enabled'] is False
    assert wrapper.gsq_training_run is run
    assert torch.isfinite(wrapper.model(torch.tensor([[1, 2, 3]]), use_cache=False).logits).all()


def test_staged_prepared_documents_preserve_order_and_remove_only_padding():
    from gptqmodel.looper.gsq_training_model import staged_documents_from_prepared

    batches = [{'input_ids': torch.tensor([[0, 1, 2], [3, 4, 0]]),
                'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 0]])},
               {'input_ids': torch.tensor([[5, 6]])}]
    assert staged_documents_from_prepared(batches) == [
        {'input_ids': [1, 2]}, {'input_ids': [3, 4]}, {'input_ids': [5, 6]}]
    batches[0]['attention_mask'][0] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match='contiguous'):
        staged_documents_from_prepared(batches)
    with pytest.raises(ValueError, match='metadata'):
        staged_documents_from_prepared([{'input_ids': torch.tensor([[1]]), 'weights': [2.]}])


@pytest.mark.parametrize('enabled', [False, True])
def test_prepared_staged_lifecycle_installs_packed_model(tmp_path, enabled):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig
    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_prepared

    model, _ = fixture()
    wrapper = LlamaQModel(model=model, quantized=False,
                          quantize_config=GPTQConfig(bits=4, group_size=32),
                          model_local_path=str(tmp_path))
    prepared = [{'input_ids': torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
                 'attention_mask': torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])}]
    run = quantize_llama_gsq_prepared(wrapper, prepared,
                                     gsq=GSQTrainingConfig(enabled=enabled, epochs=1, qk_steps=2))
    assert run['state'] == 'complete' and wrapper.gsq_training_run is run
    assert wrapper.quantized and wrapper.model is model
    assert wrapper.quantize_config.meta['gsq_training']['enabled'] is enabled
    assert all(isinstance(layer.mlp.down_proj, TorchLinear) for layer in model.model.layers)
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]]), use_cache=False).logits).all()


@pytest.mark.parametrize('optimizer', ['lion', 'adamw'])
def test_public_staged_quantize_config_and_dispatch(tmp_path, optimizer):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig, FORMAT
    from gptqmodel.utils.backend import BACKEND

    config = GPTQConfig(bits=4, group_size=32, format=FORMAT.GPTQ_V2, act_group_aware=False,
                        device='cpu', offload_to_disk=False,
                        gsq_training=dict(enabled=True, epochs=1, qk_steps=2, optimizer=optimizer))
    config.meta['calibration_provenance'] = {'seed': 7, 'fixture': 'public-lifecycle'}
    restored = GPTQConfig.from_quant_config(config.to_dict())
    assert restored.gsq_training.enabled and restored.gsq_training.epochs == 1
    model, documents = fixture()
    model.save_pretrained(tmp_path/'dense')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(128)}, unk_token='0')), unk_token='0', pad_token='0')
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=restored,
                          tokenizer=tokenizer, model_local_path=str(tmp_path/'dense'))
    result = wrapper.quantize(documents, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert len(result['gsq_training']) == 2
    assert wrapper.quantized and wrapper.model is model
    assert wrapper.quantize_config.gsq_training.enabled
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]]), use_cache=False).logits).all()

    from gptqmodel import GPTQModel

    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(tokens, use_cache=False).logits
    wrapper.save(str(tmp_path/'quantized'))
    loaded = GPTQModel.load(str(tmp_path/'quantized'), backend=BACKEND.TORCH,
                            device='cpu', dtype=torch.float16, attn_implementation='eager')
    assert loaded.quantize_config.gsq_training.to_dict() == restored.gsq_training.to_dict()
    assert loaded.quantize_config.meta['calibration_provenance'] == config.meta['calibration_provenance']
    assert loaded.quantize_config.meta['gsq_requested_quantization']['gsq_training']['enabled']
    with torch.no_grad():
        after = loaded.model(tokens, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_public_disabled_staged_config_uses_ordinary_dispatch(tmp_path, monkeypatch):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.looper import gsq_training_model

    model, documents = fixture()
    config = GPTQConfig(bits=4, group_size=32, device='cpu', offload_to_disk=False,
                        gsq_training=dict(enabled=False))
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=config,
                          model_local_path=str(tmp_path))
    sentinel = {'ordinary': []}
    monkeypatch.setattr(wrapper, '_quantize_with_calibration', lambda **kwargs: sentinel)

    def unexpected(*args, **kwargs):
        pytest.fail('Disabled staged GSQ entered staged dispatch')
    monkeypatch.setattr(gsq_training_model, 'quantize_llama_gsq_public', unexpected)
    assert wrapper.quantize(documents, backend=BACKEND.TORCH) is sentinel
    assert not hasattr(model, 'gsq_training_runs')


def test_public_staged_rejection_precedes_weight_mutation(tmp_path):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig, FORMAT
    from gptqmodel.utils.backend import BACKEND

    model, documents = fixture()
    config = GPTQConfig(bits=4, group_size=32, format=FORMAT.GPTQ_V2, act_group_aware=False,
                        device='cpu', gsq_training=dict(enabled=True, epochs=1, qk_steps=2))
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=config,
                          model_local_path=str(tmp_path))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='additional public'):
        wrapper.quantize(documents, backend=BACKEND.TORCH, validation_calibration=documents)
    assert not wrapper.quantized and not hasattr(model, 'gsq_training_runs')
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('option,value', [('rotation', 'hadamard'), ('offload_to_disk', True),
                                         ('pack_dtype', torch.int16)])
def test_public_staged_rejects_unimplemented_runtime_options(tmp_path, option, value):
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import GPTQConfig, FORMAT
    from gptqmodel.utils.backend import BACKEND

    model, documents = fixture()
    kwargs = dict(bits=4, group_size=32, format=FORMAT.GPTQ_V2, act_group_aware=False,
                  device='cpu', offload_to_disk=False, gsq_training=dict(enabled=True))
    kwargs[option] = value
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=GPTQConfig(**kwargs),
                          model_local_path=str(tmp_path))
    with pytest.raises(ValueError, match='Public staged GSQ'):
        wrapper.quantize(documents, backend=BACKEND.TORCH)
    assert not hasattr(model, 'gsq_training_runs')
