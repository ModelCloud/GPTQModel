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
