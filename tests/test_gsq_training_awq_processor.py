import pytest
import torch

from gptqmodel.quantization import AWQConfig, GPTQConfig, QuantizeConfig, GSQTrainingConfig


def test_awq_staged_config_roundtrip_and_method_boundaries():
    config = AWQConfig(bits=4, group_size=32, gsq_training=dict(enabled=True, epochs=1, qk_steps=2))
    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert restored.gsq_training.initializer == 'awq'
    assert restored.gsq_training.to_dict() == config.gsq_training.to_dict()
    assert AWQConfig().gsq_training is None
    with pytest.raises(ValueError, match='GPTQ initializer'):
        GPTQConfig(gsq_training=GSQTrainingConfig(enabled=True, initializer='awq'))
    with pytest.raises(ValueError, match='AWQ initializer'):
        AWQConfig(gsq_training=GSQTrainingConfig(enabled=True))


@pytest.mark.parametrize('sym', [False, True])
def test_awq_staged_processor_replaces_weights_and_packing_metadata_together(sym):
    from test_adjacent_awq import _TestAWQProcessor
    from test_gsq_training_capture import fixture
    from gptqmodel.looper.gsq_training_awq import capture_awq_staged_teacher, refine_awq_staged_layer
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(7)
    layer, cache = fixture()
    config = AWQConfig(bits=4, group_size=32, sym=sym,
                        gsq_training=dict(enabled=True, epochs=1, qk_steps=2))
    processor = _TestAWQProcessor(config)
    processor.inputs_cache = cache
    modules = {name: NamedModule(module, name=name, full_name='model.layers.0.'+name, layer_index=0)
               for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
    teacher = capture_awq_staged_teacher(processor, layer, modules, set())
    with torch.no_grad():
        for named in modules.values():
            named.weight.clamp_(-.02, .02)
    processor.apply_quant(modules, scales_list=[])
    before = {name: named.weight.clone() for name, named in modules.items()}
    refine_awq_staged_layer(processor, teacher, modules, layer_index=0)
    assert any(not torch.equal(named.weight, before[name]) for name, named in modules.items())
    for named in modules.values():
        named.stream_sync()
        packed = AwqTorchLinear(bits=4, group_size=32, sym=sym, desc_act=False,
                                in_features=named.module.in_features, out_features=named.module.out_features,
                                bias=False)
        packed.pack(named.module, named.state['q_scales'], named.state['q_zeros'])
        packed.scales = packed.scales.float()
        torch.testing.assert_close(packed.dequantize_weight().T, named.state['wq'], atol=0, rtol=0)
    serialized = config.to_dict()
    assert serialized['meta']['gsq_training_runs'][0]['initializer'] == 'awq'


def test_public_awq_staged_quantize_dispatch(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    model_config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                               num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2)
    model_config._attn_implementation = 'eager'
    model = LlamaForCausalLM(model_config).half().eval()
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    model.model.rotary_emb = LlamaRotaryEmbedding(model_config)
    documents = [{'input_ids': list(range(1, 17))}]
    model.save_pretrained(tmp_path/'dense')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(128)}, unk_token='0')), unk_token='0', pad_token='0')
    config = AWQConfig(bits=4, group_size=32, sym=False, device=DEVICE.CPU, offload_to_disk=False,
                        gsq_training=dict(enabled=True, epochs=1, qk_steps=2))
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=config,
                          tokenizer=tokenizer, model_local_path=str(tmp_path/'dense'))
    wrapper.quantize(documents, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert wrapper.quantized
    assert len(wrapper.quantize_config.meta['gsq_training_runs']) == 2
    from gptqmodel import GPTQModel

    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(tokens, use_cache=False).logits
    assert torch.isfinite(before).all()
    wrapper.save(str(tmp_path/'quantized'))
    loaded = GPTQModel.load(str(tmp_path/'quantized'), backend=BACKEND.TORCH,
                            device='cpu', dtype=torch.float16, attn_implementation='eager')
    assert loaded.quantize_config.gsq_training.to_dict() == config.gsq_training.to_dict()
    assert len(loaded.quantize_config.meta['gsq_training_runs']) == 2
    with torch.no_grad():
        after = loaded.model(tokens, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
