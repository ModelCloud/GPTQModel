import pytest
import torch

from gptqmodel.looper.fp8_gsq_processor import FP8GSQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import FP8Config
from gptqmodel.models._const import DEVICE


def test_processor_capture_fit_handoff(monkeypatch):
    processor = object.__new__(FP8GSQProcessor)
    processor.qcfg = FP8Config(gsq={'enabled': True, 'steps': 3, 'candidates': 3, 'modules': ['proj$']})
    processor.tasks = {}
    import threading
    processor.lock = threading.Lock()
    processor.log = []
    monkeypatch.setattr(processor, 'quantize_module', lambda module, device=None: processor.qcfg)
    dense = torch.nn.Linear(8, 4, bias=False)
    named = NamedModule(dense, name='proj', full_name='model.layers.0.proj', layer_index=0)
    processor.preprocess(named)
    assert not processor.is_skipped(named)
    task = processor.tasks['proj']
    inputs = torch.randn(2, 10, 8, generator=torch.Generator().manual_seed(7))
    processor.pre_process_fwd_hook('proj')(dense, (inputs,), None)
    processor.process(named)
    result = named.state['gsq_fp8_result']
    assert result['diagnostics']['tokens'] == 20
    assert result['after'] <= result['before']
    expected = result['weight'].float()/result['scale_inv'][:, None]
    torch.testing.assert_close(named.weight, expected)
    assert not torch.equal(named.weight, named.state['fp8_reference_weight'])
    assert task.teacher is None
    assert not processor.tasks


def test_unmatched_module_still_gets_ordinary_fp8_processing(monkeypatch):
    processor = object.__new__(FP8GSQProcessor)
    processor.qcfg = FP8Config(gsq={'enabled': True, 'modules': ['other$']})
    processor.tasks = {}
    import threading
    processor.lock = threading.Lock()
    processor.log = []
    calls = []
    def record(module, device=None):
        calls.append(module.full_name)
        return processor.qcfg

    monkeypatch.setattr(processor, 'quantize_module', record)
    named = NamedModule(torch.nn.Linear(8, 4), name='proj', full_name='model.layers.0.proj', layer_index=0)
    processor.preprocess(named)
    assert not processor.is_skipped(named)
    processor.pre_process_fwd_hook('proj')(named.module, (torch.ones(2, 8),), None)
    processor.process(named)
    assert calls == [named.full_name]
    assert named.state['gsq_fp8_result']['diagnostics']['status'] == 'skipped'



def test_hook_uses_batch_source_weights_and_padding_mask():
    import threading

    processor = object.__new__(FP8GSQProcessor)
    processor.qcfg = FP8Config(gsq={'enabled': True})
    processor.tasks = {}
    processor.lock = threading.Lock()
    processor.log = []
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor._set_current_batch_index(0)
    processor.calibration_dataset = [{'fisher_sequence_weight': [1.25, 1.]}]
    mask = torch.tensor([[True, False, True], [False, True, True]])
    processor._mask_tls.value = mask
    named = NamedModule(torch.nn.Linear(8, 4), name='proj', full_name='model.layers.0.proj', layer_index=0)
    processor.preprocess(named)
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(7))
    processor.pre_process_fwd_hook('proj')(named.module, (x,), None)
    gram, stats = processor.tasks['proj'].capture.take()
    expected = 1.25*(x[0, mask[0]].T @ x[0, mask[0]]) + x[1, mask[1]].T @ x[1, mask[1]]
    torch.testing.assert_close(gram, expected)
    assert stats == {'tokens': 4, 'weighted_tokens': 4.5}
    processor.tasks['proj'].free()



def test_calibrated_fp8_config_selects_calibration_lifecycle_and_roundtrips():
    import pytest
    from gptqmodel.quantization.config import QuantizeConfig

    cfg = FP8Config(gsq={'enabled': True}, gsq_calibration=True)
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.gsq_calibration is True
    assert not restored.uses_weight_only_lifecycle()
    assert FP8Config(gsq={'enabled': True}).uses_weight_only_lifecycle()
    assert FP8Config().uses_weight_only_lifecycle()
    with pytest.raises(ValueError, match='requires enabled'):
        FP8Config(gsq_calibration=True)
    with pytest.raises(TypeError, match='boolean'):
        FP8Config(gsq_calibration='yes')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_shared_looper_hook_capture_replay_and_real_finalizer(dtype):
    from types import SimpleNamespace

    from gptqmodel.looper.module_looper import ModuleLooper
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    cfg = FP8Config(device=DEVICE.CPU, offload_to_disk=False, gsq_calibration=True,
                    gsq={'enabled': True, 'steps': 3, 'candidates': 3})
    batch = {'input_ids': torch.arange(6).reshape(2, 3),
             'attention_mask': torch.tensor([[1, 0, 1], [0, 1, 1]]),
             'fisher_sequence_weight': [1.25, 1.]}
    processor = FP8GSQProcessor(None, cfg, [batch], lambda **kwargs: [batch])
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(8, 4, bias=False, dtype=dtype)
    wrapper = SimpleNamespace(model=model, qlinear_kernel=TorchFP8Linear, lm_head='lm_head')
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    named = NamedModule(model.proj, name='proj', full_name='proj', layer_index=0)
    processor.preprocess(named)
    processor._set_current_batch_index(0)
    looper._set_processor_mask(processor, batch['attention_mask'].bool())
    hook = looper._masked_hook_wrapper(processor, processor.pre_process_fwd_hook('proj'), 'test')
    handle = model.proj.register_forward_hook(hook)
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(7)).to(dtype)
    try:
        model.proj(x)
    finally:
        handle.remove()
    active_config = processor.process(named, device=torch.device('cpu'))
    assert named.state['gsq_diagnostics']['tokens'] == 4
    from gptqmodel.models.writer import QUANT_LOG_NSAMPLES
    assert processor.log[-1][QUANT_LOG_NSAMPLES] == '4'
    assert processor.log[-1]['lifecycle'] == 'calibrated_fp8'
    replay = model.proj(x).detach()
    packed = processor.submodule_finalize(named, wrapper, qcfg=active_config)
    assert isinstance(packed, TorchFP8Linear)
    torch.testing.assert_close(packed(x), replay, rtol=1e-6, atol=1e-7)
    assert 'gsq_fp8_result' not in named.state
    assert 'fp8_reference_weight' not in named.state
    processor._close_device_smi_handles()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('selective', [False, True])
def test_public_calibrated_fp8_quantizes_tiny_llama(monkeypatch, tmp_path, dtype, selective):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.utils.backend import BACKEND

    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, pad_token_id=0)
    config._attn_implementation = 'eager'
    native = LlamaForCausalLM(config).to(dtype=dtype).eval()
    cfg = FP8Config(device=DEVICE.CPU, offload_to_disk=False, gsq_calibration=True,
                    gsq={'enabled': True, 'steps': 2, 'candidates': 3})
    if selective:
        cfg.dynamic = {r'-:model\.layers\.[1-9][0-9]*\.': {},
                       r'-:model\.layers\.0\.mlp\.': {},
                       r'-:model\.layers\.0\.self_attn\.o_proj$': {}}
    wrapper = LlamaQModel(model=native, quantized=False, quantize_config=cfg, tokenizer=None)
    batch = {'input_ids': torch.arange(1, 17).reshape(1, 16),
             'attention_mask': torch.ones(1, 16, dtype=torch.long)}
    monkeypatch.setattr(wrapper, 'prepare_dataset', lambda **kwargs: [batch])
    result = wrapper.quantize(calibration=[batch], backend=BACKEND.FP8_TORCH)
    assert wrapper.quantized
    assert 'fp8_gsq' in result
    packed = [module for module in native.modules() if isinstance(module, TorchFP8Linear)]
    assert len(packed) == (3 if selective else 14)
    with torch.no_grad():
        logits = native(**batch, use_cache=False).logits
    assert torch.isfinite(logits).all()
    checkpoint = tmp_path / 'state.pt'
    torch.save(native.state_dict(), checkpoint)
    restored = LlamaForCausalLM(config).to(dtype=dtype).eval()
    for name, module in native.named_modules():
        if isinstance(module, TorchFP8Linear):
            destination = TorchFP8Linear(
                bits=8, group_size=-1, sym=True, desc_act=False, in_features=module.in_features,
                out_features=module.out_features, bias=module.bias is not None,
                **cfg.quant_linear_init_kwargs())
            parent, leaf = name.rsplit('.', 1)
            setattr(restored.get_submodule(parent), leaf, destination)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        reloaded_logits = restored(**batch, use_cache=False).logits
    torch.testing.assert_close(reloaded_logits, logits, rtol=0, atol=0)



def test_dynamic_disabled_calibrated_module_config_is_serializable():
    from gptqmodel.quantization.config import QuantizeConfig, clone_weight_only_config_for_module

    cfg = FP8Config(gsq={'enabled': True}, gsq_calibration=True,
                    dynamic={'+:.*proj$': {'gsq': {'enabled': False}}})
    clone = clone_weight_only_config_for_module(cfg, 'model.layers.0.proj')
    restored = QuantizeConfig.from_quant_config(clone.to_dict())
    assert not restored.gsq.enabled
    assert not restored.gsq_calibration
    assert cfg.gsq_calibration
