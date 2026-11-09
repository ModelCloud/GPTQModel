# SPDX-License-Identifier: Apache-2.0
"""CPU storage/ownership contracts; CUDA projection oracle runs separately."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils import kimi_k3
from gptqmodel.utils import kimi_k3_core_format as core


def packed(outputs=640, inputs=128, bits=6, logical_outputs=576):
    scales = torch.ones(outputs, dtype=torch.float16)
    scales[logical_outputs:] = 0
    return {"trellis": torch.zeros((inputs // 16, outputs // 16, bits * 16), dtype=torch.int16),
                "suh": torch.ones(inputs, dtype=torch.float16), "svh": scales,
                "mcg": torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)}


def test_padded_geometry_keeps_ordinary_exl3_storage():
    evidence = core.validate_core_tensors(packed(), (576, 128))
    assert evidence == {"native_shape": [576, 128], "stored_shape": [640, 128], "bits": 6, "output_padding": 64}


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(trellis=torch.zeros((8, 36, 96), dtype=torch.int16)),
    lambda p: p.update(svh=torch.ones(576, dtype=torch.float16)),
    lambda p: p.update(svh=torch.ones(640, dtype=torch.float16)),
    lambda p: p.update(mcg=torch.tensor(123, dtype=torch.int32)),
    lambda p: p.update(mul1=torch.tensor(123, dtype=torch.int32)),
    lambda p: p["suh"].fill_(float("nan")),
])
def test_reject_ambiguous_or_nonstandard_storage(mutation):
    tensors = packed()
    mutation(tensors)
    with pytest.raises(ValueError):
        core.validate_core_tensors(tensors, (576, 128))


def test_reconstruct_crops_after_full_inverse_transform(monkeypatch):
    dense = torch.arange(128 * 640, dtype=torch.float32).reshape(128, 640)
    calls = []
    def reconstruct(tensors, *, device, dtype):
        calls.append((tuple(tensors['svh'].shape), device, dtype))
        return dense.to(dtype)
    monkeypatch.setattr(core, '_quantizer', lambda: SimpleNamespace(reconstruct_exl3_tensors=reconstruct))
    actual = core.reconstruct_core_projection(packed(), (576, 128), device='cpu', dtype=torch.float32)
    torch.testing.assert_close(actual, dense[:, :576].T.contiguous(), rtol=0, atol=0)
    assert actual.is_contiguous()
    assert calls == [((640,), 'cpu', torch.float32)]


def valid_metrics(count):
    return {"quantizer_path": 'hessian_ldlq', "hessian_metric_status": 'ok',
                "hessian_numerical_contract": core._HESSIAN_CONTRACT,
                "hessian_transform_compute_dtype": 'torch.float64', "hessian_storage_dtype": 'torch.float32',
                "hessian_sample_count": count, "reported_metric_value": .001,
                "hessian_weighted_error_numerator": 1., "hessian_weighted_reference_denominator": 1000.,
                "hessian_weighted_relative_error": .001}


def test_quantizer_zero_pads_and_does_not_mutate_raw_hessian_or_weight(monkeypatch):
    weight = torch.arange(576 * 128, dtype=torch.float32).reshape(576, 128) / 1024
    original = weight.clone()
    raw = {"H": torch.eye(128), "count": 1201086, "finalized": False}
    calls = []
    def quantize(w, h, args, *, return_weight_q):
        torch.testing.assert_close(w[:, :576], original.T, rtol=0, atol=0)
        assert torch.count_nonzero(w[:, 576:]) == 0
        assert w.shape == (128, 640)
        assert args['K'] == 6 and args['seed'] == 787 and args['sigma_reg'] == .025
        assert not return_weight_q
        calls.append(h['H'].data_ptr())
        h['H'].zero_()
        h['finalized'] = True
        w.zero_()
        args['error_metrics'] = valid_metrics(h['count'])
        return None, .001, packed()
    monkeypatch.setattr(core, '_quantizer', lambda: SimpleNamespace(quantize_exl3=quantize))
    for _ in range(2):
        tensors, metrics, evidence = core.quantize_core_projection(weight, raw, device='cpu')
        assert evidence['identity_rows'] == 0 and evidence['output_padding'] == 64
        assert all(t.device.type == 'cpu' for t in tensors.values())
        torch.testing.assert_close(raw['H'], torch.eye(128), rtol=0, atol=0)
        torch.testing.assert_close(weight, original, rtol=0, atol=0)
        assert raw['finalized'] is False
    assert raw['H'].data_ptr() not in calls


@pytest.mark.parametrize('bad_metric,value', [
    ('quantizer_path', 'fallback'), ('hessian_metric_status', 'out_of_memory'),
    ('hessian_transform_compute_dtype', 'torch.float32'),
    ('hessian_numerical_contract', 'different'),
    ('hessian_sample_count', 2), ('reported_metric_value', float('nan')),
    ('hessian_weighted_relative_error', -1.),
])
def test_core_quantizer_fails_closed_on_uncalibrated_metrics(monkeypatch, bad_metric, value):
    def quantize(w, h, args, **kwargs):
        args['error_metrics'] = valid_metrics(h['count'])
        args['error_metrics'][bad_metric] = value
        return None, .001, packed()
    monkeypatch.setattr(core, '_quantizer', lambda: SimpleNamespace(quantize_exl3=quantize))
    with pytest.raises(ValueError):
        core.quantize_core_projection(torch.ones(576,128), {"H": torch.eye(128), "count": 3, "finalized": False}, device='cpu')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_generic_wrapper_crops_padded_output_and_preserves_input_dtype(monkeypatch, dtype):
    # Independent Torch linear oracle. The padded inner sees all640 channels;
    # outer cropping preserves exactly the native576 output and input dtype.
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(640,128,generator=generator,dtype=torch.float16) * .01
    x = torch.randn(3,128,generator=generator).to(dtype)
    module = ExllamaV3Linear.from_tensors(in_features=128, out_features=576, name='kv_a', tensors=packed())
    assert module._stored_features() == (128,640)
    class Inner:
        def forward(self, actual, params):
            return actual @ weight.T
    monkeypatch.setattr(module, '_ensure_inner', lambda: Inner())
    actual = module(x)
    expected = torch.nn.functional.linear(x.half(),weight[:576]).to(dtype)
    assert actual.dtype == dtype and actual.shape == (3,576)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('out_features', [500,641,768])
def test_generic_wrapper_rejects_storage_that_cannot_represent_logical_shape(out_features):
    module = ExllamaV3Linear.from_tensors(in_features=128,out_features=out_features,name='bad',tensors=packed())
    with pytest.raises(ValueError,match='logical dimensions'):
        module._stored_features()


def write_source(tmp_path, monkeypatch):
    tensors = {'native.weight':torch.arange(8,dtype=torch.bfloat16).reshape(2,4)}
    prefix='language_model.model.layers.1.block_sparse_moe.experts.0.w1'
    tensors.update({prefix+'.'+k:v for k,v in packed(outputs=128,logical_outputs=128,bits=2).items()})
    save_file(tensors,tmp_path/'model.safetensors')
    config={"model_type": 'kimi_k3',"text_config": {"num_hidden_layers": 3,"num_experts": 1,
                "hidden_size": 128,"routed_expert_hidden_size": 128,"moe_intermediate_size": 128}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({"weight_map": dict.fromkeys(tensors, 'model.safetensors')}))
    monkeypatch.setattr(kimi_k3,'load_calibration_reference',lambda _:SimpleNamespace())
    return prefix


def test_exl3_source_reads_exact_base_packed_components(tmp_path,monkeypatch):
    prefix=write_source(tmp_path,monkeypatch)
    calls=[]
    def reconstruct(tensors,logical_shape,*,device):
        assert set(tensors)=={'trellis','suh','svh','mcg'}
        assert logical_shape==(128,128)
        assert tensors['trellis'].shape==(8,8,32)
        calls.append(device)
        return torch.full(logical_shape,3,dtype=torch.bfloat16)
    monkeypatch.setattr(core,'reconstruct_core_projection',reconstruct)
    source=core.KimiK3EXL3Source(tmp_path)
    assert torch.equal(source.decoded('native.weight'),torch.arange(8,dtype=torch.bfloat16).reshape(2,4))
    assert source.decoded(prefix+'.weight').shape==(128,128)
    assert calls==['cpu']
    with pytest.raises(KeyError): source.decoded('missing.weight')


def test_lazy_source_materializes_routed_weights_only_on_request(monkeypatch):
    source=object.__new__(core.KimiK3EXL3Source)
    source.text_config={'num_hidden_layers':3}
    class Config:
        @staticmethod
        def from_dict(value): return SimpleNamespace()
    class Block(nn.Module):
        def __init__(self,config,layer):
            super().__init__()
            self.layer_idx=layer
            self.self_attn=nn.Linear(128,128,bias=False)
            self.block_sparse_moe=nn.Module()
            self.block_sparse_moe.experts=nn.ModuleList([nn.Linear(128,128,bias=False)])
    source.reference=SimpleNamespace(KimiLinearConfig=Config,KimiDecoderLayer=Block)
    names=[]
    def decoded(name,device):
        names.append(name)
        return torch.ones(128,128,dtype=torch.bfloat16,device=device)
    monkeypatch.setattr(source,'decoded',decoded)
    block=source.load_block(1,device='cpu')
    assert not block.self_attn.weight.is_meta and block.block_sparse_moe.experts[0].weight.is_meta
    assert names==['language_model.model.layers.1.self_attn.weight']
    source.materialize_routed_experts(block,device='cpu')
    assert not any(p.is_meta for p in block.parameters())
    assert names[-1]=='language_model.model.layers.1.block_sparse_moe.experts.0.weight'
    source.materialize_routed_experts(block,device='cpu')
    assert len(names)==2  # Repeated admission must not duplicate a full expert block.


def test_source_reuses_only_one_reader_and_returns_owned_tensors(tmp_path, monkeypatch):
    from pathlib import Path

    import safetensors
    original_open = safetensors.safe_open
    opened, active = [], set()
    class Reader:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.inner = original_open(path, **kwargs)
        def __enter__(self):
            assert not active
            self.inner.__enter__()
            active.add(self.path.name)
            opened.append(self.path.name)
            return self
        def __exit__(self, *args):
            active.remove(self.path.name)
            return self.inner.__exit__(*args)
        def get_tensor(self, name):
            return self.inner.get_tensor(name)
    monkeypatch.setattr(safetensors, 'safe_open', Reader)
    save_file({'a': torch.arange(16, dtype=torch.bfloat16)}, tmp_path/'one.safetensors')
    save_file({'b': torch.arange(8, dtype=torch.bfloat16)}, tmp_path/'two.safetensors')
    source = object.__new__(core.KimiK3EXL3Source)
    source.snapshot = tmp_path
    source.weight_map = {'a':'one.safetensors', 'b':'two.safetensors'}
    fd_before = len(list(Path('/proc/self/fd').iterdir()))
    for _ in range(1100):
        value = source.tensor('a')
        assert torch.equal(value, torch.arange(16, dtype=torch.bfloat16))
        # Mutating owned memory must not alter a subsequent source read.
        value.zero_()
    assert opened == ['one.safetensors']
    assert len(list(Path('/proc/self/fd').iterdir())) <= fd_before + 4
    assert torch.equal(source.tensor('b'), torch.arange(8, dtype=torch.bfloat16))
    assert opened == ['one.safetensors', 'two.safetensors']
    assert active == {'two.safetensors'}
    source.close()
    source.close()
    assert not active


def test_generic_source_clones_before_closing_and_transfers_owned_buffer(monkeypatch):
    state = {'active': False}
    class Owned:
        def to(self, *, device):
            assert not state['active']
            assert device == 'cuda:0'
            return 'owned-device-buffer'
    class Mapped:
        def clone(self):
            assert state['active']
            return Owned()
    class Reader:
        def __enter__(self):
            state['active'] = True
            return self
        def __exit__(self, *args):
            state['active'] = False
        def get_tensor(self, name):
            assert name == 'tensor'
            return Mapped()
    monkeypatch.setattr(kimi_k3, 'safe_open', lambda *args, **kwargs: Reader())
    source = object.__new__(kimi_k3.KimiK3Source)
    from pathlib import Path
    source.snapshot, source.weight_map = Path('/unused'), {'tensor':'shard'}
    assert source.tensor('tensor', 'cuda:0') == 'owned-device-buffer'
