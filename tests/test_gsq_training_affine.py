import pytest
import torch

from gptqmodel.quantization.gsq_training_affine import GSQAffineTrainingModule


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_affine_boundaries_and_relaxation_gradients(bits):
    maximum = 2**bits-1
    codes = torch.tensor([[0, maximum, 1, maximum-1]])
    scales = torch.tensor([[.125, .25]])
    zeros = torch.tensor([[0, maximum]])
    module = GSQAffineTrainingModule(codes, scales, zeros, 2, bits=bits,
                                     noise=torch.zeros(4 if bits == 2 else 5, 1, 4)).double()
    torch.testing.assert_close(module.hard_codes(), codes.double(), rtol=0, atol=0)
    expected = (codes-zeros.repeat_interleave(2, 1))*scales.repeat_interleave(2, 1)
    torch.testing.assert_close(module.hard_weight(), expected.double(), rtol=0, atol=0)
    candidate_codes = module.candidates+module.zeros[:, module.group_index]
    assert ((candidate_codes[module.valid] >= 0) & (candidate_codes[module.valid] <= maximum)).all()
    uniform = torch.linspace(.1, .9, module.logits.numel(), dtype=torch.float64).reshape_as(module.logits)
    actual = module(uniform=uniform, temperature=.7, multiplier=12.)
    noise = -torch.log(-torch.log(uniform+1e-8)+1e-8)
    probability = ((module.logits.masked_fill(~module.valid, -1e9)*12.+noise)/.7).softmax(0)
    reference = (probability*module.candidates).sum(0)*module.scales[:, module.group_index]
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (module.logits, module.scales), retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().sum(), (module.logits, module.scales))
    for got, want in zip(actual_grad, reference_grad):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('codes,scales,zeros', [
    ([[16., 0.]], [[1.]], [[0.]]),
    ([[.5, 0.]], [[1.]], [[0.]]),
    ([[1., 0.]], [[-1.]], [[0.]]),
    ([[1., 0.]], [[1.]], [[.5]]),
    ([[1., 0.]], [[1.]], [[16.]]),
])
def test_affine_rejects_unrepresentable_initializer(codes, scales, zeros):
    with pytest.raises(ValueError):
        GSQAffineTrainingModule(torch.tensor(codes), torch.tensor(scales), torch.tensor(zeros),
                                2, bits=4, noise=torch.zeros(5, 1, 2))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_affine_awq_gemm_export_pack_reload(dtype, tmp_path):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    codes = torch.arange(32*64).reshape(32, 64) % 16
    scales = torch.linspace(.01, .2, 64).reshape(32, 2)
    zeros = torch.arange(64).reshape(32, 2) % 16
    module = GSQAffineTrainingModule(codes, scales, zeros, 32, bits=4, noise=torch.zeros(5, 32, 64))
    with torch.no_grad():
        choice = module.candidates.masked_fill(~module.valid, -torch.inf).argmax(0, keepdim=True)
        module.logits.fill_(-1.)
        module.logits.scatter_(0, choice, 1.)
        module.scales.mul_(1.17)
    export = module.export_affine(packing='awq_gemm', scale_dtype=dtype)
    assert (export['codes'] != codes).any()
    assert not torch.equal(export['scales'].float(), scales)
    layer = torch.nn.Linear(64, 32, bias=False)
    layer.weight.data.copy_(export['weight'])
    def packed():
        return AwqTorchLinear(bits=4, group_size=32, sym=False, desc_act=False,
                              in_features=64, out_features=32, bias=False)
    first = packed()
    first.pack(layer, export['scales'], export['zeros'])
    checkpoint = tmp_path/'awq.pt'
    torch.save(first.state_dict(), checkpoint)
    second = packed()
    second.pack(layer, export['scales'], export['zeros'])
    for value in second.state_dict().values():
        value.zero_()
    second.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    second.scales = second.scales.float()
    torch.testing.assert_close(second.dequantize_weight().T, export['weight'], atol=0, rtol=0)


@pytest.mark.parametrize('scale', [-1., 0., 1e-12, float('inf')])
def test_affine_export_rejects_invalid_learned_storage_scales(scale):
    module = GSQAffineTrainingModule(torch.zeros(1, 2), torch.ones(1, 1), torch.zeros(1, 1),
                                     2, bits=4, noise=torch.zeros(5, 1, 2))
    with torch.no_grad():
        module.scales.fill_(scale)
    with pytest.raises(ValueError, match='scales'):
        module.export_affine(packing='awq_gemm')


def test_affine_llama_stages_keep_zero_points_and_never_initialize_gptq(monkeypatch):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization import gsq_training

    torch.manual_seed(7)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    pristine = {k: v.clone() for k, v in layer.state_dict().items()}
    initializers = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            grouped = module.weight.detach().reshape(module.out_features, -1, 32)
            scales = (grouped.amax(-1)-grouped.amin(-1)).clamp_min(1e-6)/15
            zeros = (-grouped.amin(-1)/scales).round().clamp(0, 15)
            codes = (grouped/scales.unsqueeze(-1)+zeros.unsqueeze(-1)).round().clamp(0, 15)
            initializers[name] = (codes.reshape_as(module.weight), scales, zeros)
    hidden = torch.randn(1, 8, 32)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(hidden, torch.arange(8)[None]),
                  attention_mask=torch.full((8, 8), -torch.inf).triu(1)[None, None], use_cache=False)
    def forbidden(*args, **kwargs):
        pytest.fail('Affine staged fitting replaced supplied initialization with GPTQ')
    monkeypatch.setattr(gsq_training, 'initialize_llama_gptq', forbidden)
    with pytest.raises(ValueError, match='supplied MLP'):
        gsq_training.fit_llama_stages(layer, initializers, [(hidden, kwargs)], bits=4,
                                      group_size=32, epochs=1, affine_initializers=True)
    fitted, records = gsq_training.fit_llama_stages(
        layer, initializers, [(hidden, kwargs)], bits=4, group_size=32, epochs=1,
        qk_steps=2, affine_initializers=True, reinitialize_mlp=False)
    assert torch.isfinite(fitted(hidden, **kwargs)).all()
    for name, (_, _, zeros) in initializers.items():
        stage = name if name in records else ('attention' if name.startswith('self_attn.') else 'mlp')
        key = 'weight' if stage == name else name+'.weight'
        record = records[stage]
        torch.testing.assert_close(record['zeros'][key], zeros, rtol=0, atol=0)
        codes = record['codes'][key]
        assert ((codes >= 0) & (codes <= 15) & (codes == codes.round())).all()
        expected = (codes-zeros.repeat_interleave(32, 1))*record['scales'][key].repeat_interleave(32, 1)
        torch.testing.assert_close(fitted.get_submodule(name).weight, expected, atol=0, rtol=0)
    assert all(torch.equal(v, pristine[k]) for k, v in layer.state_dict().items())
    from gptqmodel.quantization.gsq_training_affine import pack_llama_affine_stages

    packed = pack_llama_affine_stages(fitted, records, group_size=32)
    for name in initializers:
        stage = name if name in records else ('attention' if name.startswith('self_attn.') else 'mlp')
        key = 'weight' if stage == name else name+'.weight'
        record = records[stage]
        expected = ((record['codes'][key]-record['zeros'][key].repeat_interleave(32, 1))
                    *record['scales'][key].half().float().repeat_interleave(32, 1))
        linear = packed.get_submodule(name)
        linear.scales = linear.scales.float()
        torch.testing.assert_close(linear.dequantize_weight().T, expected, rtol=0, atol=0)
    assert torch.isfinite(packed(hidden, **kwargs)).all()
