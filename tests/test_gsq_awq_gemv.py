"""Actual GEMV packers versus GSQ's exported operator, without CUDA mocks."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import AWQConfig, FORMAT, GSQConfig, QuantizeConfig, RTNConfig
from gptqmodel.quantization.gsq_scalar import affine_codes, refine_affine_scalar
from gptqmodel.utils.gemv import awq_gemv_codes, calculate_zeros_width, dequantize_awq_gemv


FORMATS = ((FORMAT.GEMV, AwqGEMVLinear, "awq_gemv"),
           (FORMAT.GEMV_FAST, AwqGEMVFastLinear, "awq_gemv_fast"),
           (FORMAT.LLM_AWQ, LLMAwqLinear, "awq_gemv_fast"))


def unpack_gemv(module, fast):
    """Independent inversion of the row/interleaved layouts and stored offsets."""
    n, k = module.out_features, module.in_features
    groups = torch.arange(k) // module.group_size
    if not fast:
        codes = ((module.qweight.long().unsqueeze(-1) >> torch.arange(0, 32, 4)) & 15).reshape(n, k)
        zero = ((module.qzeros.long().unsqueeze(-1) >> torch.arange(0, 32, 4)) & 15).reshape(n, -1)
        return codes, module.scales.float()[:, groups] * (codes-zero[:, groups])
    words = (module.qweight.long().unsqueeze(-1) >> torch.arange(0, 16, 4)) & 15
    values = words.reshape(n//4, k//64, 4, 64).permute(0, 2, 1, 3).reshape(n, k)
    values = values.reshape(n, k//32, 4, 2, 4).transpose(3, 4).reshape(n, k)
    codes = values.reshape(n, k//32, 4, 4, 2).transpose(2, 3).reshape(n, k)
    return codes, module.scales.T.float()[:, groups]*codes + getattr(module, module.zeros_name).T.float()[:, groups]


def pack_on_cpu(cls, weight, scales, zeros, group_size):
    n, k = weight.shape
    layer = torch.nn.Linear(k, n, bias=False, dtype=weight.dtype)
    layer.weight.data.copy_(weight)
    # CPU-capable buffer container; call the real GEMV packing method. CUDA
    # constructor availability is unrelated to these CPU tensor transforms.
    module = AwqTorchLinear(bits=4, group_size=group_size, sym=False, desc_act=False,
                             in_features=k, out_features=n, bias=False)
    module.zeros_name = getattr(cls, "zeros_name", "qzeros")
    cls.pack(module, layer, scales, zeros)
    return module


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_gemv_gsq_actual_packed_objective(fmt, cls, packing, dtype, group_size):
    generator = torch.Generator().manual_seed(7)
    target = (torch.randn(32, 128, generator=generator)*0.2).to(dtype)
    cfg = AWQConfig(bits=4, group_size=group_size, sym=False, format=fmt,
                    gsq=GSQConfig(enabled=True, steps=10, learn_scales=True))
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.gsq == cfg.gsq
    assert restored.format == cfg.format
    baseline, scales, zeros = AWQProcessor.pseudo_quantize_tensor(SimpleNamespace(qcfg=cfg), target)
    groups = torch.arange(128, dtype=torch.int32)//(128 if group_size == -1 else group_size)
    inputs = torch.randn(37, 128, generator=generator)
    result = refine_affine_scalar(baseline, scales, zeros, groups, target=target.float(), bits=4,
                                   config=cfg.gsq, inputs=inputs, packing=packing)
    base = pack_on_cpu(cls, baseline, scales, zeros, group_size)
    packed = pack_on_cpu(cls, result.weight, result.scales, result.zeros, group_size)
    base_codes, base_dense = unpack_gemv(base, packing == "awq_gemv_fast")
    codes, dense = unpack_gemv(packed, packing == "awq_gemv_fast")
    restored_dense = dequantize_awq_gemv(
        packed.qweight, packed.scales, getattr(packed, packed.zeros_name),
        group_size=packed.group_size, in_features=128, out_features=32, fast=packing == "awq_gemv_fast")
    assert torch.equal(restored_dense.T, dense.to(packed.scales.dtype))
    canonical = dequantize_awq_gemv(
        packed.qweight, packed.scales.float(), getattr(packed, packed.zeros_name),
        group_size=packed.group_size, in_features=128, out_features=32, fast=packing == "awq_gemv_fast")
    assert torch.equal(canonical.T, dense)
    # The original column loop is an independent dtype/order reference for
    # the new shared vectorized inverse, including non-power-of-two scales.
    offset, stored = zeros*scales, scales.half()
    old_codes = torch.stack([((baseline[:, column]+offset[:, group])/stored[:, group]).round()
                             for column, group in enumerate(groups.tolist())], dim=1)
    assert torch.equal(base_codes, old_codes.clamp(0, 15))
    assert torch.equal(base_codes, affine_codes(baseline, scales, zeros, groups, 4, packing=packing))
    assert torch.equal(codes, affine_codes(result.weight, result.scales, zeros, groups, 4, packing=packing))
    denominator = (target.float() @ inputs.T).square().sum()
    assert float(((base_dense-target.float()) @ inputs.T).square().sum()/denominator) == pytest.approx(result.before, rel=1e-5)
    assert float(((dense-target.float()) @ inputs.T).square().sum()/denominator) == pytest.approx(result.after, rel=1e-5)
    assert result.after <= result.before


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
def test_gemv_pack_saturates_endpoints(fmt, cls, packing):
    scales = torch.full((32, 4), 0.125)
    zeros = torch.full_like(scales, 7)
    groups = torch.arange(128)//32
    desired = torch.tensor([-100., -1., 0., 1., 14., 15., 16., 100.]).repeat(32, 16)
    weight = scales[:, groups]*(desired-zeros[:, groups])
    packed = pack_on_cpu(cls, weight, scales, zeros, 32)
    codes, _ = unpack_gemv(packed, packing == "awq_gemv_fast")
    assert torch.equal(codes, desired.clamp(0, 15))


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
@pytest.mark.parametrize("invalid", ["nan", "zero_scale", "storage_underflow", "fractional_zero", "zero_range"])
def test_gemv_pack_rejects_invalid_metadata(fmt, cls, packing, invalid):
    weight = torch.zeros(32, 128)
    scales, zeros = torch.ones(32, 4), torch.full((32, 4), 7.)
    if invalid == "nan":
        weight[0, 0] = torch.nan
    elif invalid == "zero_scale":
        scales.zero_()
    elif invalid == "storage_underflow":
        scales.fill_(1e-20)
    elif invalid == "fractional_zero":
        zeros.fill_(0.5)
    else:
        zeros.fill_(16)
    with pytest.raises(ValueError):
        pack_on_cpu(cls, weight, scales, zeros, 32)


def test_gemv_fast_rejects_unrepresentable_offset():
    with pytest.raises(ValueError, match="offsets"):
        pack_on_cpu(AwqGEMVFastLinear, torch.zeros(32, 128), torch.full((32, 4), 60000.),
                    torch.full((32, 4), 7.), 32)


@pytest.mark.parametrize("invalid", ["shape", "empty", "device", "group_dtype", "group_range",
                                      "storage_overflow", "source_offset_overflow"])
def test_gemv_inverse_rejects_invalid_contract(invalid):
    weight = torch.zeros(32, 128)
    scales, zeros = torch.ones(32, 4), torch.full((32, 4), 7.)
    groups = torch.arange(128)//32
    if invalid == "shape":
        scales = scales.flatten()
    elif invalid == "empty":
        weight, scales, zeros, groups = torch.empty(0, 0), torch.empty(0, 0), torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    elif invalid == "device":
        scales = scales.to("meta")
    elif invalid == "group_dtype":
        groups = groups.float()
    elif invalid == "group_range":
        groups[0] = 4
    elif invalid == "storage_overflow":
        scales.fill_(1e10)
    else:
        scales, zeros = scales.half().fill_(60000), zeros.half()
    with pytest.raises(ValueError):
        awq_gemv_codes(weight, scales, zeros, groups)


def test_gemv_gsq_rejects_unrepresentable_storage():
    weight, scales, zeros, groups = torch.zeros(32, 128), torch.full((32, 4), 60000.), torch.full((32, 4), 7.), torch.arange(128)//32
    with pytest.raises(ValueError, match="offsets"):
        refine_affine_scalar(weight, scales, zeros, groups, target=torch.ones_like(weight), bits=4,
                              packing="awq_gemv_fast", config=GSQConfig(enabled=True, steps=1))
    for bits, dtype in ((3, torch.float16), (4, torch.bfloat16)):
        with pytest.raises(ValueError, match="int4 codes and FP16"):
            affine_codes(weight, scales, zeros, groups, bits, packing="awq_gemv", scale_dtype=dtype)
    with pytest.raises(NotImplementedError):
        calculate_zeros_width(128, 16)
    for cls in (AWQConfig, RTNConfig):
        with pytest.raises(ValueError, match="group_size"):
            cls(bits=4, format=FORMAT.GEMV, group_size=16, gsq=GSQConfig(enabled=True))


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
def test_rtn_gemv_config_roundtrip(fmt, cls, packing):
    cfg = RTNConfig(bits=4, group_size=32, format=fmt, gsq=GSQConfig(enabled=True))
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert isinstance(restored, RTNConfig)
    assert restored.gsq == cfg.gsq and restored.format == cfg.format


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
@pytest.mark.parametrize("method", ["rtn", "awq"])
def test_gemv_public_quantization_hooks(fmt, cls, packing, method):
    from test_adjacent_awq import _TestAWQProcessor

    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization.rtn import RTN

    generator = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
    layer.weight.data.copy_(torch.randn(32, 128, generator=generator)*0.1)
    target = layer.weight.detach().float().clone()
    gsq = GSQConfig(enabled=True, steps=10, learn_scales=True)
    cfg_cls = RTNConfig if method == "rtn" else AWQConfig
    cfg = cfg_cls(bits=4, group_size=32, sym=False, format=fmt, gsq=gsq)
    if method == "rtn":
        task = RTN(layer, cfg)
        weight, scales, zeros, *_ = task.quantize()
        stats = task.gsq_diagnostics
        inputs = torch.eye(128)
    else:
        processor = _TestAWQProcessor(cfg)
        named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
        references = processor._capture_adjacent_reference_weights({"proj": named})
        layer.weight.data.clamp_(-0.1, 0.1)
        inputs = torch.randn(37, 128, generator=generator)
        processor.apply_quant({"proj": named}, [], input_features={"proj": inputs}, adjacent_references=references)
        named.stream_sync()
        weight, scales, zeros = layer.weight.detach(), named.state["q_scales"], named.state["q_zeros"]
        stats = named.state["gsq_diagnostics"]
    packed = pack_on_cpu(cls, weight, scales, zeros, 32)
    _, dense = unpack_gemv(packed, packing == "awq_gemv_fast")
    loss = float(((dense-target) @ inputs.T).square().sum()/(target @ inputs.T).square().sum())
    assert loss == pytest.approx(stats["after"], rel=1e-5)


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemv_learned_checkpoint_survives_source_and_storage_casts(fmt, cls, packing, dtype):
    weight = torch.full((32, 128), 0.125, dtype=dtype)
    target = torch.full((32, 128), 0.2)
    scales, zeros = torch.full((32, 4), 0.125, dtype=dtype), torch.full((32, 4), 7., dtype=dtype)
    groups = torch.arange(128, dtype=torch.int32)//32
    result = refine_affine_scalar(weight, scales, zeros, groups, target=target, bits=4, packing=packing,
                                   config=GSQConfig(enabled=True, steps=50, candidates=3, learn_scales=True))
    assert result.after < result.before
    packed = pack_on_cpu(cls, result.weight, result.scales, zeros, 32)
    _, dense = unpack_gemv(packed, packing == "awq_gemv_fast")
    assert float((dense-target).square().sum()/target.square().sum()) == pytest.approx(result.after, rel=1e-5)


@pytest.mark.parametrize("fmt,cls,packing", FORMATS)
@pytest.mark.parametrize("tokens", [1, 16])
def test_gemv_native_reloaded_forward(fmt, cls, packing, tokens, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires a leased CUDA device")
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(128, 256, generator=generator).half()*0.05
    cfg = AWQConfig(bits=4, group_size=128, sym=False, format=fmt)
    weight, scales, zeros = AWQProcessor.pseudo_quantize_tensor(SimpleNamespace(qcfg=cfg), target)
    groups = torch.arange(256, dtype=torch.int32)//128
    result = refine_affine_scalar(weight.cuda(), scales.cuda(), zeros.cuda(), groups.cuda(),
                                   target=target.cuda().float(), bits=4, packing=packing,
                                   config=GSQConfig(enabled=True, steps=10, learn_scales=True))
    cpu = pack_on_cpu(cls, result.weight.cpu(), result.scales.cpu(), zeros, 128)
    _, dense = unpack_gemv(cpu, packing == "awq_gemv_fast")

    def native():
        return cls(bits=4, group_size=128, sym=False, desc_act=False, in_features=256,
                   out_features=128, bias=False, register_buffers=True)

    module = native()
    module.load_state_dict(cpu.state_dict(), strict=True)
    path = tmp_path / "packed.pt"
    torch.save(module.state_dict(), path)
    restored = native()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored = restored.cuda().eval()
    assert torch.equal(restored.dequantize_weight().T.cpu(), dense.half())
    x = torch.randn(tokens, 256, generator=generator).half().cuda()
    with torch.inference_mode():
        actual = restored(x).float()
        reference = x.float() @ dense.cuda().T
    delta = (actual-reference).abs()
    assert torch.isfinite(actual).all()
    print(f"NATIVE_PARITY {fmt.value} tokens={tokens} mean={float(delta.mean()):.9g} max={float(delta.max()):.9g}")
    assert float(delta.mean()) <= 0.002
    assert float(delta.max()) <= 0.046875
