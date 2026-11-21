# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import CalibrationConfig, CPTCConfig, GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.cptc import CPTCLinear
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.quantization.cptc import (
    CPTCSpec,
    decode_e2m5,
    decode_weight,
    gaussian_table,
    pack_states,
    snap_e2m5,
    state_indices,
    unpack_states,
    viterbi_encode,
)
from gptqmodel.quantization.cptc_hessian import HessianSketch
from gptqmodel.quantization.cptc_yaqa import _wavefront_tiles, block_feedback
from gptqmodel.utils import arch, cptc
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


def _e2m5_oracle(code):
    sign = -1 if code & 128 else 1
    exponent, fraction = divmod(code % 128, 32)
    if code % 128 == 127:
        return math.nan
    if exponent == 0:
        return sign * (fraction / 32)
    return math.ldexp(sign * (1 + fraction / 32), exponent - 1)


def _decode_oracle(packed, table, coeffs, scales, spec, in_features):
    """Decode via independent integer bitstreams and scalar E2M5 arithmetic."""
    rows = []
    coefficients = coeffs.tolist()
    lookup = [_e2m5_oracle(code) for code in table.tolist()]
    for row, row_scales in zip(packed.tolist(), scales.tolist()):
        weights = []
        for block, scale in zip(row, row_scales):
            stream = int.from_bytes(bytes(block) + bytes(block), byteorder="little")
            for t in range(spec.block_size):
                state = (stream >> (t * spec.bits)) % (2**spec.trellis_depth)
                taps = [((a * state) % 65536) // (2**spec.shift) for a in coefficients]
                index = sum(taps)
                code = lookup[index] if index < len(lookup) else -lookup[2 * len(lookup) - 2 - index]
                weights.append(code * scale)
        rows.append(weights[:in_features])
    return torch.tensor(rows, dtype=torch.float32)


def _operands(spec, k, n, *, unified=False):
    rng = torch.Generator().manual_seed(7531)
    shape = (n, spec.blocks(k))
    packed = torch.randint(256, (*shape, spec.block_bytes), dtype=torch.uint8, generator=rng)
    table = gaussian_table(spec, "cpu")
    coeffs = torch.tensor(spec.coefficients(unified=unified), dtype=torch.int32)
    scales = torch.rand(shape, generator=rng) * 0.05
    scales[0, 0] = 0
    return packed, table, coeffs, scales


def _walsh(size):
    matrix = torch.ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)))
    return matrix / math.sqrt(size)


@pytest.fixture(scope="module")
def cptc_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CPTC native kernels need a CUDA SM80+ or ROCm RDNA4 device")
    device = torch.device("cuda", torch.cuda.current_device())
    if not cptc.native_supported(device):
        pytest.skip("CPTC has native kernels for CUDA SM80+ and ROCm gfx1200/gfx1201")
    ops = cptc.native_ops(device)
    assert ops is not None, cptc._EXTENSION.last_error_message()
    return device, ops


def test_e2m5_bytes_and_rounding_boundaries():
    codes = torch.arange(256, dtype=torch.uint8)
    values = decode_e2m5(codes)
    expected = torch.tensor([_e2m5_oracle(code) for code in range(256)])
    torch.testing.assert_close(values, expected, atol=0, rtol=0, equal_nan=True)
    assert values[128].signbit() and not values[0].signbit()

    lower = torch.arange(126, dtype=torch.uint8)
    upper = lower + 1
    middle = (expected[lower.long()] + expected[upper.long()]) / 2
    before = torch.nextafter(middle, torch.full_like(middle, -math.inf))
    after = torch.nextafter(middle, torch.full_like(middle, math.inf))
    tie = torch.where(lower % 2 == 0, lower, upper)
    for points, magnitudes in ((before, lower), (middle, tie), (after, upper)):
        assert torch.equal(snap_e2m5(points), magnitudes)
        assert torch.equal(snap_e2m5(-points), magnitudes | 128)
    extremes = torch.tensor([-100000, -7.75, -0.0, 0.0, 7.75, 100000])
    assert snap_e2m5(extremes).tolist() == [254, 254, 128, 0, 126, 126]


@pytest.mark.parametrize("bits", range(1, 7))
@pytest.mark.parametrize("block_size", (128, 256))
def test_packing_and_decoder_match_independent_oracle(bits, block_size):
    spec = CPTCSpec(bits=bits, block_size=block_size)
    k = block_size + 11
    for unified in (False, True):
        operands = _operands(spec, k, 3, unified=unified)
        states = unpack_states(operands[0], spec)
        assert torch.equal(pack_states(states, spec), operands[0])
        torch.testing.assert_close(decode_weight(*operands, spec, k), _decode_oracle(*operands, spec, k), atol=0, rtol=0)


def test_scalar_trellis_threshold_and_tail_biting():
    scalar = CPTCSpec(bits=1, trellis_depth=1, table_q=1)
    midpoint = torch.tensor(0.5)
    before = torch.nextafter(midpoint, torch.tensor(0.0))
    after = torch.nextafter(midpoint, torch.tensor(1.0))
    targets = torch.tensor([before, midpoint, after, 0, 1, 1, 1, 0]).repeat(16).view(1, 128)
    # Make the rounding-threshold samples the first state of their own block
    # to avoid FP32 accumulation turning neighbors into path-cost ties.
    scalar_targets = targets[:, :3].T.contiguous().repeat(1, 128)
    states = viterbi_encode(scalar_targets, torch.tensor([0.0, 1.0]), scalar)
    assert states[:, 0].tolist() == [0, 0, 1]

    spec = CPTCSpec(bits=2, trellis_depth=7)
    rng = torch.Generator().manual_seed(39)
    # A known closed path with a distinct value for every state has zero
    # distortion and is the unique optimum conditional on that same seam.
    codes = torch.randint(4, (128,), generator=rng)
    whole = sum(int(code) << (2 * t) for t, code in enumerate(codes))
    ring = whole | (whole << 256)
    path = torch.tensor([(ring >> (2 * t)) & 127 for t in range(128)], dtype=torch.int32)[None]
    codebook = (torch.randperm(128, generator=rng) - 64).float() / 32
    assert torch.equal(viterbi_encode(codebook[path.long()], codebook, spec), path)
    assert bytes(pack_states(path, spec)[0].tolist()) == whole.to_bytes(32, "little")


@pytest.mark.parametrize("device_name", ("cpu", "native"))
def test_tail_biting_optimizes_all_states_at_fixed_overlap(device_name, request):
    device, ops = request.getfixturevalue("cptc_gpu") if device_name == "native" else (torch.device("cpu"), None)
    spec = CPTCSpec(bits=3, trellis_depth=16)
    torch_rng = torch.Generator().manual_seed(617)
    targets = torch.randn((64, 128), generator=torch_rng)[[18, 36, 51]].to(device) * 2
    coefficients = torch.tensor(spec.coefficients(), dtype=torch.int32, device=device)
    states = torch.arange(65536, device=device)
    table = gaussian_table(spec, device)
    values = torch.tensor([_e2m5_oracle(v) for v in table.tolist()], device=device)
    values = torch.cat((values, -values[:-1].flip(0)))[state_indices(states, coefficients, spec.shift)]
    paths = viterbi_encode(targets, values.half() if ops else values, spec, native=ops).long()
    assert torch.equal(paths, unpack_states(pack_states(paths, spec), spec))

    # The paper's reverse recursion groups its running costs by the low
    # L-K bits. Compare the exact optimum conditioned on the encoded overlap.
    overlap = paths[:, 0] & 8191
    cost = (targets[:, -1:].double() - values.double()).square()
    high = states >> spec.bits
    cost.masked_fill_(high != overlap[:, None], math.inf)
    for target in targets[:, :-1].flip(1).T:
        best = cost.reshape(-1, 8, 8192).amin(dim=1)
        cost = best[:, high] + (target[:, None].double() - values.double()).square()
    optimum = cost.reshape(-1, 8, 8192).amin(dim=1).gather(1, overlap[:, None]).flatten()
    actual = (targets.double() - values[paths].double()).square().sum(dim=1)
    torch.testing.assert_close(actual, optimum, rtol=0, atol=1e-10)


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("device_name", ("cpu", "native"))
def test_block_scale_preserves_isolated_outliers(bits, device_name, request):
    device = request.getfixturevalue("cptc_gpu")[0] if device_name == "native" else torch.device("cpu")
    source = torch.nn.Linear(145, 2, bias=False, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        source.weight[0].fill_(0.125)
        source.weight[0, 0] = 64
        source.weight[0, 144] = -64
        source.weight[1].zero_()
    layer = CPTCLinear(bits, 145, 2, dtype=source.weight.dtype).to(device)
    layer.pack_original(source, device=device)
    assert layer.cptc_scale_bits[1].tolist() == [0, 0]

    # Each basis vector isolates the coefficient at one side of the stream,
    # including the last valid position in a zero-padded block.
    inputs = torch.zeros((2, 145), dtype=source.weight.dtype, device=device)
    inputs[0, 0] = inputs[1, 144] = 1
    expected = source(inputs).detach()
    with torch.inference_mode():
        result = layer(inputs)
    torch.testing.assert_close(result, expected, atol=0, rtol=0.08)


@pytest.mark.parametrize("bits", range(1, 7))
def test_native_decoder_and_encoder(cptc_gpu, bits):
    device, ops = cptc_gpu
    spec = CPTCSpec(bits=bits, block_size=256)
    k = 293
    operands = _operands(spec, k, 17, unified=bits == 4)
    actual = ops.decode(*(t.to(device) for t in operands), bits, spec.trellis_depth, spec.shift, k)
    torch.testing.assert_close(actual.cpu(), _decode_oracle(*operands, spec, k), atol=0, rtol=0)

    indices = state_indices(torch.arange(65536), operands[2], spec.shift)
    full_table = torch.tensor([_e2m5_oracle(v) for v in operands[1].tolist()])
    full_table = torch.cat((full_table, -full_table[:-1].flip(0)))
    rng = torch.Generator().manual_seed(312)
    values = torch.randn((3, 256), generator=rng)
    decoded = full_table[indices]
    torch_path = viterbi_encode(values, decoded, spec)
    gpu_path = viterbi_encode(values.to(device), decoded.to(device), spec, native=ops).cpu()
    assert torch.equal(gpu_path, torch_path)
    cached_path = viterbi_encode(values.to(device), decoded.to(device, torch.float16), spec, native=ops).cpu()
    assert torch.equal(cached_path, torch_path)
    packed = pack_states(gpu_path, spec)
    assert torch.equal(unpack_states(packed, spec), torch_path)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("rows", (1, 15, 17))
def test_native_linear_matches_fp64_and_keeps_packed_state(cptc_gpu, dtype, rows, monkeypatch):
    device, ops = cptc_gpu
    k, n = 1031, 71
    spec = CPTCSpec(bits=3, trellis_depth=10)
    layer = CPTCLinear(spec.bits, k, n, True, trellis_depth=spec.trellis_depth).to(device)
    packed, table, coefficients, scales = _operands(spec, k, n)
    for name, tensor in zip(layer.CHECKPOINT_REQUIRED_BUFFERS, (packed, table, coefficients, scales.view(torch.int32))):
        getattr(layer, name).copy_(tensor)
    layer.bias.copy_(torch.linspace(-0.10003, 0.10003, n))
    layer.post_init()
    weight = _decode_oracle(packed, table, coefficients, scales, spec, k).double()
    rng = torch.Generator().manual_seed(633)
    x = torch.randn((rows, k, 1), generator=rng).transpose(1, 2).to(dtype=dtype, device=device)
    y64 = (x.cpu().double() @ weight.T) + layer.bias.cpu().double()
    before = {name: tensor.clone() for name, tensor in layer.state_dict().items()}

    calls = []

    class NativeCalls:
        def mma(self, *args):
            calls.append("mma")
            wide = ops.mma(*args)
            torch.testing.assert_close(wide.cpu(), (y64 - layer.bias.cpu().double()).view(rows, n).float(), atol=2e-4, rtol=2e-4)
            return wide

        def decode(self, *args):
            calls.append("decode")
            return ops.decode(*args)

    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.cptc.native_ops", lambda _: NativeCalls())
    for _ in range(3):
        result = layer(x).cpu()
        assert result.dtype == dtype
        torch.testing.assert_close(result, y64.to(dtype), atol=2e-3, rtol=2e-3)
    expected_op = "mma" if rows <= 16 and dtype != torch.float32 else "decode"
    assert calls == [expected_op] * 3
    assert layer.state_dict().keys() == before.keys()
    for name, actual in layer.state_dict().items():
        assert torch.equal(actual, before[name])
    assert sum(t.nbytes for t in layer.list_buffers()) == sum(t.nbytes for t in before.values())
    assert not list(layer.iter_runtime_weight_tensors())
    assert not any(isinstance(value, torch.Tensor) for value in vars(layer).values())


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_cpu_linear_fp32_bias_and_empty_input(dtype):
    layer = CPTCLinear(4, 128, 4, True)
    layer.cptc_table.fill_(32)  # +1.0 in E2M5
    layer.cptc_scale_bits.fill_(torch.tensor(1.0).view(torch.int32).item())
    layer.bias.fill_(-65535)
    x = torch.full((2, 128), 512, dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(x)
    torch.testing.assert_close(output, torch.ones((2, 4), dtype=dtype), rtol=0, atol=0)
    assert layer(x[:0]).shape == (0, 4)
    assert layer.bias.dtype == torch.float32 and layer.bias[0] == -65535


def test_effective_bits_matches_paper_design_point(cptc_gpu):
    device, ops = cptc_gpu
    spec = CPTCSpec(bits=4, block_size=256)
    table = gaussian_table(spec, device)
    values = decode_e2m5(table)
    full_table = torch.cat((values, -values[:-1].flip(0)))
    indices = state_indices(torch.arange(65536, device=device), spec.coefficients(), spec.shift)
    decoded = full_table[indices]
    weights = torch.randn((1024, 256), generator=torch.Generator().manual_seed(12345)).to(device)
    states = viterbi_encode(weights * 2, decoded, spec, native=ops)
    mse = (decoded[states.long()] / 2 - weights).square().mean().item()
    assert -0.5 * math.log2(mse) == pytest.approx(3.934, abs=0.025)


def test_bf16_native_unscaled_dot_overflow(cptc_gpu):
    device, ops = cptc_gpu
    spec = CPTCSpec()
    packed = torch.zeros((2, 1, 64), dtype=torch.uint8, device=device)
    table = torch.full((64,), 32, dtype=torch.uint8, device=device)
    coeffs = torch.tensor(spec.coefficients(), dtype=torch.int32, device=device)
    scales = torch.full((2, 1), 2.0**-126, device=device)
    x = torch.full((1, 128), 2.0**127, dtype=torch.bfloat16, device=device)
    y = ops.mma(x, packed, table, coeffs, scales, spec.bits, spec.trellis_depth, spec.shift)
    torch.testing.assert_close(y.cpu(), torch.full((1, 2), 256.0), rtol=0, atol=0)


def test_config_format_resolution_and_architecture(monkeypatch):
    raw = CPTCConfig(bits=4, trellis_depth=9, block_size=256, offload_to_disk=False).to_dict()
    parsed = QuantizeConfig.from_quant_config(raw)
    assert (parsed.method, parsed.format) == (METHOD.CPTC, FORMAT.CPTC)
    assert parsed.quant_linear_init_kwargs()["block_size"] == 256
    assert parsed.uses_weight_only_lifecycle()
    with pytest.raises(ValueError, match="trellis_depth"):
        CPTCConfig(bits=4, trellis_depth=3, offload_to_disk=False)
    with pytest.raises(ValueError, match="group_size"):
        CPTCConfig(group_size=128, offload_to_disk=False)
    kernel = select_quant_linear(bits=4, group_size=-1, desc_act=False, sym=True,
                                 backend=BACKEND.AUTO, format=FORMAT.CPTC, quant_method=METHOD.CPTC,
                                 pack=True, device=DEVICE.CPU, pack_dtype=torch.int32)
    assert kernel is CPTCLinear

    fake_arches = {0: "gfx942", 1: "gfx1201"}
    monkeypatch.setattr(cptc, "normalize_device", lambda _: DEVICE.ROCM)
    monkeypatch.setattr(cptc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cptc, "rocm_arch", lambda device: fake_arches[device.index])
    arch.clear_arch_cache()
    assert not cptc.native_supported(torch.device("cpu"))
    assert not cptc.native_supported(torch.device("cuda:0"))
    assert cptc.native_supported(torch.device("cuda:1"))


@pytest.mark.parametrize("pack_on_gpu", (False, True), ids=("cpu-encode", "native-encode"))
@pytest.mark.parametrize("yaqa", (False, True), ids=("direct", "yaqa"))
def test_gptqmodel_cptc_save_and_transformers_load(tmp_path, pack_on_gpu, yaqa, request, monkeypatch):
    dense, quantized = tmp_path / "dense", tmp_path / "quantized"
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=128, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                         attention_bias=True)
    torch.manual_seed(415)
    original = LlamaForCausalLM(config)
    name = "model.layers.0.self_attn.q_proj"
    original.get_submodule(name).bias.data.fill_(0.10003)
    original.save_pretrained(dense)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(dense)
    pack_device = request.getfixturevalue("cptc_gpu")[0] if pack_on_gpu else torch.device("cpu")
    qcfg = CPTCConfig(bits=4, trellis_depth=8, yaqa=yaqa, offload_to_disk=False, device=pack_device,
                      dynamic={"+:.*q_proj$": {"bits": 3}})
    model = GPTQModel.load(str(dense), quantize_config=qcfg, dtype=torch.float32)
    encoder_calls = []

    def encode(blocks, decoded, spec, *, native):
        encoder_calls.append((blocks.device, native is not None))
        return viterbi_encode(blocks, decoded, spec, native=native)

    monkeypatch.setattr("gptqmodel.quantization.cptc.viterbi_encode", encode)
    calibration = CalibrationConfig(dataset=[[1, 3, 5, 7, 9]], min_length=1) if yaqa else None
    if yaqa:
        with pytest.raises(ValueError, match="CalibrationConfig.dataset is required"):
            model.quantize()
        assert not encoder_calls
    model.quantize(calibration)
    assert encoder_calls and set(encoder_calls) == {(pack_device, pack_on_gpu)}
    model.save(str(quantized))

    load_options = {"device_map": "cpu", "dtype": torch.bfloat16, "local_files_only": True}
    reloaded = GPTQModel.load(str(quantized), **load_options).model
    transformers_model = AutoModelForCausalLM.from_pretrained(quantized, **load_options)
    for loaded in (reloaded, transformers_model):
        projection = loaded.get_submodule(name)
        assert type(projection) is CPTCLinear and projection.bits == 3
        assert projection.yaqa == yaqa
        source = model.model.get_submodule(name)
        for key, tensor in source.state_dict().items():
            restored = projection.state_dict()[key]
            assert tensor.dtype == restored.dtype
            assert torch.equal(tensor, restored)

    ids = torch.tensor([[1, 3, 5, 7]])
    with torch.inference_mode():
        expected = reloaded(input_ids=ids, use_cache=False).logits
        actual = transformers_model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    checkpoint = quantized / "model.safetensors"
    with safe_open(checkpoint, framework="pt") as handle:
        metadata = handle.metadata()
    assert {key: metadata[key] for key in ("format", "quant_method", "cptc_depth", "cptc_block_size")} == {
        "format": "pt", "quant_method": "cptc", "cptc_depth": "8", "cptc_block_size": "128",
    }
    payload = load_file(checkpoint)
    deleted = "cptc_input_signs" if yaqa else "cptc_packed"
    payload.pop(f"{name}.{deleted}")
    save_file(payload, checkpoint, metadata={"format": "pt"})
    with pytest.raises(ValueError, match=f"Missing required.*{deleted}"):
        AutoModelForCausalLM.from_pretrained(quantized, **load_options)


def test_yaqa_partial_trace_and_tile_feedback():
    rng = torch.Generator().manual_seed(519)
    x = torch.randn((2, 11, 48), generator=rng)
    dy = torch.randn((2, 11, 32), generator=rng)
    sketch = HessianSketch.empty(48, 32, x.device)
    sketch.add(x, dy)
    weight_grads = dy.double().mT @ x.double()
    for observed, expected in (
        (sketch.input, (weight_grads.mT @ weight_grads).sum(dim=0)),
        (sketch.output, (weight_grads @ weight_grads.mT).sum(dim=0)),
    ):
        assert torch.linalg.norm(observed.double() - expected) / torch.linalg.norm(expected) < 1e-6

    # Use power-of-two matrices so the explicit normalized Walsh transform is
    # an independent oracle for the factorization and congruence.
    factors = []
    for size, tile, value in ((32, 16, sketch.input[:32, :32]), (32, 8, sketch.output)):
        signs = torch.tensor([-1, 1] * (size // 2), dtype=torch.int8)
        factor = block_feedback(value, signs, tile, 0.01).double()
        h = _walsh(size) * signs
        hessian = h @ value.double() @ h.T
        hessian = hessian / hessian.diag().mean() + 0.01 * torch.eye(size, dtype=torch.float64)
        chol = torch.linalg.cholesky(hessian)
        d = torch.block_diag(*(chol[i:i + tile, i:i + tile] for i in range(0, size, tile)))
        unit = factor + torch.eye(size, dtype=torch.float64)
        assert torch.linalg.norm(unit @ d @ d.T @ unit.T - hessian) / torch.linalg.norm(hessian) < 1e-6
        assert factor.abs().sum() > 0
        factors.append(unit)

    input_factor, output_factor = factors
    weight = torch.randn((32, 32), generator=rng, dtype=torch.float64)
    residuals = torch.zeros_like(weight)
    visited = set()

    def encode(target, row_tiles, col_tiles):
        for block, row, col in zip(target, row_tiles.tolist(), col_tiles.tolist(), strict=True):
            y, x = slice(row * 8, (row + 1) * 8), slice(col * 16, (col + 1) * 16)
            projected = output_factor.T @ residuals @ input_factor
            torch.testing.assert_close(block, weight[y, x] + projected[y, x], atol=1e-10, rtol=1e-10)
            # Apply a coarse deterministic quantizer so previous tiles have
            # nonzero, off-diagonal contributions in the remaining targets.
            residuals[y, x] = weight[y, x] - (block * 2).round() / 2
            visited.add((row, col))
        return (target * 2).round() / 2

    _wavefront_tiles(weight, input_factor - torch.eye(32), output_factor - torch.eye(32), 8, encode)
    assert len(visited) == 8


@pytest.mark.parametrize("bits", (3, 4))
@pytest.mark.parametrize("block_size", (128, 256))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_yaqa_tile_mma_matches_dense_oracle(cptc_gpu, bits, block_size, dtype):
    device, ops = cptc_gpu
    n, k = 32, 64
    rng = torch.Generator().manual_seed(739)
    source = torch.nn.Linear(k, n, bias=True, dtype=torch.float32)
    sketch = HessianSketch.empty(k, n, "cpu")
    sketch.add(torch.randn((2, 23, k), generator=rng), torch.randn((2, 23, n), generator=rng))
    layer = CPTCLinear(bits, k, n, True, block_size=block_size, yaqa=True).to(device)
    layer.pack_original(source, sketch=sketch, device=device)
    layer.post_init()
    payload = tuple(getattr(layer, name).cpu() for name in CPTCLinear.CHECKPOINT_REQUIRED_BUFFERS)
    expected = _decode_oracle(*payload[:3], payload[3].view(torch.float32), layer.spec,
                              k * layer.tile_rows)
    expected = expected.view(n // layer.tile_rows, k // 16, layer.tile_rows, 16)
    expected = expected.transpose(1, 2).reshape(n, k).double()
    actual = layer._decode_rows(0, n, device, ops)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=0, rtol=0)
    torch.testing.assert_close(layer._decode_rows(0, n, torch.device("cpu"), None).double(), expected, atol=0, rtol=0)
    u = _walsh(k) * layer.cptc_input_signs.cpu()
    v = _walsh(n) * layer.cptc_output_signs.cpu()
    torch.testing.assert_close(layer.dequantize_weight().cpu().double(), v.T @ expected @ u, atol=1e-6, rtol=1e-6)
    inputs = torch.randn((17, k), generator=rng).to(dtype)
    target = (((inputs.double() @ u.T).to(dtype).double() @ expected.T) @ v + source.bias.double()).to(dtype)
    with torch.inference_mode():
        for rows in (1, 16, 17):
            actual = layer(inputs[:rows].to(device)).cpu()
            torch.testing.assert_close(actual, target[:rows], atol=2e-3, rtol=2e-3)
