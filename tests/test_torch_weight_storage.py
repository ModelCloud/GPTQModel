"""Storage gates with independent FP32 matmuls after native weight-dtype rounding."""

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


FORMATS = [(bits, FORMAT.GPTQ_P) for bits in range(2, 9)] + [
    (bits, FORMAT.GPTQ_V2) for bits in (2, 3, 4, 8)
]


def build_case(bits=4, dtype=torch.float16, group_size=32, fmt=FORMAT.GPTQ_V2,
               in_features=512, out_features=64):
    generator = torch.Generator().manual_seed(821 + bits)
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    codes = torch.randint(0, 1 << bits, (in_features, out_features), generator=generator)
    codes[0] = 0
    codes[1] = (1 << bits) - 1
    zeros = torch.randint(0, 1 << bits, (groups, out_features), generator=generator)
    scales = torch.randint(1, 4, (groups, out_features), generator=generator).float() / (1 << (bits + 3))
    indices = (torch.randperm(in_features, generator=generator) // effective_group).int()
    weight = (codes - zeros[indices]) * scales[indices]
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    linear.weight.data.copy_(weight.t())
    module = TorchLinear(bits=bits, group_size=group_size, sym=False, desc_act=True,
                         in_features=in_features, out_features=out_features,
                         bias=False, format=fmt)
    module.pack_block(linear, scales.t(), zeros.t().float(), indices)
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        setattr(module, name, getattr(module, name).clone())
    module._init_wf_unsqueeze_buffers()
    module.eval().to(dtype)
    return module, weight


def state_tensors(module):
    visited = set()

    def collect(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, torch.nn.Module):
            yield from collect(vars(value))
        elif isinstance(value, dict):
            for child in value.values():
                yield from collect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from collect(child)

    return list(collect(module))


def track_dequant(module, monkeypatch):
    references = []
    original = module.dequantize_weight

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize_weight", tracked)
    return references


@pytest.mark.parametrize("bits,fmt", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128, 256, 512])
def test_eager_oracle_and_no_retained_weights(monkeypatch, bits, fmt, dtype, group_size):
    module, expected_weight = build_case(bits, dtype, group_size, fmt)
    references = track_dequant(module, monkeypatch)
    original_storage = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}
    with torch.inference_mode():
        for tokens in (1, 7, 25):
            activation = torch.arange(tokens * module.in_features).reshape(tokens, -1).remainder(9).sub(4).to(dtype) / 16
            raw = activation.float() @ expected_weight.to(dtype).float()
            output = module(activation)
            assert output.dtype == dtype
            torch.testing.assert_close(output, raw.to(dtype), rtol=0, atol=0)
            assert references and all(reference() is None for reference in references)
            allowed = original_storage | {
                tensor.untyped_storage().data_ptr()
                for tensor in (module._zeros_cache, module._g_idx_long_cache)
                if tensor is not None
            }
            assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(module))
            if module._zeros_cache is not None:
                assert module._zeros_cache.untyped_storage().nbytes() == module._zeros_cache.numel() * module._zeros_cache.element_size()
            if module._g_idx_long_cache is not None:
                assert module._g_idx_long_cache.untyped_storage().nbytes() == module.in_features * 8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_legacy_opt_ins_cannot_retain_weights(monkeypatch, dtype):
    monkeypatch.setenv("GPTQ_TORCH_CACHE_WEIGHTS", "1")
    monkeypatch.setenv("GPTQ_TORCH_LOOKAHEAD", "1")
    with pytest.warns(FutureWarning, match="Dense weight"):
        module, weight = build_case(dtype=dtype)
        module.enable_weight_cache().enable_lookahead().set_lookahead_next(module)
    assert not module._cache_enabled and not module._lookahead_enabled
    assert module._lookahead_next is None
    references = track_dequant(module, monkeypatch)
    activation = torch.ones(7, module.in_features, dtype=dtype)
    with torch.inference_mode():
        for _ in range(2):
            output = module(activation)
            torch.testing.assert_close(output, (activation.float() @ weight).to(dtype), rtol=0, atol=0)
            assert all(reference() is None for reference in references)


def unpack_oracle(module):
    shifts = torch.arange(32 // module.bits) * module.bits
    codes = ((module.qweight.long()[:, None, :] >> shifts[None, :, None]) & module.maxq).reshape(-1, module.out_features)
    zeros = ((module.qzeros.long()[:, :, None] >> shifts) & module.maxq).reshape(-1, module.out_features)
    return (codes - zeros[module.g_idx.long()]) * module.scales.float()[module.g_idx.long()]


@pytest.mark.parametrize("tensor_name", ["qweight", "qzeros", "scales", "g_idx"])
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_metadata_and_forward_follow_mutations(tensor_name, replacement, compiled):
    module, _ = build_case()
    if compiled:
        module.optimize()
    activation = torch.ones(1, module.in_features, dtype=torch.float16)
    with torch.no_grad():
        first = module(activation)
        tensor = getattr(module, tensor_name)
        if replacement:
            tensor = tensor.clone()
            setattr(module, tensor_name, tensor)
        if tensor_name == "g_idx":
            tensor.add_(1).remainder_(module.scales.shape[0])
        elif tensor_name == "scales":
            tensor.mul_(2)
        else:
            tensor.zero_()
        output = module(activation)
    torch.testing.assert_close(output, (activation.float() @ unpack_oracle(module)).half(), rtol=0, atol=0)
    assert not torch.equal(first, output)


@pytest.mark.parametrize("operation", ["load", "move", "clear"])
def test_metadata_lifecycle_with_inference_tensors(operation):
    with torch.inference_mode():
        module, _ = build_case()
        activation = torch.ones(1, module.in_features, dtype=torch.float16)
        module(activation)
        references = [weakref.ref(module._zeros_cache), weakref.ref(module._g_idx_long_cache)]
        if operation == "load":
            state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
            state["qzeros"].zero_()
            module.load_state_dict(state)
        elif operation == "move":
            module.to(torch.bfloat16)
            activation = activation.bfloat16()
        else:
            module.qzeros.zero_()
            module.g_idx.zero_()
            module.clear_weight_cache()
        assert all(reference() is None for reference in references)
        output = module(activation)
        torch.testing.assert_close(output, (activation.float() @ unpack_oracle(module)).to(activation.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inference_tensors", [False, True])
def test_compiled_dequant_does_not_retain_weights(monkeypatch, dtype, inference_tensors):
    with torch.inference_mode(inference_tensors):
        module, weight = build_case(dtype=dtype)
    module.optimize()
    references = track_dequant(module, monkeypatch)
    with torch.inference_mode():
        for _ in range(3):
            activation = torch.ones(7, module.in_features, dtype=dtype)
            output = module(activation)
            torch.testing.assert_close(output, (activation.float() @ weight).to(dtype), rtol=0, atol=0)
            assert all(reference() is None for reference in references)


def test_stream_workspace_is_per_call():
    module, _ = build_case()
    first = module._stream_get_workspace(torch.device("cpu"), torch.float16, 64)
    second = module._stream_get_workspace(torch.device("cpu"), torch.float16, 64)
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first, second, strict=True))
    references = [weakref.ref(tensor) for tensor in first + second]
    del first, second
    assert all(reference() is None for reference in references)


def test_streaming_uses_eager_fallback_for_staged_weights(monkeypatch):
    module, _ = build_case()
    module.enable_streaming()
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch.HAS_CUDA", True)
    activation = SimpleNamespace(device=torch.device("cuda:0"), shape=(7, module.in_features))
    assert not module._should_use_streaming(activation)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_streaming_schedule_and_tile_lifetime_without_cuda(monkeypatch, dtype):
    module, weight = build_case(dtype=dtype, out_features=320)
    module.enable_streaming(tile_cols=64)
    operations = []

    class Stream:
        def wait_stream(self, other):
            operations.append(("stream", self, other))

        def wait_event(self, event):
            assert event in recorded
            operations.append(("reuse", event))

    class Event:
        def record(self, stream):
            assert stream is compute
            recorded.append(self)

    compute, dequant = Stream(), Stream()
    recorded = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **kwargs: compute)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(module, "_stream_get_dequant_stream", lambda device: dequant)
    references = []
    original = module._stream_get_workspace

    def tracked(*args, **kwargs):
        buffers = original(*args, **kwargs)
        references.extend(weakref.ref(buffer) for buffer in buffers)
        return buffers

    monkeypatch.setattr(module, "_stream_get_workspace", tracked)
    activation = torch.ones(7, module.in_features, dtype=dtype)
    with torch.inference_mode():
        output = module._forward_streaming(activation, (7, 320))
    torch.testing.assert_close(output, weight.sum(0).to(dtype).expand(7, -1), rtol=0, atol=0)
    assert all(reference() is None for reference in references)
    assert operations[0] == ("stream", dequant, compute)
    assert [operation[1] for operation in operations if operation[0] == "reuse"] == recorded[:3]


def test_norm_tuning_restores_streaming_without_dense_opt_ins():
    from gptqmodel.quantization.norm_tuning import _prepare_runtime_for_tuning, _restore_runtime_after_tuning

    module, _ = build_case()
    module.enable_streaming(tile_cols=64)
    module(torch.ones(1, module.in_features, dtype=torch.float16))
    states = _prepare_runtime_for_tuning(module)
    assert not module._streaming_enabled
    assert module._zeros_cache is None and module._g_idx_long_cache is None
    _restore_runtime_after_tuning(states)
    assert module._streaming_enabled and module._stream_tile_cols == 64
    assert not module._cache_enabled and not module._lookahead_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streaming hardware required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_streaming_releases_tiles_on_nondefault_stream(monkeypatch, dtype):
    module, weight = build_case(dtype=dtype, out_features=320)
    module.to("cuda").enable_streaming(tile_cols=64)
    module._triton_dequant_enabled = False
    references = []
    original = module._stream_get_workspace

    def tracked(*args, **kwargs):
        buffers = original(*args, **kwargs)
        references.extend(weakref.ref(tensor) for tensor in buffers)
        return buffers

    monkeypatch.setattr(module, "_stream_get_workspace", tracked)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        activation = torch.ones(7, module.in_features, dtype=dtype, device="cuda")
        for _ in range(3):
            output = module(activation)
            assert all(reference() is None for reference in references)
            torch.empty((module.in_features, 64), dtype=dtype, device="cuda").fill_(99)
            stream.synchronize()
            torch.testing.assert_close(output.cpu(), weight.sum(0).to(dtype).expand(7, -1), rtol=0, atol=0)
