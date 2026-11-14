# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-call host overhead in kernel dispatch (#3983).

Every change here must be bitwise neutral: the same kernels run with the same
arguments, only the Python work in front of them shrinks. The tests pin the
memoised fast paths, their invalidation, the ExLlamaV2 handle lifecycle, and
the single-kernel FP16 bias epilogue.
"""

import types

import pytest
import torch

import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb_module
import gptqmodel.nn_modules.qlinear.exllamav2 as exllamav2
import gptqmodel.nn_modules.qlinear.gemv_awq as gemv_awq
import gptqmodel.nn_modules.qlinear.gemv_fast_awq as gemv_fast_awq
import gptqmodel.utils.awq as awq_utils
import gptqmodel.utils.exllamav2 as exllamav2_utils
import gptqmodel.utils.machete as machete_utils
import gptqmodel.utils.marlin as marlin_utils
import gptqmodel.utils.qqq as qqq_utils
from gptqmodel import extension as extension_api


class _FakeLoadedExtension:
    """Loaded-extension stand-in exposing the real fast-path attributes."""

    def __init__(self, ops, *, loaded=True):
        self._ops = dict(ops)
        self._namespace_cache = types.SimpleNamespace(**self._ops) if loaded else None
        self._op_cache = {}
        self.load_calls = 0
        self.clear_calls = 0

    def load(self, *args, **kwargs):
        self.load_calls += 1
        return True

    def _ops_available(self):
        return True

    def op(self, op_name):
        return self._ops[op_name]

    def namespace_object(self):
        return self._namespace_cache

    def clear_cache(self):
        self.clear_calls += 1
        self._namespace_cache = None
        self._op_cache = {}


def _count_loader(monkeypatch, calls):
    real_op = extension_api.op

    def counting(name, op_name, **kwargs):
        calls.append((name, op_name))
        return real_op(name, op_name, **kwargs)

    monkeypatch.setattr(extension_api, "op", counting)


def test_awq_dispatch_uses_loaded_handle_without_loader(monkeypatch):
    handle = lambda *args, **kwargs: "awq-out"  # noqa: E731
    extension = _FakeLoadedExtension({"gemm_forward": handle})
    monkeypatch.setattr(awq_utils, "_AWQ_TORCH_OPS_EXTENSION", extension)
    calls = []
    _count_loader(monkeypatch, calls)

    assert awq_utils.awq_gemm_forward(None, None, None, None, 1, False) == "awq-out"
    assert awq_utils.awq_gemm_forward(None, None, None, None, 1, False) == "awq-out"
    assert calls == []


def test_awq_dispatch_falls_back_after_cache_clear(monkeypatch):
    handle = lambda *args, **kwargs: "awq-out"  # noqa: E731
    extension = _FakeLoadedExtension({"gemm_forward": handle})
    monkeypatch.setattr(awq_utils, "_AWQ_TORCH_OPS_EXTENSION", extension)

    awq_utils.clear_awq_extension_cache()
    assert extension.clear_calls == 1

    # After the clear the cached namespace is gone, so the wrapper must go
    # through the loader again instead of returning the memoised handle.
    calls = []

    def fake_op(name, op_name):
        calls.append((name, op_name))
        return handle

    monkeypatch.setattr(awq_utils, "_extension_api", lambda: types.SimpleNamespace(op=fake_op))
    assert awq_utils.awq_gemm_forward(None, None, None, None, 1, False) == "awq-out"
    assert calls == [("awq", "gemm_forward")]


@pytest.mark.parametrize(
    "module,global_name,op_name",
    [
        (marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", "marlin_scratch_sizes"),
        (qqq_utils, "_QQQ_TORCH_OPS_EXTENSION", "qqq_gemm"),
        (machete_utils, "_MACHETE_TORCH_OPS_EXTENSION", "machete_mm"),
        (exllamav2_utils, "_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION", "make_q_matrix"),
    ],
)
def test_dispatch_uses_loaded_handle_without_loader(monkeypatch, module, global_name, op_name):
    handle = lambda *args, **kwargs: "out"  # noqa: E731
    extension = _FakeLoadedExtension({op_name: handle})
    monkeypatch.setattr(module, global_name, extension)
    calls = []
    _count_loader(monkeypatch, calls)

    if module is marlin_utils:
        resolved = marlin_utils._marlin_resolve_op(dtype=torch.float16, op_name=op_name)
    elif module is qqq_utils:
        resolved = qqq_utils._qqq_op(op_name)
    elif module is machete_utils:
        resolved = machete_utils._machete_op(op_name)
    else:
        resolved = exllamav2_utils._exllamav2_op("exllamav2", op_name)

    assert resolved is handle
    assert calls == []


def test_marlin_dispatch_respects_patched_extension_api(monkeypatch):
    """The memo must never shadow a test's fake ``_extension_api``."""
    real_handle = lambda *args, **kwargs: "real"  # noqa: E731
    fake_handle = lambda *args, **kwargs: "fake"  # noqa: E731
    extension = _FakeLoadedExtension({"marlin_scratch_sizes": real_handle})
    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", extension)
    monkeypatch.setattr(
        marlin_utils,
        "_extension_api",
        lambda: types.SimpleNamespace(op=lambda name, op_name: fake_handle),
    )

    resolved = marlin_utils._marlin_resolve_op(dtype=torch.float16, op_name="marlin_scratch_sizes")

    assert resolved is fake_handle


# ---------------------------------------------------------------------------
# ExLlamaV2 handle lifecycle
# ---------------------------------------------------------------------------

def _exllamav2_layer():
    layer = types.SimpleNamespace(
        qweight=torch.zeros((2, 2), dtype=torch.int32),
        qzeros=torch.zeros((1, 2), dtype=torch.int32),
        scales=torch.ones((1, 2), dtype=torch.float16),
        g_idx=torch.zeros(8, dtype=torch.int32),
        q_handle=None,
        q_tensors=None,
        _q_handle_key=None,
        _q_fast=None,
    )
    made = []

    def make_q_matrix(tensors, **_kwargs):
        made.append(tensors["qweight"].clone())
        return len(made)

    layer.ext_make_q_matrix = make_q_matrix
    layer._release_q_handle = lambda: exllamav2.ExllamaV2Linear._release_q_handle(layer)
    return layer, made


def test_exllamav2_handle_rebuild_counts_are_unchanged(monkeypatch):
    """#3733/#3771: forward, forward, replace, mul_, .data=, inference forward."""
    freed = []
    monkeypatch.setattr(exllamav2, "exllamav2_free_q_matrix", freed.append)
    layer, made = _exllamav2_layer()
    ensure = lambda: exllamav2.ExllamaV2Linear._ensure_q_handle(layer)  # noqa: E731

    # ``post_init`` builds the first handle before any forward runs.
    ensure()
    assert len(made) == 1

    counts = []
    ensure()
    counts.append(len(made) - 1)
    ensure()
    counts.append(len(made) - 1)
    layer.qweight = torch.ones_like(layer.qweight)
    ensure()
    counts.append(len(made) - 1)
    layer.qweight.mul_(2)
    ensure()
    counts.append(len(made) - 1)
    layer.qweight = layer.qweight.data.clone()
    ensure()
    counts.append(len(made) - 1)
    with torch.inference_mode():
        ensure()
    counts.append(len(made) - 1)

    assert counts == [0, 0, 1, 2, 3, 3]


def test_exllamav2_fast_path_skips_key_rebuild(monkeypatch):
    keys = []
    real_key = exllamav2.tensor_cache_key

    def counting_key(tensor):
        keys.append(tensor)
        return real_key(tensor)

    monkeypatch.setattr(exllamav2, "tensor_cache_key", counting_key)
    layer, made = _exllamav2_layer()
    ensure = lambda: exllamav2.ExllamaV2Linear._ensure_q_handle(layer)  # noqa: E731

    ensure()
    assert len(made) == 1
    first_keys = len(keys)
    assert first_keys > 0

    ensure()
    ensure()

    assert len(keys) == first_keys
    assert len(made) == 1


def test_exllamav2_fast_path_detects_in_place_write(monkeypatch):
    freed = []
    monkeypatch.setattr(exllamav2, "exllamav2_free_q_matrix", freed.append)
    layer, made = _exllamav2_layer()
    ensure = lambda: exllamav2.ExllamaV2Linear._ensure_q_handle(layer)  # noqa: E731

    ensure()
    layer.qweight.mul_(3)
    ensure()

    assert len(made) == 2
    assert freed == [1]
    assert made[1].eq(0).all()


# ---------------------------------------------------------------------------
# bitsandbytes availability + quant-state signature
# ---------------------------------------------------------------------------

def test_bitsandbytes_version_validated_once(monkeypatch):
    parses = []
    real_parse = bnb_module.version.parse

    def counting_parse(text):
        parses.append(text)
        return real_parse(text)

    monkeypatch.setattr(bnb_module.version, "parse", counting_parse)
    fake_bnb = types.SimpleNamespace(__version__="0.49.0")
    monkeypatch.setitem(__import__("sys").modules, "bitsandbytes", fake_bnb)
    bnb_module._clear_availability_state()

    assert bnb_module.import_bitsandbytes() is fake_bnb
    assert bnb_module.import_bitsandbytes() is fake_bnb

    # One validated version comparison per process, not per forward.
    assert len(parses) == 2
    assert bnb_module._VALIDATED_BITSANDBYTES is fake_bnb

    bnb_module._clear_availability_state()
    assert bnb_module._VALIDATED_BITSANDBYTES is None


def test_bitsandbytes_quant_state_signature_uses_tensor_identity_fields(monkeypatch):
    built = []

    class _QuantState:
        @classmethod
        def from_dict(cls, payload, device):
            built.append(device)
            return object()

    fake_bnb = types.SimpleNamespace(functional=types.SimpleNamespace(QuantState=_QuantState))
    monkeypatch.setattr(bnb_module, "import_bitsandbytes", lambda: fake_bnb)

    state = types.SimpleNamespace(
        is_4bit=True,
        _cpu_packing_active=False,
        weight=torch.zeros(4, 4),
        weight_absmax=torch.zeros(1),
        weight_quant_map=torch.zeros(2),
        weight_quant_state=torch.zeros(3),
        quant_state=None,
        _quant_state_signature=None,
        _quant_state_payload=lambda: {},
    )
    refresh = lambda: bnb_module.BitsAndBytesLinear._refresh_quant_state(state)  # noqa: E731

    refresh()
    refresh()
    assert len(built) == 1
    assert state._quant_state_signature == (
        (torch.Size([4, 4]), torch.device("cpu"), torch.float32),
        (torch.Size([1]), torch.device("cpu"), torch.float32),
        (torch.Size([2]), torch.device("cpu"), torch.float32),
        (torch.Size([3]), torch.device("cpu"), torch.float32),
    )

    # A new shape invalidates the cached QuantState; a same-shape replacement
    # did not invalidate the pre-existing signature either.
    state.weight = torch.zeros(4, 8)
    refresh()
    assert len(built) == 2


# ---------------------------------------------------------------------------
# AWQ bias epilogue
# ---------------------------------------------------------------------------

def _count_float_conversions(monkeypatch):
    calls = {"float": 0}
    original = torch.Tensor.float

    def counting_float(self):
        calls["float"] += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, "float", counting_float)
    return calls


def test_awq_gemv_bias_epilogue_is_a_single_fp16_add(monkeypatch):
    module = gemv_awq.AwqGEMVLinear(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=True,
        register_buffers=True,
    )
    kernel_out = torch.randn(1, 64, dtype=torch.float16)
    monkeypatch.setattr(gemv_awq, "awq_gemv_forward", lambda *args: kernel_out)
    monkeypatch.setattr(gemv_awq, "awq_gemmv2_forward", lambda *args: kernel_out)

    calls = _count_float_conversions(monkeypatch)
    out = module(torch.randn(1, 64, dtype=torch.float16))

    oracle = (kernel_out.to(torch.float32) + module.bias.to(torch.float32)).to(torch.float16)
    assert calls["float"] == 0
    assert torch.equal(out, oracle)


def test_awq_gemv_bias_epilogue_keeps_wide_path_for_bf16(monkeypatch):
    module = gemv_awq.AwqGEMVLinear(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=True,
        register_buffers=True,
    )
    kernel_out = torch.randn(1, 64, dtype=torch.float16)
    monkeypatch.setattr(gemv_awq, "awq_gemv_forward", lambda *args: kernel_out)
    monkeypatch.setattr(gemv_awq, "awq_gemmv2_forward", lambda *args: kernel_out)

    calls = _count_float_conversions(monkeypatch)
    out = module(torch.randn(1, 64, dtype=torch.bfloat16))

    oracle = (kernel_out.to(torch.float32) + module.bias.to(torch.float32)).to(torch.bfloat16)
    assert calls["float"] == 2
    assert torch.equal(out, oracle)


def test_awq_gemv_fast_bias_epilogue_is_a_single_fp16_add(monkeypatch):
    module = gemv_fast_awq.AwqGEMVFastLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=True,
        register_buffers=True,
    )
    kernel_out = torch.randn(1, 128, dtype=torch.float16)
    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", lambda *args: kernel_out)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", lambda *args: kernel_out)

    calls = _count_float_conversions(monkeypatch)
    out = module(torch.randn(1, 128, dtype=torch.float16))

    oracle = (kernel_out.to(torch.float32) + module.bias.to(torch.float32)).to(torch.float16)
    assert calls["float"] == 0
    assert torch.equal(out, oracle)


def test_awq_gemv_fast_bias_epilogue_keeps_wide_path_for_bf16_input(monkeypatch):
    module = gemv_fast_awq.AwqGEMVFastLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=True,
        register_buffers=True,
    )
    kernel_out = torch.randn(1, 128, dtype=torch.float16)
    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", lambda *args: kernel_out)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", lambda *args: kernel_out)

    calls = _count_float_conversions(monkeypatch)
    out = module(torch.randn(1, 128, dtype=torch.bfloat16))

    oracle = (kernel_out.to(torch.float32) + module.bias.to(torch.float32)).to(torch.bfloat16)
    assert calls["float"] == 2
    assert torch.equal(out, oracle)


def _gemv_fast_module(bias: bool = False) -> gemv_fast_awq.AwqGEMVFastLinear:
    return gemv_fast_awq.AwqGEMVFastLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=bias,
        register_buffers=True,
    )


def test_gemv_fast_runtime_check_raises_on_first_call(monkeypatch):
    module = _gemv_fast_module()
    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: False)
    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_error", lambda: "missing awq jit ops")

    with pytest.raises(ModuleNotFoundError, match="missing awq jit ops"):
        module(torch.randn(1, 128, dtype=torch.float16))

    assert not getattr(module, "_runtime_ok", False)


def test_gemv_fast_runtime_check_is_memoized_after_success(monkeypatch):
    module = _gemv_fast_module()
    probes = {"n": 0}
    kernel_out = torch.randn(1, 128, dtype=torch.float16)

    def available():
        probes["n"] += 1
        return True

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", available)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", lambda *args: kernel_out)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", lambda *args: kernel_out)

    module(torch.randn(1, 128, dtype=torch.float16))
    module(torch.randn(1, 128, dtype=torch.float16))
    module(torch.randn(1, 128, dtype=torch.float16))

    assert probes["n"] == 1
    assert module._runtime_ok is True


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_add_matches_widened_add_bitwise():
    """The single-kernel epilogue must reproduce the FP32 add exactly."""
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(0)

    total = 0
    for _ in range(8):
        n = 262_144
        scale = torch.exp(torch.randn(n, device=device, generator=generator) * 3)
        a = (torch.randn(n, device=device, generator=generator) * scale).half()
        b = (torch.randn(n, device=device, generator=generator) * scale).half()
        wide = (a.float() + b.float()).half()
        assert torch.equal(wide.view(torch.int16), torch.add(a, b).view(torch.int16))
        total += n

    values = torch.tensor(
        [1.0, 0.5, 0.25, 3.0, 0.1, -0.1, 65504.0, -65504.0, 1e-7, -1e-7, 0.0, -0.0],
        dtype=torch.float16,
        device=device,
    )
    a = values.repeat_interleave(values.numel())
    b = values.repeat(values.numel())
    wide = (a.float() + b.float()).half()
    assert torch.equal(wide.view(torch.int16), torch.add(a, b).view(torch.int16))
    total += a.numel()

    assert total > 2_000_000
