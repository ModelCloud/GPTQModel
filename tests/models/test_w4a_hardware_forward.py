# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tests.models.w4a_hardware_forward import HardwareForward, hardware_value
from tests.w4a_hardware_marks import NVFP4_HARDWARE


def test_hardware_value_is_exact_and_uses_only_surrogate_gradient():
    surrogate = torch.tensor([1.2345, -2.3456], requires_grad=True)
    actual = torch.tensor([8., 3.], requires_grad=True)
    result = hardware_value(surrogate, actual)
    torch.testing.assert_close(result, actual, rtol=0, atol=0)
    result.square().sum().backward()
    torch.testing.assert_close(surrogate.grad, torch.tensor([16., 6.]), rtol=0, atol=0)
    assert actual.grad is None


def test_hardware_value_rejects_broadcasting():
    with pytest.raises(ValueError, match="matching shapes"):
        hardware_value(torch.zeros(3, 2), torch.zeros(2))


def test_producer_capture_covers_boundaries_without_nvfp4_quantizers():
    """A mixed stream stages some residuals as FP8 carriers with no quantizer.

    The student wrapper substitutes the pristine input, post-attention
    residual, and output for every layer, so the capture must not depend on an
    NVFP4 quantizer owning the boundary.
    """
    from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer

    class _FakeNorm(torch.nn.Module):
        def forward(self, value):
            return value

    class _FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _FakeNorm()

        def forward(self, hidden_states):
            self.post_attention_layernorm(hidden_states + 1)
            return hidden_states + 2

    class _FakeRuntime(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(layers)

        def forward(self, hidden_states):
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    proxy = HardwareForward.__new__(HardwareForward)
    proxy.values = {}
    proxy.handles = []
    proxy.capture_active = False
    proxy.active = False
    proxy.producer_keys = []
    runtime = _FakeRuntime([_FakeLayer(), _FakeLayer()])
    # Only one boundary owns an NVFP4 quantizer; the rest are FP8 in a mixed
    # stream and must still be captured.
    runtime.model.layers[0].add_module(
        "_w4a_attention_residual_quantizer",
        NVFP4BoundaryQuantizer("model.layers.0.post_attention_residual", "cpu"),
    )
    try:
        proxy._install_producer_capture(runtime)
        for index in range(2):
            prefix = f"model.layers.{index}"
            for suffix in ("input", "post_attention_residual", "output"):
                assert f"{prefix}.{suffix}" in proxy.producer_keys
        proxy.capture_active = True
        ids = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        runtime(ids)
        proxy.active = True
        for index in range(2):
            prefix = f"model.layers.{index}"
            for suffix in ("input", "post_attention_residual", "output"):
                key = ("producer", f"{prefix}.{suffix}")
                assert key in proxy.values, key
                surrogate = torch.zeros_like(proxy.values[key])
                proxy.replace(surrogate, key)
    finally:
        for handle in proxy.handles:
            handle.remove()


def _pair(dtype, parameterization="physical_weight", *, attention_mode=None, mlp_fp8_layers=None):
    from transformers import LlamaForCausalLM

    from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay
    from gptqmodel.nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream
    from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
    from tests.models.test_w4a_producer_calibration import _tiny_packed_nvfp4
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes

    runtime = _tiny_packed_nvfp4(dtype=dtype)
    install_w4a_llama_stream(
        runtime, "w4a_nvfp4", "least_squares",
        attention_mode=attention_mode, mlp_fp8_layers=mlp_fp8_layers,
    )
    student = LlamaForCausalLM(runtime.config).to(device="cuda", dtype=dtype).train()
    for runtime_layer, student_layer in zip(runtime.model.layers, student.model.layers, strict=True):
        runtime_layer.mlp.down_proj.online_full_had = True
        student_layer.mlp.down_proj.online_full_had = True
    modules = dict(runtime.named_modules())
    with torch.no_grad():
        for name, value in student.named_parameters():
            owner, leaf = name.rsplit(".", 1)
            if not isinstance(modules[owner], W4ANVFP4Linear):
                value.copy_(getattr(modules[owner], leaf))
            value.requires_grad_(False)
    codes, scales = {}, {}
    for name, module in modules.items():
        if isinstance(module, W4ANVFP4Linear):
            shifts = torch.arange(8, device="cuda", dtype=torch.int64)[None, :, None] * 4
            codes[name] = (((module.qweight.long()[:, None] >> shifts) & 15)
                           .reshape(module.in_features, module.out_features) - 8).to(torch.int8)
            scales[name] = module.scales.float().clone()
    params, trainable = _install_trainable_gptq_codes(
        student, codes, scales, parameterization=parameterization,
    )
    config = SimpleNamespace(
        activation_mode="w4a_nvfp4", activation_recipe="least_squares",
        activation_global_scales=None, rotation=None,
        activation_attention_mode=attention_mode,
        activation_attention_recipe=None,
        activation_mlp_fp8_layers=mlp_fp8_layers,
        dynamic_get=lambda **_kwargs: None,
    )
    install_w4a_llama_replay(student, config)
    return student, runtime, params, trainable


# The CUDA cases below drive the NVFP4 hardware-forward harness, which this
# release validates on GB10 / SM 12.1 only.
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=NVFP4_HARDWARE)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 256, 2048, 8192])
def test_online_hadamard_keeps_backend_values_and_walsh_gradient(device, dtype, width, monkeypatch):
    from gptqmodel.quantization.rotation import hadamard_utils as had

    # Exercise the actual no-autograd CUDA fallback even if a native extension
    # becomes available. CPU exercises the portable backend through the same API.
    backend = had._TritonHadamardTransform() if device == "cuda" else had._TorchHadamardTransform()
    monkeypatch.setattr(had, "fast_hadamard_transform", backend)
    generator = torch.Generator(device=device).manual_seed(1193 + width)
    x = torch.randn((3, width), dtype=dtype, device=device, generator=generator).requires_grad_()
    gradient = torch.randn(x.shape, dtype=dtype, device=device, generator=generator)
    scale = float(1. / torch.tensor(width).sqrt())
    with torch.no_grad():
        original = backend.hadamard_transform(x, scale)
    actual = had.matmul_hadU_cuda(x, None, 1)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    (actual * gradient).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Construct Walsh entries from bit parity, independently of the butterfly.
    # Chunk columns to keep the 8192-channel reference below 32 MiB.
    columns = torch.arange(width, dtype=torch.int64)
    expected = torch.empty((3, width), dtype=torch.float64)
    incoming = gradient.detach().cpu().double()
    for start in range(0, width, 128):
        products = columns[:, None] & columns[None, start:start + 128]
        parity = torch.zeros_like(products)
        for bit in range(width.bit_length() - 1):
            parity ^= (products >> bit) & 1
        expected[:, start:start + 128] = incoming @ (1 - 2 * parity).double() * scale
    if device == "cuda":
        torch.cuda.synchronize()
    # Convert the independent FP64 result on the target device. CPU's
    # double->half conversion can pass through FP32 and double-round a tie,
    # whereas CUDA directly rounds the FP64 value to half.
    expected_rounded = expected.to(device=device).to(dtype)
    torch.testing.assert_close(x.grad, expected_rounded, rtol=1e-6, atol=1e-6)


@NVFP4_HARDWARE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parameterization", ["physical_weight", "code_cell"])
def test_encoded_forward_is_exact_and_checkpointed_gradients_match(dtype, parameterization):
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay

    student, runtime, params, trainable = _pair(dtype, parameterization)
    proxy = HardwareForward(student, runtime, trainable)
    ids = torch.arange(17, device="cuda")[None]
    try:
        assert proxy.sync_weights() == 0
        with proxy.frame(ids) as frame:
            actual = student(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, frame.logits, rtol=0, atol=0)
            actual.float().square().mean().backward()
        gradients = [p.grad.detach().clone() for p in params]
        for p in params:
            p.grad = None
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with proxy.frame(ids) as frame:
            actual = student(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, frame.logits, rtol=0, atol=0)
            actual.float().square().mean().backward()
        for p, expected in zip(params, gradients, strict=True):
            assert torch.isfinite(p.grad).all() and bool((p.grad != 0).any())
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
        with proxy.frame(ids):
            expired_loss = student(input_ids=ids, use_cache=False).logits.float().square().mean()
        with pytest.raises(RuntimeError, match="live forward/backward frame"):
            expired_loss.backward()
        with pytest.raises(RuntimeError, match="live forward/backward frame"):
            student(input_ids=ids, use_cache=False)
        replay.set_w4a_replay_enabled(student, False)
        with torch.no_grad():
            native = student(input_ids=ids, use_cache=False).logits
        assert torch.isfinite(native).all()
        proxy.close()
        with torch.no_grad():
            restored = student(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(restored, native, rtol=0, atol=0)
    finally:
        proxy.close()


@NVFP4_HARDWARE
@pytest.mark.parametrize("parameterization", ["physical_weight", "code_cell"])
def test_code_refresh_and_frame_lifetime(parameterization):
    student, runtime, _params, trainable = _pair(torch.bfloat16, parameterization)
    proxy = HardwareForward(student, runtime, trainable)
    ids = torch.arange(7, device="cuda")[None]
    try:
        name, module = next(iter(trainable.items()))
        before = runtime.get_submodule(name).qweight.clone()
        old_planes = runtime.get_submodule(name)._weight_both.view(torch.uint8).clone()
        with torch.no_grad():
            scale = 1. if parameterization == "code_cell" else module._w4a_qad_scales.T[0, 0]
            value = (module._w4a_qad_latent_codes if parameterization == "code_cell"
                     else module._w4a_qad_latent_weight)[0, 0, 0]
            old_code = int((value / scale).round())
            new_code = old_code - 1 if old_code == 7 else old_code + 1
            value.copy_(new_code * scale)
        assert proxy.sync_weights() == 1
        packed = runtime.get_submodule(name).qweight
        assert int((packed[0, 0] & 15) - 8) == new_code
        assert int((before != packed).sum()) == 1
        assert not torch.equal(old_planes, runtime.get_submodule(name)._weight_both.view(torch.uint8))
        assert proxy.sync_weights() == 0
        with proxy.frame(ids) as frame:
            torch.testing.assert_close(student(input_ids=ids, use_cache=False).logits,
                                       frame.logits, rtol=0, atol=0)
            with pytest.raises(RuntimeError, match="Cannot change"):
                proxy.sync_weights()
            with pytest.raises(RuntimeError, match="cannot overlap"):
                with proxy.frame(ids):
                    pass
        assert not proxy.values and proxy.logits is None
    finally:
        proxy.close()


@NVFP4_HARDWARE
@pytest.mark.parametrize("override", [
    {"attention_mode": "w4afp8", "mlp_fp8_layers": None},
    {"attention_mode": None, "mlp_fp8_layers": (0,)},
])
def test_hardware_forward_supports_mixed_stream_boundaries(override):
    """FP8 attention or MLP boundaries own no NVFP4 quantizer for their layer.

    The student wrapper still requests the pristine input,
    post-attention-residual, and output values, so the capture must not depend
    on an NVFP4 quantizer owning the boundary.
    """
    student, runtime, _params, trainable = _pair(
        torch.bfloat16, "physical_weight", **override
    )
    proxy = HardwareForward(student, runtime, trainable)
    ids = torch.arange(17, device="cuda")[None]
    try:
        with proxy.frame(ids) as frame:
            actual = student(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, frame.logits, rtol=0, atol=0)
            actual.float().square().mean().backward()
    finally:
        proxy.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=NVFP4_HARDWARE)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parameterization", ["physical_weight", "code_cell"])
def test_native_preservation_forward_gradients_refresh_and_lifetime(device, dtype, parameterization, monkeypatch):
    import copy

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from tests.models.w4a_native_forward import NativeForward
    from tests.models.w4a_nvfp4_scale_qad import _logit_distillation_loss
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes

    torch.manual_seed(9117)
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128)
    runtime = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    student = copy.deepcopy(runtime).train()
    for p in student.parameters():
        p.requires_grad_(False)
    monkeypatch.setattr(TorchLinear, "optimize", lambda *_args, **_kwargs: None)
    codes, scales = {}, {}
    for name, dense in list(runtime.named_modules()):
        if not isinstance(dense, torch.nn.Linear) or name == "lm_head":
            continue
        module = TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                             in_features=dense.in_features, out_features=dense.out_features,
                             bias=False).to(device=device, dtype=dtype)
        module.qweight.random_(-(2**31), 2**31 - 1)
        module.qzeros.fill_(-2004318072)  # Eight native v2 zero points of 8.
        module.qzero_format(format=2)
        module.scales.fill_(.002)
        if name.endswith("down_proj"):
            module.online_full_had = True
            student.get_submodule(name).online_full_had = True
        module.post_init()
        module.eval().enable_weight_cache()
        owner, leaf = name.rsplit(".", 1)
        setattr(runtime.get_submodule(owner), leaf, module)
        # Decode nibbles independently of the training helper.
        shifts = torch.arange(8, device=device)[None, :, None] * 4
        codes[name] = (((module.qweight.long()[:, None] >> shifts) & 15)
                       .reshape(module.in_features, module.out_features) - 8).to(torch.int8)
        scales[name] = module.scales.float().clone()
    params, modules = _install_trainable_gptq_codes(student, codes, scales,
                                                  parameterization=parameterization)
    policy = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                             activation_global_scales=None,
                             dynamic_get=lambda **_kwargs: None)
    replay.install_w4a_llama_replay(student, policy)
    proxy = NativeForward(student, runtime, modules)
    ids = torch.arange(17, device=device)[None]
    try:
        assert proxy.sync_weights() == 0
        with pytest.raises(RuntimeError, match="disabled activation replay"):
            with proxy.frame(ids):
                pass
        replay.set_w4a_replay_enabled(student, False)
        with torch.no_grad():
            teacher = runtime(input_ids=ids, use_cache=False).logits.detach().clone()
        with proxy.frame(ids) as frame:
            actual = student(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, frame.logits, rtol=0, atol=0)
            assert _logit_distillation_loss(actual, teacher, 1.).item() == 0
            actual.float().square().mean().backward()
        gradients = [p.grad.detach().clone() for p in params]
        for p in params:
            p.grad = None
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with proxy.frame(ids) as frame:
            actual = student(input_ids=ids, use_cache=False).logits
            torch.testing.assert_close(actual, teacher, rtol=0, atol=0)
            actual.float().square().mean().backward()
            with pytest.raises(RuntimeError, match="Cannot change"):
                proxy.sync_weights()
            with pytest.raises(RuntimeError, match="Cannot close"):
                proxy.close()
            with pytest.raises(RuntimeError, match="cannot overlap"):
                with proxy.frame(ids):
                    pass
        for p, expected in zip(params, gradients, strict=True):
            assert torch.isfinite(p.grad).all() and bool((p.grad != 0).any())
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
        with proxy.frame(ids):
            expired = student(input_ids=ids, use_cache=False).logits.float().square().mean()
        with pytest.raises(RuntimeError, match="live forward/backward frame"):
            expired.backward()
        name, module = next(iter(modules.items()))
        target = runtime.get_submodule(name)
        assert target._cached_weights
        with torch.no_grad():
            scale = 1. if parameterization == "code_cell" else module._w4a_qad_scales.T[0, 0]
            value = (module._w4a_qad_latent_codes if parameterization == "code_cell"
                     else module._w4a_qad_latent_weight)[0, 0, 0]
            old_code = int((value / scale).round())
            new_code = old_code - 1 if old_code == 7 else old_code + 1
            value.copy_(new_code * scale)
        assert proxy.sync_weights() == 1 and not target._cached_weights
        assert int((target.qweight[0, 0] & 15) - 8) == new_code
        assert proxy.sync_weights() == 0
        with proxy.frame(ids) as frame:
            torch.testing.assert_close(student(input_ids=ids, use_cache=False).logits,
                                       frame.logits, rtol=0, atol=0)
        assert not proxy.values and proxy.logits is None
    finally:
        proxy.close()
