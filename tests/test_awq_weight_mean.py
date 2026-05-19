import os
import types


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7" #"expandable_segments:True"

import time

import pytest
import torch
from parameterized import parameterized
from pytest import MonkeyPatch
from torch import nn

import gptqmodel.looper.awq_processor as awq_processor_mod
from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _accumulate_awq_weight_mean,
    _AWQLayerState,
    _compute_awq_weight_mean,
)
from gptqmodel.models.base import generate_node_for_awq_scaling
from gptqmodel.quantization.config import FORMAT, METHOD, AWQConfig, QuantizeConfig


QWEN3_HIDDEN_SIZE = 3584

pytestmark = [pytest.mark.cpu, pytest.mark.gpu]


def _compute_legacy_w_mean(layers, group_size):
    weights = [layer.weight.detach().to(torch.float32).cpu() for layer in layers]
    weight = torch.cat(weights, dim=0)
    org_shape = weight.shape
    weight = weight.view(-1, group_size)
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_scale = w_scale.view(org_shape)
    return w_scale.mean(0)

def _compute_fast_w_mean(layers, group_size):
    return _compute_awq_weight_mean(layers, group_size)


def _compute_fast_w_mean_multi(layer_groups, group_size):
    total_sum = None
    total_rows = 0
    for layers in layer_groups:
        w_sum, rows = _accumulate_awq_weight_mean(layers, group_size)
        if total_sum is None:
            total_sum = w_sum.cpu()
        else:
            total_sum += w_sum.cpu()
        total_rows += rows
    return (total_sum / total_rows)


class _DummyQwen3SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)


class _TestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: QuantizeConfig):
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            gptq_model=types.SimpleNamespace(
                rotary_embedding=None,
            ),
            model=None,
            require_fwd=True,
            calculate_w_wq_diff=False,
            calibration_concat_separator=None,
        )

    def _module_forward(self, x: torch.Tensor, module: torch.nn.Module, module_kwargs):
        return module(x)


def test_awq_record_input_feature_preserves_sample_axis_for_2d_inputs():
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    state = _AWQLayerState(modules={"self_attn.q_proj": object()})

    processor.tasks["self_attn.q_proj"] = {"inputs": []}
    processor._record_input_feature("self_attn.q_proj", torch.randn(16, QWEN3_HIDDEN_SIZE))
    processor._record_input_feature("self_attn.q_proj", torch.randn(16, QWEN3_HIDDEN_SIZE))

    features = processor._layer_input_features(state)

    assert features["self_attn.q_proj"].shape == (2, 16, QWEN3_HIDDEN_SIZE)


def test_awq_layer_input_features_aligns_variable_length_fallback_with_cached_kwargs():
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    state = _AWQLayerState(modules={"self_attn.q_proj": object()})

    processor.inputs_cache = types.SimpleNamespace(
        attention_masks=[
            torch.ones(1, 1, 423, 423),
            torch.ones(1, 1, 36, 36),
        ],
        position_ids=[
            torch.arange(423).unsqueeze(0),
            torch.arange(36).unsqueeze(0),
        ],
        layer_input_kwargs=[{}, {}],
    )
    processor._module_forward_kwargs = {"attention_mask": processor.inputs_cache.attention_masks[-1]}
    processor.tasks["self_attn.q_proj"] = {
        "inputs": [
            torch.randn(1, 423, QWEN3_HIDDEN_SIZE),
            torch.randn(1, 36, QWEN3_HIDDEN_SIZE),
        ],
        "batch_indices": [0, 1],
    }

    features = processor._layer_input_features(state)

    assert features["self_attn.q_proj"].shape == (1, 36, QWEN3_HIDDEN_SIZE)
    assert processor._awq_feature_kwargs["self_attn.q_proj"]["attention_mask"].shape == (1, 1, 36, 36)
    assert processor._awq_feature_kwargs["self_attn.q_proj"]["position_ids"].shape == (1, 36)


def test_awq_can_concat_batch_tensors_requires_matching_trailing_shapes():
    compatible = [
        torch.randn(1, 36, QWEN3_HIDDEN_SIZE),
        torch.randn(2, 36, QWEN3_HIDDEN_SIZE),
    ]
    incompatible = [
        torch.randn(1, 423, QWEN3_HIDDEN_SIZE),
        torch.randn(1, 36, QWEN3_HIDDEN_SIZE),
    ]

    assert AWQProcessor._can_concat_batch_tensors(compatible) is True
    assert AWQProcessor._can_concat_batch_tensors(incompatible) is False


def test_awq_module_forward_slices_batch_aligned_kwargs_with_chunk_offset():
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    processor._quant_batch_size = 1

    class _Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.attention_masks = []
            self.position_ids = []

        def forward(self, x, attention_mask=None, position_ids=None):
            self.attention_masks.append(attention_mask.clone())
            self.position_ids.append(position_ids.clone())
            return x

    module = _Recorder()
    x = torch.randn(2, 8, 16)
    attention_mask = torch.stack(
        (
            torch.full((1, 8, 8), 1.0),
            torch.full((1, 8, 8), 2.0),
        ),
        dim=0,
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14, 15, 16, 17],
        ]
    )

    out = AWQProcessor._module_forward(
        processor,
        x,
        module,
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
    )

    assert out.shape == x.shape
    assert len(module.attention_masks) == 2
    assert float(module.attention_masks[0][0, 0, 0, 0]) == 1.0
    assert float(module.attention_masks[1][0, 0, 0, 0]) == 2.0
    assert module.position_ids[0].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert module.position_ids[1].tolist() == [[10, 11, 12, 13, 14, 15, 16, 17]]


def test_awq_module_forward_splits_accumulated_batches_even_when_quant_batch_size_is_one():
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    processor._quant_batch_size = 1

    class _Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, x):
            self.calls.append(int(x.shape[0]))
            return x

    module = _Recorder()
    x = torch.randn(4, 8, 16)

    out = AWQProcessor._module_forward(processor, x, module, {})

    assert out.shape == x.shape
    assert module.calls == [1, 1, 1, 1]


def test_awq_forward_signature_is_cached_for_scale_search(monkeypatch):
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    processor._quant_batch_size = 1

    class _ModuleWithKwargs(nn.Module):
        def forward(self, x, attention_mask=None, position_ids=None):
            return x

    module = _ModuleWithKwargs()
    real_signature = awq_processor_mod.inspect.signature
    signature_calls = 0

    def _record_signature(target):
        nonlocal signature_calls
        signature_calls += 1
        return real_signature(target)

    monkeypatch.setattr(awq_processor_mod.inspect, "signature", _record_signature)

    kwargs = {
        "attention_mask": None,
        "position_ids": torch.arange(4).unsqueeze(0),
        "unused": torch.ones(1),
    }
    x = torch.randn(1, 4, 8)

    for _ in range(2):
        sanitized = processor._sanitize_kwargs(kwargs, module)
        assert "unused" not in sanitized
        list(processor._iter_module_forward_outputs(x, module, sanitized))

    assert signature_calls == 1


def test_awq_activation_x_mean_uses_dtype_itemsize_without_tensor_alloc(monkeypatch):
    torch.manual_seed(0)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=4))
    processor.max_chunk_memory = 64
    x = torch.randn(2, 3, 8, dtype=torch.float16)

    def _reject_tensor_alloc(*args, **kwargs):
        raise AssertionError("x_mean chunk sizing should use dtype.itemsize instead of torch.tensor")

    monkeypatch.setattr(torch, "tensor", _reject_tensor_alloc)

    actual = processor._compute_activation_x_mean(x)
    expected = x.reshape(-1, x.shape[-1]).to(torch.float32).abs().mean(dim=0).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_generate_node_for_awq_scaling_keeps_kwargs_for_later_nodes():
    kwargs = {
        "attention_mask": torch.ones(1, 1, 36, 36),
        "position_ids": torch.arange(36).unsqueeze(0),
    }

    first_node, _ = generate_node_for_awq_scaling(
        inp=torch.randn(1, 36, 16),
        prev_op=object(),
        module_kwargs=kwargs,
        nodes_size=0,
        subset=[nn.Linear(16, 16, bias=False)],
        module2inspect=None,
    )
    later_node, _ = generate_node_for_awq_scaling(
        inp=torch.randn(1, 36, 16),
        prev_op=object(),
        module_kwargs=kwargs,
        nodes_size=1,
        subset=[nn.Linear(16, 16, bias=False)],
        module2inspect=None,
    )

    assert first_node["kwargs"] is kwargs
    assert later_node["kwargs"] is kwargs


def test_awq_align_module_kwargs_packs_mask_for_packed_feature_tensor():
    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))
    inp = torch.randn(1, 5, 16)
    attention_mask = torch.tensor(
        [
            [
                [
                    [0.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min],
                    [0.0, 0.0, torch.finfo(torch.float32).min],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min],
                    [0.0, 0.0, torch.finfo(torch.float32).min],
                    [torch.finfo(torch.float32).min, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min],
                ]
            ],
        ],
        dtype=torch.float32,
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2],
            [10, 11, 12],
        ]
    )

    aligned = processor._align_module_kwargs_to_input(
        inp,
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
    )

    packed_mask = aligned["attention_mask"]
    assert packed_mask.shape == (1, 1, 5, 5)
    assert aligned["position_ids"].tolist() == [[0, 1, 2, 10, 11]]
    assert torch.isfinite(packed_mask[0, 0, 2, 0])
    assert packed_mask[0, 0, 0, 3] == torch.finfo(torch.float32).min
    assert packed_mask[0, 0, 3, 4] == torch.finfo(torch.float32).min


def test_awq_search_best_scale_keeps_cpu_activations_off_device_until_forward_chunks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for this test run.")

    processor = AWQProcessor(
        tokenizer=None,
        qcfg=QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=16),
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=types.SimpleNamespace(rotary_embedding=None),
        model=None,
        require_fwd=True,
        calculate_w_wq_diff=False,
        calibration_concat_separator=None,
    )
    processor._quant_batch_size = 1

    module = nn.Linear(16, 16, bias=False, device="cuda:0", dtype=torch.float16)
    inp = torch.randn(4, 8, 16, device="cpu", dtype=torch.float16)

    captured = {}

    def fake_compute_best_scale(
        self,
        _inp,
        w_mean,
        x_mean,
        module2inspect,
        layers_arg,
        fp16_output,
        module_kwargs,
    ):
        captured["inp_device"] = _inp.device.type
        captured["fp16_output_devices"] = [chunk.device.type for chunk in fp16_output]
        captured["fp16_output_shapes"] = [tuple(chunk.shape) for chunk in fp16_output]
        return torch.ones_like(w_mean, dtype=w_mean.dtype).detach().cpu(), 0.0

    monkey_patcher = MonkeyPatch()
    monkey_patcher.setattr(AWQProcessor, "_compute_best_scale", fake_compute_best_scale)

    try:
        processor._search_best_scale(
            module,
            module,
            [module],
            inp,
            module2inspect=module,
            kwargs={},
        )
    finally:
        monkey_patcher.undo()

    assert captured["inp_device"] == "cpu"
    assert captured["fp16_output_devices"] == ["cpu", "cpu", "cpu", "cpu"]
    assert captured["fp16_output_shapes"] == [(1, 8, 16)] * 4


def test_awq_search_best_scale_can_disable_chunked_activation_streaming():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for this test run.")

    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(format=FORMAT.GEMM, group_size=16, scale_search_chunked_activations=False),
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=types.SimpleNamespace(rotary_embedding=None),
        model=None,
        require_fwd=True,
        calculate_w_wq_diff=False,
        calibration_concat_separator=None,
    )
    processor._quant_batch_size = 1

    module = nn.Linear(16, 16, bias=False, device="cuda:0", dtype=torch.float16)
    inp = torch.randn(4, 8, 16, device="cpu", dtype=torch.float16)

    captured = {}

    def fake_compute_best_scale(
        self,
        _inp,
        w_mean,
        x_mean,
        module2inspect,
        layers_arg,
        fp16_output,
        module_kwargs,
    ):
        captured["inp_device"] = _inp.device.type
        captured["fp16_output_type"] = type(fp16_output).__name__
        captured["fp16_output_device"] = fp16_output.device.type
        captured["fp16_output_shape"] = tuple(fp16_output.shape)
        return torch.ones_like(w_mean, dtype=w_mean.dtype).detach().cpu(), 0.0

    monkey_patcher = MonkeyPatch()
    monkey_patcher.setattr(AWQProcessor, "_compute_best_scale", fake_compute_best_scale)

    try:
        processor._search_best_scale(
            module,
            module,
            [module],
            inp,
            module2inspect=module,
            kwargs={},
        )
    finally:
        monkey_patcher.undo()

    assert captured["inp_device"] == "cuda"
    assert captured["fp16_output_type"] == "Tensor"
    assert captured["fp16_output_device"] == "cuda"
    assert captured["fp16_output_shape"] == (4, 8, 16)


def test_awq_compute_best_scale_restores_cpu_weights_without_aliasing():
    torch.manual_seed(0)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=4))
    processor._quant_batch_size = 1

    module = nn.Linear(8, 8, bias=False, dtype=torch.float32).eval()
    x = torch.randn(2, 4, 8)
    original_weight = module.weight.detach().clone()
    w_mean = _compute_awq_weight_mean([module], processor.qcfg.group_size)
    x_mean = processor._compute_activation_x_mean(x)
    fp16_output = [
        output.detach().clone()
        for output in processor._iter_module_forward_outputs(x, module, {})
    ]

    with torch.inference_mode():
        best_scales, loss = processor._compute_best_scale(
            x,
            w_mean,
            x_mean,
            module,
            [module],
            fp16_output,
            {},
        )

    torch.testing.assert_close(module.weight, original_weight, atol=0, rtol=0)
    assert best_scales.device.type == "cpu"
    assert loss >= 0


def test_awq_compute_best_scale_avoids_winning_scale_clone(monkeypatch):
    torch.manual_seed(0)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=4))
    processor._quant_batch_size = 1

    module = nn.Linear(8, 8, bias=False, dtype=torch.float32).eval()
    x = torch.randn(2, 4, 8)
    w_mean = _compute_awq_weight_mean([module], processor.qcfg.group_size)
    x_mean = processor._compute_activation_x_mean(x)
    fp16_output = [
        output.detach()
        for output in processor._iter_module_forward_outputs(x, module, {})
    ]

    clone_shapes = []
    original_clone = torch.Tensor.clone

    def _record_clone(tensor, *args, **kwargs):
        if tuple(tensor.shape) == tuple(w_mean.shape):
            clone_shapes.append(tuple(tensor.shape))
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", _record_clone)

    with torch.inference_mode():
        best_scales, loss = processor._compute_best_scale(
            x,
            w_mean,
            x_mean,
            module,
            [module],
            fp16_output,
            {},
        )

    assert best_scales.shape == w_mean.shape
    assert loss >= 0
    assert clone_shapes == []


def test_awq_chunked_scale_loss_matches_legacy_expression():
    torch.manual_seed(0)

    ref_chunk = torch.randn(2, 4, 16, dtype=torch.float16)
    int_w_output = torch.randn(2, 4, 16, dtype=torch.float16)
    int_w_output[0, 0, 0] = float("inf")

    finfo = torch.finfo(int_w_output.dtype)
    clamped = int_w_output.clip(finfo.min, finfo.max)
    expected_diff = (ref_chunk.to(dtype=clamped.dtype) - clamped).float()
    expected_loss = expected_diff.pow(2).sum().item()

    chunk_loss, chunk_elements = AWQProcessor._accumulate_awq_chunk_loss(
        ref_chunk,
        int_w_output.clone(),
    )

    assert chunk_loss == expected_loss
    assert chunk_elements == expected_diff.numel()


def test_awq_legacy_eager_loss_matches_chunked_expression():
    torch.manual_seed(0)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=4))
    processor.max_chunk_memory = 64
    fp16_output = torch.randn(2, 4, 16, dtype=torch.float16)
    int_w_output = torch.randn(2, 4, 16, dtype=torch.float16)
    device = torch.device("cpu")

    fp16_flat = fp16_output.view(-1)
    int_w_flat = int_w_output.view(-1)
    chunk_size = processor.max_chunk_memory // (fp16_output.element_size() * 2)
    chunk_size = min(chunk_size, fp16_flat.size(0))

    expected = 0.0
    for fp16_chunk, int_w_chunk in zip(torch.split(fp16_flat, chunk_size), torch.split(int_w_flat, chunk_size)):
        expected += (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
    expected /= fp16_flat.size(0)

    loss = processor._compute_loss(fp16_output, int_w_output.clone(), device)

    assert loss == expected


def test_awq_scale_search_restore_device_respects_config_and_headroom(monkeypatch):
    layer = nn.Linear(8, 8, bias=False, dtype=torch.float16).eval()
    cuda_device = torch.device("cuda", 0)

    disabled = _TestAWQProcessor(
        AWQConfig(format=FORMAT.GEMM, group_size=4, scale_search_gpu_weight_restore=False)
    )
    assert disabled._select_awq_weight_restore_device(cuda_device, [layer]) == torch.device("cpu")

    enabled = _TestAWQProcessor(
        AWQConfig(format=FORMAT.GEMM, group_size=4, scale_search_gpu_weight_restore=True)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (128 << 20, 16 << 30))
    assert enabled._select_awq_weight_restore_device(cuda_device, [layer]) == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (2 << 30, 16 << 30))
    assert enabled._select_awq_weight_restore_device(cuda_device, [layer]) == cuda_device


@pytest.mark.cuda
def test_awq_gpu_weight_restore_preserves_exact_scale_search_math(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for this test run.")

    requested = int(os.environ.get("GPTQMODEL_TEST_CUDA_INDEX", "0"))
    device = torch.device("cuda", requested if requested < torch.cuda.device_count() else 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    base = nn.Linear(16, 16, bias=False, device=device, dtype=torch.float16).eval()
    x = torch.randn(2, 4, 16, device=device, dtype=torch.float16)

    def run_case(use_gpu_restore: bool):
        qcfg = AWQConfig(
            format=FORMAT.GEMM,
            group_size=8,
            scale_search_gpu_weight_restore=use_gpu_restore,
        )
        processor = _TestAWQProcessor(qcfg)
        processor._quant_batch_size = 1
        module = nn.Linear(16, 16, bias=False, device=device, dtype=torch.float16).eval()
        module.load_state_dict(base.state_dict())
        if use_gpu_restore:
            monkeypatch.setattr(
                processor,
                "_select_awq_weight_restore_device",
                lambda _device, _layers: device,
            )
        w_mean = _compute_awq_weight_mean([module], processor.qcfg.group_size)
        x_mean = processor._compute_activation_x_mean(x)
        fp16_output = [
            output.detach().cpu()
            for output in processor._iter_module_forward_outputs(x, module, {})
        ]
        with torch.inference_mode():
            scales, loss = processor._compute_best_scale(
                x,
                w_mean,
                x_mean,
                module,
                [module],
                fp16_output,
                {},
            )
        return scales, loss, module.weight.detach().cpu()

    cpu_scales, cpu_loss, cpu_weight = run_case(False)
    gpu_scales, gpu_loss, gpu_weight = run_case(True)

    torch.testing.assert_close(gpu_scales, cpu_scales, atol=0, rtol=0)
    torch.testing.assert_close(gpu_weight, cpu_weight, atol=0, rtol=0)
    assert gpu_loss == cpu_loss


@parameterized.expand([
    ("cpu_gs32", "cpu", 32),
    ("cpu_gs64", "cpu", 64),
    ("cpu_gs128", "cpu", 128),
    ("cuda0_gs32", "cuda:0", 32),
    ("cuda0_gs64", "cuda:0", 64),
    ("cuda0_gs128", "cuda:0", 128),
    ("cuda0_cuda1_gs128", ("cuda:0", "cuda:1"), 128),
])
def test_awq_weight_mean_matches_legacy_impl(param_name, device, group_size):
    if isinstance(device, (list, tuple)):
        devices = list(device)
        for dev in devices:
            if not torch.cuda.is_available() or torch.device(dev).index >= torch.cuda.device_count():
                pytest.skip(f"{dev} is not available")
    elif isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available for this test run.")

    torch.manual_seed(0)
    if isinstance(device, (list, tuple)):
        dtype = torch.float16
        layer_groups = []
        for dev in device:
            layer_groups.append([
                nn.Linear(QWEN3_HIDDEN_SIZE, QWEN3_HIDDEN_SIZE, bias=False, device=dev, dtype=dtype)
                for _ in range(3)
            ])

        baseline_layers = [layer for group in layer_groups for layer in group]
        baseline = _compute_legacy_w_mean(baseline_layers, group_size)
        fast = _compute_fast_w_mean_multi(layer_groups, group_size)
        fast = fast.to(baseline.dtype)

        # Accuracy table
        abs_diff = (fast - baseline).abs()
        with torch.no_grad():
            safe_baseline = torch.where(baseline == 0, torch.ones_like(baseline), baseline)
            rel_diff = abs_diff / safe_baseline.abs()
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()

        header = f"{'Metric':<20}{'Measured':<20}{'Tolerance':<20}"
        separator = "-" * len(header)
        print(f"AWQ weight mean comparison (fast vs baseline) [{param_name}]")
        print(separator)
        print(header)
        print(separator)
        atol = 5e-4
        rtol = 1e-3
        print(f"{'max_abs_diff':<20}{max_abs_diff:<20.6e}{atol:<20.6e}")
        print(f"{'max_rel_diff':<20}{max_rel_diff:<20.6e}{rtol:<20.6e}")
        print(separator)
        assert torch.allclose(fast, baseline, rtol=rtol, atol=atol)

        # Timing comparison
        def _time_it(fn, runs=5, warmup=2):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(torch.device(device[0]).index)
            start = time.perf_counter()
            for _ in range(runs):
                fn()
            torch.cuda.synchronize(torch.device(device[0]).index)
            return (time.perf_counter() - start) / runs

        def fast_fn():
            _ = _compute_fast_w_mean_multi(layer_groups, group_size)

        def legacy_fn():
            _ = _compute_legacy_w_mean(baseline_layers, group_size)

        fast_time = _time_it(fast_fn)
        legacy_time = _time_it(legacy_fn)

        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"

        delta_ms = (fast_time - legacy_time) * 1e3
        rel = (fast_time / legacy_time) if legacy_time > 0 else float("inf")
        if rel <= 1.0:
            color = GREEN
            verdict = "faster"
        elif rel <= 1.05:
            color = YELLOW
            verdict = "≈ parity"
        else:
            color = RED
            verdict = "slower"

        print(f"AWQ weight mean timing [{param_name}]")
        print("+----------------+------------+--------------+-------------+---------------+")
        print("| Metric         | Fast (ms)  | Legacy (ms)  | Delta (ms)  | Relative      |")
        print("+----------------+------------+--------------+-------------+---------------+")
        print(
            f"| runtime        | {fast_time*1e3:10.3f} | {legacy_time*1e3:12.3f} | "
            f"{delta_ms:11.3f} | {color}{rel:>11.3%} {verdict:<7}{RESET}|"
        )
        print("+----------------+------------+--------------+-------------+---------------+")

        assert fast_time <= legacy_time * 1.05, (
            f"Streaming mean slower than legacy for {param_name}: "
            f"{fast_time*1e3:.3f} ms vs {legacy_time*1e3:.3f} ms"
        )
        return

    device_str = device
    dtype = torch.float16 if device_str.startswith("cuda") else torch.float32

    attn = _DummyQwen3SelfAttention(QWEN3_HIDDEN_SIZE, device_str, dtype)
    layers = [attn.q_proj, attn.k_proj, attn.v_proj]

    batch_size = 4
    inp = torch.randn(batch_size, QWEN3_HIDDEN_SIZE, device=device_str, dtype=dtype)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=group_size))

    captured = {}

    def fake_compute_best_scale(
        self,
        _inp,
        w_mean,
        x_mean,
        module2inspect,
        layers_arg,
        fp16_output,
        module_kwargs,
    ):
        captured["fast"] = w_mean.detach().to(torch.float32).cpu()
        captured["baseline"] = (
            _compute_legacy_w_mean(layers_arg, self.qcfg.group_size).detach().to(torch.float32).cpu()
        )
        return torch.ones_like(w_mean, dtype=w_mean.dtype).detach().cpu(), 0.0

    monkey_patcher = MonkeyPatch()
    monkey_patcher.setattr(AWQProcessor, "_compute_best_scale", fake_compute_best_scale)

    try:
        processor._search_best_scale(
            attn,
            layers[0],
            layers,
            inp,
            module2inspect=layers[0],
            kwargs={},
        )
    finally:
        monkey_patcher.undo()

    assert "fast" in captured and "baseline" in captured
    if dtype == torch.float32:
        atol = 2e-7
        rtol = 2e-7
    else:
        atol = 5e-4
        rtol = 1e-3
    fast = captured["fast"]
    baseline = captured["baseline"]

    abs_diff = (fast - baseline).abs()
    with torch.no_grad():
        safe_baseline = torch.where(baseline == 0, torch.ones_like(baseline), baseline)
        rel_diff = abs_diff / safe_baseline.abs()

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()

    header = f"{'Metric':<20}{'Measured':<20}{'Tolerance':<20}"
    separator = "-" * len(header)
    print(f"AWQ weight mean comparison (fast vs baseline) [{param_name}]")
    print(separator)
    print(header)
    print(separator)
    print(f"{'max_abs_diff':<20}{max_abs_diff:<20.6e}{atol:<20.6e}")
    print(f"{'max_rel_diff':<20}{max_rel_diff:<20.6e}{rtol:<20.6e}")
    print(separator)

    assert torch.allclose(fast, baseline, rtol=rtol, atol=atol)

    def _time_it(fn, runs=5, warmup=2):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize(device_str)
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        if device == "cuda":
            torch.cuda.synchronize(device_str)
        return (time.perf_counter() - start) / runs

    fast_time = _time_it(lambda: _compute_fast_w_mean(layers, group_size))
    legacy_time = _time_it(lambda: _compute_legacy_w_mean(layers, group_size))

    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    delta_ms = (fast_time - legacy_time) * 1e3
    rel = (fast_time / legacy_time) if legacy_time > 0 else float("inf")
    if rel <= 1.0:
        color = GREEN
        verdict = "faster"
    elif rel <= 1.05:
        color = YELLOW
        verdict = "≈ parity"
    else:
        color = RED
        verdict = "slower"

    print(f"AWQ weight mean timing [{param_name}]")
    print("+----------------+------------+--------------+-------------+---------------+")
    print("| Metric         | Fast (ms)  | Legacy (ms)  | Delta (ms)  | Relative      |")
    print("+----------------+------------+--------------+-------------+---------------+")
    print(
        f"| runtime        | {fast_time*1e3:10.3f} | {legacy_time*1e3:12.3f} | "
        f"{delta_ms:11.3f} | {color}{rel:>11.3%} {verdict:<7}{RESET}|"
    )
    print("+----------------+------------+--------------+-------------+---------------+")

    assert fast_time <= legacy_time * 1.05, (
        f"Streaming mean slower than legacy for {param_name}: "
        f"{fast_time*1e3:.3f} ms vs {legacy_time*1e3:.3f} ms"
    )
