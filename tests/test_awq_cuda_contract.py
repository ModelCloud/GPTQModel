import tempfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "source_path",
    [
        "gptqmodel_ext/awq/quantization/gemv_cuda.cu",
        "gptqmodel_ext/awq/quantization/gemm_cuda_gen.cu",
        "gptqmodel_ext/paro/rotation.cu",
    ],
)
def test_cuda_entrypoints_guard_device_and_launch(source_path):
    source = (ROOT / source_path).read_text()
    assert "OptionalCUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source


def test_awq_gemv_native_contract_checks_tensor_layout_and_device():
    source = (ROOT / "gptqmodel_ext/awq/quantization/gemv_cuda.cu").read_text()
    for contract in (
        "is_contiguous",
        "scalar_type() == at::kHalf",
        "must be on the same CUDA device",
        "does not support zero-row launches",
        "group_size == 64 || group_size == 128",
    ):
        assert contract in source


def test_awq_gemm_pack_backend_stays_portable_off_cuda():
    from gptqmodel.utils.backend import BACKEND, awq_gemm_pack_backend

    # The native AWQ GEMM extension only builds for CUDA, so every other host
    # (and the weight-only export lifecycle) packs with the portable kernel.
    assert (
        awq_gemm_pack_backend(device_type="cuda", is_rocm=False, weight_only=False, bits=4)
        == BACKEND.AWQ_GEMM
    )
    assert (
        awq_gemm_pack_backend(device_type="cuda", is_rocm=True, weight_only=False, bits=4)
        == BACKEND.AWQ_TORCH
    )
    assert (
        awq_gemm_pack_backend(device_type="xpu", is_rocm=False, weight_only=False, bits=4)
        == BACKEND.AWQ_TORCH
    )
    assert (
        awq_gemm_pack_backend(device_type="npu", is_rocm=False, weight_only=False, bits=4)
        == BACKEND.AWQ_TORCH
    )
    assert (
        awq_gemm_pack_backend(device_type="cpu", is_rocm=False, weight_only=False, bits=4)
        == BACKEND.AWQ_TORCH
    )
    assert (
        awq_gemm_pack_backend(device_type="cuda", is_rocm=False, weight_only=True, bits=4)
        == BACKEND.AWQ_TORCH
    )


@pytest.mark.parametrize("bits", [2, 3, 8])
def test_awq_gemm_pack_backend_uses_torch_for_non_4bit_widths(bits):
    from gptqmodel.utils.backend import BACKEND, awq_gemm_pack_backend

    # ``AwqGEMMLinear`` only supports 4 bits; the portable packer emits the
    # identical GEMM layout for the other advertised widths (#6135).
    assert (
        awq_gemm_pack_backend(
            device_type="cuda", is_rocm=False, weight_only=False, bits=bits
        )
        == BACKEND.AWQ_TORCH
    )


def _tiny_awq_model_dir():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    path = tempfile.mkdtemp(prefix="awq_gemm_gate_6135_")
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).save_pretrained(path)

    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(specials)
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 20,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=128),
    )
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>").save_pretrained(path)
    return path


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="AWQ GEMM packing is CUDA-only")
@pytest.mark.parametrize("bits", [8, 3])
def test_awq_gemm_non_4bit_global_width_quantizes_with_default_backend(bits):
    """The AUTO pack backend must validate the kernel it will actually use."""

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = _tiny_awq_model_dir()
    model = GPTQModel.load(
        source,
        quantize_config=AWQConfig(bits=bits, group_size=32, sym=False),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    texts = [
        " ".join(["the quick brown fox jumps over the lazy dog"] * (1 + index % 3))
        for index in range(8)
    ]

    model.quantize(CalibrationConfig(dataset=texts, batch_size=2))

    assert model.quantize_backend == BACKEND.AWQ_TORCH
    packed = [
        module
        for name, module in model.model.named_modules()
        if name.endswith("q_proj")
    ]
    assert packed and all(isinstance(module, AwqTorchLinear) for module in packed)


def test_awq_gemm_kernel_is_cuda_only():
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear

    assert AwqGEMMLinear.SUPPORTS_DEVICES == [DEVICE.CUDA]


def test_awq_gemm_validate_once_reports_rocm_gap(monkeypatch):
    from gptqmodel.nn_modules.qlinear import gemm_awq

    monkeypatch.setattr(gemm_awq, "IS_ROCM", True)
    ok, err = gemm_awq.AwqGEMMLinear.validate_once()

    assert ok is False
    assert isinstance(err, NotImplementedError)
    assert "ROCm" in str(err)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.version.hip is not None,
    reason="requires two NVIDIA CUDA devices (the native AWQ kernel is CUDA-only)",
)
def test_awq_gemv_uses_tensor_device_instead_of_current_device():
    from gptqmodel.utils.awq import (
        awq_gemmv2_forward,
        awq_gemv_forward,
        awq_runtime_error,
        prewarm_awq_extension,
    )

    if not prewarm_awq_extension():
        pytest.skip(awq_runtime_error())

    original_device = torch.cuda.current_device()
    input_device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    torch.cuda.set_device(original_device)

    qweight = torch.zeros((64, 16), device=input_device, dtype=torch.int32)
    qzeros = torch.zeros((64, 2), device=input_device, dtype=torch.int32)
    scales = torch.ones((64, 16), device=input_device, dtype=torch.float16)

    decode = awq_gemv_forward(
        torch.ones((1, 128), device=input_device, dtype=torch.float16),
        qweight,
        scales,
        qzeros,
        64,
    )
    prefill = awq_gemmv2_forward(
        torch.ones((9, 128), device=input_device, dtype=torch.float16),
        qweight,
        scales,
        qzeros,
        64,
        8,
    )

    assert torch.cuda.current_device() == original_device
    assert decode.device == input_device and decode.shape == (1, 64)
    assert prefill.device == input_device and prefill.shape == (9, 64)
    assert torch.count_nonzero(decode) == 0
    assert torch.count_nonzero(prefill) == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_paro_rotation_uses_tensor_device_instead_of_current_device():
    from gptqmodel import extension
    from gptqmodel.utils.paro import (
        _load_rotation_extension,
        build_identity_rotation_buffers,
    )

    if not _load_rotation_extension():
        pytest.skip("Paro CUDA extension is unavailable")

    original_device = torch.cuda.current_device()
    input_device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    torch.cuda.set_device(original_device)
    x = torch.ones((1, 128), device=input_device, dtype=torch.float16)
    pairs, theta, scales = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=1,
        device=input_device,
        dtype=torch.float16,
    )

    out = extension.op("paro", "rotate")(
        x,
        pairs,
        theta,
        scales,
        128,
        -1,
        -1,
    )

    assert torch.cuda.current_device() == original_device
    assert out.device == input_device
    torch.testing.assert_close(out, x)
