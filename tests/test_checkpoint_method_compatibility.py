# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from gptqmodel import CheckpointConfig, QuantizeConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization import FORMAT, METHOD


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_unsupported_checkpoint_fails_before_model_or_disk_mutation(
    tmp_path, existing_checkpoint
):
    root = tmp_path / "checkpoint"
    if existing_checkpoint:
        root.mkdir()
        (root / "CURRENT").write_bytes(b"must not be opened or overwritten")
    config = QuantizeConfig(
        method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        lm_head=True,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    linear = torch.nn.Linear(4, 4)
    linear.config = SimpleNamespace(model_type="llama")
    model = SimpleNamespace(
        quantize_config=config,
        model=linear,
        _normalize_embed_quant_config=lambda **kwargs: None,
    )
    weights = {key: tensor.clone() for key, tensor in linear.state_dict().items()}
    config_before = config.to_dict()
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(
        NotImplementedError,
        match="checkpoint requires sequential quantization",
    ):
        BaseQModel.quantize(
            model,
            calibration=None,
            checkpoint=CheckpointConfig(root),
        )
    assert config.to_dict() == config_before
    assert all(
        torch.equal(weights[key], tensor) for key, tensor in linear.state_dict().items()
    )
    assert {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before
    assert root.exists() == existing_checkpoint


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
@pytest.mark.parametrize(
    "mode",
    [
        "term",
        "int",
        "kill-before",
        "kill-after",
        "error",
        "kill-hessian",
        "kill-hessian-early",
    ],
)
def test_eora_checkpoint_recovery(tmp_path, family, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    test_subprocess_recovery(
        tmp_path, family, mode, device="cuda:0", required_gpus=2, eora=True
    )


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("mode", ["term", "kill-before", "kill-after", "error"])
@pytest.mark.parametrize("eora", [False, True])
def test_awq_checkpoint_recovery(tmp_path, family, mode, eora):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(
        tmp_path, family, mode, device="cuda:0", method="awq", eora=eora
    )


@pytest.mark.parametrize("method", ["rtn", "fp8", "gguf", "bitsandbytes"])
@pytest.mark.parametrize("mode", ["term", "kill-before", "kill-after", "error"])
def test_weight_only_checkpoint_recovery(tmp_path, method, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(tmp_path, "llama", mode, device="cuda:0", method=method)


@pytest.mark.parametrize("method", ["qqq", "paro", "exl3"])
@pytest.mark.parametrize("mode", ["term", "kill-before", "kill-after", "error"])
def test_specialized_checkpoint_recovery(tmp_path, method, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(tmp_path, "llama", mode, device="cuda:0", method=method)


def test_paro_eager_checkpoint_recovery(tmp_path):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(
        tmp_path,
        "llama",
        "term",
        device="cuda:0",
        method="paro",
        driver_args=("--paro-no-cudagraph",),
    )


@pytest.mark.parametrize("method", ["qqq", "paro"])
@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
@pytest.mark.parametrize("mode", ["term", "kill-before"])
def test_specialized_eora_checkpoint_recovery(tmp_path, method, family, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(
        tmp_path, family, mode, device="cuda:0", method=method, eora=True
    )


@pytest.mark.parametrize(
    "method", ["rtn", "fp8", "gguf", "bitsandbytes", "qqq", "paro", "exl3"]
)
@pytest.mark.parametrize("mode", ["term", "kill-before"])
def test_moe_method_checkpoint_recovery(tmp_path, method, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    test_subprocess_recovery(
        tmp_path, "qwen3_moe", mode, device="cuda:0", method=method
    )


@pytest.mark.parametrize(
    "method,format",
    [
        ("gptq", "gptq_v2"),
        ("gptq", "gptq_p"),
        ("awq", "gemv"),
        ("awq", "gemv_fast"),
        ("awq", "llm-awq"),
        ("awq", "bitblas"),
        ("fp8", "float8_e5m2"),
        ("bitsandbytes", "fp4"),
        ("bitsandbytes", "int8"),
    ],
)
def test_format_checkpoint_recovery(tmp_path, method, format):
    from test_checkpoint_quantization import test_subprocess_recovery

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if format == "bitblas":
        from gptqmodel.nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE

        if not BITBLAS_AVAILABLE:
            pytest.skip("requires the optional BitBLAS compiler/runtime")
    test_subprocess_recovery(
        tmp_path, "llama", "kill-before", device="cuda:0", method=method, format=format
    )


def test_gptq_bitblas_is_repack_only_not_a_quantization_export():
    from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
    from gptqmodel.utils.importer import _supports_pack_api

    assert not _supports_pack_api(BitblasLinear)
    assert callable(BitblasLinear.repack_from_gptq)
