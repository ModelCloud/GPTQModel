# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import load_checkpoint_in_model
from safetensors.torch import save_file

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.models.loader import _checkpoint_load_dtype, _external_preload_backend
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import FORMAT, METHOD, QVQConfig
from gptqmodel.utils.backend import BACKEND


def _prepared_calibration(**kwargs):
    return kwargs["calibration_dataset"]


@pytest.mark.parametrize("format_code", (FORMAT.QVQ, FORMAT.QVQ_V4))
def test_qvq_mlx_load_uses_qvq_tensor_owner_before_native_conversion(format_code):
    assert _external_preload_backend(BACKEND.MLX, METHOD.QVQ, format_code) == BACKEND.QVQ
    assert _external_preload_backend(BACKEND.QVQ, METHOD.QVQ, format_code) == BACKEND.QVQ


def _quantized_bf16_named_module(
    seed: int,
    *,
    full_name: str = "proj",
    grouped_p32_candidates=None,
    transform_axis_overrides=None,
):
    torch.manual_seed(seed)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=True, dtype=torch.bfloat16)
    root.eval()
    named = NamedModule(
        root.proj,
        name=full_name.rsplit(".", 1)[-1],
        full_name=full_name,
        layer_index=0,
    )
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=QVQConfig(bits=2, rounding="block_ldlq", device="cpu", offload_to_disk=False),
        calibration=[
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }
        ],
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        grouped_p32_candidates=grouped_p32_candidates,
        transform_axis_overrides=transform_axis_overrides,
    )
    processor.preprocess(named)
    source = torch.randn((1, 4, 16), dtype=torch.bfloat16)
    output = root.proj(source)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 4), dtype=torch.bool)
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook(named.name)(root.proj, (source,), output)
    original_weight = root.proj.weight.detach().clone()
    processor.process(named, device=torch.device("cpu"))
    return root, named, processor, source, original_weight


def test_qvq_processor_applies_declared_shared_input_and_axis_policy_at_quantization():
    groups = {"gate_up": (("gate_proj", "up_proj"),)}
    axes = {
        "mlp.gate_proj": (True, False),
        "mlp.up_proj": (True, False),
    }
    gate = _quantized_bf16_named_module(
        401,
        full_name="model.layers.3.mlp.gate_proj",
        grouped_p32_candidates=groups,
        transform_axis_overrides=axes,
    )
    up = _quantized_bf16_named_module(
        402,
        full_name="model.layers.3.mlp.up_proj",
        grouped_p32_candidates=groups,
        transform_axis_overrides=axes,
    )

    gate_named, gate_processor = gate[1], gate[2]
    up_named, up_processor = up[1], up[2]
    assert torch.equal(gate_named.state["SU"], up_named.state["SU"])
    assert gate_named.state["_qvq_runtime_config"][8:10] == (True, False)
    assert up_named.state["_qvq_runtime_config"][8:10] == (True, False)
    assert (
        gate_processor.qcfg.meta["qvq_transform_axis_overrides"]
        == up_processor.qcfg.meta["qvq_transform_axis_overrides"]
    )


@pytest.mark.parametrize("seed", (11, 29, 47, 83))
def test_qvq_installation_preserves_fp32_auxiliaries_across_accelerate_reload(seed, tmp_path):
    root, named, processor, source, _ = _quantized_bf16_named_module(seed)
    original_post_init = QVQLinear.post_init
    installed_before_post_init = []

    def tracked_post_init(qmodule):
        installed_before_post_init.append(root.proj is qmodule)
        return original_post_init(qmodule)

    with patch.object(QVQLinear, "post_init", tracked_post_init):
        live = processor.submodule_finalize(named, SimpleNamespace(model=root))

    assert installed_before_post_init == [True]
    assert root.proj is live
    assert live.training is False
    assert hasattr(live, "_qvq_mps_compander")
    assert live.SU.dtype == live.SV.dtype == torch.float32
    assert live.bias.dtype == torch.bfloat16

    checkpoint_tensors = {name: tensor.clone() for name, tensor in live.state_dict().items()}
    checkpoint = tmp_path / f"qvq-{seed}.safetensors"
    save_file({f"proj.{name}": tensor.contiguous() for name, tensor in checkpoint_tensors.items()}, checkpoint)

    reloaded_root = torch.nn.Module()
    with torch.device("meta"):
        reloaded_root.proj = QVQLinear(
            bits=2,
            in_features=16,
            out_features=16,
            bias=True,
            dtype=torch.bfloat16,
        )
    assert reloaded_root.proj.SU.dtype == reloaded_root.proj.SV.dtype == torch.float32
    assert reloaded_root.proj.bias.dtype == torch.bfloat16

    load_checkpoint_in_model(
        reloaded_root,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=_checkpoint_load_dtype(format_code=FORMAT.QVQ, dtype=torch.bfloat16),
    )
    reloaded = reloaded_root.proj.eval()
    reloaded.post_init()

    for name, expected in checkpoint_tensors.items():
        actual = reloaded.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reloaded(source), live(source), rtol=0, atol=0)


def test_qvq_installation_uses_authoritative_model_eval_mode_when_materialized_leaf_is_stale():
    root, named, processor, _, _ = _quantized_bf16_named_module(97)
    assert root.training is False
    named.module.train()
    assert named.module.training is True

    installed = processor.submodule_finalize(named, SimpleNamespace(model=root))

    assert installed.training is False
    assert root.proj is installed


def test_qvq_post_init_failure_restores_original_installed_module():
    root, named, processor, _, original_weight = _quantized_bf16_named_module(101)
    original = root.proj
    staged_keys = {
        "trellis",
        "SU",
        "SV",
        "bias",
        "_qvq_original_weight",
        "_qvq_runtime_config",
    }
    staged_before = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in named.state.items()
        if key in staged_keys
    }

    with (
        patch.object(QVQLinear, "post_init", side_effect=RuntimeError("post init failed")),
        pytest.raises(RuntimeError, match="post init failed"),
    ):
        processor.submodule_finalize(named, SimpleNamespace(model=root))

    assert root.proj is original
    assert root.proj.training is False
    torch.testing.assert_close(root.proj.weight, original_weight, rtol=0, atol=0)
    assert staged_keys.issubset(named.state)
    for key, expected in staged_before.items():
        actual = named.state[key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected

    installed = processor.submodule_finalize(named, SimpleNamespace(model=root))
    assert root.proj is installed
    assert not staged_keys.intersection(named.state)


def test_qvq_constructor_failure_keeps_dense_weight_and_retry_payload():
    root, named, processor, _, original_weight = _quantized_bf16_named_module(103)
    original = root.proj
    staged_keys = {
        "trellis",
        "SU",
        "SV",
        "bias",
        "_qvq_original_weight",
        "_qvq_runtime_config",
    }

    with (
        patch("gptqmodel.looper.qvq_processor.QVQLinear", side_effect=RuntimeError("construction failed")),
        pytest.raises(RuntimeError, match="construction failed"),
    ):
        processor.submodule_finalize(named, SimpleNamespace(model=root))

    assert root.proj is original
    torch.testing.assert_close(root.proj.weight, original_weight, rtol=0, atol=0)
    assert staged_keys.issubset(named.state)


def test_qvq_delayed_stream_failure_restores_exact_dense_weight_and_rejects_payload():
    root, named, processor, _, original_weight = _quantized_bf16_named_module(107)
    original = root.proj
    staged_keys = {
        "trellis",
        "SU",
        "SV",
        "bias",
        "_qvq_original_weight",
        "_qvq_runtime_config",
    }
    assert not torch.equal(original.weight, original_weight)

    with (
        patch.object(NamedModule, "stream_sync", side_effect=RuntimeError("delayed transfer failed")),
        pytest.raises(RuntimeError, match="delayed transfer failed"),
    ):
        processor.submodule_finalize(named, SimpleNamespace(model=root))

    assert root.proj is original
    torch.testing.assert_close(root.proj.weight, original_weight, rtol=0, atol=0)
    assert not staged_keys.intersection(named.state)


def test_accelerate_global_dtype_coercion_would_damage_qvq_auxiliary_state():
    trellis = torch.zeros((1, 16), dtype=torch.int32)
    source = torch.linspace(-1.0, 1.0, 16).reshape(1, 16).to(torch.bfloat16)
    tensors = {
        "trellis": trellis,
        "SU": torch.linspace(0.111111, 1.777777, 16),
        "SV": torch.linspace(0.222222, 1.888888, 16),
    }
    fp32_live = QVQLinear(bits=2, in_features=16, out_features=16, dtype=torch.float32, tensors=tensors).eval()
    bf16_reload = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        dtype=torch.bfloat16,
        tensors={name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor for name, tensor in tensors.items()},
    ).eval()

    drift = (fp32_live(source).float() - bf16_reload(source).float()).abs().max()
    assert drift > 0


def test_checkpoint_load_dtype_only_defers_to_mixed_dtype_shell_for_qvq():
    assert _checkpoint_load_dtype(format_code=FORMAT.QVQ, dtype=torch.bfloat16) is None
    assert _checkpoint_load_dtype(format_code=FORMAT.GPTQ, dtype=torch.bfloat16) == torch.bfloat16
