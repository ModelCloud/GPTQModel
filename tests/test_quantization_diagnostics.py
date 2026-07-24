# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.diagnostics import (
    QUANTIZATION_DIAGNOSTICS_ENV,
    QuantizationDiagnosticsMode,
    analyze_group_index,
    analyze_quantization_losses,
    analyze_reconstruction_error,
    analyze_scale_channels,
    compare_quant_code_samples,
    render_quantization_diagnostics_markdown,
    resolve_quantization_diagnostics_mode,
    sample_packed_quant_codes,
    sample_reconstructed_quant_codes,
    summarize_quant_code_fingerprints,
)
from gptqmodel.utils.model_dequant import pack_cols


def test_quantization_diagnostics_config_defaults_and_round_trips():
    config = QuantizeConfig(offload_to_disk=False)

    assert config.quantization_diagnostics == QuantizationDiagnosticsMode.AUTO

    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert restored.quantization_diagnostics == QuantizationDiagnosticsMode.AUTO

    channel = QuantizeConfig(quantization_diagnostics="channel", offload_to_disk=False)
    restored_channel = QuantizeConfig.from_quant_config(channel.to_dict())
    assert restored_channel.quantization_diagnostics == QuantizationDiagnosticsMode.CHANNEL


def test_quantization_diagnostics_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="quantization_diagnostics"):
        QuantizeConfig(quantization_diagnostics="expensive", offload_to_disk=False)


def test_quantization_diagnostics_environment_override(monkeypatch):
    monkeypatch.setenv(QUANTIZATION_DIAGNOSTICS_ENV, "channel")

    assert (
        resolve_quantization_diagnostics_mode(QuantizationDiagnosticsMode.OFF)
        == QuantizationDiagnosticsMode.CHANNEL
    )


def test_loss_summary_localizes_a_severe_module_outlier():
    entries = [
        {"layer": layer, "module": "mlp.down_proj", "loss": f"{loss:.6f}"}
        for layer, loss in enumerate([0.0003] * 35 + [0.2265])
    ]
    entries.extend(
        {"layer": layer, "module": "self_attn.q_proj", "loss": "0.0002"}
        for layer in range(36)
    )

    summary = analyze_quantization_losses(entries)

    assert summary["module_count"] == 72
    assert summary["severe_concentration"] is True
    assert summary["top"][0]["layer"] == 35
    assert summary["top"][0]["module"] == "mlp.down_proj"
    assert summary["top"][0]["role_median_ratio"] == pytest.approx(755.0)
    assert summary["top"][0]["total_loss_share"] > 0.90


def test_scale_channel_summary_localizes_output_channel():
    scales = torch.tensor(
        [
            [0.2, 0.3, 0.1],
            [0.3, 0.2, 0.4],
            [35.0, 12.0, 11.0],
            [0.4, 0.5, 0.2],
        ],
        dtype=torch.float32,
    )

    summary = analyze_scale_channels(scales)

    assert summary["max_scale_output_channel"] == 2
    assert summary["output_channel_count"] == 4
    assert summary["group_count"] == 3
    assert summary["max_scale"] == pytest.approx(35.0)
    assert summary["max_scale_group"] == 0
    assert summary["scale_count_above_10"] == 3
    assert summary["nonfinite_scale_count"] == 0
    assert summary["nonpositive_scale_count"] == 0


def test_scale_channel_summary_does_not_report_group_index_as_channel():
    scales = torch.zeros((4096, 384), dtype=torch.float32)
    scales[2276, 161] = 35.0

    summary = analyze_scale_channels(scales)

    assert summary["max_scale_output_channel"] == 2276
    assert summary["output_channel_count"] == 4096


def test_group_index_summary_preserves_zero_based_mapping_and_reports_decreases():
    summary = analyze_group_index(torch.tensor([0, 0, 2, 1], dtype=torch.int32), group_count=3)

    assert summary["minimum_group"] == 0
    assert summary["maximum_group"] == 2
    assert summary["unique_group_count"] == 3
    assert summary["negative_group_count"] == 0
    assert summary["out_of_range_group_count"] == 0
    assert summary["monotonic_decrease_count"] == 1
    assert summary["is_monotonic_non_decreasing"] is False


def test_reconstruction_error_localizes_actual_tensor_axes_in_bounded_chunks():
    source = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32)
    reconstructed = torch.tensor([[1.5, -1.0], [2.0, -2.0]], dtype=torch.float32)

    summary = analyze_reconstruction_error(
        source,
        reconstructed,
        max_chunk_values=2,
        top_k_axes=2,
    )

    assert summary["available"] is True
    assert summary["shape"] == [2, 2]
    assert summary["axis_semantics"]["index_base"] == 0
    assert summary["relative_rmse"] == pytest.approx(0.5 / (10.0 ** 0.5))
    assert summary["rmse"] == pytest.approx(0.25)
    assert summary["maximum_absolute_error"] == pytest.approx(0.5)
    assert summary["maximum_error_output_row"] == 0
    assert summary["maximum_error_input_feature"] == 0
    assert summary["top_output_rows"][0]["output_row"] == 0
    assert summary["top_input_features"][0]["input_feature"] == 0


def test_during_quantization_markdown_is_human_readable_and_zero_based():
    reconstruction = analyze_reconstruction_error(
        torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
        torch.tensor([[1.5, -1.0], [2.0, -2.0]]),
    )
    reconstruction.update(
        {
            "layer": 0,
            "module": "self_attn.q_proj",
            "full_name": "model.layers.0.self_attn.q_proj",
            "bits": 4,
            "group_size": 128,
        }
    )
    loss = analyze_quantization_losses(
        [
            {
                "layer": 0,
                "module": "self_attn.q_proj",
                "full_name": "model.layers.0.self_attn.q_proj",
                "loss": 0.25,
                "bits": 4,
                "group_size": 128,
            }
        ]
    )
    markdown = render_quantization_diagnostics_markdown(
        {
            "mode": "channel",
            "module_grouping": {
                "basis": "gptqmodel_model_definition_module_group",
                "source": "gptqmodel.models.definitions.qwen3.Qwen3QModel.module_tree",
                "groups": [
                    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                    ["self_attn.o_proj"],
                    ["mlp.gate_proj", "mlp.up_proj"],
                    ["mlp.down_proj"],
                ],
            },
            "quantization_config": {
                "method": "gptq",
                "format": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": True,
                "pack_impl": "original",
            },
            "loss": loss,
            "reconstruction": {"module_count": 1, "records": [reconstruction]},
            "scale_channels": {"module_count": 0, "records": []},
            "code_fingerprints": {
                "module_count": 0,
                "prepack_mismatch_count": 0,
                "packed_mismatch_count": 0,
                "records": [],
            },
        }
    )

    assert "# During-quantization error analysis" in markdown
    assert "## Canonical GPTQ reconstruction error" in markdown
    assert "## Localized reconstruction rows and input features" in markdown
    assert "## Logical-code lifecycle" in markdown
    assert "## GPTQModel definition-group recommendations" in markdown
    assert "qkv_projection_group" in markdown
    assert "Qwen3QModel.module_tree" in markdown
    assert "`model.layers.0.self_attn.k_proj`" in markdown
    assert "`model.layers.0.self_attn.v_proj`" in markdown
    assert "not** evidence" in markdown
    assert "`model.layers.0.self_attn.q_proj`" in markdown
    assert "no `+1` conversion" in markdown


@pytest.mark.parametrize("bits,in_features", [(2, 16), (3, 32), (4, 8), (8, 4)])
def test_quant_code_fingerprint_matches_reconstructed_and_packed_codes(bits, in_features):
    out_features = 3
    group_size = in_features // 2
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    codes = (
        torch.arange(out_features * in_features, dtype=torch.int32).reshape(out_features, in_features)
        % (1 << bits)
    )
    scales = torch.tensor(
        [[0.25, 0.5], [0.5, 0.75], [0.75, 1.0]],
        dtype=torch.float32,
    )
    zeros = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    weight = (codes.float() - zeros[:, g_idx.long()]) * scales[:, g_idx.long()]

    fingerprint = sample_reconstructed_quant_codes(
        weight,
        scales,
        zeros,
        g_idx,
        bits=bits,
        sample_size=max(in_features, out_features),
    )
    assert fingerprint is not None
    assert torch.equal(fingerprint["codes"], codes.to(dtype=torch.int16))

    qweight = pack_cols(codes.contiguous(), bits, pack_dtype=torch.int32).T.contiguous()
    packed_codes = sample_packed_quant_codes(
        qweight,
        bits=bits,
        input_indexes=fingerprint["input_indexes"],
        output_indexes=fingerprint["output_indexes"],
    )
    assert torch.equal(packed_codes, fingerprint["codes"])


def test_quant_code_fingerprint_detects_prepack_and_packed_mismatches():
    reference = torch.tensor([[0, 1], [2, 3]], dtype=torch.int16)
    prepack = reference.clone()
    prepack[0, 0] = 1
    packed = reference.clone()
    packed[1, 1] = 0
    record = {
        "layer": 6,
        "module": "mlp.down_proj",
        "prepack": compare_quant_code_samples(reference, prepack),
        "packed": compare_quant_code_samples(reference, packed),
    }

    summary = summarize_quant_code_fingerprints([record])

    assert summary["sample_code_count"] == 4
    assert summary["prepack_mismatch_count"] == 1
    assert summary["packed_mismatch_count"] == 1
    assert summary["prepack_mismatch_rate"] == pytest.approx(0.25)
    assert summary["packed_mismatch_rate"] == pytest.approx(0.25)
