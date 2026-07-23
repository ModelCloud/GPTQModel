# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.diagnostics import (
    QUANTIZATION_DIAGNOSTICS_ENV,
    QuantizationDiagnosticsMode,
    analyze_quantization_losses,
    analyze_scale_channels,
    compare_quant_code_samples,
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
    assert summary["max_scale"] == pytest.approx(35.0)
    assert summary["scale_count_above_10"] == 3
    assert summary["nonfinite_scale_count"] == 0


def test_scale_channel_summary_does_not_report_group_index_as_channel():
    scales = torch.zeros((4096, 384), dtype=torch.float32)
    scales[2276, 161] = 35.0

    summary = analyze_scale_channels(scales)

    assert summary["max_scale_output_channel"] == 2276
    assert summary["output_channel_count"] == 4096


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
