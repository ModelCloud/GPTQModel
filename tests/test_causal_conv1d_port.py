# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from types import ModuleType

import causal_conv1d
import gptqmodel
import gptqmodel.hf_kernels.mamba_ssm as local_mamba_ssm


def test_causal_conv1d_fn_matches_grouped_conv_without_seq_idx():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.5, -0.25], [0.2, -0.1, 0.3]], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32)

    out = causal_conv1d.causal_conv1d_fn(x, weight, bias=bias, activation=None)
    expected = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[:, :, : x.shape[-1]]

    torch.testing.assert_close(out, expected)


def test_causal_conv1d_fn_respects_seq_idx_boundaries():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)

    out = causal_conv1d.causal_conv1d_fn(x, weight, seq_idx=seq_idx, activation=None)
    expected = torch.tensor([[[10.0, 21.0, 30.0, 43.0]]], dtype=torch.float32)

    torch.testing.assert_close(out, expected)


def test_causal_conv1d_update_supports_state_indices_and_cache_seqlens():
    conv_state = torch.tensor(
        [
            [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]],
            [[40.0, 50.0, 60.0], [4.0, 5.0, 6.0]],
            [[70.0, 80.0, 90.0], [7.0, 8.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[[100.0], [10.0]], [[200.0], [20.0]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    cache_seqlens = torch.tensor([1, 2], dtype=torch.int32)
    conv_state_indices = torch.tensor([2, 0], dtype=torch.int32)

    out = causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )

    expected_out = torch.tensor([[[70.0], [10.0]], [[20.0], [20.0]]], dtype=torch.float32)
    expected_state = torch.tensor(
        [
            [[10.0, 20.0, 200.0], [1.0, 2.0, 20.0]],
            [[40.0, 50.0, 60.0], [4.0, 5.0, 6.0]],
            [[70.0, 100.0, 90.0], [7.0, 10.0, 9.0]],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(conv_state, expected_state)


def test_gptqmodel_routes_transformers_hub_kernel_to_local_port(monkeypatch):
    import transformers.integrations as hf_integrations
    from transformers.integrations import hub_kernels

    gptqmodel._patch_transformers_causal_conv1d_hub_kernel_compat()

    def _unexpected_get_kernel(*args, **kwargs):
        raise AssertionError("HF hub get_kernel should not be called for causal-conv1d")

    monkeypatch.setattr(hub_kernels, "get_kernel", _unexpected_get_kernel)
    hub_kernels._KERNEL_MODULE_MAPPING.pop("causal-conv1d", None)

    kernel = hf_integrations.lazy_load_kernel("causal-conv1d")

    assert kernel is causal_conv1d
    assert hub_kernels._KERNEL_MODULE_MAPPING["causal-conv1d"] is causal_conv1d


def test_gptqmodel_hub_kernel_loader_falls_back_without_local_override(monkeypatch):
    import transformers.integrations as hf_integrations
    from transformers.integrations import hub_kernels

    gptqmodel._patch_transformers_causal_conv1d_hub_kernel_compat()

    kernel_name = "gptqmodel-fallback-kernel"
    fallback_kernel = ModuleType("fallback_kernel")

    monkeypatch.setitem(hub_kernels._HUB_KERNEL_MAPPING, kernel_name, {"repo_id": "kernels-community/fallback", "version": 1})
    monkeypatch.setattr(hub_kernels, "get_kernel", lambda *args, **kwargs: fallback_kernel)
    hub_kernels._KERNEL_MODULE_MAPPING.pop(kernel_name, None)

    kernel = hf_integrations.lazy_load_kernel(kernel_name)

    assert kernel is fallback_kernel
    assert hub_kernels._KERNEL_MODULE_MAPPING[kernel_name] is fallback_kernel


def test_local_mamba_chunk_scan_combined_returns_expected_state():
    x = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32)
    dt = torch.ones(1, 2, 1, dtype=torch.float32)
    A = torch.zeros(1, dtype=torch.float32)
    B = torch.ones(1, 2, 1, 1, dtype=torch.float32)
    C = torch.ones(1, 2, 1, 1, dtype=torch.float32)

    out, final_state = local_mamba_ssm.mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=2,
        return_final_states=True,
    )

    expected_out = torch.tensor([[[[1.0]], [[3.0]]]], dtype=torch.float32)
    expected_state = torch.tensor([[[[3.0]]]], dtype=torch.float32)
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(final_state, expected_state)


def test_gptqmodel_routes_mamba_ssm_hub_kernel_to_namespaced_local_port(monkeypatch):
    import transformers.integrations as hf_integrations
    from transformers.integrations import hub_kernels

    gptqmodel._patch_transformers_causal_conv1d_hub_kernel_compat()

    def _unexpected_get_kernel(*args, **kwargs):
        raise AssertionError("HF hub get_kernel should not be called for mamba-ssm")

    monkeypatch.setattr(hub_kernels, "get_kernel", _unexpected_get_kernel)
    hub_kernels._KERNEL_MODULE_MAPPING.pop("mamba-ssm", None)

    kernel = hf_integrations.lazy_load_kernel("mamba-ssm")

    assert kernel is local_mamba_ssm
    assert hub_kernels._KERNEL_MODULE_MAPPING["mamba-ssm"] is local_mamba_ssm
