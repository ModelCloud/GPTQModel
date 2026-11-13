# SPDX-License-Identifier: Apache-2.0
"""Independent CPU oracles for Kimi's bounded dense calibration stages."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


_path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/kimi_k3_core_stream.py"
_spec = importlib.util.spec_from_file_location("kimi_k3_core_stream_under_test", _path)
stages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stages)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.7, 1.3, width))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        dtype = value.dtype
        value = value.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(dtype)


def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    # Independent checkpoint formula, not a call into the staged helpers.
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    fp32 = values.float()
    keys = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    scores = (keys * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(-1)
    return torch.matmul(scores.softmax(-1).unsqueeze(1), fp32).squeeze(1).to(values.dtype)


class Situ(nn.Module):
    def forward(self, value):
        dtype = value.dtype
        gate, up = value.float().chunk(2, dim=-1)
        return ((4 * torch.tanh(gate / 4) * torch.sigmoid(gate))
                * (25 * torch.tanh(up / 25))).to(dtype)


class SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="situ")
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = Situ()

    def forward(self, value):
        joined = torch.cat([self.gate_proj(value), self.up_proj(value)], dim=-1)
        return self.down_proj(self.act_fn(joined))


class Gate(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def forward(self, value):
        self.log.append(("gate", tuple(value.shape)))
        count = value.numel() // value.shape[-1]
        ids = torch.tensor([[1, 0], [0, 1]], device=value.device).repeat((count + 1) // 2, 1)[:count]
        weights = torch.tensor([.3125, .6875], device=value.device).expand(count, -1)
        return ids, weights


class LoggedLinear(nn.Linear):
    def __init__(self, inputs, outputs, log, name):
        super().__init__(inputs, outputs, bias=False)
        self.log, self.name = log, name

    def forward(self, value):
        self.log.append((self.name, tuple(value.shape)))
        return super().forward(value)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.log = []
        self.use_latent_moe = True
        self.latent_moe_use_norm = True
        self.gate = Gate(self.log)
        self.routed_expert_down_proj = LoggedLinear(4, 2, self.log, "latent_down")
        self.routed_expert_up_proj = LoggedLinear(2, 4, self.log, "latent_up")
        self.routed_expert_norm = RMSNorm(2)
        self.shared_experts = SharedMLP()

    def moe_infer(self, value, ids, weights):
        self.log.append(("expert_reduce", tuple(value.shape)))
        # Two different immutable expert functions; route-slot reduction remains
        # FP32 followed by a cast, just as in the checkpoint reference.
        experts = torch.stack((value * 1.75, value * -.625), dim=1)
        selected = torch.gather(experts, 1, ids.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
        return (selected.float() * weights.unsqueeze(-1)).sum(1).to(value.dtype)

    def forward(self, hidden):
        identity = hidden
        original_shape = hidden.shape
        ids, weights = self.gate(hidden)
        latent = self.routed_expert_down_proj(hidden.view(-1, hidden.shape[-1]))
        output = self.moe_infer(latent, ids, weights)
        if self.latent_moe_use_norm:
            output = self.routed_expert_norm(output)
        output = self.routed_expert_up_proj(output).view(*original_shape)
        return output + self.shared_experts(identity)


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer_idx = layer
        self.attn_res_block_size = 12
        self.use_attn_residuals = True
        self.self_attention_res_proj = nn.Linear(4, 1, bias=False)
        self.self_attention_res_norm = RMSNorm(4)
        self.mlp_res_proj = nn.Linear(4, 1, bias=False)
        self.mlp_res_norm = RMSNorm(4)
        self.input_layernorm = RMSNorm(4)
        self.post_attention_layernorm = RMSNorm(4)
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.block_sparse_moe = MoE()

    def forward(self, hidden, bank):
        batch, length, width = hidden.shape
        prefix = hidden
        if bank.shape[1]:
            hidden = _apply_attn_res(prefix.view(-1, width), bank,
                                     self.self_attention_res_proj, self.self_attention_res_norm).view(batch, length, width)
        if self.layer_idx % self.attn_res_block_size == 0:
            bank = torch.cat([bank, prefix.view(-1, width).unsqueeze(1)], dim=1)
            prefix = None
        hidden = self.self_attn(self.input_layernorm(hidden))
        prefix = hidden if prefix is None else prefix + hidden
        hidden = _apply_attn_res(prefix.view(-1, width), bank,
                                self.mlp_res_proj, self.mlp_res_norm).view(batch, length, width)
        hidden = self.block_sparse_moe(self.post_attention_layernorm(hidden))
        return prefix + hidden, bank


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layer,banks", [(0, 0), (1, 1), (12, 1), (13, 2), (92, 8)])
@torch.inference_mode()
def test_stages_equal_independent_native_decoder(dtype, layer, banks):
    torch.manual_seed(19)
    block = Block(layer).to(dtype)
    prefix = torch.randn(1, 3, 4, dtype=dtype)
    bank = torch.randn(3, banks, 4, dtype=dtype)
    expected, expected_bank = block(prefix, bank)
    normalized, residual_prefix, next_bank = stages.attention_inputs(block, prefix, bank)
    attention = block.self_attn(normalized)
    attention_prefix, next_bank, ffn_input = stages.attention_finish(block, residual_prefix, next_bank, attention)
    shared = stages.shared_down_input(block.block_sparse_moe.shared_experts, ffn_input)
    routed = stages.routed_up_input(block.block_sparse_moe, ffn_input)
    actual, actual_bank = stages.complete_moe(attention_prefix, next_bank, shared, routed, block.block_sparse_moe)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert torch.equal(actual_bank, expected_bank)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_routed_input_is_weighted_then_normalized_and_router_precedes_down(dtype):
    torch.manual_seed(31)
    moe = MoE().to(dtype)
    hidden = torch.randn(1, 5, 4, dtype=dtype)
    ids, weights = moe.gate(hidden)
    latent = moe.routed_expert_down_proj(hidden.view(-1, 4))
    aggregated = moe.moe_infer(latent, ids, weights)
    expected = moe.routed_expert_norm(aggregated)
    moe.log.clear()
    actual = stages.routed_up_input(moe, hidden)
    assert torch.equal(actual, expected)
    assert moe.log == [("gate", (1, 5, 4)), ("latent_down", (5, 4)), ("expert_reduce", (5, 2))]
    moe.latent_moe_use_norm = False
    assert torch.equal(stages.routed_up_input(moe, hidden), aggregated)


def test_moe_preserves_addition_parentheses():
    # A BF16 example distinguishing prefix + (routed + shared) from
    # (prefix + routed) + shared.
    moe = MoE().to(torch.bfloat16)
    moe.shared_experts.down_proj = nn.Identity()
    moe.routed_expert_up_proj = nn.Identity()
    prefix = torch.tensor([[[256., 256.]]], dtype=torch.bfloat16)
    shared = torch.tensor([[[1., 1.]]], dtype=torch.bfloat16)
    routed = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    actual, _ = stages.complete_moe(prefix, None, shared, routed, moe)
    assert torch.equal(actual, torch.tensor([[[258., 258.]]], dtype=torch.bfloat16))
    assert not torch.equal(actual, (prefix + routed) + shared)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_final_head_input_includes_output_mixer_then_norm(dtype):
    torch.manual_seed(51)
    proj, mixer_norm, norm = nn.Linear(4, 1, bias=False).to(dtype), RMSNorm(4).to(dtype), RMSNorm(4).to(dtype)
    prefix, bank = torch.randn(1, 3, 4, dtype=dtype), torch.randn(3, 8, 4, dtype=dtype)
    expected = norm(_apply_attn_res(prefix.view(-1, 4), bank, proj, mixer_norm).view(1, 3, 4))
    actual = stages.final_head_input(prefix, bank, output_attn_res_proj=proj,
                                    output_attn_res_norm=mixer_norm, model_norm=norm,
                                    apply_attn_res=_apply_attn_res)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)


def test_capture_aborts_before_target_and_removes_hook():
    target = nn.Linear(4, 4, bias=False)
    value = torch.randn(2, 4)
    executed = []
    target.register_forward_hook(lambda *args: executed.append("target_executed"))

    def native_forward():
        before = value.sigmoid()
        target(before)
        executed.append("after")

    actual = stages.capture_linear_input(target, native_forward)
    assert torch.equal(actual, value.sigmoid())
    assert not actual.requires_grad
    assert not target._forward_pre_hooks
    assert not executed
    target(value)
    assert executed == ["target_executed"]


def test_capture_supports_keyword_linear_input_and_context():
    target = nn.Linear(4, 4)
    value = torch.randn(2, 4)
    with stages.LinearInputCapture(target) as capture:
        target(input=value)
    assert torch.equal(capture.input, value)
    assert not target._forward_pre_hooks
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with capture:
            pass


def test_capture_is_not_swallowed_by_model_exception_recovery():
    target = nn.Linear(4, 4)
    reached = []

    def model():
        try:
            target(torch.ones(2, 4))
        except Exception:
            reached.append("unexpected recovery")
        reached.append("after target")

    assert torch.equal(stages.capture_linear_input(target, model), torch.ones(2, 4))
    assert not reached
    assert not target._forward_pre_hooks


@pytest.mark.parametrize("exception", [RuntimeError("unexpected model failure"), KeyboardInterrupt()])
def test_capture_propagates_unexpected_failures_after_cleanup(exception):
    target = nn.Linear(4, 4)

    def failure():
        raise exception

    with pytest.raises(type(exception)) as caught:
        stages.capture_linear_input(target, failure)
    assert caught.value is exception
    assert not target._forward_pre_hooks


def test_capture_rejects_unreached_target_and_non_tensor_input():
    target = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="did not reach"):
        stages.capture_linear_input(target, lambda: torch.ones(2))
    assert not target._forward_pre_hooks
    with pytest.raises(TypeError, match="tensor input"):
        stages.capture_linear_input(target, target, None)
    assert not target._forward_pre_hooks


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_hessian_collects_every_row_with_fp64_oracle(dtype):
    torch.manual_seed(63)
    first, second = torch.randn(1, 7, 8).to(dtype), torch.randn(1, 11, 8).to(dtype)
    accumulator = stages.DenseRawHessian(8, device="cpu", chunk_rows=3)
    accumulator.add(first)
    accumulator.update(second)
    rows = torch.cat((first.view(-1, 8), second.view(-1, 8))).double()
    oracle = rows.T @ rows
    raw = accumulator.raw_dict(minimum_rows=18)
    assert raw["count"] == 18
    assert raw["finalized"] is False
    assert raw["H"] is accumulator.H
    assert torch.linalg.vector_norm(raw["H"].double() - oracle) / torch.linalg.vector_norm(oracle) < 1e-6
    assert torch.isfinite(raw["H"]).all()
    assert accumulator.as_dict()["H"] is raw["H"]


def test_hessian_rejects_missing_insufficient_or_zero_observations_without_augmentation():
    accumulator = stages.DenseRawHessian(4, device="cpu")
    with pytest.raises(ValueError, match="insufficient"):
        accumulator.raw_dict()
    accumulator.add(torch.ones(2, 4))
    before = accumulator.H.clone()
    with pytest.raises(ValueError, match="insufficient"):
        accumulator.raw_dict(minimum_rows=3)
    assert accumulator.count == 2
    assert torch.equal(accumulator.H, before)
    assert torch.equal(before, torch.full((4, 4), 2.))
    zero = stages.DenseRawHessian(4, device="cpu")
    zero.add(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="covariance"):
        zero.raw_dict()


@pytest.mark.parametrize("value", [torch.empty(0, 4), torch.ones(2, 3),
                                  torch.ones(2, 4, dtype=torch.int64),
                                  torch.full((2, 4), float("nan")), torch.full((2, 4), float("inf"))])
def test_hessian_rejects_invalid_input(value):
    accumulator = stages.DenseRawHessian(4, device="cpu")
    with pytest.raises(ValueError):
        accumulator.add(value)
    assert accumulator.count == 0
    assert not accumulator.H.any()


@pytest.mark.parametrize("width,chunk_rows", [(0, 2), (True, 2), (4, 0), (4, True)])
def test_hessian_rejects_invalid_configuration(width, chunk_rows):
    with pytest.raises(ValueError):
        stages.DenseRawHessian(width, device="cpu", chunk_rows=chunk_rows)


def test_non_situ_shared_path_and_unsupported_stage_rejections():
    shared = SharedMLP()
    shared.config.hidden_act = "silu"
    shared.act_fn = nn.SiLU()
    hidden = torch.randn(1, 3, 4)
    assert torch.equal(stages.shared_down_input(shared, hidden), shared.act_fn(shared.gate_proj(hidden)) * shared.up_proj(hidden))
    block = Block(0)
    block.use_attn_residuals = False
    with pytest.raises(ValueError, match="attention residuals"):
        stages.attention_inputs(block, hidden, torch.empty(3, 0, 4))
    block.block_sparse_moe.use_latent_moe = False
    with pytest.raises(ValueError, match="latent MoE"):
        stages.routed_up_input(block.block_sparse_moe, hidden)
