# SPDX-License-Identifier: Apache-2.0
"""CPU execution/ownership contracts and independent FP64 covariance oracles."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn


_root = Path(__file__).parents[1] / "gptqmodel/utils"
_package = ModuleType("_k31_utils_under_test")
_package.__path__ = [str(_root)]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(_package.__name__ + ".kimi_k31", _root / "kimi_k31.py")
k31 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = k31
_spec.loader.exec_module(k31)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dense_real_rows_match_independent_fp64_oracle(dtype):
    x = torch.linspace(-2, 2, 513 * 16).reshape(513, 16).to(dtype)
    state = k31.DenseRawHessian(16, device="cpu")
    state.add(x[:241])
    state.update(x[241:].reshape(1, 272, 16))
    raw = state.raw_dict(minimum_rows=513)
    oracle = x.double().T @ x.double()
    assert raw["count"] == 513 and raw["finalized"] is False
    assert raw["H"].dtype == torch.float32
    assert float(torch.linalg.norm(raw["H"].double() - oracle) / torch.linalg.norm(oracle)) < 1e-6
    with torch.inference_mode():
        raw["H"].zero_()
    assert torch.count_nonzero(state.as_dict()["H"]) > 0
    with pytest.raises(ValueError, match="insufficient"):
        state.raw_dict(minimum_rows=514)


class Spool:
    def __init__(self, rows, natural, recovery):
        self.inputs = rows
        self.route_rows, self.recovery_rows = [np.array(natural)], [np.array(recovery)]
        self.gathered = []

    def gather(self, selection, device="cpu"):
        self.gathered.extend(selection.tolist())
        return self.inputs[selection].to(device)


@pytest.mark.parametrize("down", [False, True])
def test_routed_selection_floor_and_down_covariance_match_fp64(down):
    torch.manual_seed(787)
    rows = torch.randn(260, 16).to(torch.bfloat16)
    natural, recovery = list(range(193)), [211, 251, 199, 207]
    spool = Spool(rows, natural, recovery)
    projections = {name: torch.randn(8, 16).to(torch.bfloat16) for name in ("w1", "w3")} if down else None
    raw, evidence = k31.capture_routed_hessian(spool, 0, projections=projections, chunk_rows=129, device="cpu")
    selected = rows[natural + recovery]
    if down:
        gate = torch.nn.functional.linear(selected, projections["w1"]).float()
        up = torch.nn.functional.linear(selected, projections["w3"]).float()
        selected = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
                    * (25 * torch.tanh(up / 25))).to(torch.bfloat16)
    oracle = selected.double().T @ selected.double()
    oracle.diagonal().add_(1024 - len(natural) - len(recovery))
    assert float(torch.linalg.norm(raw["H"].double() - oracle) / torch.linalg.norm(oracle)) < 1e-6
    assert spool.gathered == natural + recovery
    assert raw["count"] == evidence["effective_rows"] == 1024
    assert evidence["natural_rows"] == 193 and evidence["augmented_rows"] == 4
    assert evidence["identity_rows"] == 827
    assert evidence["ranks"] == [17, 32]


class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = nn.Linear(4, 2, bias=False)

    def forward(self, hidden, latent):
        assert hidden.dtype == latent.dtype == torch.float32
        return self.wk(hidden)


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.layer_idx = layer
        self.self_attn = nn.Module()
        self.self_attn.indexer = Indexer()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.experts = nn.ModuleList([nn.Module()])
        self.block_sparse_moe.experts[0].w1 = nn.Linear(32, 2, bias=False)


def test_source_deferred_experts_one_active_reader_and_f32_indexer(tmp_path, monkeypatch):
    snapshot = tmp_path / k31.REVISION
    snapshot.mkdir()
    prefix = "language_model.model.layers.3."
    name = prefix + "self_attn.indexer.wk.weight"
    packed_name = prefix + "block_sparse_moe.experts.0.w1.weight_packed"
    scale_name = prefix + "block_sparse_moe.experts.0.w1.weight_scale"
    weight = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
    save_file({name: weight, packed_name: torch.full((2, 16), 0x32, dtype=torch.uint8),
               scale_name: torch.full((2, 1), 127, dtype=torch.uint8)}, str(snapshot / "a.safetensors"))
    save_file({"other": torch.ones(1)}, str(snapshot / "b.safetensors"))
    config = {"model_type": "kimi_k31", "architectures": ["KimiK31ForConditionalGeneration"],
                  "text_config": {"num_hidden_layers": 4}}
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        name: "a.safetensors", packed_name: "a.safetensors", scale_name: "a.safetensors", "other": "b.safetensors"}}))
    reference = ModuleType("families.kimi_k3.model")
    reference.load_official = lambda _: (SimpleNamespace(KimiDecoderLayer=Block), SimpleNamespace(),
                                         {"format": "mxfp4-pack-quantized"})
    reference.source_identity = lambda _: {"revision": k31.REVISION}
    monkeypatch.setitem(sys.modules, "families.kimi_k3.model", reference)
    import safetensors
    original, opens = safetensors.safe_open, []
    def observed(path, **kwargs):
        opens.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(safetensors, "safe_open", observed)
    source = k31.KimiK31Source(snapshot, reference_root=tmp_path)
    block = source.load_block(3, "cpu", include_experts=False)
    assert block.block_sparse_moe.experts[0].w1.weight.is_meta
    assert block.self_attn.indexer.wk.weight.dtype == torch.float32
    hidden, latent = torch.ones(2, 4, dtype=torch.bfloat16), torch.ones(2, 1, dtype=torch.bfloat16)
    actual = block.self_attn.indexer(hidden, latent)
    assert torch.equal(actual, torch.nn.functional.linear(hidden.float(), weight.float()))
    source.materialize_routed_experts(block, "cpu")
    expected = torch.tensor([1., 1.5]).repeat(16).repeat(2, 1).to(torch.bfloat16)
    assert torch.equal(block.block_sparse_moe.experts[0].w1.weight, expected)
    # The core replay phase asks to materialize experts again. Already-installed
    # serialized K2 reconstructions must survive that call unchanged.
    with torch.no_grad():
        block.block_sparse_moe.experts[0].w1.weight.fill_(9)
    source.materialize_routed_experts(block, "cpu")
    assert torch.equal(block.block_sparse_moe.experts[0].w1.weight, torch.full_like(expected, 9))
    assert opens == ["a.safetensors"]
    cloned = source.tensor(name)
    cloned.zero_()
    assert torch.equal(source.tensor(name), weight)
    source.tensor("other")
    assert opens == ["a.safetensors", "b.safetensors"]
    source.close()
    assert source._reader is None
    assert not block.training


@pytest.mark.parametrize("bad", ["nonzero", "negative-zero", "wrong-dtype", "wrong-length"])
def test_padded_a_log_requires_exact_zero_f32_tail(bad):
    source = object.__new__(k31.KimiK31Source)
    source.text_config = {"linear_attn_config": {"num_heads": 96}}
    name = "language_model.model.layers.0.self_attn.A_log"
    source.weight_map = {name: "unused"}
    value = torch.cat((torch.linspace(-2, 2, 96), torch.zeros(32)))
    source.tensor = lambda *a: value
    assert torch.equal(source.decoded(name), value)
    if bad == "nonzero": value[127] = 1
    elif bad == "negative-zero": value[127] = -0.
    elif bad == "wrong-dtype": value = value.to(torch.bfloat16)
    elif bad == "wrong-length": value = value[:127]
    with pytest.raises(ValueError, match="A_log padding"):
        source.decoded(name)
