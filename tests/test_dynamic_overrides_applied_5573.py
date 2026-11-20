"""#5573: per-module ``dynamic`` overrides must reach the quantizer.

``dynamic`` accepts every known quantize-config field, range-checks the
numeric ones and serializes the rule, but each processor used to copy only a
hand-written subset. These tests pin the config -> consumer chain for RTN,
GPTQ and Paro, including the grouped Paro scopes that can only apply a subset
and must reject the rest instead of silently dropping it.
"""

import threading

import pytest
import torch

from gptqmodel.looper import paro_processor as paro_processor_module
from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.quantization.config import (
    GPTQConfig,
    ParoConfig,
    RTNConfig,
    clone_weight_only_config_for_module,
)
from gptqmodel.quantization.rtn import RTN


MODULE_NAME = "model.layers.0.self_attn.q_proj"


class _Captured(Exception):
    """Stop a processor after it has handed its per-module values downstream."""


def _fake_linear(full_name: str = MODULE_NAME, in_features: int = 8) -> torch.nn.Linear:
    module = torch.nn.Linear(in_features, in_features, bias=False)
    module.full_name = full_name
    module.name = full_name.rsplit(".", 1)[-1]
    module.layer_index = 0
    module.state = {}
    return module


def _bare_paro_processor(qcfg: ParoConfig) -> ParoProcessor:
    processor = object.__new__(ParoProcessor)
    processor.qcfg = qcfg
    processor._module_qcfg_cache = {}
    processor.tasks = {}
    processor.fallback = None
    processor.calculate_w_wq_diff = False
    processor.lock = threading.Lock()
    return processor


@pytest.mark.cpu
def test_rtn_dynamic_mse_override_changes_the_quantized_weight():
    """A per-module ``mse`` rule must produce the same weight as top-level ``mse``."""

    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 64, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(64, 256) * 0.02)

    base = RTNConfig(bits=4, group_size=128, offload_to_disk=False)
    top_level = RTNConfig(bits=4, group_size=128, mse=2.0, offload_to_disk=False)
    dynamic = RTNConfig(
        bits=4,
        group_size=128,
        dynamic={"+:.*": {"mse": 2.0}},
        offload_to_disk=False,
    )
    clone = clone_weight_only_config_for_module(dynamic, MODULE_NAME)

    assert clone is not None
    assert float(clone.mse.norm) == pytest.approx(2.0)
    assert float(dynamic.mse.norm) == pytest.approx(0.0)

    def quantize(cfg):
        return RTN(module=linear, qcfg=cfg).quantize()[0].float()

    base_q = quantize(base)
    top_q = quantize(top_level)
    dynamic_q = quantize(clone)

    assert not torch.equal(base_q, top_q)
    assert torch.equal(top_q, dynamic_q)


@pytest.mark.cpu
def test_gptq_dynamic_damp_auto_increment_is_copied_to_the_module_clone():
    """GPTQ rebuilds the Hessian from the clone, so ``damp_auto_increment`` must be there."""

    dynamic = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={"+:.*q_proj": {"damp_auto_increment": 0.5}},
        offload_to_disk=False,
    )
    clone = clone_gptq_config_for_module(dynamic, MODULE_NAME)

    assert clone is not None
    assert clone is not dynamic
    assert clone.damp_auto_increment == pytest.approx(0.5)
    assert dynamic.damp_auto_increment == pytest.approx(0.01)

    unmatched = clone_gptq_config_for_module(dynamic, "model.layers.0.mlp.down_proj")
    assert unmatched is not None
    assert unmatched.damp_auto_increment == pytest.approx(0.01)


@pytest.mark.cpu
def test_paro_module_clone_applies_optimizer_fields_without_mutating_the_run_config():
    """The Paro clone must expose every applied field while the run config stays intact."""

    qcfg = ParoConfig(
        bits=4,
        group_size=128,
        offload_to_disk=False,
        dynamic={
            "+:.*q_proj": {
                "krot": 1,
                "opt_rotation_epochs": 0,
                "opt_rotation_lr": 10.0,
                "opt_train_samples": 64,
                "opt_validation_samples": 16,
                "opt_seed": 1234,
            }
        },
    )
    processor = _bare_paro_processor(qcfg)
    module = _fake_linear()

    clone = processor._module_qcfg(module)

    assert clone is not processor.qcfg
    assert clone.krot == 1
    assert clone.opt_rotation_epochs == 0
    assert float(clone.opt_rotation_lr) == pytest.approx(10.0)
    assert processor._target_sample_rows(clone) == 80
    assert processor._stage_learning_rates(clone)[0] == pytest.approx(10.0)
    # The default 64-row reference batch is not scaled.
    assert processor._stage_learning_rates()[0] == pytest.approx(0.05)

    # The cache returns the same clone and the run-level config is untouched.
    assert processor._module_qcfg(module) is clone
    assert processor.qcfg.krot == 8
    assert processor.qcfg.opt_rotation_epochs == 10


@pytest.mark.cpu
def test_paro_quantize_one_module_forwards_the_per_module_clone(monkeypatch):
    """Guard the module-scope optimizer call against reading ``self.qcfg`` again."""

    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(paro_processor_module, "optimize_paro_linear", fake_optimize)

    qcfg = ParoConfig(
        bits=4,
        group_size=128,
        opt_stage_impl="reference",
        offload_to_disk=False,
        dynamic={
            "+:.*q_proj": {
                "krot": 1,
                "opt_rotation_epochs": 0,
                "opt_finetune_epochs": 2,
                "opt_rotation_lr": 10.0,
                "opt_weight_lr": 0.5,
                "opt_quantizer_lr": 0.25,
                "opt_batch_size": 32,
                "opt_train_samples": 64,
                "opt_validation_samples": 16,
                "opt_pair_ratio": 0.5,
                "opt_optimizer": "sgd",
                "opt_seed": 1234,
            }
        },
    )
    processor = _bare_paro_processor(qcfg)
    module = _fake_linear()
    inputs = torch.randn(16, 8)

    with pytest.raises(_Captured):
        processor._quantize_one_module(module, inputs)

    clone = processor._module_qcfg(module)
    assert captured["krot"] == 1
    assert captured["rotation_epochs"] == 0
    assert captured["finetune_epochs"] == 2
    assert captured["rotation_lr"] == pytest.approx(10.0)
    assert captured["weight_lr"] == pytest.approx(0.5)
    assert captured["quantizer_lr"] == pytest.approx(0.25)
    assert captured["batch_size"] == 32
    assert captured["train_rows"] == 64
    assert captured["val_rows"] == 16
    assert captured["pair_ratio"] == pytest.approx(0.5)
    assert captured["optimizer_name"] == "sgd"
    assert captured["seed"] == processor._module_seed(0, MODULE_NAME, clone)


@pytest.mark.cpu
def test_paro_grouped_scope_applies_per_module_rotation_keys(monkeypatch):
    """Grouped scopes must still apply the keys they can build per module."""

    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(paro_processor_module, "build_random_rotation_buffers", fake_builder)

    qcfg = ParoConfig(
        bits=4,
        group_size=8,
        opt_scope="layer",
        offload_to_disk=False,
        dynamic={"+:.*q_proj": {"krot": 1, "opt_pair_ratio": 0.25, "opt_seed": 7}},
    )
    processor = _bare_paro_processor(qcfg)
    module = _fake_linear()

    with pytest.raises(_Captured):
        processor._build_group_optim_linear(module)

    clone = processor._module_qcfg(module)
    assert captured["krot"] == 1
    assert captured["pair_ratio"] == pytest.approx(0.25)
    assert captured["seed"] == processor._module_seed(0, MODULE_NAME, clone)


@pytest.mark.cpu
def test_paro_module_scope_accepts_shared_optimizer_keys_per_module():
    """Module scope runs one optimizer per linear, so every optimizer key applies."""

    cfg = ParoConfig(
        bits=4,
        group_size=128,
        offload_to_disk=False,
        dynamic={"+:.*": {"opt_rotation_epochs": 0, "opt_rotation_lr": 0.5}},
    )

    assert dict(cfg.dynamic)["+:.*"]["opt_rotation_epochs"] == 0


@pytest.mark.cpu
def test_paro_grouped_scope_rejects_optimizer_keys_it_cannot_apply():
    """A grouped scope has one optimizer, so a per-module LR/epoch must not be accepted."""

    with pytest.raises(ValueError, match="opt_rotation_epochs"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_scope="layer",
            offload_to_disk=False,
            dynamic={"+:.*": {"opt_rotation_epochs": 0}},
        )


@pytest.mark.cpu
def test_paro_rejects_per_module_opt_scope():
    """``opt_scope`` selects the processor-wide mode and cannot be per module."""

    with pytest.raises(ValueError, match="opt_scope"):
        ParoConfig(
            bits=4,
            group_size=128,
            offload_to_disk=False,
            dynamic={"+:.*": {"opt_scope": "module"}},
        )
