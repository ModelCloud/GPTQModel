"""The RTN fallback path of GPTQ.quantize() and free() release the Hessian
partials add_batch accumulated; before, only materialize_global_hessian did,
and the task object lingers in processor.tasks until layer end. CPU-only.
"""

import pytest
import torch
import torch.nn as nn
import transformers

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import FallbackStrategy
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.fallback import should_use_fallback


COLUMNS = 64
ROWS = 32
GROUP_SIZE = 16
EXPECTED_ROWS = 64
FALLBACK_ROWS = 8  # < 75% of EXPECTED_ROWS -> fallback branch
SOLVE_ROWS = EXPECTED_ROWS + 64  # >= 75% -> GPTQ solve


def _task(*, rows_fed: int, named: bool = True) -> GPTQ:
    """A GPTQ task over a tiny Linear fed `rows_fed` calibration rows.

    fallback="75%" of EXPECTED_ROWS, mirroring test_fallback.py
    `test_gptq_fallback_threshold_triggers_rtn_when_samples_below_percent`,
    plus the NamedModule wrap the looper uses.
    """
    torch.manual_seed(0)
    linear = nn.Linear(COLUMNS, ROWS, bias=False)
    module = linear
    if named:
        module = NamedModule(
            linear, name="mlp.up_proj", full_name="model.layers.0.mlp.up_proj",
            layer_index=0,
        )
    qcfg = QuantizeConfig(bits=4, group_size=GROUP_SIZE, fallback="75%")
    task = GPTQ(module, qcfg)
    task.fallback = qcfg.fallback
    task.expected_nsamples = EXPECTED_ROWS
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.randn(rows_fed, COLUMNS), None)
    return task


def _assert_partials_released(task: GPTQ) -> None:
    assert task._device_hessian_partials == {}
    assert task._device_sample_counts == {}
    assert task._hessian_dirty is False


class TestFallbackReleasesPartials:
    def test_fixture_is_on_the_fallback_branch(self):
        task = _task(rows_fed=FALLBACK_ROWS)
        assert should_use_fallback(task.fallback, float(task.nsamples), task.expected_nsamples)
        assert len(task._device_hessian_partials) == 1
        assert sum(task._device_sample_counts.values()) == FALLBACK_ROWS
        assert task._hessian_dirty is True

    def test_quantize_fallback_releases_partials(self):
        task = _task(rows_fed=FALLBACK_ROWS)
        result = task.quantize(blocksize=GROUP_SIZE)

        _assert_partials_released(task)
        # nsamples feeds the fallback log line and the returned tuple; the
        # release must not touch it.
        assert task.nsamples == FALLBACK_ROWS
        assert result[7] == FALLBACK_ROWS
        assert result[5].startswith("fallback(rtn): ")

    def test_quantize_fallback_matches_pre_fix_arithmetic(self):
        """The old branch allocated a zero Hessian with create_H() and let
        _fallback_quantize read its device. RTN never reads the Hessian
        values, so the result must be identical when the allocation is
        skipped and the partials released instead.
        """
        baseline = _task(rows_fed=FALLBACK_ROWS)
        baseline.H = baseline.create_H(getattr(baseline.module, "target_device", None))
        expected = baseline._fallback_quantize(FallbackStrategy.RTN, GROUP_SIZE)

        task = _task(rows_fed=FALLBACK_ROWS)
        actual = task.quantize(blocksize=GROUP_SIZE)

        assert len(actual) == len(expected)
        for got, want in zip(actual[:4], expected[:4]):
            assert torch.equal(got, want)
            assert got.dtype == want.dtype
            assert got.device == want.device
        # avg_loss, damp, nsamples
        assert actual[5] == expected[5]
        assert actual[6] == expected[6]
        assert actual[7] == expected[7] == FALLBACK_ROWS

    def test_fallback_device_resolved_before_partials_are_cleared(self, monkeypatch):
        """The compute device is decided while the partials still exist (the
        clear changes what _select_hessian_target_device returns) and is
        handed to _fallback_quantize, which sees the dicts already empty.
        """
        seen: dict[str, object] = {}
        resolve = GPTQ._select_hessian_target_device

        def _spy(self: GPTQ, requested: torch.device | None) -> torch.device:
            seen["partials_at_resolve"] = len(self._device_hessian_partials)
            return resolve(self, requested)

        def _record(
            self: GPTQ,
            strategy: FallbackStrategy,
            blocksize: int,
            target_device: torch.device | None = None,
        ) -> tuple:
            seen["device"] = target_device
            seen["partials_at_entry"] = dict(self._device_hessian_partials)
            seen["counts_at_entry"] = dict(self._device_sample_counts)
            return ("stub",) * 8

        monkeypatch.setattr(GPTQ, "_select_hessian_target_device", _spy)
        monkeypatch.setattr(GPTQ, "_fallback_quantize", _record)

        task = _task(rows_fed=FALLBACK_ROWS)
        task.module.target_device = torch.device("meta")

        assert task.quantize(blocksize=GROUP_SIZE) == ("stub",) * 8
        assert seen["partials_at_resolve"] == 1
        assert seen["device"] == torch.device("meta")
        assert seen["partials_at_entry"] == {}
        assert seen["counts_at_entry"] == {}
        _assert_partials_released(task)

    def test_fallback_quantize_without_device_keeps_old_resolution(self):
        task = _task(rows_fed=FALLBACK_ROWS)
        task.H = None
        result = task._fallback_quantize(FallbackStrategy.RTN, GROUP_SIZE)
        assert result[0].device == task.module.weight.device

    def test_gptq_path_still_clears_and_solves(self):
        task = _task(rows_fed=SOLVE_ROWS)
        result = task.quantize(blocksize=GROUP_SIZE)
        _assert_partials_released(task)
        assert isinstance(result[5], float)


class TestFreeReleasesPartials:
    def test_free_clears_partials(self):
        task = _task(rows_fed=FALLBACK_ROWS)
        assert task._device_hessian_partials
        task.free()
        _assert_partials_released(task)

    def test_free_on_plain_module_clears_partials(self):
        task = _task(rows_fed=FALLBACK_ROWS, named=False)
        task.free()
        _assert_partials_released(task)


class TestColumnsIsPlainInt:
    @pytest.mark.parametrize("named", [True, False])
    def test_linear_columns_is_int(self, named):
        task = _task(rows_fed=FALLBACK_ROWS, named=named)
        assert type(task.columns) is int
        assert type(task.rows) is int
        assert task.columns == COLUMNS

    def test_conv1d_columns_is_int(self):
        conv = transformers.Conv1D(nf=ROWS, nx=COLUMNS)
        task = GPTQ(conv, QuantizeConfig(bits=4, group_size=GROUP_SIZE))
        assert type(task.columns) is int
        assert task.columns == COLUMNS
