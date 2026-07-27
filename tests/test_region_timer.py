# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import time

import pytest

from gptqmodel.utils.logger import QuantizationRegionTimer


def test_region_timer_includes_module_and_sync_regions():
    timer = QuantizationRegionTimer()

    assert "module_load" in timer._region_labels
    assert "module_move" in timer._region_labels
    assert "torch_sync" in timer._region_labels

    timer.record("module_load", 0.123, source="test_module")
    timer.record("module_move", 0.234, source="test_move")
    timer.record("torch_sync", 0.005, source="test_sync")

    stats = timer._stats
    assert stats["module_load"]["count"] == 1
    assert abs(stats["module_load"]["total"] - 0.123) < 1e-9
    assert stats["module_move"]["count"] == 1
    assert abs(stats["module_move"]["total"] - 0.234) < 1e-9
    assert stats["torch_sync"]["count"] == 1
    assert abs(stats["torch_sync"]["total"] - 0.005) < 1e-9


def test_region_timer_record_accumulates_and_tracks_source():
    timer = QuantizationRegionTimer()

    timer.record("module_load", 0.1, source="a")
    timer.record("module_load", 0.2, source="b")

    stats = timer._stats["module_load"]
    assert stats["count"] == 2
    assert abs(stats["total"] - 0.3) < 1e-9
    assert abs(stats["last"] - 0.2) < 1e-9
    assert stats["source"] == "b"
