# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import pytest
import torch
import torch.nn as nn

from gptqmodel.utils.hf import suspend_hf_weight_init


def test_suspend_hf_weight_init_restores_globals_after_exception():
    original_kaiming_uniform = torch.nn.init.kaiming_uniform_
    original_uniform = torch.nn.init.uniform_
    original_normal = torch.nn.init.normal_

    with pytest.raises(RuntimeError, match="boom"):
        with suspend_hf_weight_init():
            assert torch.nn.init.kaiming_uniform_ is not original_kaiming_uniform
            assert torch.nn.init.uniform_ is not original_uniform
            assert torch.nn.init.normal_ is not original_normal
            raise RuntimeError("boom")

    assert torch.nn.init.kaiming_uniform_ is original_kaiming_uniform
    assert torch.nn.init.uniform_ is original_uniform
    assert torch.nn.init.normal_ is original_normal

    linear = nn.Linear(32, 32, bias=False)
    assert torch.isfinite(linear.weight).all()


def _init_state():
    return (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )


def test_suspend_hf_weight_init_is_reference_counted_across_threads():
    """One caller leaving must not re-enable init for another (issue #4129)."""

    import threading

    original_state = _init_state()
    entered_a = threading.Event()
    entered_b = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()
    errors = []

    def worker(entered, release):
        try:
            with suspend_hf_weight_init():
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("init suspension was never released")
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    thread_a = threading.Thread(target=worker, args=(entered_a, release_a))
    thread_b = threading.Thread(target=worker, args=(entered_b, release_b))
    thread_a.start()
    assert entered_a.wait(timeout=10)
    thread_b.start()
    assert entered_b.wait(timeout=10)

    try:
        release_a.set()
        thread_a.join(timeout=10)
        # A left first: the patches must still be installed for B.
        assert torch.nn.init.kaiming_uniform_ is not original_state[0]
        assert torch.nn.init.uniform_ is not original_state[1]
        assert torch.nn.init.normal_ is not original_state[2]
    finally:
        release_b.set()
        thread_b.join(timeout=10)

    assert not errors
    assert _init_state() == original_state


def test_suspend_hf_weight_init_nested_entries_restore_once():
    original_state = _init_state()

    with suspend_hf_weight_init():
        patched = _init_state()
        with suspend_hf_weight_init():
            assert _init_state() == patched
        # The inner exit must not restore the real initializers.
        assert _init_state() == patched

    assert _init_state() == original_state


def test_suspend_hf_weight_init_restores_after_overlapping_exception():
    import threading

    original_state = _init_state()
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def failing_worker():
        try:
            with suspend_hf_weight_init():
                entered.set()
                release.wait(timeout=10)
                raise RuntimeError("boom")
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=failing_worker)
    thread.start()
    assert entered.wait(timeout=10)
    with suspend_hf_weight_init():
        assert _init_state() != original_state
    # The surviving caller still holds the patches.
    assert _init_state() != original_state
    release.set()
    thread.join(timeout=10)

    assert len(errors) == 1
    assert _init_state() == original_state
