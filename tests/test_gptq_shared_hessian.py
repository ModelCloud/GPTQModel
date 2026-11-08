from __future__ import annotations

import torch

from gptqmodel.looper.exllamav3_processor import prepare_exl3_hessian
from gptqmodel.quantization.gptq import GPTQ


def test_gate_up_share_one_capture_without_double_accumulation() -> None:
    gate = GPTQ(torch.nn.Linear(4, 3, bias=False))
    up = GPTQ(torch.nn.Linear(4, 3, bias=False))
    up.share_hessian_state_from(gate)

    rows = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0]],
        dtype=torch.float32,
    )
    gate.add_batch(rows, torch.empty(0), batch_index=0)
    up.add_batch(rows.mul(7), torch.empty(0), batch_index=0)

    assert gate.nsamples == up.nsamples == 2
    assert gate._device_hessian_partials is up._device_hessian_partials
    assert len(gate._device_hessian_partials) == 1

    normalized = gate.snapshot_hessian(device := torch.device("cpu"))
    expected = rows.T @ rows
    assert torch.equal(normalized, expected)

    gate_raw = prepare_exl3_hessian(
        gate,
        target_device=device,
        module_full_name="gate",
    )
    up_raw = prepare_exl3_hessian(
        up,
        target_device=device,
        module_full_name="up",
    )
    assert gate_raw.data_ptr() != up_raw.data_ptr()
    assert torch.equal(gate_raw, expected)
    assert torch.equal(up_raw, expected)
    assert torch.equal(gate.H, expected)


def test_shared_hessian_survives_until_last_projection_releases_it() -> None:
    gate = GPTQ(torch.nn.Linear(2, 2, bias=False))
    up = GPTQ(torch.nn.Linear(2, 2, bias=False))
    up.share_hessian_state_from(gate)
    gate.H = torch.eye(2)
    state = gate._hessian_state

    gate.free()
    assert state.hessian is not None
    up.free()
    assert state.hessian is None
