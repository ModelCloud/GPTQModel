# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import _random_signs


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_exl3_signs_are_thread_local_and_do_not_reset_global_rng(device):
    if device.startswith("cuda") and torch.cuda.device_count() <= int(device[-1]):
        pytest.skip("CUDA device unavailable")
    torch.manual_seed(123)
    before_cpu = torch.random.get_rng_state()
    before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    expected = _random_signs(128, device=device, seed=787)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(
            pool.map(lambda _: _random_signs(128, device=device, seed=787), range(16))
        )
    assert all(torch.equal(expected, value) for value in values)
    assert not torch.equal(
        expected, _random_signs(128, device=device, seed=787, stream=1)
    )
    assert torch.equal(before_cpu, torch.random.get_rng_state())
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            before_cuda, torch.cuda.get_rng_state_all() if before_cuda else []
        )
    )
