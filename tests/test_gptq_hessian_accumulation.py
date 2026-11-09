# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_gptq_hessian_finalization_accumulates_later_batches():
    task = GPTQ(
        torch.nn.Linear(2, 1, bias=False),
        QuantizeConfig(bits=4, group_size=-1, fallback=None),
    )
    task.quantizer.configure(perchannel=True)
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    task.add_batch(first, None)
    task.finalize_hessian()
    task.add_batch(second, None)
    actual = task.finalize_hessian()
    all_rows = torch.cat((first, second)).double()
    expected = 2 * all_rows.T @ all_rows / len(all_rows)

    torch.testing.assert_close(actual.double(), expected)
    assert task.nsamples == len(all_rows)


def test_gptq_adopted_hessian_accumulates_later_batches():
    config = QuantizeConfig(bits=4, group_size=-1, fallback=None)
    leader = GPTQ(torch.nn.Linear(2, 1, bias=False), config)
    follower = GPTQ(torch.nn.Linear(2, 1, bias=False), config)
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    leader.add_batch(first, None)
    leader.finalize_hessian()
    follower.adopt_hessian_from(leader)
    follower.add_batch(second, None)
    actual = follower.finalize_hessian()
    all_rows = torch.cat((first, second)).double()
    expected = 2 * all_rows.T @ all_rows / len(all_rows)

    torch.testing.assert_close(actual.double(), expected)
    assert follower.nsamples == len(all_rows)
    assert follower._hessian_total_samples == len(all_rows)
