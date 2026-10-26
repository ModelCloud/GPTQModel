# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.module_looper import ModuleLooper


class _Processor:
    pass


def test_set_processor_mask_caches_per_sample_flags():
    """The batch mask's emptiness flags are derived once, not per module hook."""

    looper = ModuleLooper.__new__(ModuleLooper)
    processor = _Processor()
    mask = torch.tensor([[True, True, False], [False, False, False]])

    looper._set_processor_mask(processor, mask)
    assert processor._mask_tls.value is mask
    assert processor._mask_tls.sample_has_tokens == [True, False]

    looper._set_processor_mask(processor, torch.ones(3, dtype=torch.bool))
    assert processor._mask_tls.sample_has_tokens is None

    looper._set_processor_mask(processor, None)
    assert processor._mask_tls.value is None
    assert processor._mask_tls.sample_has_tokens is None
