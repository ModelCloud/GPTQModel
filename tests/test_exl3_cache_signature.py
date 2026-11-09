# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


def test_exllamav3_cache_signature_tracks_in_place_buffer_updates():
    module = object.__new__(ExllamaV3Linear)
    module.trellis = torch.zeros((1, 1, 32), dtype=torch.int16)
    module.suh = torch.ones(16, dtype=torch.float16)
    module.svh = torch.ones(16, dtype=torch.float16)
    module.su = None
    module.sv = None
    module.bias = None
    module.mcg = torch.zeros((), dtype=torch.int32)
    module.mul1 = None

    before = module._current_signature()
    module.suh.add_(1)
    after = module._current_signature()

    assert before != after
