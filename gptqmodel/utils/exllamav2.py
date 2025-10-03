# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch


class ScratchSpace:
    def __init__(self, scratch_bytes, dev):
        self.scratch_bytes = scratch_bytes
        self.scratch = torch.empty(
            self.scratch_bytes // 2,
            dtype=torch.float16,
            device=dev,
        )

    def get_slice(self, size_bytes):
        size_halfs = next_multiple(size_bytes, 128) // 2
        scratch_slice = self.scratch.narrow(0, 0, size_halfs)

        return scratch_slice


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple
