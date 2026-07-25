# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Environment helpers for the GPU allocator."""

from __future__ import annotations

import os


def force_pci_bus_order() -> None:
    """Force CUDA devices to be ordered by PCI bus id.

    This must run before any torch/cuda imports so the CUDA runtime sees the
    correct ordering.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Unable to set CUDA_DEVICE_ORDER=PCI_BUS_ID")
