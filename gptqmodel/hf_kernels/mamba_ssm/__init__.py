# Adapted from HF kernel https://huggingface.co/kernels-community/mamba-ssm/tree/main
# Copyright (c) 2024, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0

from .ops import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update
from .ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

__version__ = "2.2.4-local"
falcon_mamba_inner_fn = mamba_inner_fn

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "falcon_mamba_inner_fn",
    "selective_state_update",
    "mamba_chunk_scan_combined",
    "mamba_split_conv1d_scan_combined",
]
