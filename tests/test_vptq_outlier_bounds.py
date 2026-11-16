# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""VPTQ outlier-index bounds (issue #3971).

Outlier codes index the outlier codebook in the CUDA kernel. The quantizer
bounds them; a checkpoint does not have to.
"""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear


def _state(outlier_indices):
    return SimpleNamespace(
        name="proj",
        indices=torch.zeros((1, 1, 1), dtype=torch.int32),
        group_num=1,
        num_indices=1,
        packed_words=1,
        enable_outlier=True,
        outlier_indices=torch.tensor(outlier_indices, dtype=torch.int32),
        outlier_num_centroids=3,
        enable_perm=False,
    )


def test_vptq_outlier_indices_accept_in_range():
    VPTQLinear._validate_loaded_state(_state([0, 2]))


@pytest.mark.parametrize("bad", [-1, 3])
def test_vptq_outlier_indices_reject_out_of_codebook(bad):
    with pytest.raises(ValueError, match="outlier index"):
        VPTQLinear._validate_loaded_state(_state([0, bad]))
