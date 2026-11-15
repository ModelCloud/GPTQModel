# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The post-load safety net must fail loudly on a corrupted model.

Issue #4129: overlapping ``GPTQModel.load()`` calls interleaved transformers'
process-wide default-dtype and meta-init state, so a model could come back with
mixed fp16/fp32 parameters, a tied weight left on the meta device, or the
process default dtype stuck at the load dtype. The loaders assert these
invariants after every build; a silent mixed model is worse than a failed load.
"""

import pytest
import torch
from torch import nn

from gptqmodel.utils.hf import assert_loaded_model_state


class _Tiny(nn.Module):
    """Model stand-in with one floating parameter per requested dtype."""

    def __init__(self, dtypes, meta_indices=()):
        super().__init__()
        for index, dtype in enumerate(dtypes):
            device = "meta" if index in meta_indices else "cpu"
            self.register_parameter(
                f"p{index}",
                nn.Parameter(torch.empty(2, dtype=dtype, device=device)),
            )


def test_accepts_a_consistent_load():
    model = _Tiny([torch.float16, torch.float16])

    assert_loaded_model_state(
        model,
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
        expected_dtype=torch.float16,
        require_uniform_dtype=True,
    )


def test_rejects_mixed_dtypes():
    model = _Tiny([torch.float16, torch.float32])

    with pytest.raises(RuntimeError, match="mixed dtypes"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            require_uniform_dtype=True,
        )


def test_rejects_a_parameter_outside_the_requested_dtype():
    model = _Tiny([torch.float16, torch.float32])

    with pytest.raises(RuntimeError, match="not all torch.float16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_rejects_a_meta_tied_weight():
    """The measured failure: ``lm_head.weight`` stayed on the meta device."""

    model = _Tiny([torch.float16, torch.float16], meta_indices=(1,))

    with pytest.raises(RuntimeError, match="stayed on the meta device"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_rejects_a_changed_process_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = _Tiny([torch.float16])
        with pytest.raises(RuntimeError, match="get_default_dtype"):
            assert_loaded_model_state(
                model,
                api_name="test.load",
                default_dtype_before=torch.float32,
                expected_dtype=torch.float16,
            )
    finally:
        torch.set_default_dtype(previous)


def test_meta_shell_only_checks_the_process_default_dtype():
    """``offload_to_disk`` builds a meta shell that is filled in later."""

    model = _Tiny([torch.float16, torch.float16], meta_indices=(0, 1))

    assert_loaded_model_state(
        model,
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
        require_materialized=False,
        require_uniform_dtype=False,
    )


def test_non_torch_model_only_checks_the_process_default_dtype():
    """The MLX runtime object has no ``named_parameters``."""

    assert_loaded_model_state(
        {"weight": torch.zeros(2)},
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
    )
