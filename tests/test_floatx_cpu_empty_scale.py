# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FloatX CPU dequantization with an empty scale tensor (issue #3971).

``make_scale_spec_*`` divided by ``scale.numel()``, so an empty scale raised
SIGFPE and killed the process. It must fail with an error instead.
"""

import pytest
import torch

from gptqmodel.quantization.dtype import _load_floatx_cpu_ops, dequantize_f4_e2m1


def test_floatx_cpu_empty_scale_raises_instead_of_dividing_by_zero():
    if _load_floatx_cpu_ops() is None:
        pytest.skip("floatx CPU extension is not built")
    with pytest.raises(RuntimeError, match="must not be empty"):
        dequantize_f4_e2m1(
            torch.zeros(16, dtype=torch.uint8),
            scale=torch.empty(0, dtype=torch.float32),
            target_dtype=torch.bfloat16,
        )
