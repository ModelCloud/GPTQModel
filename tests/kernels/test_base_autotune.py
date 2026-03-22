# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.models._const import DEVICE, PLATFORM
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import METHOD
from gptqmodel.utils.backend import BACKEND


class _AutotuneTestKernel(BaseQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [4]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = True
    SUPPORTS_TRAINING_USE_TORCH_KERNEL = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

    def __init__(self, *, autotune: bool):
        self.AUTOTUNE = autotune
        self.autotune_calls = 0
        super().__init__(
            bits=4,
            in_features=4,
            out_features=4,
            bias=False,
            backend=BACKEND.TORCH,
            adapter=None,
            register_buffers=False,
            validate_kwargs={
                "group_size": 4,
                "desc_act": False,
                "sym": True,
                "pack_dtype": torch.int32,
            },
        )

    def _autotune(self, x: torch.Tensor):
        self.autotune_calls += 1
        return (x.shape, x.dtype)

    def forward(self, x: torch.Tensor):
        self.maybe_autotune(x)
        return x


def test_base_quant_linear_autotune_runs_once_per_instance():
    module = _AutotuneTestKernel(autotune=True).eval()
    x = torch.randn(2, 4)

    module(x)
    module(x)

    assert module.autotune_calls == 1
    assert module.get_autotune_result() == (x.shape, x.dtype)


def test_base_quant_linear_autotune_disabled_by_default():
    module = _AutotuneTestKernel(autotune=False).eval()
    x = torch.randn(2, 4)

    module(x)
    module(x)

    assert module.autotune_calls == 0
    assert module.get_autotune_result() is None


def test_base_quant_linear_clear_autotune_reenables_autotune():
    module = _AutotuneTestKernel(autotune=True).eval()
    x = torch.randn(2, 4)

    module(x)
    assert module.autotune_calls == 1

    module.clear_autotune()
    module(x)

    assert module.autotune_calls == 2


def test_base_quant_linear_train_mode_clears_autotune_state():
    module = _AutotuneTestKernel(autotune=True).eval()
    x = torch.randn(2, 4)

    module(x)
    module.train(True)
    module.train(False)
    module(x)

    assert module.autotune_calls == 2
