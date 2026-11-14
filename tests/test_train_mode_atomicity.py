# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A failed ``model.train()`` must not leave a partially-updated module tree."""

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear


class _TrainableKernel(BaseQuantLinear):
    SUPPORTS_TRAINING = True


class _InferenceOnlyKernel(BaseQuantLinear):
    SUPPORTS_TRAINING = False


class _TestModel(BaseQModel):
    pass


def _kernel(cls):
    kernel = object.__new__(cls)
    torch.nn.Module.__init__(kernel)
    return kernel


def _model(*modules):
    model = object.__new__(_TestModel)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Sequential(*modules)
    return model


def test_failed_train_switch_leaves_no_partial_training_state():
    trainable = _kernel(_TrainableKernel)
    inference_only = _kernel(_InferenceOnlyKernel)
    model = _model(trainable, inference_only)
    model.eval()
    assert not trainable.training
    assert not inference_only.training

    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(True)

    # The rejection happens before nn.Module.train() recurses, so the earlier
    # trainable kernel is not left in training mode.
    assert not model.training
    assert not trainable.training
    assert not inference_only.training


def test_train_switch_succeeds_when_every_kernel_is_trainable():
    kernel = _kernel(_TrainableKernel)
    model = _model(kernel)
    model.eval()

    model.train(True)

    assert model.training
    assert kernel.training
