# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Micro regression tests for the native_kernel_replay quantization toggle.

Runs OPT-125m with native_kernel_replay=False and True and compares the
result against the built-in arc_challenge baseline. This is a small, fast
end-to-end quantization path; the full Qwen3-8B sweep lives in
``test_qwen3_8b_default_sweep.py``.
"""

from model_test import ModelTest


class _TestOptNativeKernelReplayBase(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1920
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2253
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    INPUTS_MAX_LENGTH = 2048


class TestOptNativeKernelReplayFalse(_TestOptNativeKernelReplayBase):
    """Baseline: dense reconstructed-weight replay (legacy behavior)."""

    NATIVE_KERNEL_REPLAY = False

    def test_opt_native_kernel_replay_false(self):
        self.quantize_and_evaluate()


class TestOptNativeKernelReplayTrue(_TestOptNativeKernelReplayBase):
    """Native-kernel replay path: the per-layer replay runs through a packed
    TorchLinear/Marlin module and is then restored to the original dense module
    so final packing can proceed normally."""

    NATIVE_KERNEL_REPLAY = True

    def test_opt_native_kernel_replay_true(self):
        self.quantize_and_evaluate()
