# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from unittest.mock import patch

import pytest
import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import macaq as macaq_linear_module
from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization import macaq as macaq_core
from gptqmodel.quantization.config import MacaqConfig
from gptqmodel.utils import macaq_triton
from gptqmodel.utils.macaq_triton import triton_mxfp4_available


# MACaQ MXFP4 W4A4 on Llama-3.2-1B-Instruct.
#
# Fast-mode NVIDIA baseline (64 calibration rows, 64 flip sweeps,
# 128 ARC rows, chat template, eager attention, Triton W4A4):
#   arc_challenge :: accuracy,loglikelihood       0.3281
#   arc_challenge :: accuracy,loglikelihood_norm  0.2969
# R9700 ROCm 10.1, GIL=0, compiled flips by default (two runs):
#   arc_challenge :: accuracy,loglikelihood       0.3438, 0.3516
#   arc_challenge :: accuracy,loglikelihood_norm  0.3281
#
# Full-model Wikitext-2 perplexity: 128 Wikitext-2 chunks, 128 flips,
# pre_alpha=0.75, and all 112 projections quantized:
#   BF16 20.50 | MACaQ W4A16 21.81 | MACaQ W4A4 24.34
class TestLlama3_2_MACaQ(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MODEL",
        "/monster/data/model/Llama-3.2-1B-Instruct",
    )
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MACAQ_SAVE_PATH",
        "/tmp/llama3_2_macaq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    DATASET_SIZE_FAST = 64
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False

    METHOD = METHOD.MACAQ
    FORMAT = FORMAT.MXFP4_MACAQ
    BITS = 4
    GROUP_SIZE = 32
    LOAD_BACKEND = BACKEND.MACAQ_TRITON
    QUANT_BACKEND = BACKEND.AUTO
    KERNEL_QUANT = {MacaqTritonLinear}
    KERNEL_INFERENCE = {MacaqTritonLinear}

    MACAQ_N_FLIP_ITERS = 64
    MACAQ_HAD_BLOCK = 32
    MACAQ_PRE_ALPHA = 0.5

    # The stock ModelTest config builder is GPTQ-shaped; MACaQ has its own
    # settings, so build the method config directly.
    def _build_quantize_config(self):
        return MacaqConfig(
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            n_flip_iters=self.MACAQ_N_FLIP_ITERS,
            had_block=self.MACAQ_HAD_BLOCK,
            pre_alpha=self.MACAQ_PRE_ALPHA,
            offload_to_disk=self.OFFLOAD_TO_DISK,
        )

    def run_eval_tasks(self, model, backend, trust_remote_code=False):
        """Require the reloaded ARC model to execute native W4A4 activations."""

        assert backend == BACKEND.MACAQ_TRITON
        linears = [module for module in model.model.modules() if isinstance(module, MacaqTritonLinear)]
        assert linears, "The reloaded checkpoint has no MACaQ Triton projections."
        for module in linears:
            assert module.backend == BACKEND.MACAQ_TRITON
            assert module.activation_precision == "w4a4"
            assert module.W_packed.dtype == module.W_scales.dtype == torch.uint8
            assert module.W_packed.is_cuda and module.W_scales.is_cuda
            assert module.W_packed.shape == (module.out_features, module.in_features // 2)
            assert module.W_scales.shape == (module.out_features, module.in_features // 32)

        original_encoder = macaq_triton.rot_scale_quantize_mxfp4

        def checked_encoder(x, pre_scale, bh):
            packed, scales = original_encoder(x, pre_scale, bh)
            assert packed.dtype == scales.dtype == torch.uint8
            assert packed.shape == (x.shape[0], x.shape[1] // 2)
            assert scales.shape == (x.shape[0], x.shape[1] // 32)
            return packed, scales

        seen = set()
        hooks = [
            module.register_forward_hook(lambda current, _inputs, _output: seen.add(current))
            for module in linears
        ]
        try:
            with (
                patch.object(
                    macaq_linear_module,
                    "mxfp4_w4a4_linear",
                    wraps=macaq_linear_module.mxfp4_w4a4_linear,
                ) as w4a4_gemm,
                patch.object(
                    macaq_triton,
                    "rot_scale_quantize_mxfp4",
                    side_effect=checked_encoder,
                ) as activation_encoder,
            ):
                result = super().run_eval_tasks(model, backend, trust_remote_code=trust_remote_code)
        finally:
            for hook in hooks:
                hook.remove()

        assert w4a4_gemm.call_count > 0, "ARC evaluation did not run the W4A4 GEMM."
        assert activation_encoder.call_count > 0, "ARC evaluation did not pack 4-bit activations."
        assert seen == set(linears), "ARC evaluation did not exercise every MACaQ projection."
        assert all(not module._triton_disabled for module in linears), "A MACaQ projection fell back to Torch."
        return result

    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "evalution_suite_kwargs": {"max_rows": 128},
            "acc": {
                "value": 0.328125,
                "floor_pct": 0.10,
                "ceil_pct": 0.30,
            },
            "acc_norm": {
                "value": 0.296875,
                "floor_pct": 0.10,
                "ceil_pct": 0.30,
            },
        },
    }

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.296875,
                "floor_pct": 0.15,
                "ceil_pct": 0.30,
            },
            "acc_norm": {
                "value": 0.296875,
                "floor_pct": 0.15,
                "ceil_pct": 0.30,
            },
        },
    }

    def test_llama3_2_macaq(self):
        if not triton_mxfp4_available():
            pytest.skip("MACaQ Triton inference requires NVIDIA CUDA or ROCm gfx1201")
        assert self._build_quantize_config().compile_flip is True
        compiled_expected = macaq_core._compiled_flip_supported(
            torch.device("cuda", torch.cuda.current_device())
        )
        with patch.object(macaq_core, "_COMPILED_FLIP_SWEEP", None):
            self.quantize_and_evaluate()
            if compiled_expected:
                assert macaq_core._COMPILED_FLIP_SWEEP is not None, (
                    "MACaQ did not compile the default flip sweep during quantization."
                )
