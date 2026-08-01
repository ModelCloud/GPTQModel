# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-model Qwen3-8B default-config sweep for 4-bit / group_size=128.

Each class is one experimental arm. Run them one per GPU with
    GPTQMODEL_MODEL_TEST_MODE=slow
    CUDA_VISIBLE_DEVICES=N

Keep bits=4 and group_size=128 fixed; all other quantization and calibration
parameters are open to experimentation.
"""

from gptqmodel import ScaleSearchConfig
from gptqmodel.quantization import FORMAT
from model_test import ModelTest


class _TestQwen3_8BSweepBase(ModelTest):
    """Base class for the Qwen3-8B 4-bit / g128 full-model sweep."""

    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-8B"
    EVAL_BATCH_SIZE = 64
    QUANT_BATCH_SIZE = 1
    DELETE_QUANTIZED_MODEL = True
    USE_FLASH_ATTN = True
    OFFLOAD_TO_DISK = True
    TRUST_REMOTE_CODE = False
    EVAL_SINGLE_GPU = True

    # Calibration data defaults (overridable per arm)
    DATASET_SIZE = 512
    DATASET_CONCAT_SIZE = 2048
    DATASET_SORT = "desc"

    # Extra GPTQConfig knobs not exposed by ModelTest
    STATIC_GROUPS = False
    TRUE_SEQUENTIAL = True
    NATIVE_KERNEL_REPLAY = False

    # Fixed by sweep charter
    BITS = 4
    GROUP_SIZE = 128
    FORMAT = FORMAT.GPTQ
    SYM = True

    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": False,
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {"value": 0.85, "floor_pct": 0.30, "ceil_pct": 0.30},
        },
        "mmlu_stem": {
            "chat_template": False,
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 64,
                "max_rows": 1024,
            },
            "acc": {"value": 0.70, "floor_pct": 0.30, "ceil_pct": 0.30},
        },
        "mmlu": {
            "chat_template": False,
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "max_rows": 1024,
            },
            "acc": {"value": 0.65, "floor_pct": 0.30, "ceil_pct": 0.30},
        },
        "arc_challenge": {
            "chat_template": True,
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 64,
            },
            "acc": {"value": 0.35, "floor_pct": 0.30, "ceil_pct": 0.30},
            "acc_norm": {"value": 0.35, "floor_pct": 0.30, "ceil_pct": 0.30},
        },
    }

    def _build_quantize_config(self):
        cfg = super()._build_quantize_config()
        cfg.static_groups = self.STATIC_GROUPS
        cfg.true_sequential = self.TRUE_SEQUENTIAL
        cfg.native_kernel_replay = self.NATIVE_KERNEL_REPLAY
        return cfg

    def _run_quant(self):
        self.quantize_and_evaluate()


class TestQwen3_8B_ConfigA_GarBaseline(_TestQwen3_8BSweepBase):
    """Arm A: current default-style GAR, no scale search."""

    SAVE_PATH = "/tmp/qwen3_8b_configA_gar_baseline"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = None
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigB_DescAct(_TestQwen3_8BSweepBase):
    """Arm B: activation ordering (desc_act) instead of GAR."""

    SAVE_PATH = "/tmp/qwen3_8b_configB_descact"
    ACT_GROUP_AWARE = False
    DESC_ACT = True
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = None
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigC_DescActStatic(_TestQwen3_8BSweepBase):
    """Arm C: desc_act with static_groups (saves g_idx memory while keeping ordering)."""

    SAVE_PATH = "/tmp/qwen3_8b_configC_descact_static"
    ACT_GROUP_AWARE = False
    DESC_ACT = True
    STATIC_GROUPS = True
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = None
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigD_GarHessianScaleSearch(_TestQwen3_8BSweepBase):
    """Arm D: GAR with Hessian scale-search objective."""

    SAVE_PATH = "/tmp/qwen3_8b_configD_gar_hessian"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.HESSIAN
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigE_GarActivationScaleSearch(_TestQwen3_8BSweepBase):
    """Arm E: GAR with activation-diagonal scale-search objective."""

    SAVE_PATH = "/tmp/qwen3_8b_configE_gar_activation"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.ACTIVATION
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigF_GarHybridScaleSearch(_TestQwen3_8BSweepBase):
    """Arm F: GAR with hybrid Hessian/activation scale-search objective."""

    SAVE_PATH = "/tmp/qwen3_8b_configF_gar_hybrid"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.HYBRID
    MSE = 0.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigG_GarMseScaleSearch(_TestQwen3_8BSweepBase):
    """Arm G: GAR with MSE scale-search objective."""

    SAVE_PATH = "/tmp/qwen3_8b_configG_gar_mse"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MSE
    MSE = 2.0

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigH_GarActivationScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm H: GAR + activation scale-search with packed native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configH_gar_activation_native_replay"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.ACTIVATION
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigI_GarMarlinScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm I: GAR + Marlin scale-search (kernel output loss) + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configI_gar_marlin_native_replay"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigJ_GarMarlinActivationScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm J: GAR + Marlin activation-diagonal scale-search + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configJ_gar_marlin_activation_native_replay"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN_ACTIVATION
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigK_GarMarlinMseScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm K: GAR + Marlin MSE scale-search + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configK_gar_marlin_mse_native_replay"
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN_MSE
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigL_NoGarMarlinScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm L: GAR disabled + Marlin scale-search + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configL_no_gar_marlin_native_replay"
    ACT_GROUP_AWARE = False
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigM_NoGarMarlinActivationScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm M: GAR disabled + Marlin activation scale-search + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configM_no_gar_marlin_activation_native_replay"
    ACT_GROUP_AWARE = False
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN_ACTIVATION
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()


class TestQwen3_8B_ConfigN_NoGarMarlinMseScaleSearchNativeReplay(_TestQwen3_8BSweepBase):
    """Arm N: GAR disabled + Marlin MSE scale-search + native-kernel replay."""

    SAVE_PATH = "/tmp/qwen3_8b_configN_no_gar_marlin_mse_native_replay"
    ACT_GROUP_AWARE = False
    DESC_ACT = False
    DAMP_PERCENT = 0.05
    SCALE_SEARCH = ScaleSearchConfig.MARLIN_MSE
    MSE = 0.0
    NATIVE_KERNEL_REPLAY = True

    def test_qwen3_8b_quant(self):
        self._run_quant()
