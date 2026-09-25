# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Layout reference: MLX (Apple Inc., MIT), mlx/nn/layers/quantized.py.
"""Validated GPTQ/AWQ checkpoint holders for MLX quantized linear inference."""

import platform
from importlib import import_module

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.mlx_packing import repack_awq_4bit, repack_gptq_4bit
from .torch import TorchLinear
from .torch_awq import AwqTorchLinear


class _MlxLinearContract:
    """Share the MLX runtime and packed-layout checks across GPTQ and AWQ."""

    def __init__(self, *args, **kwargs):
        if kwargs.setdefault("backend", BACKEND.MLX) != BACKEND.MLX:
            raise ValueError("MLX quantized linear requires backend=BACKEND.MLX")
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_once(cls):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False, NotImplementedError("MLX quantized linear requires Apple Silicon")
        try:
            import mlx.core as mx
            import_module("mlx_lm")
        except ImportError:
            return False, ImportError("Install gptqmodel[mlx] for MLX inference")
        if not mx.metal.is_available():
            return False, NotImplementedError("MLX quantized linear requires Metal")
        return True, None

    @classmethod
    def _validate(cls, **kwargs):
        ok, err = super()._validate(**kwargs)
        if not ok:
            return ok, err
        group_size = kwargs.get("group_size", 128)
        in_features = kwargs.get("in_features")
        effective_group_size = in_features if group_size == -1 else group_size
        if effective_group_size not in (32, 64, 128):
            return False, NotImplementedError(
                f"MLX affine 4-bit requires group_size 32, 64, or 128; got {effective_group_size}"
            )
        return True, None

    @classmethod
    def source_compatible(cls, module):
        if not isinstance(module, cls.SOURCE_LINEAR):
            return False
        ok, _ = cls.validate(
            bits=module.bits,
            group_size=module.requested_group_size,
            desc_act=module.desc_act,
            sym=module.sym,
            in_features=module.in_features,
            out_features=module.out_features,
            pack_dtype=module.pack_dtype,
            dtype=module.scales.dtype,
            device=DEVICE.MPS,
            adapter=module.adapter,
        )
        return ok and cls._source_layout_compatible(module)

    @classmethod
    def pack_source(cls, module):
        """Return MLX affine weight arrays and the layer's quantization config."""
        if not cls.source_compatible(module):
            raise ValueError(f"{cls.__name__} cannot consume this packed layer")
        weight, scales, biases = cls.REPACK(
            module.qweight.detach().to("cpu").numpy(),
            module.qzeros.detach().to("cpu").numpy(),
            module.scales.detach().to("cpu", torch.float16).numpy(),
            module.in_features,
            module.out_features,
        )
        group_size = module.in_features if module.requested_group_size == -1 else module.group_size
        return weight, scales, biases, {"group_size": group_size, "bits": 4, "mode": "affine"}

    def forward(self, _x):
        raise RuntimeError("MLX checkpoint holders must be converted to mlx.nn.QuantizedLinear before inference")


class MlxQuantLinear(_MlxLinearContract, TorchLinear):
    """GPTQ v2 holder validated by GPT-QModel, then transferred to MLX."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Loader AUTO selects MLX only for inference; direct backend selection uses this registry entry.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    REQUIRES_FORMAT_V2 = True

    SOURCE_LINEAR = TorchLinear
    REPACK = staticmethod(repack_gptq_4bit)

    @staticmethod
    def _source_layout_compatible(module):
        if module.qzero_format() != 2 or module.planar:
            return False
        groups = module.in_features // module.group_size
        if (tuple(module.qweight.shape) != (module.in_features // 8, module.out_features)
                or tuple(module.qzeros.shape) != (groups, module.out_features // 8)
                or tuple(module.scales.shape) != (groups, module.out_features)
                or tuple(module.g_idx.shape) != (module.in_features,)):
            return False
        expected = torch.arange(module.in_features, device=module.g_idx.device) // module.group_size
        return torch.equal(module.g_idx, expected)


class AwqMlxQuantLinear(_MlxLinearContract, AwqTorchLinear):
    """AWQ GEMM holder validated by GPT-QModel, then transferred to MLX."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = AwqTorchLinear
    REPACK = staticmethod(repack_awq_4bit)

    @staticmethod
    def _source_layout_compatible(module):
        groups = module.in_features // module.group_size
        return (
            tuple(module.qweight.shape) == (module.in_features, module.out_features // 8)
            and tuple(module.qzeros.shape) == (groups, module.out_features // 8)
            and tuple(module.scales.shape) == (groups, module.out_features)
        )
