# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Layout reference: MLX (Apple Inc., MIT), mlx/nn/layers/quantized.py.
# Format references: ParoQuant (z-lab), QQQ (vLLM), GGUF (ggml-org),
# and bitsandbytes (Tim Dettmers and contributors); their licenses are noted
# in the source implementations and method-specific runtime modules.
"""Validated GPT-QModel checkpoint holders for MLX native linear inference."""

import platform
from importlib import import_module

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.mlx_packing import (repack_awq_4bit, repack_awq_gemv,
                                  repack_awq_gemv_fast, repack_gptq)
from .torch import TorchLinear
from .torch_awq import AwqTorchLinear
from .paroquant import ParoLinear
from .qqq import QQQTorchLinear
from .gguf import GGUFTorchLinear
from .fp8 import TorchFP8Linear
from .bitsandbytes import BitsAndBytesLinear
from .gemv_awq import AwqGEMVLinear
from .gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from . import BaseQuantLinear


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
        supported = (effective_group_size in (16, 32, 64, 128, 256, 512, 1024)
                     or (group_size == -1 and (in_features is None or (in_features >= 128 and in_features % 128 == 0))))
        if not supported:
            return False, NotImplementedError(
                f"MLX affine conversion cannot map group_size {effective_group_size} to a supported group"
            )
        if effective_group_size == 16 and in_features is not None and in_features % 32:
            return False, NotImplementedError("MLX group-16 inference requires in_features divisible by 32")
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
    def mlx_params(cls, module):
        # MLX affine accepts 32–128 values per group and no native 7-bit codes.
        # Group 16 is split across two group-32 matmuls in MlxGroup16Linear.
        return {"group_size": max(32, min(module.group_size, 128)),
                "bits": 8 if module.bits == 7 else module.bits, "mode": "affine"}

    @classmethod
    def pack_source(cls, module):
        """Return MLX affine weight arrays and the layer's quantization config."""
        if not cls.source_compatible(module):
            raise ValueError(f"{cls.__name__} cannot consume this packed layer")
        args = (
            module.qweight.detach().to("cpu").numpy(),
            module.qzeros.detach().to("cpu").numpy(),
            module.scales.detach().to("cpu", torch.float16).numpy(),
            module.in_features,
            module.out_features,
        )
        weight, scales, biases = cls.REPACK(*args, module.bits, getattr(module, "planar", False))
        params = cls.mlx_params(module)
        group_size = params["group_size"]
        if module.group_size > group_size:
            repeats = module.group_size // group_size
            scales = scales.repeat(repeats, axis=1)
            biases = biases.repeat(repeats, axis=1)
        return weight, scales, biases, params

    def forward(self, _x):
        raise RuntimeError("MLX checkpoint holders must be converted to mlx.nn.QuantizedLinear before inference")


class MlxQuantLinear(_MlxLinearContract, TorchLinear):
    """GPTQ v2 holder validated by GPT-QModel, then transferred to MLX."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Loader AUTO selects MLX only for inference; direct backend selection uses this registry entry.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0, FORMAT.GPTQ_P: 0}
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = True

    SOURCE_LINEAR = TorchLinear
    REPACK = staticmethod(repack_gptq)

    @staticmethod
    def _source_layout_compatible(module):
        # GPTQ_P checkpoints store raw zero codes despite the holder's default
        # qzero_format metadata of 1; only legacy GPTQ subtracts one on disk.
        if module.qzero_format() != 2 and module.format != FORMAT.GPTQ_P:
            return False
        groups = module.in_features // module.group_size
        if (tuple(module.qweight.shape) != (module.in_features * module.bits // 32, module.out_features)
                or tuple(module.qzeros.shape) != (groups, module.out_features * module.bits // 32)
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
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
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
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = AwqTorchLinear
    REPACK = staticmethod(repack_awq_4bit)

    @staticmethod
    def _source_layout_compatible(module):
        if isinstance(module, ParoLinear):
            return False
        groups = module.in_features // module.group_size
        return (
            tuple(module.qweight.shape) == (module.in_features, module.out_features // 8)
            and tuple(module.qzeros.shape) == (groups, module.out_features // 8)
            and tuple(module.scales.shape) == (groups, module.out_features)
        )


class AwqGemvMlxQuantLinear(_MlxLinearContract, AwqGEMVLinear):
    """AWQ GEMV output-major checkpoint holder converted to MLX affine."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMV: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = AwqGEMVLinear

    @classmethod
    def validate(cls, *args, **kwargs):
        # The CUDA GEMV holder limits normalized groups to 64/128. MLX can
        # split a full-width (-1) group into native affine groups exactly.
        return BaseQuantLinear.validate.__func__(cls, *args, **kwargs)

    @staticmethod
    def _source_layout_compatible(module):
        groups = module.in_features // module.group_size
        return (tuple(module.qweight.shape) == (module.out_features, module.in_features // 8)
                and module.qzeros.shape[0] == module.out_features
                and module.qzeros.shape[1] >= (groups + 7) // 8
                and module.scales.shape[0] == module.out_features
                and module.scales.shape[1] >= groups)

    @classmethod
    def pack_source(cls, module):
        import numpy as np

        if not cls.source_compatible(module):
            raise ValueError("AWQ GEMV layout cannot be transferred to MLX")
        weight, scales, biases = repack_awq_gemv(
            module.qweight.detach().cpu().numpy(), module.qzeros.detach().cpu().numpy(),
            module.scales.detach().to("cpu", torch.float16).numpy(),
            module.in_features, module.out_features, module.group_size,
        )
        params = cls.mlx_params(module)
        if module.group_size > params["group_size"]:
            repeats = module.group_size // params["group_size"]
            scales, biases = np.repeat(scales, repeats, axis=1), np.repeat(biases, repeats, axis=1)
        return weight, scales, biases, params


class _MlxGemvFastTransfer(_MlxLinearContract):
    """Shared packed transfer for GEMV_FAST and LLM_AWQ checkpoint names."""

    @staticmethod
    def _source_layout_compatible(module):
        groups = module.in_features // module.group_size
        return (tuple(module.qweight.shape) == (module.out_features // 4, module.in_features)
                and tuple(module.scales.shape) == tuple(module._runtime_zeros().shape)
                and module.scales.shape[1] == module.out_features
                and module.scales.shape[0] >= groups)

    @classmethod
    def pack_source(cls, module):
        import numpy as np

        if not cls.source_compatible(module):
            raise ValueError("AWQ GEMV_FAST layout cannot be transferred to MLX")
        weight, scales, biases = repack_awq_gemv_fast(
            module.qweight.detach().cpu().numpy(),
            module._runtime_zeros().detach().cpu().numpy(),
            module.scales.detach().cpu().numpy(),
            module.in_features, module.out_features, module.group_size,
        )
        params = cls.mlx_params(module)
        if module.group_size > params["group_size"]:
            repeats = module.group_size // params["group_size"]
            scales, biases = np.repeat(scales, repeats, axis=1), np.repeat(biases, repeats, axis=1)
        return weight, scales, biases, params


class AwqGemvFastMlxQuantLinear(_MlxGemvFastTransfer, AwqGEMVFastLinear):
    """AWQ GEMV_FAST holder with exact packed MLX transfer."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMV_FAST: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = AwqGEMVFastLinear


class LLMAwqMlxQuantLinear(_MlxGemvFastTransfer, LLMAwqLinear):
    """LLM_AWQ holder with exact packed MLX transfer."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.LLM_AWQ: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = LLMAwqLinear


class ParoMlxQuantLinear(_MlxLinearContract, ParoLinear):
    """ParoQuant holder with the source AWQ codes and learned rotation state."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.PARO]
    SUPPORTS_FORMATS = {FORMAT.PAROQUANT: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = ParoLinear
    REPACK = staticmethod(repack_awq_4bit)

    @staticmethod
    def _source_layout_compatible(module):
        groups = module.in_features // module.group_size
        if (tuple(module.qweight.shape) != (module.in_features, module.out_features // 8)
                or tuple(module.qzeros.shape) != (groups, module.out_features // 8)
                or tuple(module.scales.shape) != (groups, module.out_features)
                or tuple(module.theta.shape) != (module.krot, module.in_features // 2)
                or tuple(module.pairs.shape) != (module.krot, module.in_features)
                or tuple(module.channel_scales.shape) != (1, module.in_features)):
            return False
        pairs = module.pairs.detach().to("cpu", torch.int64).reshape(module.krot, groups, module.group_size)
        expected = torch.arange(module.group_size).expand(module.krot, groups, -1)
        return torch.equal(pairs.sort(dim=-1).values, expected)


class QQQMlxQuantLinear(_MlxLinearContract, QQQTorchLinear):
    """QQQ holder for exact INT8 weights and dynamic input quantization on MLX."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.QQQ]
    SUPPORTS_FORMATS = {FORMAT.QQQ: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = QQQTorchLinear

    @classmethod
    def source_compatible(cls, module):
        if not isinstance(module, QQQTorchLinear):
            return False
        ok, _ = cls.validate(
            bits=module.bits, group_size=-1 if module.group_size == module.in_features else module.group_size,
            desc_act=module.desc_act, sym=module.sym,
            in_features=module.in_features, out_features=module.out_features,
            pack_dtype=torch.int32, dtype=torch.float16,
            device=DEVICE.MPS, adapter=module.adapter,
        )
        return ok and cls._source_layout_compatible(module)

    @staticmethod
    def _source_layout_compatible(module):
        return (tuple(module.B.shape) == (module.in_features // 16, module.out_features * 2)
                and tuple(module.s_channel.shape) == (1, module.out_features)
                and tuple(module.s_group.shape) ==
                ((module.in_features // module.group_size, module.out_features)
                 if module.group_size != module.in_features else (0,)))

    @classmethod
    def mlx_params(cls, module):
        # Whole-row QQQ codes are signed nibbles scaled by 16. Grouped QQQ
        # rounds each group's scaled nibbles to INT8 and needs the 8-bit path.
        bits = 4 if module.group_size == module.in_features else 8
        return {"group_size": 64 if module.in_features == 64 else 128, "bits": bits, "mode": "affine"}

    @classmethod
    def pack_source(cls, module):
        import numpy as np

        if not cls.source_compatible(module):
            raise ValueError("QQQ layer cannot be transferred to MLX")
        if module.group_size == module.in_features:
            # XOR the sign bit so MLX's unsigned affine codes decode to
            # QQQ's signed two's-complement nibbles without rounding.
            codes = module._unpack_weight_codes().to("cpu", torch.uint8).numpy().T
            codes = np.ascontiguousarray(codes ^ np.uint8(8))
            lanes = codes.reshape(module.out_features, module.in_features // 8, 8)
            packed = np.zeros((module.out_features, module.in_features // 8), dtype=np.uint32)
            for lane in range(8):
                packed |= lanes[:, :, lane].astype(np.uint32) << (4 * lane)
            group_size = cls.mlx_params(module)["group_size"]
            shape = (module.out_features, module.in_features // group_size)
            scales = np.full(shape, 16, dtype=np.float32)
            biases = np.full(shape, -128, dtype=np.float32)
            return packed, scales, biases, cls.mlx_params(module)
        weight, _ = module._dequantize_weight_for_torch()
        weight = weight.detach().to("cpu", torch.int16).numpy().T
        if np.any((weight < -128) | (weight > 127)):
            raise ValueError("QQQ weights exceed the signed INT8 range")
        codes = (weight + 128).astype(np.uint8)
        packed = np.ascontiguousarray(codes).view(np.uint32).reshape(module.out_features, -1)
        group_size = cls.mlx_params(module)["group_size"]
        groups = module.in_features // group_size
        scales = np.ones((module.out_features, groups), dtype=np.float32)
        biases = np.full_like(scales, -128)
        return packed, scales, biases, cls.mlx_params(module)


class GGUFMlxQuantLinear(_MlxLinearContract, GGUFTorchLinear):
    """GGUF affine checkpoint holder using exact source block codes on MLX."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 0}
    SUPPORTS_BITS = [1, 2, 4, 5, 6, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = GGUFTorchLinear

    @classmethod
    def source_compatible(cls, module):
        from ...utils.mlx_gguf_packing import MLX_GGUF_QTYPES

        if not isinstance(module, GGUFTorchLinear) or module.gguf_tensor_qtype not in MLX_GGUF_QTYPES:
            return False
        ok, _ = cls.validate(
            bits=int(module.bits), group_size=-1, desc_act=False, sym=True,
            in_features=module.in_features, out_features=module.out_features,
            pack_dtype=torch.int32, dtype=torch.float16, device=DEVICE.MPS,
            adapter=module.adapter,
        )
        return (ok and module.in_features % module.gguf_block_size == 0
                and tuple(module.qweight.shape) ==
                (module.out_features, module.in_features // module.gguf_block_size * module.gguf_type_size))

    @classmethod
    def mlx_params(cls, module):
        qtype = module.gguf_tensor_qtype
        return {
            "group_size": {"Q2_0": 64, "Q1_0": 128, "Q1_0_g128": 128,
                           "TQ1_0": 128, "TQ2_0": 128, "NVFP4": 16}.get(qtype, 32),
            "bits": {"Q1_0": 2, "Q1_0_g128": 2, "Q2_0": 2,
                     "TQ1_0": 2, "TQ2_0": 2}.get(qtype, int(module.bits)),
            "mode": qtype.lower() if qtype in {"MXFP4", "NVFP4"} else "affine",
        }

    @classmethod
    def pack_source(cls, module):
        from ...utils.mlx_gguf_packing import repack_gguf_affine, repack_gguf_float4

        if not cls.source_compatible(module):
            raise ValueError("GGUF block layout cannot be transferred to MLX")
        repack = repack_gguf_float4 if module.gguf_tensor_qtype in {"MXFP4", "NVFP4"} else repack_gguf_affine
        return repack(
            module.qweight.detach().cpu().numpy(), module.gguf_tensor_qtype, module.in_features,
        )


class _MlxDenseContract(_MlxLinearContract):
    """Validate MLX native dense runtime used after one-time source decoding."""

    @classmethod
    def _validate(cls, **kwargs):
        # Dense weight-only formats still use BaseQuantLinear validation, but
        # have no affine group constraints from _MlxLinearContract.
        return BaseQuantLinear._validate.__func__(cls, **kwargs)

    @classmethod
    def source_compatible(cls, module):
        if not isinstance(module, cls.SOURCE_LINEAR) or module.adapter is not None:
            return False
        ok, _ = cls.validate(
            bits=module.bits, group_size=-1, desc_act=False, sym=True,
            in_features=module.in_features, out_features=module.out_features,
            pack_dtype=torch.int32, dtype=torch.float16, device=DEVICE.MPS,
        )
        return ok


class FP8MlxQuantLinear(_MlxDenseContract, TorchFP8Linear):
    """FP8 checkpoint holder decoded once for MLX native dense matmul."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.FP8]
    SUPPORTS_FORMATS = {FORMAT.FP8: 0}
    SUPPORTS_BITS = [8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = TorchFP8Linear

    @classmethod
    def source_compatible(cls, module):
        return (super().source_compatible(module)
                and module.weight_scale_semantics == "inverse"
                and tuple(module.weight.shape) == (module.out_features, module.in_features))

    @classmethod
    def dense_weight(cls, module):
        return module.dequantize_weight(device="cpu", dtype=torch.float16).T.contiguous()


class BitsAndBytesMlxQuantLinear(_MlxDenseContract, BitsAndBytesLinear):
    """BNB checkpoint holder decoded once for MLX native dense matmul."""

    SUPPORTS_BACKENDS = [BACKEND.MLX]
    SUPPORTS_METHODS = [METHOD.BITSANDBYTES]
    SUPPORTS_FORMATS = {FORMAT.BITSANDBYTES: 0}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

    SOURCE_LINEAR = BitsAndBytesLinear

    @classmethod
    def validate_once(cls):
        ok, err = _MlxLinearContract.validate_once()
        return (ok, err) if not ok else BitsAndBytesLinear.validate_once()

    @classmethod
    def dense_weight(cls, module):
        return module.dequantize_weight().detach().to("cpu", torch.float16).contiguous()
