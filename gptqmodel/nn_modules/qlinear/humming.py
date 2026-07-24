# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger


try:
    from gptqmodel.humming.layer import HummingLayer
    from gptqmodel.humming.schema.awq import AWQWeightSchema
    from gptqmodel.humming.schema.gptq import GPTQWeightSchema

    humming_import_exception: Optional[str] = None
except Exception as _humming_exc:  # pragma: no cover - vendored code not available
    HummingLayer = None  # type: ignore[misc,assignment]
    GPTQWeightSchema = None  # type: ignore[misc,assignment]
    AWQWeightSchema = None  # type: ignore[misc,assignment]
    humming_import_exception = str(_humming_exc)


log = setup_logger()


_HUMMING_PAD_N = 64
_HUMMING_PAD_K = 32


def _humming_dtype_supported(dtype: torch.dtype) -> bool:
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if dtype == torch.bfloat16:
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i)[0] < 8:
                return False
    return True


class _HummingLinearBase:
    """Shared helpers for Humming-backed GPTQ and AWQ quantized linear layers."""

    def _humming_device(self) -> torch.device:
        for name in ("qweight", "scales", "qzeros", "bias"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                return tensor.device
        raise RuntimeError(f"{self.__class__.__name__} requires CUDA tensors before post_init().")

    def _build_humming_layer(
        self,
        weight_schema,
        device: torch.device,
    ) -> HummingLayer:
        if HummingLayer is None:
            raise RuntimeError(f"Humming kernel is not available: {humming_import_exception}")

        torch_dtype = self.compute_dtype
        if torch_dtype not in (torch.float16, torch.bfloat16):
            torch_dtype = torch.bfloat16 if _humming_dtype_supported(torch.bfloat16) else torch.float16

        has_bias = getattr(self, "bias", None) is not None
        layer = HummingLayer(
            shape_n=self.out_features,
            shape_k=self.in_features,
            weight_config=weight_schema,
            torch_dtype=torch_dtype,
            pad_n_to_multiple=_HUMMING_PAD_N,
            pad_k_to_multiple=_HUMMING_PAD_K,
            has_bias=has_bias,
        )
        return layer.to(device)

    def _load_humming_tensors(self, layer: HummingLayer, tensors: dict) -> None:
        layer.load_from_tensors(tensors)
        layer.transform()

    def _humming_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "humming_layer") or self.humming_layer is None:
            raise RuntimeError(
                f"{self.__class__.__name__} `humming_layer` is not initialized; "
                "call post_init() after loading weights."
            )

        input_shape = x.shape
        x = x.to(dtype=self.compute_dtype).contiguous()
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        out = self.humming_layer(x)
        if input_shape[-1] != self.in_features:
            raise ValueError(
                f"{self.__class__.__name__} expected input width {self.in_features}, got {input_shape[-1]}."
            )
        if input_shape[:-1] != out.shape[:-1]:
            out = out.reshape(*input_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype, device=out.device)

        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)

        return out


class HummingGptqLinear(_HummingLinearBase, GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_HUMMING]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 85}
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "humming_gptq"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise RuntimeError(
                f"Trying to use the Humming GPTQ backend, but the runtime requirements were not met: "
                f"{humming_import_exception}"
            )

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_HUMMING),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        self.humming_layer: Optional[HummingLayer] = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if humming_import_exception is not None:
            return False, ImportError(humming_import_exception)
        if not torch.cuda.is_available():
            return False, RuntimeError("Humming kernel requires a CUDA device.")
        return True, None

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err

        if dtype is not None and not _humming_dtype_supported(dtype):
            return False, NotImplementedError(
                f"{cls}: Humming does not support dtype `{dtype}` on the current CUDA device."
            )

        return True, None

    def post_init(self):
        device = self._humming_device()

        weight_schema = GPTQWeightSchema(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=False,
            sym=self.sym,
        )

        self.humming_layer = self._build_humming_layer(weight_schema, device)

        tensors = {
            "qweight": self.qweight,
            "scales": self.scales,
            "qzeros": self.qzeros,
            "g_idx": self.g_idx,
        }
        if self.bias is not None:
            tensors["bias"] = self.bias

        self._load_humming_tensors(self.humming_layer, tensors)
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


class HummingAwqLinear(_HummingLinearBase, AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_HUMMING]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 85}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "humming_awq"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise RuntimeError(
                f"Trying to use the Humming AWQ backend, but the runtime requirements were not met: "
                f"{humming_import_exception}"
            )

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_HUMMING),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        self.humming_layer: Optional[HummingLayer] = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if humming_import_exception is not None:
            return False, ImportError(humming_import_exception)
        if not torch.cuda.is_available():
            return False, RuntimeError("Humming kernel requires a CUDA device.")
        return True, None

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err

        if dtype is not None and not _humming_dtype_supported(dtype):
            return False, NotImplementedError(
                f"{cls}: Humming does not support dtype `{dtype}` on the current CUDA device."
            )

        return True, None

    def post_init(self):
        device = self._humming_device()

        weight_schema = AWQWeightSchema(
            bits=self.bits,
            group_size=self.group_size,
            zero_point=not self.sym,
        )

        self.humming_layer = self._build_humming_layer(weight_schema, device)

        tensors = {
            "qweight": self.qweight,
            "scales": self.scales,
            "qzeros": self.qzeros,
        }
        if self.bias is not None:
            tensors["bias"] = self.bias

        self._load_humming_tensors(self.humming_layer, tensors)
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


__all__ = ["HummingGptqLinear", "HummingAwqLinear"]
