# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import torch
from packaging import version

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils import BACKEND
from ...utils.logger import setup_logger


log = setup_logger()

MINIMUM_BITBLAS_VERSION = "0.1.0.post1"
BITBLAS_OPTIMIZE_FEATURES: List[int] = [1, 16, 32, 64, 128, 256, 512, 1024]
BITBLAS_SUPPORTED_GROUP_SIZES: List[int] = [-1, 32, 64, 128]
BITBLAS_SUPPORTED_BITS: List[int] = [1, 2, 4, 8]
BITBLAS_SUPPORTED_SYM: List[bool] = [False, True]
BITBLAS_DEFAULT_ZEROS_MODE = "quantized"
BITBLAS_PROPAGATE_WEIGHTS = False

BITBLAS_TARGET = None
BITBLAS_DATABASE_PATH = None

# TODO FIXME. ugly hack to bypass nv lib loadig for bitlbas
def _load_cuda_libraries() -> bool:
    loaded_any = False
    candidate_dirs = []

    env_dirs = []
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        paths = os.environ.get(var, "")
        if paths:
            env_dirs.extend(Path(p) for p in paths.split(":") if p)
    candidate_dirs.extend(env_dirs)

    candidate_dirs.extend(
        [
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/cuda/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
    )

    try:
        import nvidia  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        nvidia_paths = []
    else:
        nvidia_paths = [Path(p) for p in getattr(nvidia, "__path__", [])]

    for base in nvidia_paths:
        candidate_dirs.extend(
            [
                base / "cuda_runtime" / "lib",
                base / "cuda_nvrtc" / "lib",
            ]
        )
        candidate_dirs.extend(path for path in base.glob("cu*/lib"))

    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    candidate_dirs.append(site_packages)

    seen_dirs = set()
    for directory in candidate_dirs:
        if not directory or not directory.is_dir():
            continue
        resolved = directory.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)

        for pattern in ("libcudart.so*", "libnvrtc.so*"):
            for candidate in sorted(directory.glob(pattern)):
                if not candidate.is_file():
                    continue
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError:
                    continue

    return loaded_any


def _is_bitblas_available() -> bool:
    try:
        import bitblas
    except Exception as exc:
        error_text = str(exc)
        if "libcu" not in error_text:
            log.debug("BitBLAS import failed: %s", exc)
            return False
        if not _load_cuda_libraries():
            log.debug("CUDA libraries missing, BitBLAS import failed: %s", exc)
            return False
        try:
            import bitblas
        except Exception as retry_exc:
            log.debug("BitBLAS import retry failed: %s", retry_exc)
            return False
    parsed_version = version.parse(bitblas.__version__)
    minimum_version = version.parse(MINIMUM_BITBLAS_VERSION)
    if parsed_version < minimum_version:
        log.debug(
            "BitBLAS version %s below minimum required %s",
            bitblas.__version__,
            MINIMUM_BITBLAS_VERSION,
        )
        return False
    return True


BITBLAS_AVAILABLE = _is_bitblas_available()


BITBLAS_INSTALL_HINT = (
    "bitblas is not installed or the version is incompatible. "
    f"Please install via `pip install bitblas>={MINIMUM_BITBLAS_VERSION}`."
)


def import_bitblas():
    global BITBLAS_DATABASE_PATH, BITBLAS_TARGET

    import bitblas

    parsed_version = version.parse(bitblas.__version__)
    minimum_version = version.parse(MINIMUM_BITBLAS_VERSION)
    if parsed_version < minimum_version:
        raise ImportError(BITBLAS_INSTALL_HINT)

    bitblas.set_log_level("INFO")

    if BITBLAS_TARGET is None:
        from .bitblas_target_detector import patched_auto_detect_nvidia_target

        bitblas.auto_detect_nvidia_target = patched_auto_detect_nvidia_target
        visible = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        BITBLAS_TARGET = patched_auto_detect_nvidia_target(visible)
        os.environ["TVM_TARGET"] = f"{BITBLAS_TARGET}"
        log.debug("BITBLAS_TARGET %s", BITBLAS_TARGET)

    if BITBLAS_DATABASE_PATH is None:
        from bitblas.cache import get_database_path

        BITBLAS_DATABASE_PATH = f"{get_database_path()}_{bitblas.__version__}"
        log.debug("BITBLAS_DATABASE_PATH %s", BITBLAS_DATABASE_PATH)


def unpack_gptq_qzeros(qzeros: torch.Tensor, bits: int, is_gptq_v2: bool = False) -> torch.Tensor:
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    if not is_gptq_v2:
        return unpacked_zeros + 1
    return unpacked_zeros


def unpack_gptq_qweight(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    qweight = qweight.view(torch.int8)
    elems_per_int8 = 8 // bits
    unpacked_weight = torch.zeros(
        (qweight.shape[0], qweight.shape[1] * elems_per_int8),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )

    for col in range(unpacked_weight.shape[1]):
        i = col % elems_per_int8
        unpacked_weight[:, col] = (qweight[:, col // elems_per_int8] >> (bits * i))

    return torch.bitwise_and(unpacked_weight, 2**bits - 1)


def _num_groups(group_size: int, in_features: int) -> int:
    if group_size in (-1, in_features):
        return 1
    return in_features // group_size


@dataclass
class BitblasQuantizationConfig:
    weight_bits: int
    group_size: int
    desc_act: bool
    is_sym: bool
    zeros_mode: str = BITBLAS_DEFAULT_ZEROS_MODE
    storage_dtype: str = "int8"
    quant_method: str = "gptq"

    def __post_init__(self) -> None:
        if self.desc_act and self.group_size == -1:
            self.desc_act = False
        if self.weight_bits not in BITBLAS_SUPPORTED_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Supported values: {BITBLAS_SUPPORTED_BITS}."
            )
        if self.is_sym not in BITBLAS_SUPPORTED_SYM:
            raise ValueError(
                f"BitBLAS does not support sym = {self.is_sym}. "
                f"Supported values: {BITBLAS_SUPPORTED_SYM}."
            )
        if 32 % self.weight_bits != 0:
            raise ValueError("weight_bits must divide 32 for GPTQ packing")
        self.pack_factor = 32 // self.weight_bits
        self.torch_storage_dtype = getattr(torch, self.storage_dtype)
        self.torch_dtype = torch.float16

    @property
    def with_zeros(self) -> bool:
        return not self.is_sym and self.zeros_mode == "quantized"


class BitblasQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = BITBLAS_SUPPORTED_BITS
    SUPPORTS_GROUP_SIZE = BITBLAS_SUPPORTED_GROUP_SIZES
    SUPPORTS_DESC_ACT = [False, True]
    SUPPORTS_SYM = BITBLAS_SUPPORTED_SYM
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [16]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    QUANT_TYPE = "gptq_bitblas"
    SUPPORTS_QUANT_METHODS = ["gptq"]

    OPT_FEATURES = BITBLAS_OPTIMIZE_FEATURES
    TORCH_DTYPE = torch.float16

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
        enable_tuning: bool = False,
        fast_decoding: bool = True,  # kept for API compatibility
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        layout: str = "nt",
        register_buffers: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.BITBLAS),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        del fast_decoding  # unused, kept for signature compatibility

        if not BITBLAS_AVAILABLE:
            raise ImportError(BITBLAS_INSTALL_HINT)

        self.quant_config = BitblasQuantizationConfig(
            weight_bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            is_sym=sym,
        )
        self.enable_tuning = enable_tuning
        self.layout = layout
        self.opt_features = list(opt_features) if isinstance(opt_features, list) else [opt_features]
        self.propagate_b = propagate_b

        import_bitblas()

        self._validate_parameters(in_features, out_features)
        self._configure_bitblas_matmul(
            in_features,
            out_features,
            self.TORCH_DTYPE,
            enable_tuning,
            False,
            layout,
            bits,
        )
        self._initialize_buffers(in_features, out_features, bias)

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if not BITBLAS_AVAILABLE:
            return False, ValueError(BITBLAS_INSTALL_HINT)
        try:
            import_bitblas()
        except Exception as exc:  # pragma: no cover - import errors handled above
            return False, exc
        return cls._validate(**args)

    def _validate_parameters(self, in_features: int, out_features: int) -> None:
        if in_features % 16 != 0:
            raise ValueError("`in_features` must be divisible by 16 for BitBLAS")
        if out_features % 16 != 0:
            raise ValueError("`out_features` must be divisible by 16 for BitBLAS")
        if self.group_size not in (-1, in_features) and in_features % self.group_size != 0:
            raise ValueError("`in_features` must be divisible by `group_size`.")

    def _buffer_device(self) -> torch.device:
        for tensor in self.list_buffers():
            if isinstance(tensor, torch.Tensor) and not tensor.is_meta:
                return tensor.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _initialize_buffers(self, in_features: int, out_features: int, bias: bool) -> None:
        num_groups = _num_groups(self.group_size, in_features)
        storage_dtype = self.quant_config.torch_storage_dtype

        weight_shape = self.bitblas_matmul.retrieve_weight_shape()
        self.register_buffer(
            "qweight",
            torch.empty(weight_shape, dtype=storage_dtype),
        )
        self.register_buffer(
            "scales",
            torch.empty((out_features, num_groups), dtype=self.TORCH_DTYPE),
        )

        if self.quant_config.with_zeros:
            zeros_shape = (num_groups, out_features // self.quant_config.pack_factor)
            self.register_buffer(
                "qzeros",
                torch.empty(zeros_shape, dtype=storage_dtype),
            )
        else:
            self.register_buffer("qzeros", torch.empty(0, dtype=storage_dtype))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=self.TORCH_DTYPE))
        else:
            self.bias = None

        # Backward compatibility with older code paths expecting `zeros`.
        self.zeros = self.qzeros

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "qzeros") and self.qzeros is not None:
            buf.append(self.qzeros)
        return buf

    def _configure_bitblas_matmul(
        self,
        infeatures: int,
        outfeatures: int,
        params_dtype: torch.dtype,
        enable_tuning: bool,
        bias: bool,
        layout: str,
        bits: int,
    ) -> None:
        from bitblas import MatmulConfig

        bitblas_dtype = "float16" if params_dtype == torch.float16 else "bfloat16"
        W_dtype = f"uint{bits}" if self.quant_config.is_sym is False else f"int{bits}"
        matmul_config = MatmulConfig(
            M=self.opt_features,
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.storage_dtype,
            with_scaling=True,
            with_zeros=self.quant_config.with_zeros,
            group_size=self.group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=self.quant_config.zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=enable_tuning)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                log.info(
                    "BitBLAS operator tuned and added to cache for %s", config
                )
        else:
            log.debug("BitBLAS operator cache hit for %s", config)
        return bitblas_matmul

    def reset_parameters(self) -> None:
        if hasattr(self, "qweight") and isinstance(self.qweight, torch.Tensor) and not self.qweight.is_meta:
            self.qweight.zero_()
        if hasattr(self, "scales") and isinstance(self.scales, torch.Tensor) and not self.scales.is_meta:
            self.scales.zero_()
        if hasattr(self, "qzeros") and isinstance(self.qzeros, torch.Tensor) and not self.qzeros.is_meta:
            self.qzeros.zero_()
        if self.bias is not None and not self.bias.is_meta:
            self.bias.zero_()

    def post_init(self) -> None:
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(self.TORCH_DTYPE)

        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])

        args = [x_2d, self.qweight, self.scales]
        if self.quant_config.with_zeros:
            args.append(self.qzeros)
        out_2d = self.bitblas_matmul(*args)

        if self.bias is not None:
            out_2d = out_2d + self.bias

        out = out_2d.view(*orig_shape, self.out_features)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def repack_from_gptq(self, gptq_module: BaseQuantLinear) -> None:
        from bitblas.quantization.utils import general_compress

        device = self._buffer_device()

        bits = self.bits
        packed_weight = (
            gptq_module.qweight.detach().T.contiguous().view(self.quant_config.torch_storage_dtype)
        )
        intweight = unpack_gptq_qweight(packed_weight, bits).contiguous()

        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(intweight.cpu()).to(device)
        else:
            from bitblas.quantization.utils import general_compress

            compressed = general_compress(intweight.cpu().numpy(), bits)
            qweight = torch.from_numpy(compressed).to(
                device=device, dtype=self.quant_config.torch_storage_dtype
            )

        self._buffers["qweight"] = qweight.contiguous()

        scales = gptq_module.scales.detach().T.contiguous().to(self.TORCH_DTYPE)
        self._buffers["scales"] = scales.to(device)

        if self.quant_config.with_zeros and hasattr(gptq_module, "qzeros") and gptq_module.qzeros is not None:
            intzeros = unpack_gptq_qzeros(gptq_module.qzeros.detach(), bits).T.contiguous()
            intzeros = intzeros - 1  # GPTQ stores qzeros offset by +1
            compressed = general_compress(intzeros.T.contiguous().cpu().numpy(), bits)
            zeros = torch.from_numpy(compressed).to(device=device, dtype=self.quant_config.torch_storage_dtype)
            self._buffers["qzeros"] = zeros.contiguous()
        else:
            self._buffers["qzeros"] = torch.empty(0, dtype=self.quant_config.torch_storage_dtype, device=device)

        if self.bias is not None and hasattr(gptq_module, "bias") and gptq_module.bias is not None:
            self._buffers["bias"] = gptq_module.bias.detach().to(device=device, dtype=self.TORCH_DTYPE)

        self.zeros = self.qzeros


BitBLASQuantLinear = BitblasQuantLinear

__all__ = ["BitblasQuantLinear", "BitBLASQuantLinear"]
