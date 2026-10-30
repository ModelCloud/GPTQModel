# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# BitBLAS reference: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS

from __future__ import annotations

import ctypes
import os
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pcre
import torch
from packaging import version

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, GroupedQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils import BACKEND
from ...utils.env import env_flag
from ...utils.logger import setup_logger


log = setup_logger()

MINIMUM_BITBLAS_VERSION = "0.1.0.post1"
BITBLAS_OPTIMIZE_FEATURES: List[int] = [1, 16, 32, 64, 128, 256, 512, 1024]
BITBLAS_SUPPORTED_GROUP_SIZES: List[int] = [-1, 32, 64, 128]
BITBLAS_SUPPORTED_BITS: List[int] = [2, 4, 8]
BITBLAS_SUPPORTED_SYM: List[bool] = [False, True]
# bf16 stays an interface dtype, but BitBLAS 0.1.0.post1 cannot compile bf16 low-bit dequant
# kernels: nvcc rejects the generated `bfloat16_t` cast in the bundled cutlass 3.2.2, and
# `Matmul()` then returns with `lib = None` and `torch_func` unset without raising. This is a
# host-side build failure, independent of the GPU and unrelated to runtime crashes (handled by
# `_invoke_bitblas_matmul`). bf16 modules therefore run the operator in fp16 and cast at the
# module boundary; accumulation stays fp32.
BITBLAS_BF16_COMPUTE_DTYPE = torch.float16
BITBLAS_DEFAULT_ZEROS_MODE = "quantized"
BITBLAS_PROPAGATE_WEIGHTS = False
BITBLAS_MAX_SUPPORTED_SM = 90
BITBLAS_FALLBACK_TARGET = f"cuda -arch=sm_{BITBLAS_MAX_SUPPORTED_SM}"

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
    # Allow disabling BitBLAS probing in environments where TVM import is unstable.
    if env_flag("GPTQMODEL_DISABLE_BITBLAS", default="0"):
        return False

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
    # Capture the attribute once. Reading ``bitblas.__version__`` inside the
    # handler below would raise again for a module that has no ``__version__``
    # at all, escaping ``validate_once``'s ``(bool, exception)`` contract.
    raw_version = getattr(bitblas, "__version__", None)
    try:
        parsed_version = version.parse(raw_version)
        minimum_version = version.parse(MINIMUM_BITBLAS_VERSION)
    except Exception as exc:
        # An unparsable or missing ``__version__`` means the build is unusable.
        log.debug("BitBLAS version %r could not be parsed: %s", raw_version, exc)
        return False
    if parsed_version < minimum_version:
        log.debug(
            "BitBLAS version %s below minimum required %s",
            raw_version,
            MINIMUM_BITBLAS_VERSION,
        )
        return False
    return True


_UNSET = object()


@lru_cache(maxsize=1)
def _probe_bitblas_available() -> bool:
    """Probe BitBLAS once, on first real use.

    Importing BitBLAS pulls in TVM and costs a large fraction of a second, so it
    must not run while the kernel registry is merely being discovered.
    """

    return _is_bitblas_available()


_AVAILABILITY_OVERRIDES = {}


def bitblas_available() -> bool:
    """Return whether a supported BitBLAS build is importable.

    An explicitly assigned module attribute named ``BITBLAS_AVAILABLE`` still
    overrides the probe so tests and downstream tooling can force the value.
    Overrides live in a private dict (see ``_LazyAvailabilityModule``) so that
    ``monkeypatch`` undo and ``delattr`` fall back to the real probe.
    """

    if "BITBLAS_AVAILABLE" in _AVAILABILITY_OVERRIDES:
        return bool(_AVAILABILITY_OVERRIDES["BITBLAS_AVAILABLE"])
    return _probe_bitblas_available()


def __getattr__(name: str):
    if name == "BITBLAS_AVAILABLE":
        return bitblas_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _LazyAvailabilityModule(types.ModuleType):
    """Module type that keeps ``*_AVAILABLE`` overrides out of ``__dict__``."""

    def __setattr__(self, name, value):
        if name == "BITBLAS_AVAILABLE":
            _AVAILABILITY_OVERRIDES[name] = value
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name == "BITBLAS_AVAILABLE" and name in _AVAILABILITY_OVERRIDES:
            del _AVAILABILITY_OVERRIDES[name]
            return
        if name == "BITBLAS_AVAILABLE":
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | {"BITBLAS_AVAILABLE"})


sys.modules[__name__].__class__ = _LazyAvailabilityModule


_BITBLAS_TARGET_ARCH_RE = pcre.compile(r"\bsm_(\d+)[a-z]*\b")
_BITBLAS_TARGET_SM_RE = pcre.compile(r"sm_(\d+)")


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
        BITBLAS_TARGET = _normalize_bitblas_target(patched_auto_detect_nvidia_target(visible))
        os.environ["TVM_TARGET"] = f"{BITBLAS_TARGET}"
        log.debug("BITBLAS_TARGET %s", BITBLAS_TARGET)

    if BITBLAS_DATABASE_PATH is None:
        from bitblas.cache import get_database_path

        BITBLAS_DATABASE_PATH = f"{get_database_path()}_{bitblas.__version__}"
        log.debug("BITBLAS_DATABASE_PATH %s", BITBLAS_DATABASE_PATH)


def _bitblas_target_arch(target) -> Optional[str]:
    if target is None:
        return None

    target_text = str(target)

    try:
        from bitblas import tvm

        return str(tvm.target.Target(target_text).arch)
    except Exception:
        match = _BITBLAS_TARGET_ARCH_RE.search(target_text)
        if match:
            return f"sm_{match.group(1)}"
        return None


def _bitblas_target_sm(target) -> Optional[int]:
    arch = _bitblas_target_arch(target)
    if arch is None:
        return None

    match = _BITBLAS_TARGET_SM_RE.search(arch)
    if match:
        return int(match.group(1))
    return None


def _current_cuda_sm() -> Optional[int]:
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major * 10 + props.minor


def _bitblas_fallback_target() -> str:
    current_sm = _current_cuda_sm()
    if current_sm is None:
        return BITBLAS_FALLBACK_TARGET

    fallback_sm = min(current_sm, BITBLAS_MAX_SUPPORTED_SM)
    return f"cuda -arch=sm_{fallback_sm}"


def _normalize_bitblas_target(target):
    if target is None:
        return None

    arch = _bitblas_target_arch(target)
    sm_version = _bitblas_target_sm(target)
    if sm_version is None:
        return target

    canonical_target = f"cuda -arch=sm_{sm_version}"

    if sm_version > BITBLAS_MAX_SUPPORTED_SM:
        fallback_target = _bitblas_fallback_target()
        log.warning(
            "BitBLAS target %s resolves to unsupported CUDA arch %s; falling back to %s.",
            target,
            arch,
            fallback_target,
        )
        return fallback_target

    if arch != f"sm_{sm_version}":
        log.info(
            "Canonicalizing BitBLAS target %s (%s) to %s.",
            target,
            arch,
            canonical_target,
        )
        return canonical_target

    return target


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


def remap_gptq_symmetric_codes_to_bitblas(qweight_codes: torch.Tensor, bits: int) -> torch.Tensor:
    # Some in-memory TorchLinear symmetric pack paths still encode qweight as GPTQ-style
    # two's-complement nibbles while leaving qzeros packed as all zeros. BitBLAS' signed intN path
    # expects the corresponding biased code range instead, so flip the sign bit for that narrow
    # producer case before loading the quant state.
    sign_bit = 1 << (bits - 1)
    remapped = torch.bitwise_xor(qweight_codes.to(torch.int16), sign_bit)
    return remapped.to(torch.int8).contiguous()


def _should_remap_symmetric_gptq_codes(gptq_module: BaseQuantLinear) -> bool:
    qzeros = getattr(gptq_module, "qzeros", None)
    if qzeros is None or qzeros.numel() == 0:
        return False

    qzero_format = getattr(gptq_module, "qzero_format", None)
    if callable(qzero_format):
        format_id = qzero_format()
        if format_id == 2:
            return False
        if format_id != 1:
            return False

    # The remap is only needed for the pre-v2 TorchLinear symmetric pack path. That producer
    # keeps qzeros packed as all zeros while storing qweight in GPTQ's two's-complement nibble
    # layout. GPT-QModel converts external checkpoints to qzero_format=2 before BitBLAS repacking,
    # and those tensors must be left untouched.
    return qzeros.count_nonzero().item() == 0


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
    torch_dtype: torch.dtype = torch.float16
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
        if self.torch_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("BitBLAS only supports torch.float16 and torch.bfloat16 compute dtypes")
        self.pack_factor = 32 // self.weight_bits
        self.torch_storage_dtype = getattr(torch, self.storage_dtype)

    @property
    def with_zeros(self) -> bool:
        return not self.is_sym and self.zeros_mode == "quantized"


class BitblasBaseQuantLinear(GroupedQuantLinear):
    OPT_FEATURES = BITBLAS_OPTIMIZE_FEATURES
    TORCH_DTYPE = torch.float16

    @property
    def compute_dtype(self) -> torch.dtype:
        """Dtype the BitBLAS operator is built with (see `_resolve_compute_dtype`)."""
        return getattr(self, "_compute_dtype", self.TORCH_DTYPE)

    @classmethod
    def _resolve_compute_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        # The installed BitBLAS build cannot compile bf16 low-bit dequant kernels, so bf16
        # modules run the operator in fp16 and cast at the module boundary. See the module header.
        if dtype == torch.bfloat16:
            return BITBLAS_BF16_COMPUTE_DTYPE
        return dtype

    def _build_quant_config(
        self,
        *,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        dtype: torch.dtype,
    ) -> BitblasQuantizationConfig:
        return BitblasQuantizationConfig(
            weight_bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            is_sym=sym,
            torch_dtype=dtype,
        )

    @classmethod
    def _validate_kernel_combo(
        cls,
        *,
        bits: int,
        sym: bool,
        dtype: Optional[torch.dtype],
        dynamic: Optional[dict] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        del bits, sym, dtype, dynamic
        return True, None

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
        dtype: torch.dtype = torch.float16,
        adapter: Adapter = None,
        enable_tuning: bool = False,
        fast_decoding: bool = True,  # kept for API compatibility
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        layout: str = "nt",
        register_buffers: bool = False,
        **kwargs,
    ) -> None:
        if dtype not in self.SUPPORTS_DTYPES:
            raise ValueError(f"{self.__class__.__name__} only supports dtypes {self.SUPPORTS_DTYPES}: actual dtype = {dtype}")

        ok, err = self.__class__._validate_kernel_combo(bits=bits, sym=sym, dtype=dtype)
        if not ok:
            raise err

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_BITBLAS),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        del fast_decoding  # unused, kept for signature compatibility

        if not bitblas_available():
            raise ImportError(BITBLAS_INSTALL_HINT)

        self.TORCH_DTYPE = dtype
        self._compute_dtype = self._resolve_compute_dtype(dtype)
        if self._compute_dtype != dtype:
            log.warn.once(
                f"BitBLAS: the installed BitBLAS build cannot compile `{dtype}` low-bit dequant "
                f"kernels; running `{self.__class__.__name__}` in `{self._compute_dtype}` and "
                "casting activations/outputs back to the requested dtype."
            )
        self.quant_config = self._build_quant_config(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            dtype=self._compute_dtype,
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
            self.compute_dtype,
            enable_tuning,
            False,
            layout,
            bits,
        )
        self._initialize_buffers(in_features, out_features, bias)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not bitblas_available():
            return False, ValueError(BITBLAS_INSTALL_HINT)

        try:
            import_bitblas()
        except Exception as exc:  # pragma: no cover - import errors handled above
            return False, exc
        return True, None

    @classmethod
    def validate(
        cls,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int = None,
        out_features: int = None,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = cls._validate_kernel_combo(bits=bits, sym=sym, dtype=dtype, dynamic=dynamic)
        if not ok:
            return False, err

        return super().validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )

    def _validate_parameters(self, in_features: int, out_features: int) -> None:
        # This wrapper keeps a conservative 16-wide gate because the current GPTQ/AWQ BitBLAS
        # integration is built around TensorCore-oriented 16x16 micro-kernel paths. BitBLAS itself
        # is looser in some cases: local probes showed odd N/out_features can still build, while
        # the hard upstream packing constraint we confirmed is K/in_features divisible by 8 / bits
        # for quant-compressed weights. We keep the stricter gate here until the relaxed shapes are
        # regression-tested across the full load/repack/forward path.
        if any(in_features % divisor != 0 for divisor in self.SUPPORTS_IN_FEATURES_DIVISIBLE_BY):
            raise ValueError(
                f"`in_features` must be divisible by {self.SUPPORTS_IN_FEATURES_DIVISIBLE_BY} for BitBLAS"
            )
        if any(out_features % divisor != 0 for divisor in self.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY):
            raise ValueError(
                f"`out_features` must be divisible by {self.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY} for BitBLAS"
            )
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
        compute_dtype = self.compute_dtype

        weight_shape = self.bitblas_matmul.retrieve_weight_shape()
        self.register_buffer(
            "qweight",
            torch.empty(weight_shape, dtype=storage_dtype),
        )
        self.register_buffer(
            "scales",
            torch.empty((out_features, num_groups), dtype=compute_dtype),
        )

        if self.quant_config.with_zeros:
            # BitBLAS stores compressed zeros in storage_dtype (usually int8),
            # not the int32 words described by GPTQ's pack_factor.
            storage_bits = torch.iinfo(storage_dtype).bits
            zeros_shape = (num_groups, out_features * self.bits // storage_bits)
            self.register_buffer(
                "qzeros",
                torch.empty(zeros_shape, dtype=storage_dtype),
            )
        else:
            self.register_buffer("qzeros", torch.empty(0, dtype=storage_dtype))

        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=compute_dtype))
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
        # FP16/BF16 accumulation drifted enough to derail autoregressive decoding.
        accum_dtype = "float32"
        W_dtype = f"uint{bits}" if self.quant_config.is_sym is False else f"int{bits}"
        matmul_config = MatmulConfig(
            M=self.opt_features,
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else accum_dtype,
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
        self._ensure_runnable_bitblas_operator(self.bitblas_matmul, matmul_config)

    def _ensure_runnable_bitblas_operator(self, bitblas_matmul, config) -> None:
        if getattr(bitblas_matmul, "lib", None) is not None:
            return
        if callable(getattr(bitblas_matmul, "torch_func", None)):
            return
        raise NotImplementedError(
            "BitBLAS could not build a runnable matmul for "
            f"A_dtype={config.A_dtype}, W_dtype={config.W_dtype}, out_dtype={config.out_dtype}, "
            f"accum_dtype={config.accum_dtype}, group_size={config.group_size}."
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache
        target = _normalize_bitblas_target(BITBLAS_TARGET)

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, target
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=target, enable_tuning=enable_tuning)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, target
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

    def _transform_bitblas_weight(self, intweight_out_in: torch.Tensor, device: torch.device) -> torch.Tensor:
        from bitblas.quantization.utils import general_compress

        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(intweight_out_in.cpu()).to(device)
        else:
            compressed = general_compress(intweight_out_in.cpu().numpy(), self.bits)
            qweight = torch.from_numpy(compressed).to(
                device=device,
                dtype=self.quant_config.torch_storage_dtype,
            )
        return qweight.contiguous()

    def _compress_bitblas_zeros(
        self,
        intzeros_group_out: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        from bitblas.quantization.utils import general_compress

        if not self.quant_config.with_zeros or intzeros_group_out is None:
            return torch.empty(0, dtype=self.quant_config.torch_storage_dtype, device=device)

        compressed = general_compress(intzeros_group_out.contiguous().cpu().numpy(), self.bits)
        return torch.from_numpy(compressed).to(
            device=device,
            dtype=self.quant_config.torch_storage_dtype,
        ).contiguous()

    def _load_bitblas_quant_state(
        self,
        intweight_out_in: torch.Tensor,
        scales_out_group: torch.Tensor,
        intzeros_group_out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        device = self._buffer_device()

        self._buffers["qweight"] = self._transform_bitblas_weight(intweight_out_in, device)
        self._buffers["scales"] = scales_out_group.to(device=device, dtype=self.compute_dtype).contiguous()
        self._buffers["qzeros"] = self._compress_bitblas_zeros(intzeros_group_out, device)

        if self.bias is not None and bias is not None:
            self._buffers["bias"] = bias.detach().to(device=device, dtype=self.compute_dtype)

        self.zeros = self.qzeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if input_dtype != self.TORCH_DTYPE:
            x = x.to(self.TORCH_DTYPE)

        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != self.compute_dtype:
            # The operator runs in fp16 when bf16 compute is unavailable (see module header);
            # keep `x` itself in the module dtype for adapters.
            x_2d = x_2d.to(self.compute_dtype)

        args = [x_2d, self.qweight, self.scales]
        if self.quant_config.with_zeros:
            args.append(self.qzeros)
        out_2d = self._invoke_bitblas_matmul(args)

        if self.bias is not None:
            out_2d = out_2d + self.bias

        out = out_2d.view(*orig_shape, self.out_features)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        if input_dtype in self.SUPPORTS_DTYPES and out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)

        return out

    def _invoke_bitblas_matmul(self, args: List[torch.Tensor]) -> torch.Tensor:
        # BitBLAS launches its prebuilt CUDA library through ctypes, so the kernel executes in the
        # CUDA context that is current for this thread. GPT-QModel's default layer-wise device map
        # spreads the model over every visible GPU; without entering the module's device context,
        # layers on `cuda:1+` would launch in GPU 0's context while holding pointers to another
        # GPU's buffers and raise `CUDA error: an illegal memory access was encountered`.
        device = self.scales.device
        if device.type == "cuda" and torch.cuda.current_device() != device.index:
            with torch.cuda.device(device):
                return self.bitblas_matmul(*args)
        return self.bitblas_matmul(*args)


# BitBLAS repacks incoming GPTQ/AWQ tensors into its own operator layout, so the
# destination module only needs grouped quantization state, not GPTQ qzero-format state.
class BitblasLinear(BitblasBaseQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_BITBLAS]
    SUPPORTS_FORMATS = {FORMAT.BITBLAS: 30, FORMAT.GPTQ: 30, FORMAT.GPTQ_V2: 30}
    SUPPORTS_BITS = BITBLAS_SUPPORTED_BITS
    SUPPORTS_GROUP_SIZE = BITBLAS_SUPPORTED_GROUP_SIZES
    # BitBLAS' public matmul API does not expose GPTQ activation-order metadata (`g_idx` /
    # permutation tensors). Keep desc_act disabled until upstream adds a supported act-order path.
    SUPPORTS_DESC_ACT = [False]
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
    SUPPORTS_METHODS = [METHOD.GPTQ]

    def repack_from_gptq(self, gptq_module: BaseQuantLinear) -> None:
        bits = self.bits
        packed_weight = (
            gptq_module.qweight.detach().T.contiguous().view(self.quant_config.torch_storage_dtype)
        )
        intweight = unpack_gptq_qweight(packed_weight, bits).contiguous()
        if self.quant_config.is_sym and _should_remap_symmetric_gptq_codes(gptq_module):
            intweight = remap_gptq_symmetric_codes_to_bitblas(intweight, bits)

        intzeros = None
        if self.quant_config.with_zeros and hasattr(gptq_module, "qzeros") and gptq_module.qzeros is not None:
            intzeros = unpack_gptq_qzeros(gptq_module.qzeros.detach(), bits).contiguous()
            intzeros = intzeros - 1  # GPTQ stores qzeros offset by +1

        bias = gptq_module.bias.detach() if self.bias is not None and getattr(gptq_module, "bias", None) is not None else None
        self._load_bitblas_quant_state(
            intweight_out_in=intweight,
            scales_out_group=gptq_module.scales.detach().T.contiguous(),
            intzeros_group_out=intzeros,
            bias=bias,
        )


BitBLASLinear = BitblasLinear

__all__ = ["BitblasLinear", "BitBLASLinear"]
