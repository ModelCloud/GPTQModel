# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Audit the entire kernel registry's ROCm family contract without GPU execution."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.models._const as model_const
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear
from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.ttnn_blockfloat import TTNNBlockFloat8Linear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer, macaq_triton
from gptqmodel.utils.backend import BACKEND


# Explicit audit categories: adding a kernel requires reviewing its implementation,
# not inferring ROCm support from a name containing "Torch", "CUDA", or "Triton".
ROCM_KERNELS = set("""
    AllSparkLinear AllSparkAwqLinear AwqGEMMTritonLinear AwqGEMVTritonLinear
    GGUFTritonKernel TorchFP8Linear BitsAndBytesLinear
    TorchLinear TorchQuantEmbeddings AwqTorchLinear QQQTorchLinear
    GGUFTorchLinear ParoLinear ParoTritonLinear VPTQLinear NF12Linear
    MacaqLinear MacaqTritonLinear
    AwqGEMVFastTorchLinear LLMAwqTorchLinear
    TTBlockfloatLinear
""".split())
CUDA_ONLY_KERNELS = set("""
    TritonV2Linear W4AFP8Linear W4ANVFP4Linear ExllamaV2Linear AwqExllamaV2Linear
    MarlinLinear AwqMarlinLinear MacheteLinear AwqMacheteLinear
    SwordfishLinear AwqSwordfishLinear BitblasLinear AWQBitBlasKernel
    AwqGEMMLinear AwqGEMVLinear AwqGEMVFastLinear LLMAwqLinear QQQLinear
    VPTQCudaLinear GGUFCudaKernel
""".split())
OTHER_KERNELS = set("""
    TorchFusedLinear TorchFusedAwqLinear TorchAtenLinear TorchAtenAwqLinear
    TorchCpuAvx512Linear TorchCpuAvx512AwqLinear TorchInt8Linear TorchInt8AwqLinear
    AwqGEMVTorchLinear
    GGUFCppKernel MlxQuantLinear GPTQMarlinMlxQuantLinear GPTQBitBLASMlxQuantLinear
    AwqMlxQuantLinear AWQMarlinMlxQuantLinear AWQBitBLASMlxQuantLinear
    AwqGemvMlxQuantLinear AwqGemvFastMlxQuantLinear LLMAwqMlxQuantLinear
    ParoMlxQuantLinear QQQMlxQuantLinear GGUFMlxQuantLinear
    BitsAndBytesMlxQuantLinear
    XLALinear
    TTNNLinear TTNNBlockFloat8Linear
""".split())
# Kernels registered for a non-Linux platform only. An MPS/Darwin kernel can
# never execute on a ROCm host, so it is classified by its declared platform
# instead of by device family. `_platform_restricted_kernels()` derives the
# bucket from `SUPPORTS_PLATFORM`, so the next MPS-only kernel is covered
# without editing this inventory; the explicit entries below are the ones
# reviewed for #4130 and are asserted to really be platform-restricted.
PLATFORM_RESTRICTED_KERNELS = set("""
    MacaqMlxQuantLinear NF12MlxQuantLinear
""".split())
KERNELS = sorted(
    [*importer.iter_quant_linear_kernels(), TorchQuantEmbeddings, TTNNBlockFloat8Linear],
    key=lambda c: c.__name__,
)


def _platform_restricted_kernels() -> set[str]:
    """Registered kernels that declare no Linux platform.

    These cannot run on the Linux ROCm hosts this audit targets. The set is
    derived from the shipped declaration rather than a hand-written list so a
    newly registered MPS-only kernel is classified automatically.
    """

    return {
        cls.__name__
        for cls in KERNELS
        if model_const.PLATFORM.LINUX not in cls.SUPPORTS_PLATFORM
        and model_const.PLATFORM.ALL not in cls.SUPPORTS_PLATFORM
    }


@pytest.fixture
def hip_runtime(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.16")
    monkeypatch.setattr(model_const, "IS_ROCM", True)
    monkeypatch.setattr(importer, "IS_ROCM", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(gcnArchName="gfx1201"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: pytest.fail("NVIDIA capability probe on HIP"))
    for cls in KERNELS:
        module = sys.modules[cls.__module__]
        if hasattr(module, "IS_ROCM"):
            monkeypatch.setattr(module, "IS_ROCM", True)
    monkeypatch.setattr(macaq_triton, "IS_ROCM", True)
    macaq_triton._rdna4_macaq_supported.cache_clear()
    yield
    macaq_triton._rdna4_macaq_supported.cache_clear()


def test_macaq_triton_rejects_mixed_rocm_architectures(hip_runtime, monkeypatch):
    def properties(target=None):
        index = target if isinstance(target, int) else (
            0 if target is None else torch.device(target).index
        )
        return SimpleNamespace(gcnArchName="gfx942" if index == 1 else "gfx1201")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    macaq_triton._rdna4_macaq_supported.cache_clear()
    MacaqTritonLinear.validate_device(torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="gfx1201"):
        MacaqTritonLinear.validate_device(torch.device("cuda:1"))
    with pytest.raises(NotImplementedError, match="gfx1201"):
        MacaqTritonLinear.validate_device(model_const.DEVICE.ROCM)


def test_audit_covers_every_registered_kernel():
    platform_restricted = _platform_restricted_kernels()
    # The explicitly reviewed entries must really be platform-restricted; a
    # kernel listed here without a non-Linux platform is a stale classification.
    assert PLATFORM_RESTRICTED_KERNELS <= platform_restricted
    assert {c.__name__ for c in KERNELS} == (
        ROCM_KERNELS | CUDA_ONLY_KERNELS | OTHER_KERNELS | platform_restricted
    )


def test_platform_restricted_kernels_declare_reachable_families():
    # A kernel that excludes Linux cannot advertise ROCm, and CUDA does not
    # exist on Darwin. Without these checks the derived bucket would let an
    # MPS-only kernel claim an accelerator family it can never reach and still
    # pass the audit.
    platform_restricted = _platform_restricted_kernels()
    assert platform_restricted
    for cls in KERNELS:
        if cls.__name__ not in platform_restricted:
            continue
        assert model_const.PLATFORM.LINUX not in cls.SUPPORTS_PLATFORM, cls.__name__
        assert model_const.DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__
        if model_const.PLATFORM.DARWIN in cls.SUPPORTS_PLATFORM:
            assert model_const.DEVICE.CUDA not in cls.SUPPORTS_DEVICES, cls.__name__


@pytest.mark.parametrize("cls,devices", [
    (ExllamaV3Linear, [model_const.DEVICE.CUDA, model_const.DEVICE.ROCM]),
    (ExllamaV3TorchLinear, [model_const.DEVICE.ALL]),
    (TorchFP4Linear, [model_const.DEVICE.CPU, model_const.DEVICE.CUDA]),
])
def test_specialized_loader_kernel_declarations(cls, devices):
    # These nn.Module implementations use dedicated loader paths rather than
    # BaseQuantLinear's registry; their device contracts must be explicit too.
    assert cls.SUPPORTS_DEVICES == devices


@pytest.mark.parametrize("is_rocm", [False, True])
def test_explicit_rocm_family_never_probes_nvidia_capability(monkeypatch, is_rocm):
    monkeypatch.setattr(importer, "IS_ROCM", is_rocm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: pytest.fail("AMD family has no NVIDIA SM"))
    assert importer._device_capability(model_const.DEVICE.ROCM) is None


@pytest.mark.parametrize("cls", KERNELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("device", [model_const.DEVICE.ROCM, model_const.DEVICE.CUDA, "rocm", "cuda", "cuda:1", torch.device("cuda:1")])
def test_rocm_device_contract(cls, device, hip_runtime):
    expected = cls.__name__ in ROCM_KERNELS
    advertised = model_const.DEVICE.ROCM in cls.SUPPORTS_DEVICES or model_const.DEVICE.ALL in cls.SUPPORTS_DEVICES
    assert advertised is expected
    # Exercise shared validation, including class overrides, while avoiding
    # optional extension imports and unrelated bit/group/shape contracts.
    ok, error = cls._validate_shared(device=device, pack_dtype=cls.SUPPORTS_PACK_DTYPES[0])
    assert ok is expected, (cls.__name__, device, error)
    if not expected:
        assert isinstance(error, NotImplementedError)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("backend", [BACKEND.AUTO, BACKEND.BITSANDBYTES])
@pytest.mark.parametrize("device", [model_const.DEVICE.ROCM, model_const.DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_bitsandbytes_rocm_selection(bits, backend, device, hip_runtime, monkeypatch):
    monkeypatch.setattr(BitsAndBytesLinear, "cached_validate_once", classmethod(lambda cls: (True, None)))
    importer.clear_validation_cache()
    try:
        assert importer.select_quant_linear(
            bits=bits, group_size=-1, desc_act=False, sym=True,
            backend=backend, format=FORMAT.BITSANDBYTES, quant_method=METHOD.BITSANDBYTES,
            device=device, pack_dtype=torch.int32,
        ) is BitsAndBytesLinear
    finally:
        importer.clear_validation_cache()


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
@pytest.mark.parametrize("device", ["cuda:1", torch.device("cuda:1"), model_const.DEVICE.ROCM, model_const.DEVICE.CUDA])
def test_allspark_rejects_non_rdna4_target(cls, device, hip_runtime, monkeypatch):
    def properties(target=None):
        index = 0 if target is None else torch.device(target).index
        return SimpleNamespace(gcnArchName="gfx942" if index == 1 else "gfx1201")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    # The current device is supported. A different target, or a family spanning
    # the unsupported target, must still be rejected before repacking/launch.
    assert cls.validate_once() == (True, None)
    cls.validate_device(torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="RDNA4"):
        cls.validate_device(device)


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
def test_allspark_accepts_rdna4_target_when_current_device_is_unsupported(cls, hip_runtime, monkeypatch):
    def properties(target=None):
        index = 0 if target is None else torch.device(target).index
        return SimpleNamespace(gcnArchName="gfx1201" if index == 1 else "gfx942")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(cls, "cached_validate_once", classmethod(lambda kernel: kernel.validate_once()))
    ok, error = cls.validate(bits=8, group_size=-1, sym=True, desc_act=False,
                             pack_dtype=torch.int32, device=torch.device("cuda:1"))
    assert ok, error
    with pytest.raises(NotImplementedError, match="RDNA4"):
        cls.validate_device(torch.device("cuda:0"))


# ---------------------------------------------------------------------------
# HIP shim build contract (#5208)
# ---------------------------------------------------------------------------
# hipcc compiles every source file of a ROCm-enabled extension, so a CUDA-only
# intrinsic in a file that declares HIP support is a *build* break: the whole
# extension fails to compile and every kernel in it silently falls back. The
# registry audit above cannot see that, and it passed while
# `gptqmodel_exllamav3_ops` was unbuildable on ROCm (#5208).
#
# This scanner is the cheap, CPU-only half of the contract. It walks the
# preprocessor regions of each HIP-compatible source and requires every
# `__`-prefixed symbol that hipcc will actually see to be either exposed by the
# HIP headers, rewritten by torch's hipify pass, or declared by the file's own
# HIP shim (`#define`, `using`, `typedef` or a `__device__` shim function).
#
# Scope is the hipified sources: files that declare a HIP branch and are also
# compiled by nvcc, so a CUDA-only spelling passes every NVIDIA check and only
# breaks ROCm. Native `.hip` sources are hipcc-only, so the same mistake there
# fails the only build it has and cannot hide.

_KERNEL_SOURCE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc")
_HIP_MARKERS = ("__HIP_PLATFORM_AMD__", "USE_ROCM")

# Attributes, keywords and preprocessor plumbing. Not HIP symbols, and not
# something a shim can or should provide.
_CUDA_KEYWORDS = frozenset("""
    __device__ __host__ __global__ __shared__ __restrict__ __forceinline__
    __launch_bounds__ __align__ __constant__ __managed__ __inline__ __noinline__
    __LINE__ __FILE__ __func__ __FUNCTION__ __PRETTY_FUNCTION__ __VA_ARGS__
    __VA_OPT__ __CUDA_ARCH__ __HIP_PLATFORM_AMD__ __HIP_DEVICE_COMPILE__
    __CUDACC__ __HIPCC__ __clang__ __GNUC__ __has_include __has_feature
""".split())

# CUDA spellings that torch's hipify pass renames in the sources it copies, so
# hipcc never sees them: torch/utils/hipify/cuda_to_hip_mappings.py maps
# __nv_bfloat16 -> __hip_bfloat16 and __nv_bfloat162 -> __hip_bfloat162. It has
# no entry for the conversion intrinsics, which is why #5208 broke the build.
_HIPIFY_REWRITTEN_SYMBOLS = frozenset({"__nv_bfloat16", "__nv_bfloat162"})

# Symbols the HIP headers expose under the *same* spelling, so a HIP-compiled
# file may use them without a shim (hip_runtime.h / amd_device_functions.h,
# hip_fp16.h, hip_bf16.h). Everything else has to be shimmed; in particular
# `__floats2bfloat162_rn` and the scalar `__float2bfloat16_rn` are absent,
# which is why rotation.cuh and bf16_scale.cu declare them.
_HIP_NATIVE_SYMBOLS = frozenset("""
    __syncthreads __activemask __shfl_down_sync __shfl_sync __shfl_up_sync __shfl_xor_sync
    __int_as_float __float_as_int __uint_as_float __float_as_uint __byte_perm
    __funnelshift_r __funnelshift_l __funnelshift_rc __funnelshift_lc
    __fadd_rn __fsub_rn __fmul_rn __fdiv_rn __frcp_rn __fsqrt_rn __fmaf_rn
    __fminf __fmaxf __min __max __sincosf __saturatef
    __half __half2 __half2float __float2half __float2half_rn __float2half_rz
    __halves2half2 __low2half __high2half __half22float2 __floats2half2_rn
    __ushort_as_half __hmul __hadd __hsub __hlt __hgt __heq __hne __hle __hge
    __hip_bfloat16 __hip_bfloat162 __bfloat162float __float2bfloat16
    __halves2bfloat162 __low2bfloat16 __high2bfloat16 __bfloat1622float2
    __float22bfloat162_rn __bfloat162bfloat162
""".split())

_DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)\s*(?:\(([^)]*)\))?")
_USING_RE = re.compile(r"\busing\s+(__\w+)\s*=")
_TYPEDEF_RE = re.compile(r"\btypedef\b[^;]*?\b(__\w+)\s*;")
_DEVICE_FUNCTION_RE = re.compile(r"\b(?:__device__|__host__)\b[^;{}]*?\b(__\w+)\s*\([^;{}]*\)\s*(?:const\s*)?\{")
_SYMBOL_RE = re.compile(r"\b(__[A-Za-z_]\w*)\b")


def _strip_comments(text: str) -> str:
    """Blank out `//` and `/* */` comments while preserving every line number."""

    output = []
    index = 0
    in_block = False
    in_string = None
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_block:
            if char == "*" and following == "/":
                output.append("  ")
                index += 2
                in_block = False
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if in_string is not None:
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 2
            else:
                if char == in_string:
                    in_string = None
                index += 1
            continue
        if char in ("'", '"'):
            in_string = char
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            while index < len(text) and text[index] != "\n":
                output.append(" ")
                index += 1
            continue
        if char == "/" and following == "*":
            output.append("  ")
            index += 2
            in_block = True
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _hip_branch_possible(condition: str) -> bool:
    """Whether `condition` can hold when hipcc compiles the file.

    Only HIP markers decide it: a branch that requires one of them absent (for
    example `#if !defined(USE_ROCM)` or `#if !defined(A) && defined(B)`) is
    never compiled on ROCm, while anything else is treated as reachable so the
    scan errs towards checking more code.
    """

    expression = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)", r"\1", condition).replace(" ", "")
    if not any(marker in expression for marker in _HIP_MARKERS):
        return True

    def impossible(conjunct: str) -> bool:
        return any(conjunct == "!" + marker for marker in _HIP_MARKERS)

    return any(
        not any(impossible(conjunct) for conjunct in disjunct.split("&&"))
        for disjunct in expression.split("||")
    )


def _hip_compiled_lines(text: str):
    """Yield `(line_number, hip_compiled, line, is_directive)` for each line."""

    frames = []
    hip = True
    for number, line in enumerate(text.splitlines(), 1):
        directive = _DIRECTIVE_RE.match(line)
        if directive is None:
            yield number, hip, line, False
            continue
        keyword, condition = directive.group(1), directive.group(2).strip()
        if keyword in ("if", "ifdef", "ifndef"):
            if keyword == "if":
                possible = _hip_branch_possible(condition)
            elif keyword == "ifdef":
                possible = True
            else:
                possible = condition not in _HIP_MARKERS
            frames.append((possible, hip))
            hip = hip and possible
        elif keyword == "elif":
            if frames:
                _, parent = frames[-1]
                possible = _hip_branch_possible(condition)
                frames[-1] = (possible, parent)
                hip = parent and possible
        elif keyword == "else":
            if frames:
                possible, parent = frames[-1]
                hip = parent and not possible
        elif frames:
            _, parent = frames.pop()
            hip = parent
        yield number, hip, line, True


def _hip_shim_violations(sources: dict[str, str]) -> list[str]:
    """Report CUDA-only symbols that hipcc would see without a HIP definition."""

    violations = []
    for name, raw_text in sorted(sources.items()):
        text = _strip_comments(raw_text)
        hip_by_line = {}
        declared = {}
        for number, hip, line, _is_directive in _hip_compiled_lines(text):
            hip_by_line[number] = hip
            if not hip:
                continue
            for symbol in _USING_RE.findall(line) + _TYPEDEF_RE.findall(line):
                declared.setdefault(symbol, number)
            define = _DEFINE_RE.match(line)
            if define:
                if define.group(1).startswith("__"):
                    declared.setdefault(define.group(1), number)
                # Macro parameters are local to the macro body, and kernel
                # headers routinely name them `__x`/`__y`.
                for parameter in (define.group(2) or "").split(","):
                    parameter = parameter.strip()
                    if parameter.startswith("__"):
                        declared.setdefault(parameter, number)
        for match in _DEVICE_FUNCTION_RE.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if hip_by_line.get(number, False):
                declared.setdefault(match.group(1), number)

        for number, hip, line, is_directive in _hip_compiled_lines(text):
            if not hip:
                continue
            define = _DEFINE_RE.match(line)
            if is_directive and define is None:
                continue
            for symbol in _SYMBOL_RE.findall(line[define.end():] if define else line):
                if symbol in _CUDA_KEYWORDS or symbol in _HIP_NATIVE_SYMBOLS:
                    continue
                if symbol in _HIPIFY_REWRITTEN_SYMBOLS:
                    continue
                declared_at = declared.get(symbol)
                if declared_at is None:
                    violations.append(
                        f"{name}:{number}: {symbol} is CUDA-only and not declared by the file's HIP shim"
                    )
                elif declared_at > number:
                    violations.append(
                        f"{name}:{number}: {symbol} is used before the HIP shim declares it at line {declared_at}"
                    )
    return list(dict.fromkeys(violations))


def _hip_compatible_kernel_sources() -> dict[str, str]:
    ext_root = Path(__file__).resolve().parents[1] / "gptqmodel_ext"
    sources = {}
    for path in sorted(ext_root.rglob("*")):
        if not path.is_file() or path.suffix not in _KERNEL_SOURCE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in _HIP_MARKERS):
            sources[path.relative_to(ext_root.parent).as_posix()] = text
    return sources


def test_hip_shims_provide_every_cuda_only_intrinsic_used_under_hip():
    sources = _hip_compatible_kernel_sources()
    # Guard the guard: a scan that silently finds nothing could never fail.
    assert "gptqmodel_ext/exllamav3/bf16_scale.cu" in sources
    assert "gptqmodel_ext/paro/rotation.cuh" in sources
    assert _hip_shim_violations(sources) == []


def _hipify_pairs(entries):
    """Flatten torch's hipify table, which is a list of dicts, a dict, or tuples."""

    if isinstance(entries, dict):
        for source, target in entries.items():
            if isinstance(target, str):
                yield source, target
            else:
                yield from _hipify_pairs(target)
        return
    for entry in entries:
        if isinstance(entry, dict):
            yield from _hipify_pairs(entry)
        elif isinstance(entry, tuple) and len(entry) >= 2:
            yield entry[0], entry[1]


def test_hipify_still_rewrites_the_bf16_types_the_contract_exempts():
    # The scanner trusts torch's hipify table for __nv_bfloat16/__nv_bfloat162
    # and relies on it having no conversion-intrinsic entry. Pin both halves.
    hipify = pytest.importorskip("torch.utils.hipify.cuda_to_hip_mappings")
    pairs = set(_hipify_pairs(hipify.CUDA_TO_HIP_MAPPINGS))
    assert ("__nv_bfloat16", "__hip_bfloat16") in pairs
    assert ("__nv_bfloat162", "__hip_bfloat162") in pairs
    assert not any(source == "__float2bfloat16_rn" for source, _ in pairs)


def test_hip_shim_contract_detects_the_issue_5208_regression():
    # Pre-#5208 shim: it aliased the bf16 *type* only, so the kernel still
    # called CUDA's scalar __float2bfloat16_rn and hipcc rejected the file.
    pre_fix = {
        "gptqmodel_ext/exllamav3/bf16_scale.cu": """\
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_bf16.h>
using __nv_bfloat16 = __hip_bfloat16;
#else
#include <cuda_bf16.h>
#endif

__global__ void rescale(const float* value, __nv_bfloat16* out)
{
    out[threadIdx.x] = __float2bfloat16_rn(value[threadIdx.x]);
}
""",
    }
    violations = _hip_shim_violations(pre_fix)
    assert len(violations) == 1, violations
    assert "__float2bfloat16_rn" in violations[0]


def test_hip_shim_contract_accepts_a_corrected_scalar_mapping():
    # Control: the same file with the HIP branch mapping the scalar conversion
    # (the rotation.cuh pattern) is clean, and the CUDA branch may still name
    # the CUDA intrinsic because hipcc never compiles it.
    fixed = {
        "gptqmodel_ext/exllamav3/bf16_scale.cu": """\
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_bf16.h>
using __nv_bfloat16 = __hip_bfloat16;
static __device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float value)
{
    return __float2bfloat16(value);
}
#else
#include <cuda_bf16.h>
#endif

__global__ void rescale(const float* value, __nv_bfloat16* out)
{
    out[threadIdx.x] = __float2bfloat16_rn(value[threadIdx.x]);
}
""",
    }
    assert _hip_shim_violations(fixed) == []
