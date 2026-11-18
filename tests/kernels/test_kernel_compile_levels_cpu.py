# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measured CPU ``torch.compile`` levels for every CPU-capable quantized kernel.

Each case builds a real packed module on CPU, then measures it with
``tests/kernels/compile_probe.py`` on torch 2.16.0.dev20261021+cpu. The CPU
level is read from ``SUPPORTS_COMPILE_LEVEL_BY_DEVICE["cpu"]`` when the kernel
declares one, otherwise from ``SUPPORTS_COMPILE_LEVEL``.

``fullgraph=True`` is the "graph safe" bar here: reaching ``DYNAMO`` or above
means dynamo traced the whole forward with zero graph breaks. A kernel whose
CPU forward is documented as ``EAGER`` (``GGUFCppKernel``) only has to run
eagerly; the technical reason is recorded next to its declaration. The biased
and BF16 int4pack layers are exercised separately, because the CPU AUTO dtype
is BF16 and both used to be the documented #4068 exceptions (#5110).

Kernels whose optional runtime is absent on this host are skipped with that
reason, not silently counted as passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

from gptqmodel.nn_modules.qlinear._compile_support import COMPILE_LEVEL, compile_level
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTorchLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastTorchLinear, LLMAwqTorchLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.quantization import FORMAT
from tests.kernels.compile_probe import probe


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
ROWS = 4
DTYPE = torch.float16


def _dense(*, bias: bool = False, seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias, dtype=DTYPE)


def _gptq_kwargs(*, bits: int = 4, bias: bool = False, group_size: int = GROUP_SIZE):
    return {
        "bits": bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": bias,
        "format": FORMAT.GPTQ_V2,
    }


def _pack_gptq(klass, *, bits: int = 4, bias: bool = False):
    module = klass(**_gptq_kwargs(bits=bits, bias=bias))
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        _dense(bias=bias),
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=DTYPE),
        torch.full((OUT_FEATURES, groups), 1 << (bits - 1), dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval()


def _pack_int8():
    """``TorchInt8Linear`` is not packable; build it from an 8-bit checkpoint.

    ``post_init()`` converts the stored 8-bit GPTQ tensors to the INT8 layout.
    """

    source = _pack_gptq(TorchLinear, bits=8)
    module = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src)
    module.post_init()
    return module.eval()


def _pack_cpu_avx512():
    """Build the BF16 AVX512-BF16 fused kernel and run its one-time transform."""

    torch.manual_seed(0)
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16)
    module = TorchCpuAvx512Linear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        dtype=torch.bfloat16,
    )
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        dense,
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.bfloat16),
        torch.full((OUT_FEATURES, groups), 1 << 3, dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    module = module.to("cpu").eval()
    module.post_init()
    module.transform_cpu()
    return module


def _awq_new(klass, *, bits: int = 4, bias: bool = True):
    return klass(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        register_buffers=True,
    )


def _pack_awq(klass, *, bits: int = 4, bias: bool = True):
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.full((OUT_FEATURES, groups), 1 / 128, dtype=DTYPE)
    zeros = torch.full((OUT_FEATURES, groups), 1 << (bits - 1), dtype=torch.int32)

    source = _awq_new(AwqTorchLinear, bits=bits, bias=bias)
    source.pack(_dense(bias=bias), scales, zeros)
    if hasattr(source, "post_init"):
        source.post_init()
    source.eval()
    if klass is AwqTorchLinear:
        return source

    module = _awq_new(klass, bits=bits, bias=bias)
    for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval()


def _pack_gguf():
    module = GGUFTorchLinear(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    module.pack_original(_dense(), scales=None, zeros=None)
    return module.eval()


def _pack_qqq():
    module = QQQTorchLinear(**_gptq_kwargs(group_size=IN_FEATURES))
    module.pack(_dense(), torch.full((OUT_FEATURES, 1), 1 / 128, dtype=DTYPE))
    return module.eval()


def _pack_original(klass, *, group_size: int = -1, bits: int = 4, none_scales: bool = False):
    module = klass(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    if none_scales:
        module.pack_original(_dense(), None, None)
    else:
        module.pack_original(_dense())
    return module.eval()


def _pack_ttbfp():
    from gptqmodel.utils.backend import BACKEND

    module = TTBlockfloatLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
    )
    module.pack(_dense())
    return module.eval()


def _pack_vptq():
    module = VPTQLinear(
        bits=1,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=GROUP_SIZE,
        group_num=-1,
        enable_norm=False,
        enable_perm=False,
        dtype=DTYPE,
    ).eval()
    module.indices.zero_()
    return module


def _unavailable(cls) -> Optional[str]:
    validator = getattr(cls, "cached_validate_once", None)
    if not callable(validator):
        return None
    try:
        ok, error = validator()
    except Exception as exc:  # a missing optional runtime raises lazily
        return f"{cls.__name__} unavailable: {type(exc).__name__}: {exc}"
    return None if ok else f"{cls.__name__} unavailable: {error}"


@dataclass(frozen=True)
class Case:
    cls: type
    build: Optional[Callable[[], torch.nn.Module]]
    unavailable: Callable[[], Optional[str]] = lambda: None
    dtype: torch.dtype = DTYPE


CASES = [
    Case(TorchLinear, lambda: _pack_gptq(TorchLinear)),
    Case(TorchFusedLinear, lambda: _pack_gptq(TorchFusedLinear)),
    # FP32 activations must take the same no-cast int4pack path as BF16, so a
    # compiled FP32 model does not break on every quantized linear (#5087).
    Case(TorchFusedLinear, lambda: _pack_gptq(TorchFusedLinear), dtype=torch.float32),
    Case(TorchInt8Linear, _pack_int8),
    Case(TorchAtenLinear, lambda: _pack_gptq(TorchAtenLinear)),
    # BF16 takes the AVX512-BF16 fused extension GEMM, not the dense fallback.
    Case(TorchCpuAvx512Linear, _pack_cpu_avx512, dtype=torch.bfloat16),
    Case(AwqTorchLinear, lambda: _pack_awq(AwqTorchLinear, bias=True)),
    # The bias-free FP16 case is the plain extern-op path. The BF16 and biased
    # variants (whose product needs an FP32 cast) are covered by
    # ``INT4PACK_MATRIX`` below.
    Case(TorchFusedAwqLinear, lambda: _pack_awq(TorchFusedAwqLinear, bias=False)),
    Case(TorchFusedAwqLinear, lambda: _pack_awq(TorchFusedAwqLinear, bias=False), dtype=torch.float32),
    Case(TorchAtenAwqLinear, lambda: _pack_awq(TorchAtenAwqLinear, bias=False)),
    Case(
        TorchCpuAvx512AwqLinear,
        lambda: _pack_awq(TorchCpuAvx512AwqLinear, bias=True).to(torch.bfloat16),
        dtype=torch.bfloat16,
    ),
    Case(AwqGEMVTorchLinear, lambda: _pack_awq(AwqGEMVTorchLinear, bias=True)),
    Case(LLMAwqTorchLinear, lambda: _pack_awq(LLMAwqTorchLinear, bias=True)),
    Case(AwqGEMVFastTorchLinear, lambda: _pack_awq(AwqGEMVFastTorchLinear, bias=True)),
    Case(GGUFTorchLinear, _pack_gguf),
    Case(TTBlockfloatLinear, _pack_ttbfp),
    Case(TorchFP8Linear, lambda: _pack_original(TorchFP8Linear, bits=8, none_scales=True)),
    Case(
        BitsAndBytesLinear,
        lambda: _pack_original(BitsAndBytesLinear, group_size=GROUP_SIZE, none_scales=True),
    ),
    Case(NF12Linear, lambda: _pack_original(NF12Linear, bits=12)),
    # BF16 selects Paro's AVX512-BF16 fused path (bias included); FP16 is the
    # ATen int4pack path, where a biased layer asks for the FP32 product.
    Case(
        ParoLinear,
        lambda: _pack_awq(ParoLinear, bias=True).to(torch.bfloat16),
        dtype=torch.bfloat16,
    ),
    Case(QQQTorchLinear, _pack_qqq),
    Case(VPTQLinear, _pack_vptq),
    Case(GGUFCppKernel, None, lambda: _unavailable(GGUFCppKernel)),
]

#: The int4pack CPU kernels (GPTQ and AWQ) declare ``INDUCTOR`` for their whole
#: supported dtype/bias matrix, not just the fp16 bias-free case the ``CASES``
#: table exercises. BF16 is the CPU AUTO dtype for them, and a biased layer
#: takes the FP32-product path, so every combination was measured and has to
#: reach the declared level (#5110).
INT4PACK_KERNELS = (
    (TorchFusedLinear, "gptq"),
    (TorchAtenLinear, "gptq"),
    (TorchFusedAwqLinear, "awq"),
    (TorchAtenAwqLinear, "awq"),
)
INT4PACK_MATRIX = [
    pytest.param(
        cls,
        biased,
        dtype,
        kind,
        id=f"{cls.__name__}-{'biased' if biased else 'bias-free'}-"
        f"{str(dtype).replace('torch.', '')}",
    )
    for cls, kind in INT4PACK_KERNELS
    for biased in (False, True)
    for dtype in (torch.float16, torch.bfloat16)
]


@pytest.mark.parametrize(
    "cls,biased,dtype,kind",
    INT4PACK_MATRIX,
)
def test_int4pack_cpu_kernel_reaches_declared_level_for_bf16_and_bias(
    cls, biased, dtype, kind
):
    """BF16 (the CPU AUTO dtype) and biased layers trace too (#5110).

    Before the casting product became a fake-able custom op, these layers broke
    the graph on every quantized linear while the class still declared
    ``INDUCTOR``.
    """

    if kind == "awq":
        build = lambda: _pack_awq(cls, bias=biased)  # noqa: E731
    else:
        build = lambda: _pack_gptq(cls, bias=biased)  # noqa: E731

    reason = _unavailable(cls)
    if reason is not None:
        pytest.skip(reason)
    try:
        module = build().to(dtype)
    except Exception as error:
        pytest.skip(
            f"{cls.__name__}: cannot build on CPU: "
            f"{type(error).__name__}: {str(error)[:200]}"
        )

    x = torch.randn(ROWS, IN_FEATURES, dtype=dtype)
    result = probe(module, (x,), check_cuda_graph=False)
    assert result.eager, f"{cls.__name__}: CPU eager forward failed: {result.eager_error}"
    assert result.graph_breaks == 0, (
        f"{cls.__name__} (dtype={dtype}, biased={biased}): "
        f"graph_breaks={result.graph_breaks} {result.break_reasons[:2]}"
    )
    declared = compile_level(cls, device_type="cpu")
    assert result.reached(declared), (
        f"{cls.__name__} (dtype={dtype}, biased={biased}): declares CPU "
        f"{declared.name} but the probe reported {result.summary()} "
        f"errors={result.errors}"
    )
    assert result.output_close.get("inductor", True), (
        f"{cls.__name__} (dtype={dtype}, biased={biased}): inductor output "
        "does not match eager on CPU"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.cls.__name__)
def test_cpu_kernel_reaches_declared_cpu_level(case):
    reason = case.unavailable()
    if reason is not None:
        pytest.skip(reason)
    if case.build is None:
        pytest.skip(f"{case.cls.__name__}: no CPU builder on this host")

    try:
        module = case.build()
    except Exception as error:
        # A builder that needs an optional runtime is an environment skip, not
        # a stale declaration.
        pytest.skip(
            f"{case.cls.__name__}: cannot build on CPU: "
            f"{type(error).__name__}: {str(error)[:200]}"
        )

    x = torch.randn(ROWS, IN_FEATURES, dtype=case.dtype)
    result = probe(module, (x,), check_cuda_graph=False)
    declared = compile_level(case.cls, device_type="cpu")

    assert result.eager, f"{case.cls.__name__}: CPU eager forward failed: {result.eager_error}"
    assert result.reached(declared), (
        f"{case.cls.__name__}: declares CPU {declared.name} but the probe reported "
        f"{result.summary()} errors={result.errors}"
    )

    if declared >= COMPILE_LEVEL.DYNAMO:
        # ``fullgraph=True`` already proves zero graph breaks; keep the
        # explicit check so a future probe change cannot silently drop it.
        assert result.levels.get("dynamo"), (
            f"{case.cls.__name__}: declared CPU {declared.name} without a clean "
            f"fullgraph trace: {result.errors.get('dynamo')}"
        )
    if declared >= COMPILE_LEVEL.INDUCTOR:
        assert result.output_close.get("inductor", True), (
            f"{case.cls.__name__}: inductor output does not match eager on CPU"
        )


def _compile_first(build, *, dtype: torch.dtype = DTYPE):
    """Compile before any eager forward, then compare the compiled output."""

    module = build()
    x = torch.randn(ROWS, IN_FEATURES, dtype=dtype)
    torch._dynamo.reset()
    compiled = torch.compile(module.forward, fullgraph=True, backend="inductor")
    with torch.inference_mode():
        out = compiled(x)
    with torch.inference_mode():
        ref = module.forward(x)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2), (
        "a compile-before-eager forward differs from the eager reference"
    )


def test_qqq_cpu_compiles_before_the_first_eager_call():
    """Without a warmup the packed cache is absent; the int8 fallback traces."""

    _compile_first(_pack_qqq)


def test_paro_biased_fp16_compiles_before_the_first_eager_call():
    """A biased FP16 layer takes the traceable dense reference before a layout exists."""

    _compile_first(lambda: _pack_awq(ParoLinear, bias=True))


def test_nf12_cpu_compiles_after_post_init_without_an_eager_call():
    """``post_init`` resolves the exponent base, so a cold trace has no host read."""

    def build():
        module = _pack_original(NF12Linear, bits=12)
        module.post_init()
        return module

    _compile_first(build)


def _pack_embedding(*, force_row_decode: bool) -> TorchQuantEmbeddings:
    module = TorchQuantEmbeddings(**_gptq_kwargs())
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        _dense(),
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=DTYPE),
        torch.full((OUT_FEATURES, groups), 1 << 3, dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    module.post_init()
    if force_row_decode:
        # The default threshold is 64 MiB; force the row-decode branch so a
        # small test module exercises it without a huge vocabulary.
        module.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 0
    return module.eval()


@pytest.mark.parametrize(
    "force_row_decode", [False, True], ids=["whole_table", "row_decode"]
)
def test_torch_quant_embeddings_reaches_declared_cpu_level(force_row_decode):
    module = _pack_embedding(force_row_decode=force_row_decode)
    ids = torch.randint(0, IN_FEATURES, (2, 8), dtype=torch.int64)
    result = probe(module, (ids,), check_cuda_graph=False)
    declared = compile_level(TorchQuantEmbeddings, device_type="cpu")

    assert declared is COMPILE_LEVEL.INDUCTOR
    assert result.eager, result.eager_error
    assert result.reached(declared), (
        f"TorchQuantEmbeddings ({force_row_decode=}): declared {declared.name} "
        f"but the probe reported {result.summary()} errors={result.errors}"
    )
