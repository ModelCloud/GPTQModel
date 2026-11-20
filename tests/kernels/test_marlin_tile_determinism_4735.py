# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4735: the FP32 Marlin bias stage must fit the budget and stay out of the reduction.

#4657 widened the bias epilogue stage to FP32 in ``marlin_template.h``
(``thread_n_blocks * 16 / 4`` int4 instead of ``/ 8``) but left the host-side
shared-memory model in ``get_kernel_cache_size`` at the old 16-bit size
(``tb_n * 2`` bytes). It also kept the bias region at the *start* of the
B/reduce union, so the wider stage shares memory with the split-K reduction.
A thread config the host accepts can therefore need more dynamic shared memory
than the launch reserves, and the bias copy overlaps reduction scratch.

The reported symptom is nondeterministic, wrong output tiles for the AUTO
kernel: GPTQ ``MarlinLinear`` bf16 on sm_120 and AWQ ``AwqMarlinLinear`` fp16
on sm_89. The bad cells are whole 16x64/64x64 tiles off by 2-5 against an
independent FP64 oracle, while a correct call is within ~0.1 (bf16).

This file pins the two halves of the contract:

* a source check that the host byte count, the kernel int4 count, and the
  dedicated bias offset agree (this is what makes the budget model sound), and
* a real-kernel check: the same input through the same launch repeated 30
  times must be bitwise identical, and within an FP64 oracle bound. The
  determinism half is the issue's own repro and fails on the affected
  architectures even when the source contract is fixed.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.marlin import marlin_import_exception, marlin_runtime_available, marlin_runtime_error


ROOT = Path(__file__).resolve().parents[2]

# One launch per forward; the reporter measured the bad tiles within 30 calls.
REPEATS = 30


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _assigned_expression(source: str, pattern: str) -> str:
    match = re.search(pattern, source, flags=re.DOTALL)
    assert match is not None, f"pattern {pattern!r} not found"
    return " ".join(match.group(1).split())


def test_host_bias_stage_matches_the_fp32_kernel_layout():
    """The host budget must use the FP32 stage size the kernel actually uses."""

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    kernel = _source("gptqmodel_ext/marlin/marlin_template.h")

    # ``sh_bias_size`` in the kernel is int4; the host model is bytes.
    kernel_bias = _assigned_expression(
        kernel, r"constexpr int sh_bias_size = \(([^;]*)\);"
    )
    assert "thread_n_blocks * 16 / 4" in kernel_bias, kernel_bias

    host_bias = _assigned_expression(host, r"int sh_bias_size = ([^;]*);")
    assert "tb_n * 4" in host_bias, host_bias

    # The bias stage must not overlap the B/reduce union: a split-K reduction
    # writes that union after the bias copy is issued.
    bias_offset = _assigned_expression(
        kernel.replace(" ", ""), r"int4\*sh_bias=sh\+([^;]*);"
    )
    assert "sh_size_b_red_max" in bias_offset, bias_offset

    # ``tmp_size`` must reserve the union *plus* the dedicated bias stage.
    tmp_size = _assigned_expression(
        host, r"int tmp_size = (\([^;]*\))\s*\+\s*sh_bias_size;"
    )
    assert "sh_b_size > sh_red_size" in tmp_size, tmp_size


def test_host_g_idx_stage_is_accounted_in_bytes():
    """``g_idx_stage`` is ``tb_k / 4`` int4, i.e. ``tb_k * 4`` bytes per stage."""

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    g_idx = _assigned_expression(host, r"int sh_g_idx_size = ([^;]*);")
    assert "stages * tb_k * 4" in g_idx, g_idx


# ---------------------------------------------------------------------------
# Derived shared-memory accounting
#
# The two checks above compare source *strings*. That catches the exact #4657
# drift, but it does not recompute the extent the kernel actually declares, so
# it cannot tell whether a host budget covers the whole layout (B/reduce union,
# FP32 bias, g_idx, zero points, scales, A) for the configs ``is_valid_config``
# accepts. The checks below translate the real declarations in
# ``marlin_template.h`` / ``gptq_marlin.cu`` into Python, evaluate them over the
# real thread-config matrix, and assert
# ``host dynamic shared memory >= kernel requirement``.
# ---------------------------------------------------------------------------


def _split_top_level(expr: str, sep: str) -> int | None:
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            return i
    return None


def _replace_ternary(expr: str) -> str:
    """Rewrite C++ ``cond ? a : b`` (including nesting) as a Python conditional."""

    expr = expr.strip()
    # Translate parenthesised sub-expressions first, so any ``?`` left at depth
    # zero belongs to this level.
    out = []
    i = 0
    while i < len(expr):
        if expr[i] == "(":
            depth = 1
            j = i + 1
            while j < len(expr) and depth:
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                j += 1
            out.append("(" + _replace_ternary(expr[i + 1 : j - 1]) + ")")
            i = j
        else:
            out.append(expr[i])
            i += 1
    expr = "".join(out)

    question = _split_top_level(expr, "?")
    if question is None:
        return expr
    colon = _split_top_level(expr[question + 1 :], ":")
    assert colon is not None, expr
    colon += question + 1
    cond = _replace_ternary(expr[:question])
    then = _replace_ternary(expr[question + 1 : colon])
    other = _replace_ternary(expr[colon + 1 :])
    return f"({then} if {cond} else {other})"


@lru_cache(maxsize=None)
def _translate_c(expr: str) -> str:
    """Translate the small C++ subset used by the two layout models to Python."""

    expr = " ".join(expr.split())
    expr = expr.replace("sizeof(int)", "4")
    expr = expr.replace("vllm::kFE2M1f", "False")
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    expr = _replace_ternary(expr)
    # Every operand is a non-negative int, so C++ truncating division is ``//``.
    return expr.replace("/", "//")


def _eval_c(expr: str, **env):
    return eval(_translate_c(expr), {"__builtins__": {}}, {"max": max, "min": min, **env})


@lru_cache(maxsize=None)
def _decl(source: str, name: str) -> str:
    match = re.search(
        r"(?:constexpr\s+)?int\s+" + re.escape(name) + r"\s*=\s*(.+?);",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"{name} is not declared in the source"
    return match.group(1)


def _thread_configs(source: str) -> list[tuple[int, int, int]]:
    configs = []
    for block in re.finditer(
        r"thread_config_t\s+\w+\[\]\s*=\s*\{(.*?)\};", source, flags=re.DOTALL
    ):
        for match in re.finditer(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", block.group(1)):
            config = tuple(int(part) for part in match.groups())
            if config not in configs:
                configs.append(config)
    assert configs, "no thread configs found"
    return configs


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _kernel_shmem_bytes(
    kernel: str, th_k: int, th_n: int, tmb: int, num_bits: int, group_size: int,
    has_act_order: bool, is_k_full: bool, has_zp: bool, is_zp_float: bool,
    stages: int, m_block_size_8: bool,
) -> tuple[int, dict[str, int]]:
    """Recompute the int4 regions ``marlin_template.h`` declares, in bytes."""

    env = {
        "thread_k_blocks": th_k // 16,
        "thread_n_blocks": th_n // 16,
        "thread_m_blocks": tmb,
        "pack_factor": 32 // num_bits,
        "stages": stages,
        "has_act_order": has_act_order,
        "has_zp": has_zp,
        "is_zp_float": is_zp_float,
        "m_block_size_8": m_block_size_8,
        "w_type": "kU4",
        # The modelled kernels use 16-bit scales; 8-bit (FP4/MX) scales halve the scale stage.
        "is_8bit_scale": False,
        "group_blocks": 0
        if has_act_order
        else (-1 if group_size == -1 else (2 if group_size == 0 else group_size // 16)),
        "tb_k": _eval_c(_decl(kernel, "tb_k"), **{"thread_k_blocks": th_k // 16}),
    }
    env["m_block_size"] = _eval_c(_decl(kernel, "m_block_size"), **env)
    env["sh_red_size"] = _eval_c(_decl(kernel, "sh_red_size"), **env)
    env["b_sh_stride"] = _eval_c(_decl(kernel, "b_sh_stride"), **env)
    env["b_sh_stage"] = _eval_c(_decl(kernel, "b_sh_stage"), **env)
    env["sh_b_size"] = _eval_c(_decl(kernel, "sh_b_size"), **env)
    env["sh_size_b_red_max"] = _eval_c(_decl(kernel, "sh_size_b_red_max"), **env)
    env["sh_bias_size"] = _eval_c(_decl(kernel, "sh_bias_size"), **env)
    if re.search(r"int\s+sh_size_b_red_min\s*=", kernel):
        # The pre-#4735 layout put the bias at the start of the union. Keep the
        # port able to read that source so the sensitivity check reports a byte
        # gap instead of a parse error.
        env["sh_size_b_red_min"] = _eval_c(_decl(kernel, "sh_size_b_red_min"), **env)
    env["sh_b_red_bias_size"] = _eval_c(_decl(kernel, "sh_b_red_bias_size"), **env)
    env["g_idx_stage"] = _eval_c(_decl(kernel, "g_idx_stage"), **env)
    env["s_sh_stride"] = _eval_c(_decl(kernel, "s_sh_stride"), **env)
    env["s_tb_groups"] = _eval_c(_decl(kernel, "s_tb_groups"), **env)
    env["s_sh_stage"] = _eval_c(_decl(kernel, "s_sh_stage"), **env)
    env["act_s_max_num_groups"] = _eval_c(
        _decl(kernel, "act_s_max_num_groups"), **env
    )
    env["sh_s_size"] = _eval_c(_decl(kernel, "sh_s_size"), **env)
    env["zp_sh_stride"] = _eval_c(_decl(kernel, "zp_sh_stride"), **env)
    env["zp_tb_groups"] = env["s_tb_groups"]
    env["zp_sh_stage"] = _eval_c(_decl(kernel, "zp_sh_stage"), **env)
    env["a_sh_stride"] = _eval_c(_decl(kernel, "a_sh_stride"), **env)
    env["a_sh_stage"] = _eval_c(_decl(kernel, "a_sh_stage"), **env)

    regions = {
        # ``sh_b_red_bias_size`` is the g_idx offset: the extent reserved for
        # the B/reduce union plus the bias stage. ``span`` is what the kernel
        # actually declares; ``union`` + ``bias`` only equals it when the bias
        # has its own region after the union (the #4735 layout).
        "span": env["sh_b_red_bias_size"],
        "union": env["sh_size_b_red_max"],
        "bias": env["sh_bias_size"],
        "g_idx": stages * env["g_idx_stage"],
        "zp": stages * env["zp_sh_stage"],
        "scales": env["sh_s_size"],
        "a": stages * env["a_sh_stage"],
    }
    total_int4 = (
        regions["span"]
        + regions["g_idx"]
        + regions["zp"]
        + regions["scales"]
        + regions["a"]
    )
    return total_int4 * 16, {name: value * 16 for name, value in regions.items()}


def _host_scales_bytes(
    host: str, th_k: int, th_n: int, group_size: int, has_act_order: bool,
    is_k_full: bool, stages: int,
) -> int:
    """Port of ``get_scales_cache_size`` (cross-checked against its source)."""

    flat = " ".join(host.split())
    assert "tb_groups = div_ceil(tb_k, 32)" in flat
    assert "load_groups = tb_groups * stages * 2" in flat
    assert "load_groups = max(load_groups, 32)" in flat
    assert "return load_groups * tb_n * 2" in flat
    assert "int tb_scales = tb_groups * tb_n * 2" in flat
    assert "return tb_scales * stages" in flat

    tb_groups = (
        1
        if group_size == -1
        else (_div_ceil(th_k, 32) if group_size == 0 else _div_ceil(th_k, group_size))
    )
    if has_act_order and not is_k_full:
        return max(tb_groups * stages * 2, 32) * th_n * 2
    return tb_groups * th_n * 2 * stages


def _host_shmem_bytes(
    host: str, th_k: int, th_n: int, tmb: int, num_bits: int, group_size: int,
    has_act_order: bool, is_k_full: bool, has_zp: bool, is_zp_float: bool,
    stages: int, *, bias_bytes: int | None = None,
) -> tuple[int, dict[str, int]]:
    """Recompute ``get_kernel_cache_size()`` in bytes."""

    env = {
        "tb_m": tmb * 16,
        "tb_k": th_k,
        "tb_n": th_n,
        "pack_factor": 32 // num_bits,
        "stages": stages,
        "has_act_order": has_act_order,
        "is_k_full": is_k_full,
        "has_zp": has_zp,
        "is_zp_float": is_zp_float,
    }
    env["sh_a_size"] = _eval_c(_decl(host, "sh_a_size"), **env)
    env["sh_b_size"] = _eval_c(_decl(host, "sh_b_size"), **env)
    env["sh_red_size"] = _eval_c(_decl(host, "sh_red_size"), **env)
    env["sh_bias_size"] = _eval_c(_decl(host, "sh_bias_size"), **env)
    if bias_bytes is not None:
        env["sh_bias_size"] = bias_bytes
    env["tmp_size"] = _eval_c(_decl(host, "tmp_size"), **env)
    env["sh_s_size"] = _host_scales_bytes(
        host, th_k, th_n, group_size, has_act_order, is_k_full, stages
    )
    env["sh_g_idx_size"] = _eval_c(_decl(host, "sh_g_idx_size"), **env)
    env["sh_zp_size"] = 0
    if has_zp:
        if is_zp_float:
            assert "sh_zp_size = sh_s_size;" in host
            env["sh_zp_size"] = env["sh_s_size"]
        elif num_bits == 4:
            assert "sh_zp_size = sh_s_size / 4;" in host
            env["sh_zp_size"] = env["sh_s_size"] // 4
        elif num_bits == 8:
            assert "sh_zp_size = sh_s_size / 2;" in host
            env["sh_zp_size"] = env["sh_s_size"] // 2
    total = _eval_c(_decl(host, "total_size"), **env)
    return total, {
        "union": env["sh_b_size"] if env["sh_b_size"] > env["sh_red_size"] else env["sh_red_size"],
        "bias": env["sh_bias_size"],
        "g_idx": env["sh_g_idx_size"],
        "zp": env["sh_zp_size"],
        "scales": env["sh_s_size"],
        "a": env["sh_a_size"],
    }


def _accepted_configs(kernel: str):
    """Every (thread config, M tile, bits, group, act/zp) the host can accept."""

    for th_k, th_n, _threads in _thread_configs(kernel):
        for tmb in (1, 2, 3, 4):
            for num_bits in (4, 8):
                for group_size in (128, 64, 32, -1):
                    for has_act_order in (False, True):
                        for has_zp in (False, True):
                            for is_zp_float in ((False, True) if has_zp else (False,)):
                                yield (
                                    th_k, th_n, tmb, num_bits, group_size,
                                    has_act_order, has_zp, is_zp_float,
                                )


def test_host_shared_memory_budget_covers_the_kernel_layout():
    """For every accepted config the host must reserve at least what the kernel uses."""

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    kernel = _source("gptqmodel_ext/marlin/marlin_template.h")
    stages = int(_decl(_source("gptqmodel_ext/marlin/marlin.cuh"), "pipe_stages"))

    # The FP32 stage is int4 in the kernel and bytes in the host model.
    kernel_bias = _eval_c(
        _decl(kernel, "sh_bias_size"), thread_n_blocks=256 // 16
    ) * 16
    host_bias = _eval_c(_decl(host, "sh_bias_size"), tb_n=256)
    assert host_bias == kernel_bias == 1024

    checked = 0
    for th_k, th_n, tmb, num_bits, group_size, has_act_order, has_zp, is_zp_float in _accepted_configs(host):
        for m_block_size_8 in (False, True):
            kernel_bytes, kregions = _kernel_shmem_bytes(
                kernel, th_k, th_n, tmb, num_bits, group_size, has_act_order,
                False, has_zp, is_zp_float, stages, m_block_size_8,
            )
            host_bytes, hregions = _host_shmem_bytes(
                host, th_k, th_n, tmb, num_bits, group_size, has_act_order,
                False, has_zp, is_zp_float, stages,
            )
            assert host_bytes >= kernel_bytes, (
                "host dynamic shared memory is smaller than the kernel layout: "
                f"thread_k={th_k} thread_n={th_n} thread_m_blocks={tmb} "
                f"bits={num_bits} group={group_size} act={has_act_order} "
                f"zp={has_zp}/{is_zp_float} m8={m_block_size_8}: "
                f"host={host_bytes} kernel={kernel_bytes} "
                f"host_regions={hregions} kernel_regions={kregions}"
            )
            # The regions #4657 changed must match exactly, not merely fit.
            assert hregions["bias"] == kregions["bias"], (hregions, kregions)
            assert hregions["g_idx"] == kregions["g_idx"], (hregions, kregions)
            assert hregions["union"] == kregions["union"], (hregions, kregions)
            # The FP32 bias must not be carved out of the B/reduce union: the
            # declared span is the union followed by the bias, nothing shared.
            assert kregions["span"] == kregions["union"] + kregions["bias"], (
                "the bias stage overlaps the B/reduce union",
                kregions,
            )
            checked += 1

    assert checked > 0


def test_the_pre_4657_bias_budget_would_under_count():
    """Sensitivity: the old ``tb_n * 2`` bias term must fail the same check.

    Without this, a future edit could make both models agree on the wrong
    number and the invariant test above would silently pass.
    """

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    kernel = _source("gptqmodel_ext/marlin/marlin_template.h")
    stages = int(_decl(_source("gptqmodel_ext/marlin/marlin.cuh"), "pipe_stages"))

    short = []
    for th_k, th_n, tmb, num_bits, group_size, has_act_order, has_zp, is_zp_float in _accepted_configs(host):
        kernel_bytes, _ = _kernel_shmem_bytes(
            kernel, th_k, th_n, tmb, num_bits, group_size, has_act_order,
            False, has_zp, is_zp_float, stages, False,
        )
        old_bytes, _ = _host_shmem_bytes(
            host, th_k, th_n, tmb, num_bits, group_size, has_act_order, False,
            has_zp, is_zp_float, stages, bias_bytes=th_n * 2,
        )
        if old_bytes < kernel_bytes:
            short.append((th_k, th_n, tmb, num_bits, group_size, has_act_order,
                          has_zp, is_zp_float, kernel_bytes - old_bytes))

    assert short, "the old 16-bit bias term unexpectedly covers the FP32 layout"
    # tn=256 is the config that shows the widest gap: the 512-byte bias
    # under-count, exactly as reported for #4735.
    assert max(entry[-1] for entry in short) >= 512


def _skip_reason(dtype: torch.dtype) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the Marlin determinism test"
    if marlin_import_exception is not None:
        return f"Marlin kernel unavailable: {marlin_import_exception}"
    if not marlin_runtime_available(dtype):
        return marlin_runtime_error(dtype)
    return None


def _build_gptq(k: int, n: int, dtype: torch.dtype):
    """Build a v1 GPTQ ``MarlinLinear`` the way the loader does."""

    torch.manual_seed(4735)
    codes = torch.randint(0, 16, (k, n))
    zeros = torch.full((k // 128, n), 8, dtype=torch.int32)
    scales = (torch.randn(k // 128, n) * 0.01 + 0.02).abs().clamp_min(0.01).to(dtype)
    indices = torch.arange(k, dtype=torch.int32) // 128
    weight = (codes.float() - zeros[indices].float()) * scales.float()[indices]

    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data.copy_(weight.t().to(dtype))

    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "in_features": k,
        "out_features": n,
        "bias": False,
        "register_buffers": True,
        "dtype": dtype,
    }
    source = TorchLinear(**kwargs, format=FORMAT.GPTQ_V2)
    source.pack_block(linear, scales.t(), zeros.t().float(), indices)

    module = MarlinLinear(**kwargs).cuda()
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
            dst.data.copy_(src.cuda())
    module.post_init()
    module.eval()
    return module, weight.cuda()


def _build_awq(k: int, n: int, dtype: torch.dtype):
    """Build a mock AWQ 4-bit g128 ``AwqMarlinLinear`` (no zero-point float)."""

    from gptqmodel.quantization.awq.utils.packing_utils import pack_awq

    torch.manual_seed(4735)
    codes = torch.randint(0, 16, (k, n))
    zeros = torch.full((k // 128, n), 8, dtype=torch.int32)
    scales = (torch.randn(k // 128, n) * 0.01 + 0.02).abs().clamp_min(0.01).to(dtype)
    indices = torch.arange(k, dtype=torch.int32) // 128
    weight = (codes.float() - zeros[indices].float()) * scales.float()[indices]

    qweight, qzeros = pack_awq(codes, zeros, 4)
    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": False,
        "desc_act": False,
        "in_features": k,
        "out_features": n,
        "bias": False,
        "register_buffers": True,
        "dtype": dtype,
    }
    module = AwqMarlinLinear(**kwargs, backend="marlin").cuda()
    with torch.no_grad():
        module.qweight.data.copy_(qweight.cuda())
        module.qzeros.data.copy_(qzeros.cuda())
        module.scales.data.copy_(scales.cuda())
    module.post_init()
    module.eval()
    return module, weight.cuda()


def _assert_deterministic_and_correct(module, weight, *, size_m, size_k, dtype, atol):
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)
    with torch.inference_mode():
        first = module(x).clone()
        digests = {hashlib.sha1(first.float().cpu().numpy().tobytes()).hexdigest()}
        worst = torch.zeros((), device="cuda")
        for _ in range(REPEATS - 1):
            out = module(x)
            digests.add(hashlib.sha1(out.float().cpu().numpy().tobytes()).hexdigest())
            worst = torch.maximum(worst, (out - first).abs().amax())
    torch.cuda.synchronize()

    assert len(digests) == 1, (
        f"Marlin is not deterministic for M={size_m} K={size_k} {dtype}: "
        f"{len(digests)} distinct outputs in {REPEATS} calls, max|delta|={worst.item():.6f}"
    )

    reference = x.double() @ weight.double()
    error = (first.double() - reference).abs().max().item()
    assert error < atol, (
        f"Marlin disagrees with the FP64 oracle for M={size_m} K={size_k} {dtype}: "
        f"max|err|={error:.4f} (bound {atol})"
    )


# The issue's GPTQ table: K x N at M in {64, 256, 1000}, bf16 on sm_120.
_GPTQ_SHAPES = ((2048, 8192), (4096, 4096))
_GPTQ_M = (64, 256, 1000)

# The issue's AWQ table: fp16 M>=512, N>=2048 on sm_89.
_AWQ_SHAPES = ((2048, 2048), (8192, 2048))
_AWQ_M = (512, 2048)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m", _GPTQ_M)
@pytest.mark.parametrize("size_k,size_n", _GPTQ_SHAPES)
def test_gptq_marlin_bf16_repeated_launches_are_bitwise_identical(size_k, size_n, size_m):
    reason = _skip_reason(torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)

    module, weight = _build_gptq(size_k, size_n, torch.bfloat16)
    _assert_deterministic_and_correct(
        module, weight, size_m=size_m, size_k=size_k, dtype=torch.bfloat16, atol=0.5
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m", _AWQ_M)
@pytest.mark.parametrize("size_k,size_n", _AWQ_SHAPES)
def test_awq_marlin_fp16_repeated_launches_are_bitwise_identical(size_k, size_n, size_m):
    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    module, weight = _build_awq(size_k, size_n, torch.float16)
    _assert_deterministic_and_correct(
        module, weight, size_m=size_m, size_k=size_k, dtype=torch.float16, atol=0.05
    )
