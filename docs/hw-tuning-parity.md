<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hardware tuning parity: RTX 4090 (Ada) and RDNA4

GPT-QModel ships two first-class consumer/workstation GPU targets and every
tuned kernel pipeline must have a measured path for **both**:

| Family id | Hardware | Detection |
|---|---|---|
| `ada` | NVIDIA RTX 4090 class, SM 8.9 | `gptqmodel.utils.arch.is_ada()` |
| `rdna4` | AMD gfx1200 / gfx1201 / gfx1250 | `gptqmodel.utils.arch.is_rdna4()` |

`gptqmodel.utils.arch.tuning_family(target)` is the single vocabulary shared by
every launch-constant table. It returns `ada`, `hopper`, `blackwell`, `cdna4`,
`rdna4`, `rdna3`, or `other`, and accepts a device, a `gfx*` / `sm_*` string, or
a `(major, minor)` capability tuple. Tables are keyed on that string, so a new
pipeline cannot silently inherit another architecture's tile.

## Why the two are not interchangeable

* **Wave/warp width.** RDNA4 is wave32; CDNA is wave64; NVIDIA warps are 32
  lanes. A CDNA `num_warps=8` tile covers 512 lanes and is a poor fit on wave32.
* **Matrix core generation.** Ada has 4th-gen tensor cores and FP8 `mma`, but
  no Hopper TMA / warp specialization. SM90+ tiles must never be applied to
  SM 8.9, and `torch.cuda.get_device_capability` must never classify AMD
  hardware (it returns an unrelated pair under HIP).
* **ROCm library availability.** `hipBLASLt` Tensile libraries exist for
  `gfx1200`/`gfx1201`/`gfx1250` but are not tuned; rocBLAS remains the default.

## Acceptance gate

`tests/test_hw_parity_contract.py` is the CI gate. It is hardware-free and
asserts that each tuned pipeline's launch table contains both `ada` and
`rdna4`. Adding a tuned pipeline means adding it to `TUNED_PIPELINES` in that
test at the same time as adding the table.

Two pipelines do not fit that shape and have dedicated assertions instead:

* **GGUF** (`gguf_triton`) is keyed by quant type and shape, not by family.
  The SM 8.9 fixed tiles are asserted to fire only on capability `(8, 9)`; on
  ROCm the kernel keeps its own autotuned generic launch.
* **AWQ fused GEMM** (`awq/modules/triton/scheduler.py`) keys rules on an exact
  device profile rather than a family token, because neighbouring shapes were
  not stable enough to bucket. The contract builds a six-field CUDA identity
  and a seven-field HIP identity from the shipped constants, asserts each
  resolves to its own measured rule, and asserts that a wrong gfx arch, board
  name, CU count, memory size, or non-FP16 dtype resolves to the generic
  policy instead.

**bitsandbytes 4-bit** (`nn_modules/qlinear/bitsandbytes_triton.py`) does fit
the table shape — `_BNB_GEMM_LAUNCH` carries an `ada` and an `rdna4` tile and
the pipeline is registered in `TUNED_PIPELINES` — but its *default* is
family-dependent rather than unconditional. On ROCm the fused Triton GEMM
replaces the vendor HIP kernel for `M >= 16` (2.4x-6.3x on gfx1201). On Ada
the vendor cuBLAS path wins on every shape tested, so the `ada` entry exists
for the contract and for forced A/B runs behind `GPTQMODEL_BNB_TRITON=1`, not
as the default. Below `M = 16` bitsandbytes' own GEMV is kept on both
families. Parity here means *a measured tile exists and is gated*, not that
every family ships it enabled.

## Pipeline status

Per-pipeline status is tracked in `docs/rocm-rdna4.md` (RDNA4 side) and in the
docstring of each kernel module (tile provenance and measurement conditions).
A table value that has not been measured on real hardware is marked as
unmeasured in a comment and defaults to the previously shipped safe value; a
speedup is never accepted without a measured number.

## Measurement protocol

Follow `scripts/benchmark_rdna4_inference.py`:

1. warm up, then take the median of >= 30 iterations;
2. bracket every timing with `torch.cuda.synchronize()`;
3. report hot (L2-resident) and L2-flushed numbers separately;
4. report the exact shape, dtype, bits, and group size.

## Numerical requirements

Every changed kernel must be validated against an independent Torch oracle per
`.agents/skills/gptqmodel-torch-oracle-numerics/SKILL.md`: exact packed codes,
relative and absolute float tolerance of `1e-6` or tighter for quantization
math, up to `2e-3` for inference output, and `output.dtype == input.dtype` for
each supported activation dtype. Weights stay in their quantized representation
between forwards; no dense or expanded weight cache may persist (see
`AGENTS.md`).
