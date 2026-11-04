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
