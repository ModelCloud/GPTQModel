# Machete / Swordfish external ABI (experimental)

Build with `bash gptqmodel_ext/quant_abi/build_abi.sh /absolute/output/path`.
This produces `libqvq_quant.so`, a C boundary linked against the installed
LibTorch/CUDA libraries. It is **not** a Torch-free implementation. External
hosts must load the matching QvQ Machete/Swordfish operator shared libraries
before preparation; no Python interpreter is needed by the C entrypoints.
The Swordfish implementation retains its AGPL-3.0-or-later licensing.

`quant_abi.h` is the ABI authority. Version 1 accepts contiguous device tensors
with explicit shapes, dtypes and byte counts. Buffers must refer to live
allocations of at least the declared size. Inputs may alias each other but not
the output. Packing formats and GPTQ/AWQ scale/zero conventions are unchanged;
this API does not convert checkpoint formats. Swordfish activations must already
be group-sorted. Its AWQ zeros are `(8 - zp) * scale`, not raw zero-point integers.

Prepare outside capture on a nondefault stream. Preparation executes three
warmups (including writes to the output), synchronizes, then captures the selected
operator and output copy into a retained LibTorch allocation pool. Launch replays
that graph or attaches it as a child of a caller capture. This avoids allocator
and tuning calls during replay but costs an output copy and retains workspace.
It is not a raw, caller-workspace CUTLASS launcher.

Plans bind buffer addresses, device and stream. Values at input addresses may
change between ordered launches; shapes/addresses/configuration may not. Keep
the buffers, stream, operator libraries, and plan alive until all launches finish
and all enclosing CUDA graphs are destroyed. Serialize destroy with every use.
Destroy synchronizes the owner stream and is forbidden during capture. Host
callers must not pass stale handles. Prepare/capture must be serialized with
other device activity as required by LibTorch CUDA graph capture.

## Selection and tuning coverage

| Path | Exposed | Still internal / unavailable |
| --- | --- | --- |
| Machete MM | Exact compiled schedule string; query all compiled schedules for dtype combination; group size and optional scale/zero tensors | Tile/cluster/scheduler/stage choices encoded or fixed by each generated schedule; no arbitrary recompilation |
| Machete prepack | Activation, weight and group-scale dtypes | Native packing implementation |
| Swordfish decode | Deterministic / atomic / Stream-K; M tiles; exact split-K; exact CTA grid; quad claims; checked threads/stages | Threads fixed at 128; stages constrained by compiled dtype/tile/mode; no silent correction |
| Swordfish prefill | Explicit prefill route; bits/group size; exact compiled tile N128/256 and M-chunk size | Pipeline/cluster configuration still compile-time; W8 has only tile N128 |
| Swordfish prepack / dequant | Bits, group size, K/N, permutation for pack, transpose for dequant | Native packing/dequant geometry |

Swordfish decode bypasses dense/prefill crossovers, environment overrides and
occupancy-derived grid selection. Deterministic mode requires T=1, split=1,
CTAs=0, quad=false, stages=1. Atomic mode supports T=1..3, CTAs=0, quad=false.
Stream-K supports T=1..4, split=1, CTAs>0; quad supports T=2/3 with N%256=0.
Atomic and Stream-K stage constraints: T1=5, T2=4 (W4) or 5 (W8), T3=3,
T4=2 (W4) or 4 (W8). These are exposed constraints, not freely tunable values.
Atomic and Stream-K reduction are not deterministic.

Machete is restricted to SM90; Swordfish to SM100/103/110. Rejection never selects
Marlin implicitly. The host owns fallback policy. MoE and dense-tier GEMM are
not yet in this ABI. No claim of all-knob exposure, ZML loader integration,
command-buffer compatibility, numerical correctness or speedup is made by
host-only compilation and rejection tests.

Run ABI tests:

```
QVQ_QUANT_ABI_LIBRARY=/absolute/output/path/libqvq_quant.so \
  python tests/kernels/test_quant_abi_contract.py
```

Before promotion: compile actual backend libraries for every advertised target,
test GPTQ/AWQ values and tails against references on Hopper/Blackwell, exercise
changing inputs across repeated graphs and external CUDA PJRT capture, verify
workspace lifetime/stream ordering, then measure matched end-to-end timing.

## Recorded local validation (2026-09-09)

- LibTorch-linked C ABI builds with `--no-undefined`; all five C symbols exported.
- Nine Python ABI tests pass, including invalid headers/unused tuning/default
  stream/null lifecycle, isolated metadata transport, and actual SM80 rejection.
- `g++ -std=c++17 -Wall -Wextra -Werror -I. tests/kernels/swordfish_decode_config_contract.cc`
  compiles; its valid/invalid decode configuration and C layout assertions pass.
- Modified Swordfish decode, FP16 prefill and BF16 prefill TUs compile to SM100a
  objects using CUDA 13.3 and CUTLASS v4.7.1 (`cb4247394dd82148787aed73e5dc7cef33cbf862`).
  This is not a full backend link or runtime test. Reproduce with
  `bash gptqmodel_ext/quant_abi/check_swordfish_build.sh CUTLASS_DIR OUTPUT_DIR 100a`.
- New Python test lint passes. The existing Swordfish Python loader has 31 Ruff
  findings, identical before and after the two required-op additions.

Physical GPU 0: PG506-230, SM80, UUID
`GPU-3a4bf14f-fa28-df88-f6e8-00ef6b13d473`, PCI `00000000:A5:00.0`.
Target numerical, nondeterminism, capture/replay and performance evidence remains
**unavailable**, not passed. No kernel math or production selection defaults were
changed; new explicit entrypoints are opt-in. Build checks also found and fixed
the pre-existing device-pass helper visibility and CUTLASS scheduler signature
incompatibilities. Prefill hardware queries now use the input device ordinal.
