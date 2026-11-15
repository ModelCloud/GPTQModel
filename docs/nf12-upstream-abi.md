# NF12: consuming the ABI upstream exposes

The NF12 kernels are vendored from [dphnAI/nf12](https://github.com/dphnAI/nf12)
under `gptqmodel_ext/nf12` (CUDA) and `gptqmodel_ext/nf12_hip` (ROCm). This tree
adds operations, but every name upstream exposes keeps its signature, struct
layout, operator schema and export, so a library built from either tree is
callable through the same ABI.

`gptqmodel_nf12/abi.py` pins that surface for one upstream commit and
`tests/test_nf12_abi_parity.py` re-parses the vendored copies to prove it still
matches. A changed signature, struct field or schema is an ABI break for every
prebuilt library and shows up as a test failure instead of a runtime mismatch.

## What is pinned

| surface | source | parity |
|---|---|---|
| C entry points | `include/nf12/nf12.h` | 25 upstream functions, identical signatures; this tree adds 6 |
| struct layouts | `nf12_weight_t` (6 fields), `nf12_moe_weight_t` (7 fields) | identical, field order and types |
| status values | `nf12_status_t` | identical (`NF12_OK` 0 … `NF12_ERROR_CUDA` 4) |
| version macros | `NF12_VERSION_MAJOR/MINOR` | 1.1 on both sides |
| operator schemas | `csrc/torch/ops.cpp` | 14 upstream ops in namespace `nf12`, identical; this tree adds 6 |
| stable ABI | `STABLE_TORCH_LIBRARY` | both sides register through torch's stable ABI |
| exported symbols | `csrc/nf12.map` | identical (`nf12_*` global, everything else local) |
| Python API | `python/nf12/__init__.py` | 21 upstream names, identical signatures |

The six additions are `nf12_decode_fp32`, `nf12_sub_fp32`,
`nf12_embedding_repeat4`, `nf12_embedding_i64_repeat4`, `nf12_linear_staged_b1`
and `nf12_linear_staged_b1_workspace_bytes`, with the matching
`decode_fp32`/`sub_fp32`/`embedding_repeat4`/`linear_staged_b1` operator schemas
and the `rowmajor_*` pair. `tests/test_nf12_abi_parity.py` pins that list, so a
new entry point cannot appear without an explicit edit.

## Known divergence: the ROCm operator namespace

Upstream builds ROCm and CUDA from one header (`NF12_HIP` aliases
`hipStream_t`/`hipError_t` onto the CUDA spellings) and registers both under
`torch.ops.nf12`. This tree registers ROCm under `torch.ops.nf12_hip`
(`gptqmodel_ext/nf12_hip/nf12_torch.cpp`), and `gptqmodel_nf12/kernels.py`
records both registries. The schemas match op for op, so moving to one
namespace is a rename on the Python side plus a registration change in the HIP
translation unit — not an ABI change. Until then, a consumer that switches
backends must switch the namespace too.

## Re-pinning after an upstream bump

```bash
git clone https://github.com/dphnAI/nf12 /tmp/nf12
git -C /tmp/nf12 checkout <commit>
python tools/generate_nf12_abi.py --source /tmp/nf12
git diff gptqmodel_nf12/abi.py        # review: any change here is an ABI change
pytest tests/test_nf12_abi_parity.py
NF12_SOURCE=/tmp/nf12 pytest tests/test_nf12_abi_parity.py -k regenerator
```

The regenerator reproduces `abi.py` byte for byte, so a stale pin is visible as
a test failure rather than as a silent mismatch. `NF12_SOURCE` additionally
checks that round trip in CI-style runs; without it the check is skipped.

## Metal and MLX

The Metal/MLX path does not link the C API: `gptqmodel/nn_modules/qlinear/mlx_nf12.py`
compiles MSL at runtime and reads the same packed storage (`1536`-byte
super-atoms, 512-byte planes, `escapes` rows of 8 int16, one `exp_base` byte).
Upstream's Metal kernels (`csrc/metal/linear.metal`) read that layout directly,
so the format ABI is shared even though the code is not.
