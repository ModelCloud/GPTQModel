# NF12 row-major embedding vector experiment on GPU1

Baseline: `e3141059`. This experiment extends the GPU2 row-major prototype
with native vector loads, contiguous stores, cache variants, and CTA ID packing.
It does not change the installed NF12 operator.

The follow-up [layout prototype and end-to-end load report](nf12-rowmajor-layout-prototype-gpu1.md)
adds CPU/GPU conversion, a versioned packed state, and full-checkpoint timings.

## Candidate and validation

The candidate keeps the prototype's 96-byte packed row/super-atom: 32 metadata
bytes, 64 mantissa bytes, and a separate four-BF16-value escape table. Four
lanes now decode a 64-output block. Each lane loads one 8-byte metadata chunk
and one 16-byte mantissa chunk, decodes four groups, and writes two contiguous
16-byte BF16 vectors. The winning code-ID variants use ordinary global stores
or `__stcg` stores that bypass L1. CTA ID packing varies by row count and ID
dtype. L2-only packed loads (`__ldcg`) and two-lane blocks lost in the sweep.

The candidate retains `992,870,400` packed bytes and `2,444,016` escape bytes,
`995,314,416` bytes total, the same total as current NF12. No reconstructed
BF16 weight or expanded code tensor persists. The kernel decodes into registers
and the BF16 output; temporary dequantized weight tensor storage is zero.
After repeated forwards, the candidate object still contains only those two
packed tensors and scalar metadata. A side-by-side packed cache would add
`995,314,416` resident bytes; replacing the old packed layout preserves its
storage size. This standalone experiment has no model mutation or checkpoint
cache lifecycle.

- Physical GPU1, NVIDIA PG506-232, SM80; `CUDA_VISIBLE_DEVICES=1`,
  `PYTHON_GIL=0`, `/root/venv-lite-torch215/bin/python`.
- Released `embed.weight` is BF16 `[129280, 5120]`, SHA256
  `aa1ac979edab508950a871c370fa645a875203a1a35c20ba9c1d45bb1ed46e3c`.
  Code IDs use the prefix of
  [the Lite fixture](nf12-lite-cuda-token-ids-20261111.json); random IDs use
  seed `30911 + M`. The two code prefixes contain 303 and 413 unique IDs.
- All swept variants produced exact BF16 bits against current NF12 for the
  complete released-weight outputs. Selected variants also matched independent
  dense BF16 lookup exactly for code and random IDs at both M values and both
  ID dtypes. Extreme negative and out-of-range IDs returned exact zero bits.
- A forced-base `[16,128]` Torch codec fixture checked two original escape
  groups, 254 selected-half sentinel collisions, signed zero, subnormals, and
  NaN payloads. All seven modes and valid 64/128/256/512-thread,
  1/2/4/8/16-ID CTA combinations matched exact BF16 bits for int32 and int64.
  The released table itself has zero selected-half collisions.
- The sweep used four CUDA graph timing samples per mode/geometry. Each pair
  used eight calls per graph replay and six ABBA rounds, with synchronized CUDA
  events. NF12/candidate and dense/candidate were separate same-process
  phases, so the three full representations were never simultaneously resident.
  Peak CUDA allocated and reserved memory were `2,523,998,208` and
  `2,531,262,464` bytes, below the 3 GB added-VRAM limit.

## Confirmed paired medians (microseconds)

The two slash-separated columns are separate paired ABBA measurements. The
candidate geometry was selected on code IDs and carried unchanged into the
random-ID cases of the same M and dtype.

| IDs | dtype | M | mode:threads × IDs/CTA | current NF12 / candidate | dense BF16 / candidate |
| --- | --- | ---: | --- | ---: | ---: |
| code | int64 | 1536 | `1:256×8` | 11.968 / 11.264 | 9.600 / 11.264 |
| code | int64 | 3072 | `2:128×2` | 24.576 / 19.776 | 18.560 / 19.648 |
| code | int32 | 1536 | `2:256×8` | 11.904 / 11.264 | 9.600 / 11.264 |
| code | int32 | 3072 | `2:128×2` | 23.936 / 19.968 | 18.560 / 19.968 |
| random | int64 | 1536 | `1:256×8` | 24.256 / 12.160 | 11.840 / 12.032 |
| random | int64 | 3072 | `2:128×2` | 45.568 / 31.744 | 35.456 / 32.128 |
| random | int32 | 1536 | `2:256×8` | 24.064 / 11.904 | 11.968 / 11.840 |
| random | int32 | 3072 | `2:128×2` | 45.440 / 31.744 | 35.840 / 32.000 |

The previous eight-lane row-major kernel was also paired directly against the
candidate on code IDs: int64 M1536 `13.760 / 11.520`, int64 M3072
`25.856 / 19.712`, int32 M1536 `13.568 / 11.264`, and int32 M3072
`26.112 / 19.648` microseconds. A separate first full-weight run selected
nearby geometries and reproduced the code-ID wins over current NF12:
int64 M1536 `12.032 / 11.264`, int64 M3072 `24.576 / 19.776`,
int32 M1536 `11.904 / 11.200`, and int32 M3072 `23.936 / 20.736`.

## Decision

Keep the four-lane vector kernel as an **experimental packed-only candidate**.
It beats current NF12 on the repeated code IDs by 5–24% in the paired runs,
and it is bit exact even under escape collisions. Dense BF16 remains 6–19%
faster on those code cases. Production adoption needs a row-major checkpoint
layout and weight mutation lifecycle; the experiment does not add dispatch or
model-owned cache state.

Source: [kernel](../gptqmodel_ext/nf12/experiments/rowmajor_embedding.cu)
and [benchmark](../scripts/bench_nf12_rowmajor_embedding_next.py).
Raw samples: [first run](nf12-rowmajor-next-gpu1-20261111.json) and
[confirmation](nf12-rowmajor-next-gpu1-confirm-20261111.json).

Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON_GIL=0 PYTHONPATH=/root/nf12-next-gpu1 \
  /root/venv-lite-torch215/bin/python scripts/bench_nf12_rowmajor_embedding_next.py \
  --json-out docs/nf12-rowmajor-next-gpu1-repeat.json
```
