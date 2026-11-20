# Native EXL3 CUDA dispatch

GPT-QModel runs EXL3 projections directly from their packed trellis. The CUDA
GEMV and GEMM kernels come from ExLlamaV3 v1.6.7 by TurboDerp and contributors.
GPT-QModel supplies the Torch operator boundary, packed launch-selection
table, and per-call scratch. Its EXL3 port also adapts AlexBrandes's Q4NX
codebook, beam encoder, and CUDA fragment decoder. [ExLlamaV3 v1.6.7 source](https://github.com/turboderp-org/exllamav3/tree/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0) · [EXL3 source credits](../gptqmodel/exllamav3/CREDITS.md)

The `3inst`, `mcg` and `mul1` codebooks handle integer
rates from 1 to 8 bits per weight. The `mul1` codebook also handles
1.5, 2.5 and 3.5 bits per weight, with 24, 40 and 56 16-bit words per
16×16 tile. The half-bit tiles alternate widths and use the same codebook
values as their integer relatives.

## Q4NX

Q4NX was designed for routed MoE experts at 2.5 and 3.0 bits per weight.
Qubitium extended Q4NX to 3.5 bits per weight using the same layout. [Original Q4NX format specification](https://github.com/AlexBrandes/exllamav3/blob/41ef7819945fe421626da3c40c678053d240d051/doc/q4nx.md)

Set `EXL3Config(bits=2.5, codebook="q4nx")` for all selected projections,
or assign both the rate and codebook to a subset through `dynamic`.

```python
from gptqmodel.quantization import EXL3Config

qcfg = EXL3Config(
    bits=4,
    codebook="mcg",
    dynamic={
        r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj$": {
            "bits": 2.5,
            "codebook": "q4nx",
        },
    },
)
```

| Tile property | `3inst`, `mcg`, `mul1` | `q4nx` |
| --- | --- | --- |
| 16×16 payload at 2.5, 3.0, and 3.5 bpw | 40, 48, and 56 int16 words, respectively. The half-bit rates use `mul1`. | 40, 48, and 56 int16 words, respectively |
| State | 256 scalar steps with overlapping 16-bit states | 64 steps with four weights per state and 22, 20, or 18 state bits |
| Tile boundary | Window wraps to the end of the same tile | Bits before the start are zero |
| Checkpoint selection | `.mcg`, `.mul1` or no marker | A scalar int32 `.q4nx` marker with multiplier `0x83DCD12D` |

Each step adds a 10-, 12-, or 14-bit suffix, most significant bit first. Safetensors
stores each 32-bit word as its low and then high 16-bit halfword. The nibble
sums of the low 32 bits of `state * 0x83DCD12D` give four
integers in `[-15, 15]`. The decoder divides them by `sqrt(42.5)`. NVIDIA
CUDA kernels feed the integer values into FP16 tensor-core operands and
apply the normalization in FP32 after accumulation. W2.5 and W3.0 use a
16-state beam for scale search and a 128-state beam for final encoding.
W3.5 preserves all 16 overlap classes at each step and uses exact Viterbi
search for both operations.

```mermaid
flowchart LR
    C["EXL3Config and dynamic rules"] --> R{"Supported rate?"}
    R -->|yes| V["quantize_tiles_q4nx"]
    R -->|no| E["ValueError"]
    V --> P["pack_trellis_q4nx"]
    P --> S["Safetensors trellis, scales and q4nx marker"]
    S --> L{"Marker keys agree with tensor_storage?"}
    L -->|yes| G["Native CUDA projection or routed decode"]
    L -->|yes| T["Chunked EXL3_TORCH reconstruction"]
    L -->|no| F["ValueError names the marker key"]
    classDef error fill:#f9d5d5,stroke:#af3b3b,color:#222
    class E,F error
```

Quantization requires NVIDIA CUDA. Inference uses native CUDA kernels for
FP16 and BF16 on NVIDIA, or `BACKEND.EXL3_TORCH` for portable reconstruction.
Q4NX CUDA dispatch on Ampere and Ada (sm_89) selects a packed GEMV for one to
eight rows when the column groups fit in a co-resident wave. On Ada the wide
config is limited to the measured small-k, large-n band (2.5 and 3.5 bpw,
`k <= 2048`, `n >= 12288`), and a 3.0 bpw tile of 64 Mi elements or more keeps
the packed kernel, which reads the trellis from L2 there. Each warp unpacks
AlexBrandes's zero-prefixed windows, including Qubitium's W3.5 extension, from
registers.
The kernel accumulates in FP32 and applies normalization with the output
Hadamard transform.

For large Q4NX prefill on Ampere, the native decoder writes the integer
weights into temporary FP16 storage, then HGEMM and the output transform
produce FP32 results. The path handles projections of 4 to 24 Mi weight
elements and begins at 64 rows up to 12 Mi elements, or at 128 rows for
larger projections. Weight scratch peaks at 48 MiB per call and is released
after HGEMM. Shapes outside the tuned range and calls short of workspace use
packed GEMM.

Set `EXL3CUDAConfig(fused_gemm=True, fp16_gemv=False)` to pin Q4NX to packed
GEMM. With `fused_gemm=True` and `fp16_gemv=True`, Q4NX can select GEMV while
prefill uses packed GEMM. INT8 GEMV remains a `mul1` policy.
The FP32 packed GEMM variants also tune the launch grid on Ampere for the
5120 → 2304, 5120 → 4608, and 2304 → 5120 expert projections. The selected
row ranges use 128-column tiles and distribute work across all available SMs.
Q4NX weights also work with the explicit [routed decode operator](exl3-routed-decode.md).
The saved quantization config keeps the default `codebook` and the
`dynamic` recipe. Its `tensor_storage` entry lists each projection's
`.q4nx` tensor and its dtype and shape. The loader selects the decoder
from the per-projection marker, so a checkpoint can mix Q4NX and other EXL3
codebooks. Both loaders verify marker keys against the actual safetensors
contents before constructing a decoder. A dynamic output-head rule can
select a different codebook. The effective head rate is validated against
that codebook.

| Checkpoint state | Loader result |
| --- | --- |
| The storage manifest and safetensors agree on the markers | Construct the selected decoder. |
| Either side is missing a marker present on the other | Raise a `ValueError` identifying the tensor. |

Packed FP16 forward uses FP32 for split-K partials, the output Hadamard
transform, scales and bias, then casts once. The BF16 bridge rescales the raw
projection, adds the canonical bias in FP32, and writes BF16. [Activation dispatch](exl3-wide-activation-dispatch.md)

### Blackwell (sm_120)

Blackwell was unmeasured when the Ada band landed, so `exl3_gemv_try_launch`
declines Q4NX there. The pre-#5873 sweep on an RTX 5090 (170 SMs, 96 MiB L2)
measured the Q4NX GEMV with the broken output epilogue: in all 180 cells it
disagreed with the reconstructed FP32 weight by 0.16 to 0.82 relative, while
the block-pipelined kernel stayed within 3.3e-4. #5873 fixes that epilogue for
the non-Q4NX FP16 GEMV (below). Q4NX still keeps the packed kernel because the
same sweep measured only a 1.06x cold and 1.18x warm geometric-mean speedup,
with 61 of the 180 cells regressing (0.77x warm at the worst).
`tests/kernels/test_exllamav3_q4nx_gemv_blackwell.py` pins the decline until a
fresh sweep justifies enabling Q4NX.

| bpw | Cells | GEMV vs FP32 reference (broken epilogue) | Packed vs reference | Cold geomean | Warm geomean | Regressions |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 2.5 | 60 | 0.16–0.74, median 0.37 | ≤ 3.2e-4 | 1.04x | 1.18x | 24/60 |
| 3.0 | 60 | 0.17–0.82, median 0.37 | ≤ 3.1e-4 | 1.05x | 1.17x | 20/60 |
| 3.5 | 60 | 0.16–0.68, median 0.36 | ≤ 3.2e-4 | 1.07x | 1.18x | 17/60 |

The timing rows are the pre-#5873 Q4NX measurement, taken with the gate opened
for the duration of the sweep; the committed tree declines every Q4NX cell, so
the table below is the Q4NX kernel Blackwell actually runs. Times are medians
in microseconds per projection: `cold` flushes 160 MiB of L2 before each of 20
launches, `warm` is a 100-launch batch after five warmups. `GB/s` is the packed
trellis, `k * n * bits / 8` bytes over the measured time.

| bpw | k | n | m | Cold | GB/s | Warm | GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 | 2048 | 2048 | 1 | 15.70 | 83.5 | 14.40 | 91.1 |
| 2.5 | 4096 | 4096 | 1 | 19.46 | 269.5 | 16.75 | 312.9 |
| 2.5 | 8192 | 8192 | 1 | 35.82 | 585.4 | 32.28 | 649.7 |
| 2.5 | 2048 | 16384 | 8 | 23.55 | 445.2 | 20.22 | 518.6 |
| 3.0 | 2048 | 2048 | 1 | 16.37 | 96.1 | 13.52 | 116.4 |
| 3.0 | 4096 | 4096 | 1 | 19.44 | 323.6 | 16.46 | 382.2 |
| 3.0 | 8192 | 8192 | 1 | 37.89 | 664.2 | 32.22 | 781.0 |
| 3.0 | 2048 | 16384 | 8 | 25.54 | 492.8 | 20.09 | 626.3 |
| 3.5 | 2048 | 2048 | 1 | 15.36 | 119.5 | 13.32 | 137.7 |
| 3.5 | 4096 | 4096 | 1 | 21.09 | 348.1 | 16.48 | 445.5 |
| 3.5 | 8192 | 8192 | 1 | 39.94 | 735.2 | 32.27 | 909.9 |
| 3.5 | 2048 | 16384 | 8 | 25.55 | 574.5 | 20.08 | 731.1 |

The packed Q4NX GEMM tuning grid (`exl3_packed::samples_*` and
`map_cc_to_index`) has no Blackwell row: `cc_values` covers Ampere, Ada and
Hopper. `CC_BLACKWELL` is defined as `CC_HOPPER`, so sm_120 indexes the Hopper
row and prefill is a tuned launch, not a generic fallback. At 3.0 bpw and 64
rows the 5090 takes 45.49 us for 5120 → 4608 and 76.77 us for 8192 → 8192 cold
(42.54 and 73.26 us warm, dispatch shape 7); 128 rows doubles that.

Run the sweep with:

```bash
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    /root/venv/bin/python tests/benchmark_exl3_q4nx_gemv_blackwell.py --check
```

### FP16 GEMV epilogue on Blackwell

The non-Q4NX FP16 GEMV (`exl3_gemv_kernel`) writes one C tile per group. A
128-column output tile spans groups written by different blocks, so the output
Hadamard transform could not run safely inside the cooperative kernel after
`grid.sync()` on Blackwell: the post-barrier reads returned whole raw tiles.
#5873 moves the transform to `exl3_gemv_epilogue_kernel`, launched next in the
same stream after the cooperative kernel completes. Stream ordering makes the
C stores visible without a device-scope barrier, and the kernel applies the
same `had_ff_r_128_inner` / `had_hf_r_128_inner` scaling per 128-column tile.
The FP16 GEMV dispatch (90) now serves eligible non-Q4NX shapes on sm_120 and
sm_121.

Before:

```mermaid
flowchart TD
    A["exl3_gemv_try_launch"] --> B["exl3_gemv_kernel"]
    B --> C["grid.sync()"]
    C --> D["in-kernel had_ff_r_128_inner reads C"]
    D --> E["Blackwell: stale 128-column tiles"]
    classDef error fill:#f9d5d5,stroke:#af3b3b,color:#222
    class E error
```

After:

```mermaid
flowchart TD
    A["exl3_gemv_try_launch"] --> B["cudaLaunchCooperativeKernel exl3_gemv_kernel"]
    B --> C["stream order: cooperative kernel completes"]
    C --> D["exl3_gemv_epilogue_kernel"]
    D --> E["had_ff_r_128_inner per 128-column tile"]
    E --> F["Projection output"]
```

On the 48-SM GB10 test host, the fixed GEMV matched the FP64 oracle within
6.7e-4 across 80 eligible cells (m 1/2/4/8, k and n 2048/4096/8192). The
separate launch measured 1.05x cold and 1.00x warm geometric-mean speedup over
the packed kernel; the 170-SM RTX 5090 warm result in #5873 was not
re-measured on this host.

## Runtime policy

```python
from gptqmodel import GPTQModel
from gptqmodel.backend import EXL3CUDAConfig

fast = EXL3CUDAConfig(fp16_gemv=True, fused_gemm=True)
model = GPTQModel.load(checkpoint, backend=fast)
model_with_int8 = GPTQModel.load(checkpoint, backend=EXL3CUDAConfig(int8_gemv=1))
```

`fp16_gemv` enables the upstream small-batch heuristic for supported shapes
with up to eight rows. `fused_gemm` enables the upstream GEMM, which shares
packed weight reads across rows and applies the output transform in shared
memory. Existing integer checkpoints default to the original GEMM for its
summation and rounding order. Half-bit checkpoints default to fused GEMM.
Both GEMM variants use FP32 tensor-core accumulators. The small-batch QTIP
kernel folds short FP16 accumulation intervals into FP32, as in upstream.
Use the GEMM path for inputs that need FP32 range through the inner
product. On a W3 `mul1` stress fixture with one row, K=512 and N=256,
constant activations of 4,000 and 5,500 made all 256 QTIP outputs
non-finite. Both GEMM paths stayed finite. At a constant activation of
1,000, QTIP was finite with a 0.26938 maximum absolute projection difference
against the original GEMM. These stress values describe the tested fixture.

`int8_gemv=0` is the default. Mode `1` uses error-feedback residuals in
eligible single-row `mul1` calls. Mode `2` uses plain INT8 activations in
eligible one- or two-row `mul1` calls. These modes change the arithmetic.
Eligibility depends on device resources, bitrate, and aligned dimensions.
GEMV requests outside those limits use the applicable FP16 path.

## Dispatch

| Operation | Earlier dense path | Current native path |
| --- | --- | --- |
| Layer policy | One integer packed GEMM entry | Per-layer FP16 and INT8 GEMV policy |
| Small batch | GEMM | Configured GEMV when eligible, or GEMM |
| Larger batch | Each 16-row slab rereads the packed weights | Selected 16-, 32-, 48-, or 64-row tile |
| Output transform | Separate device pass inside the cooperative kernel | Transform from the final shared-memory output tile |
| CUDA stream state | Per-launch Hadamard and split-K scratch | Per-launch Hadamard, split-K and INT8 scratch |
| Q4NX small batch | Packed GEMM | Register decode and FP32 GEMV on eligible Ampere shapes |
| Q4NX large prefill | Packed GEMM for every row tile | One native reconstruction per call and FP32 HGEMM on tuned Ampere shapes |

```mermaid
flowchart TD
    A["BC_LinearEXL3.run"] --> B["bc_linear_exl3_run_configured"]
    B --> C["exl3_gemm validates geometry and selects the device"]
    C --> Q{"Eligible Q4NX prefill?"}
    Q -->|yes| W["reconstruct_q4nx, hgemm and output Hadamard"]
    Q -->|no| D{"INT8 policy and eligible mul1 shape?"}
    W -->|workspace unavailable| H["select_exl3_gemm_kernel_tuned"]
    W -->|complete| J["Projection output"]
    D -->|yes| E["exl3_gemv_int8"]
    D -->|no| F{"FP16 GEMV policy and eligible shape?"}
    E -->|declined| F
    F -->|yes| G["exl3_gemv_try_launch"]
    F -->|no| H
    G -->|declined| H
    H --> I["Launch original or fused EXL3 GEMM"]
    E -->|accepted| J["Projection output"]
    G -->|accepted| J
    I --> J
```

The original integer Torch operator and default Python call remain available.
The configured operator accepts a float bitrate and returns the selected path
for diagnostics. NVIDIA CUDA serves the native tensor-core paths. The EXL3
Torch backend provides the CPU inference path; on ROCm, RDNA4 (gfx120x) also
has native HIP GEMV and WMMA GEMM paths (see [RDNA4 HIP dispatch](#rdna4-hip-dispatch)),
while CDNA keeps reconstruct + hipBLAS. NVIDIA CUDA also serves the new
half-bit Viterbi quantizer.

Only immutable launch geometry and function attributes are cached across
streams. CUDA graph allocations retain the pointer addresses of each graph's
working buffers. Inference retains the packed trellis, scale and bias
buffers. The kernels decode weights into temporary tiles or into the
call-private matrix consumed by prefill HGEMM.

For gate, up and down projections in an expert layer, the extension also
provides the [routed decode operator](exl3-routed-decode.md).

### Q4NX launch table on Ada

The packed launch table stores one sample per `(cc, bits, k, n)`. Q4NX rates
are 10, 12 or 14 bits per step, so they do not line up with the integer
`bits = K + half_k` rows. Before this table, every Q4NX launch read the integer
sample for the nearest bitrate: `lookup_packed_sample(table, cc_idx, K + half_k,
k_idx, n_idx)` at `exl3_kernel_map.cu:266` in the parent commit. The retune
adds a `[cc][rate][k][n]` grid (`exl3_packed::samples_*_q4nx`) that only the
Q4NX codebook reads.

```mermaid
flowchart TD
    subgraph before["Before: shared bitrate grid"]
        A1["select_exl3_gemm_kernel_tuned<br/>cb = EXL3_Q4NX_CODEBOOK"] --> B1["lookup_packed_sample<br/>bits = K + half_k"]
        B1 --> C1["integer-tuned shape and num_sms"]
    end
    subgraph after["After: measured Q4NX grid"]
        A2["select_exl3_gemm_kernel_tuned<br/>cb = EXL3_Q4NX_CODEBOOK"] --> R2{"q4nx_rate_index(K, half_k)"}
        R2 -->|"0, 1, 2"| B2["lookup_q4nx_sample<br/>[cc][rate][k][n]"]
        R2 -->|"-1"| E2["TORCH_CHECK: Q4NX requires 2.5, 3.0, or 3.5 bpw"]
        B2 --> C2["measured shape and num_sms<br/>else the declared bitrate fallback"]
    end
    classDef error fill:#f9d5d5,stroke:#af3b3b,color:#222
    class E2 error
```

| Mechanism | Parent commit | This change |
| --- | --- | --- |
| Q4NX sample | `lookup_packed_sample(..., K + half_k, ...)`, `exl3_kernel_map.cu:266` | `lookup_q4nx_sample(..., q4nx_rate_index(K, half_k), ...)` |
| Unknown Q4NX rate | nearest integer row | `TORCH_CHECK` fails closed |
| Grid | integer samples only | 585 measured Ada Q4NX cells, 0 fallbacks |
| Coverage | implicit | the generator refuses an undeclared missing cell |

`tests/benchmark_exl3_q4nx_tuning_ada.py` measures each shape with five
warmups and the median of 100 launches at m=1 and m=16, picks the shape with
the smallest worst-case slowdown across those rows, and scores the `num_sms`
cap across m=16 and m=2048. `scripts/generate_exl3_kernel_map_packed.py`
requires every Ada `(rate, k, n)` cell to be measured or listed in the JSON
`fallback` array; a declared fallback copies the integer `bits = K + half_k`
sample, which is exactly the launch the parent commit used. The generator also
rejects a JSON measured on another capability or device, and the generated
header records the measured and fallback counts. The checked-in table has
585/585 measured cells and no fallbacks.

`exl3_gemm_force_tuned_config(shape_idx, num_sms)` pins one `(shape, cap)` pair
so a sweep can measure every shape from a single build. Production dispatch
never calls it, and a non-positive shape restores the tuned lookup.

Reproduce with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    python tests/benchmark_exl3_q4nx_tuning_ada.py sweep --max-weight-gb 6 \
    --emit-json /tmp/q4nx_ada.json --raw-json /tmp/q4nx_ada_raw.json
cp /tmp/q4nx_ada.json scripts/exl3_kernel_map_q4nx_ada.json
python scripts/generate_exl3_kernel_map_packed.py
```

## RDNA4 HIP dispatch

ROCm builds compile `quant/exl3_gemm_hip.cu` instead of the CUDA tensor-core
tables. The HIP entry point owns the whole decision: it runs the fused
trellis -> WMMA kernel when the device and geometry are covered, and otherwise
reconstructs the weight and calls hipBLAS, so `bc_linear_exl3_run_configured`
never raises for a supported geometry. `EXL3_HIP_DISPATCH_NATIVE` (93) and
`EXL3_HIP_DISPATCH_FALLBACK` (94) report which path ran.

```mermaid
flowchart TD
    A["LinearEXL3.forward (ROCm)"] --> B{"reconstruct flag, R4NX, or (Q4NX and bsz > 8)?"}
    B -->|yes| T["EXL3_TORCH reconstruct + hgemm"]
    B -->|no| C{"bsz <= 8?"}
    C -->|yes| D["bc_linear_exl3_run_hip_gemv"]
    D -->|accepted| J["Projection output"]
    D -->|declined| Q{"Q4NX?"}
    C -->|no| Q
    Q -->|yes| T
    Q -->|no| E["bc.run -> bc_linear_exl3_run_configured -> exl3_gemm"]
    E --> F["exl3_gemm_hip: had_r_128, then validate K and N % 128"]
    F --> G{"c_fp32 and 0 < m <= 512 and gfx120x?"}
    G -->|yes| H["exl3_hip_gemm_kernel WMMA decode, FP32 accumulate, output Hadamard"]
    G -->|no| I["reconstruct, hgemm, output Hadamard"]
    H --> J
    I --> J
    T --> J
```

The native GEMM covers integer rates 1-8 bpw for the `3inst`, `mcg` and
`mul1` codebooks and the 1.5/2.5/3.5 bpw `mul1` half rates, with K and N
multiples of 128 and at most 512 rows. Q4NX has no fused ROCm decoder, so it
keeps the Torch path above the GEMV's batch size. R4NX always reconstructs
(there is no R4NX GEMM kernel on any backend). CDNA and every non-gfx120x ROCm
target fall back to reconstruct + hipBLAS.

### AUTO selection on RDNA4

Before #5836, `ExllamaV3Linear.SUPPORTS_DEVICES` was `[DEVICE.CUDA]`, so
`_resolve_exl3_auto_backend` (`gptqmodel/models/loader.py:2184`) always fell
through to `EXL3_TORCH` on ROCm even though `validate_device` already accepted
gfx120x. The native HIP GEMV and WMMA GEMM were reachable only by passing
`BACKEND.EXL3_EXLLAMA_V3` explicitly.

```mermaid
flowchart TD
    subgraph before["Before #5836"]
        B1["backend=AUTO<br/>_resolve_exl3_auto_backend"] --> B2{"family in<br/>ExllamaV3Linear.SUPPORTS_DEVICES?"}
        B2 -->|"CUDA"| B3["EXL3_EXLLAMA_V3"]
        B2 -->|"ROCm: list is [DEVICE.CUDA]"| B4["EXL3_TORCH<br/>packed HIP GEMV unreachable"]
    end
    subgraph after["After #5836"]
        A1["backend=AUTO<br/>_resolve_exl3_auto_backend"] --> A2{"cuda_available?"}
        A2 -->|no| A6["EXL3_TORCH"]
        A2 -->|yes| A3{"selector_device_family"}
        A3 -->|CUDA| A5["EXL3_EXLLAMA_V3"]
        A3 -->|ROCm| A4{"all selected ordinals<br/>supports_device()?"}
        A4 -->|"gfx120x and extension usable"| A5
        A4 -->|"CDNA, other gfx, no HIP ops"| A6
    end
```

`ExllamaV3Linear.SUPPORTS_DEVICES` is now the static family list
`[DEVICE.CUDA, DEVICE.ROCM]` (`gptqmodel/nn_modules/exllamav3.py:71`).
`supports_device` (`exllamav3.py:82`) keeps the per-architecture gate: NVIDIA
CUDA is always eligible, and ROCm is eligible only when
`exl3_native_hip_gemm_usable` (`gptqmodel/exllamav3/ext.py:287`) returns true.
That probe extends the `gfx120*` metadata check with loadability: already
registered ops, a cached or configured prebuilt library, or a HIP JIT
toolchain (`ninja` plus `hipcc` through `ROCM_HOME` or `PATH`). A ROCm runtime
wheel without the development toolkit therefore stays on `EXL3_TORCH` instead
of selecting a kernel that cannot load; the probe never compiles.

The explicit path now calls `ExllamaV3Linear.validate_device(device)`
(`loader.py:3079`) with the real selector, so every selected ordinal is probed
instead of the abstract `DEVICE.ROCM` family (which converted to `cuda:0` and
validated GPU 0 only). `validate_device` keeps the cheap architecture metadata
probe because the load itself will build the extension.

`tests/test_rocm_exl3_auto.py` covers the AUTO matrix (CUDA native yes/no,
ROCm native yes/no, no CUDA, CPU) and the all-ordinals requirement.
`tests/kernels/test_exl3_public_load_hip_5836.py` is the gfx1201 end-to-end
regression: it quantizes a 256->512 EXL3 3.0 bpw `mul1` projection, reloads it
through `GPTQModel.load` with AUTO, asserts `ExllamaV3Linear`, spies on
`bc_linear_exl3_run_hip_gemv` for M=1/8 (and its absence at M=9), and matches
an independent FP64 decode at `rtol=atol=2e-3`.

## A100 validation

The W3 `mcg` Llama 3.2 1B checkpoint produced 20 bit-identical logit
tensors on the integer default path, against GPT-QModel `a54f05d4` with
the same Transformers 5.23.0 and PyTorch 2.15.0 runtime. The opt-in FP16
path had a maximum absolute final-logit difference of 0.07421875 in that
probe. It selected a different next token at one exact baseline top-logit
tie and changed two interior prefill argmax positions at margins of 0.015625.
Those observations motivated retaining the original integer default.

The projection benchmark used identical packed payloads, scales and FP16
inputs on one A100, and rotated more than three times the 48 MiB L2 over
41 warmed CUDA graph replays. Times are medians in microseconds per W3
`mul1` projection. The INT8 columns include call-private workspace setup.

| Projection | Rows | Original | Opt-in FP16 | Residual INT8 | Plain INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate + up (5120 → 4608) | 1 | 73.67 | 31.35 | 28.50 | 26.40 |
| Gate + up (5120 → 4608) | 2 | 73.96 | 31.52 | 31.80 | 34.70 |
| Gate + up (5120 → 4608) | 6 | 64.51 | 32.14 | 32.37 | 32.37 |
| Gate + up (5120 → 4608) | 24 | 123.96 | 80.84 | 80.95 | 80.84 |
| Gate (5120 → 2304) | 1 | 44.06 | 23.44 | 25.20 | 22.39 |
| Down (2304 → 5120) | 1 | 40.73 | 18.20 | 21.33 | 19.20 |

These are projection timings on synthetic packed inputs. On the W2.5/W3/W4
shape grid, versus upstream FP16 on the same fixtures, the worst relative
RMS was 0.0612% for FP16, 0.0650% for residual INT8 and 0.7537% for plain
INT8. The corresponding maximum absolute projection differences were
0.0078125, 0.0078125 and 0.046875. INT8 dispatched only on eligible
row counts, so some columns include the FP16 path.

The packaged extension also passed 148 focused A100 tests, including changed
inputs on graph replay, live graphs on independent streams, round trips for
half-bit tiles, and grouped decode. ROCm, Turing, and newer NVIDIA
architectures remain unmeasured for this port.

## Llama 3.2 1B Instruct at 3.5 bits

The full-model test quantizes all 112 attention and MLP projections with
`mul1` using 256 C4 training windows of 512 tokens. It saves the model as
safetensors, verifies the 56-word trellises, and reloads the native CUDA
backend. It compares both models on all WikiText-2 test tokens and the same
256 seeded GSM8K questions. A second reload with the Torch backend checks
prefill loss and cached decode logits using the same packed weights.

| Metric | BF16 | EXL3 W3.5 |
| --- | ---: | ---: |
| WikiText-2 test perplexity | 11.6343 | 12.2846 |
| GSM8K correct | 125/256 | 116/256 |
| Safetensors payload | 2.472 GB | 0.953 GB |

The A100 results and sampling details are in the
[PR discussion](https://github.com/ModelCloud/GPTQModel/pull/5619#issuecomment-7029571986).

Run the test with a local Llama 3.2 1B Instruct BF16 checkpoint, an Ampere
or newer NVIDIA GPU, and a directory containing these evaluation assets.

- `c4-train-*.arrow` from Hugging Face Datasets
  `load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train", revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2")`.
  This shard contains 356,318 documents.
- `wikitext-test.arrow` from `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision="b08601e04326c79dfdd32d625aee71d232d685c3")`.
- `gsm8k-test.jsonl` from the [official GSM8K test split](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl).

Use the Arrow files listed in each dataset's `cache_files` in their shard order.

```bash
GPTQMODEL_EXL3_E2E_MODEL_DIR=/path/to/Llama-3.2-1B-Instruct \
GPTQMODEL_EXL3_E2E_DATA_DIR=/path/to/eval-data \
python -m pytest -m exl3_cuda_e2e tests/models/test_llama3_2_exl3_cuda.py
```

The test permits up to a 10% perplexity increase and a ten percentage-point
GSM8K drop relative to the BF16 baseline measured in the same run. The regular
CUDA suite covers W3.5 save and reload on a tiny Llama without external assets.
