<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Computed Permutation Trellis Coding

CPTC, introduced by Muhammed Fatih Balın, quantizes blocks of weights
jointly. A two-pass Viterbi search picks a tail-biting bitstream for each
block. During inference, the decoder multiplies each overlapping state by
two fixed coefficients and adds the top bits of their 16-bit products to
index a small E2M5 codebook. [CPTC paper](https://arxiv.org/abs/2611.22019)

The encoder uses the tail-biting bitshift trellis construction from QTIP by
Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa. [QTIP paper](https://arxiv.org/abs/2406.11235)

## Quantize and load

```python
from gptqmodel import CPTCConfig, GPTQModel

qcfg = CPTCConfig(bits=4, trellis_depth=16, block_size=128, device="cuda:0")
model = GPTQModel.load("path/to/dense-model", quantize_config=qcfg)
model.quantize()
model.save("path/to/cptc-checkpoint")

model = GPTQModel.load("path/to/cptc-checkpoint", device="cuda:0")
```

`bits` accepts 1–6. `trellis_depth` ranges from `bits` to 16, and `table_q`
ranges from `bits` to 6. The block size is 128 or 256 values. Dynamic rules
can change the bit rate or exclude layers. For the installed PyTorch runtime,
`cuda:0` is the execution device on both NVIDIA and AMD. Use `device="cpu"`
for CPU quantization. A shallower trellis reduces the encoder's state count.

Load CPTC safetensors checkpoints with `GPTQModel.load`.

## YAQA calibration

The calibrated path uses Sketch B and two-sided adaptive rounding from Yet
Another Quantization Algorithm (YAQA), introduced by Albert Tseng, Zhaofeng
Sun, and Christopher De Sa. [YAQA paper](https://arxiv.org/abs/2505.22988) · [YAQA reference implementation](https://github.com/Cornell-RelaxML/yaqa-quantization)

```python
from gptqmodel import CPTCConfig, CalibrationConfig, GPTQModel

qcfg = CPTCConfig(bits=3, block_size=256, yaqa=True, yaqa_seed=2718, device="cuda:0")
model = GPTQModel.load("path/to/dense-model", quantize_config=qcfg)
model.quantize(CalibrationConfig(dataset=calibration_samples, batch_size=8))
model.save("path/to/calibrated-cptc")
```

`yaqa=True` gathers input and output partial-trace Fisher sketches from the
dense model, using next-token samples drawn from its own distribution.
It rotates the weights and sketches with signed Hadamard transforms, damps
each sketch by 0.01 of its mean diagonal, and uses two-sided block LDL error
feedback to form the tile targets for Viterbi. The tiles are 8 × 16 with
128-weight blocks and 16 × 16 with 256-weight blocks. The Gaussian
codebook uses 0.9 loading and the packer applies the E2M5 range bound per tile.
Calibration holds the complete teacher and its FP32 sketch pairs on the
quantization device. Both projection dimensions need a supported Hadamard
width and must be divisible by their tile dimensions.

The CPTC paper's LLM experiment uses these transforms and two-sided rounding
with a larger Sketch B corpus. GPT-QModel uses its E2M5 tables and traverses each
tile in row-major order. The CUDA and RDNA4 kernels consume both checkpoint
layouts. [CPTC paper](https://arxiv.org/abs/2611.22019)

```mermaid
flowchart TD
    A["GPTQModel.quantize"] --> B{"CPTCConfig.yaqa"}
    B -->|False| C["Encode source rows"]
    B -->|True| D["collect_hessian_sketches"]
    D --> E["Rotate weights and both sketches"]
    E --> F["Two-sided LDL feedback and tile encoding"]
    C --> G["Save packed CPTC checkpoint"]
    F --> G
```

## Quantization and checkpoint flow

```mermaid
flowchart TD
    A["CPTCConfig validates bits, depth, taps and block size"] -->|Valid| B["WeightOnlyProcessor.submodule_finalize"]
    A -->|Invalid| E["Raise ValueError"]
    B --> C["CPTCLinear.pack_original on assigned quantization device"]
    C --> D["quantize_weight normalizes blocks and calls viterbi_encode"]
    D -->|Finite weights| P["pack_states and save_quantized write the CPTC payload"]
    D -->|Nonfinite weights| E
    P --> L["GPTQModel.load creates CPTCLinear"]
    L -->|Complete payload| F["CPTCLinear.post_init checks layout"]
    L -->|Missing packed fields| E
    F -->|Malformed layout| E
    F -->|Valid layout| R["CPTCLinear.forward"]
    classDef error fill:#ffdddd,color:#411,stroke:#a22
    class E error
```

| Stage | Base `46e5539e5` | CPTC |
| --- | --- | --- |
| Format resolution | `gptqmodel/quantization/config.py:10013` resolves supported checkpoint methods. | `METHOD.CPTC` resolves to `CPTCConfig` and `FORMAT.CPTC`. |
| Weight-only packing | `gptqmodel/looper/weight_only_processor.py:110` selects direct pack for GGUF, BitsAndBytes, and NF12. | CPTC joins direct pack and encodes on the selected worker device. |
| Checkpoint load | The native loader verifies the buffers declared by each quantized kernel. | CPTC declares `.cptc_packed`, `.cptc_table`, `.cptc_coeffs`, and `.cptc_scale_bits` as required. |

For each output row, the input axis is divided into contiguous blocks and
the last block is padded with zeros. Each block stores `block_size * bits / 8`
bytes, little-endian, with its final windows wrapping to its own first bytes.
The two Viterbi passes search in FP32 and pin the shared boundary bits in
the second pass.
The default depth and tap width use fitted Gaussian E2M5 tables from
`fit_gaussian_table`. Other geometries initialize their tables from Gaussian
conditional means. Direct weight encoding starts from a per-block RMS
scale. Before Viterbi, it raises the scale as needed to bring the block's
positive and negative extremes into the reachable codebook range.

| Tensor | Checkpoint representation |
| --- | --- |
| `cptc_packed` | `uint8`, `[out_features, blocks, block_bytes]` |
| `cptc_table` | `uint8`, E2M5 codes, `2**table_q` entries. The upper indices use a mirrored lookup and sign flip. |
| `cptc_coeffs` | `int32`, two 16-bit multiplication constants |
| `cptc_scale_bits` | `int32`, IEEE FP32 scale bits, `[out_features, blocks]` |
| `bias` | Optional stored bias in its original dtype |

With YAQA calibration, `cptc_packed` and `cptc_scale_bits` are indexed by
tile row and tile column. The checkpoint also carries two `int8` Hadamard
sign vectors, `cptc_input_signs` and `cptc_output_signs`. Inference applies
the input transform before the packed multiplication and inverts the
output transform before bias and final rounding.

The scale bit pattern survives model dtype conversion. FP32 scales add 0.25
bits per weight with 128-value blocks and 0.125 with 256-value blocks, before
padding and per-layer metadata. See the [forward dtype contract](../contracts/kernel/forward-dtype.md)
for the module's activation dtype policy.

## Storage device contract

`quantize_weight` separates the device that runs the trellis search from the
device that owns the returned checkpoint payload. `device` selects the
encoder; `storage_device` selects the returned packed blocks, table,
coefficients and scales. The default is `weight.device`, which keeps the model
path's CPU checkpoint storage while the encoder runs on an accelerator.
An explicit `meta` storage target raises `ValueError` before encoding.
When a caller supplies a codebook, the payload keeps its own copy so the
caller can refit the table without changing previously encoded weights.

Before this contract, `quantize_weight` allocated `coeffs`, `packed` and
`scales` on `weight.device` and returned `table` there, even when `device`
was another accelerator. Each row batch was copied to `device` for encoding
and then copied back, so a cross-accelerator call silently paid a
device-to-device copy per batch and returned the payload on the weight's
device.

```mermaid
flowchart TD
    A["quantize_weight(weight, spec, device)"] --> B["Allocate coeffs, packed, scales on weight.device"]
    B --> C["Copy each row batch to device"]
    C --> D["Encode and pack on device"]
    D --> E["Copy packed and scales back to weight.device"]
    E --> F["Return table on weight.device"]
    G["weight.device != device"] -.->|"implicit per-batch copy"| C
    classDef removed fill:#ffdddd,color:#411,stroke:#a22
    class B,E,G removed
```

The encoder resolves the storage device before allocation. It rejects a
`meta` target or an implicit cross-accelerator placement.

```mermaid
flowchart TD
    A["quantize_weight(weight, spec, device, storage_device)"] --> B["as_torch_device normalizes device"]
    B --> C{"storage_device given?"}
    C -->|Yes| D{"storage_device is meta?"}
    D -->|Yes| F["Raise ValueError"]
    D -->|No| M["Resolve storage_device"]
    C -->|No| E{"weight.device and device are distinct accelerators?"}
    E -->|Yes| F
    E -->|No| G["storage_device = weight.device"]
    M --> H["Allocate coeffs, packed, scales on storage_device"]
    G --> H
    H --> I["Copy each row batch to device"]
    I --> J["Encode and pack on device"]
    J --> K["Copy packed and scales to storage_device"]
    K --> L["Return packed blocks, table, coefficients and scales on storage_device"]
    classDef error fill:#ffdddd,color:#411,stroke:#a22
    class F error
```

| Caller | Before | After |
| --- | --- | --- |
| CPU weight, accelerator encoder | Payload stayed on CPU | Payload stays on CPU by default; `storage_device=device` can place it on the encoder |
| Accelerator weight, same encoder device | Payload stayed on that accelerator | Unchanged |
| Accelerator weight, different encoder device | Payload returned on the weight's accelerator after implicit copies | `ValueError` unless `storage_device` is explicit |
| `CPTCLinear.pack_original` (direct) | Payload allocated on `weight.device`, then moved to the layer target | Allocates regular tensors on the resolved layer target for direct registration, including when called inside inference mode |

## Inference dispatch

```mermaid
flowchart TD
    A["CPTCLinear.forward"] --> B{"native_ops supports this device ordinal and JIT loads?"}
    B -->|Yes| C{"Resident packed weights and 1 to 16 FP16 or BF16 inference rows?"}
    C -->|Yes| D["cptc_mma decodes into CUDA MMA or RDNA4 WMMA fragments"]
    C -->|No| E["cptc_decode builds one FP32 weight tile at a time"]
    B -->|CPU or unsupported GPU| F["decode_weight builds a Torch FP32 weight tile"]
    B -->|JIT build fails| W["Log the extension error"] --> F
    E --> G["linear_wide_bias uses a wide reduction and bias"]
    F --> G
    D --> H["FP32 split reduction and bias"]
    H --> R["Round once to the input dtype"]
    G --> R
    classDef fallback fill:#fff1d6,color:#431,stroke:#b70
    class W fallback
```

| Path | Device contract | Runtime evidence |
| --- | --- | --- |
| Fused FP16 and BF16 | NVIDIA SM80+ | RTX 4090 (`sm_89`) and RTX 5090 (`sm_120`) |
| Fused FP16 and BF16 | ROCm RDNA4 wave32, `gfx1200` and `gfx1201` | Built both targets, ran on Radeon AI PRO R9700 (`gfx1201`) |
| Tiled native decode with FP32 matmul | Same GPU families, prefill or FP32 input | RTX 4090, RTX 5090, R9700 |
| Tiled Torch decode with FP32 matmul | CPU and fallback on other CUDA or ROCm architectures | CPU validated |

The fused path multiplies the exact E2M5 values on the matrix units and
applies the per-block scales in FP32. Bias joins the complete FP32 result.
Prefill holds a decoded tile only for its matmul. Every path retains the
four packed checkpoint tensors and optional bias between forwards. The
projection-fusion inventory classifies CPTC as a separate format.

## Measurements

The fitted 4-bit, depth-16, 256-value codebook reached 3.934 effective bits
on 524,288 held-out standard-normal values. The decoder matched the
independent bitstream oracle exactly for all six rates. The two native
Viterbi schedules produced the same packed codes as the FP32 Torch reference
in the tested blocks. Native FP16 and BF16 outputs met `rtol=atol=2e-3`
against the independently decoded FP64 result rounded to the output dtype.
A small Llama checkpoint produced identical BF16 logits through the native
and Transformers loaders.

The 4-bit, 4096 × 4096 fixture with FP32 bias retains 8,929,352 bytes on
each device. The table shows FP16 warm-input medians from seven CUDA event
timing batches of 40 calls at 1 and 16 rows and 10 calls at 128 rows.
The allocation column measures peak live bytes above the pre-call baseline,
including temporary outputs and BLAS workspace.

| Device | Rows | Latency | Peak extra allocation |
| --- | ---: | ---: | ---: |
| RTX 4090 | 1 | 50 µs | 0.27 MiB |
| RTX 4090 | 16 | 104 µs | 4.25 MiB |
| RTX 4090 | 128 | 260 µs | 27.38 MiB |
| Radeon AI PRO R9700 | 1 | 177 µs | 0.27 MiB |
| Radeon AI PRO R9700 | 16 | 201 µs | 4.25 MiB |
| Radeon AI PRO R9700 | 128 | 1,158 µs | 99.25 MiB |

For 512 resident blocks of 128 values at 4 bits and depth 16, Viterbi reached
18.65 million values/s on the RTX 4090 and 1.53 million values/s on the R9700
with the register-cached E2M5 codebook. The generic FP32-codebook search
reached 0.31 and 0.13 million values/s on the same devices. The timings cover
the two passes and the backtrace. Test runtimes were PyTorch `2.15.0+cu132`
and `2.16.0.dev20261025+rocm10.1` respectively.

Run `python -m pytest tests/test_cptc.py` on each GPU runtime and on CPU.
GPU cases skip on other architectures and fail when a supported device
cannot build or run the native extension.

[Implementation PR #5959](https://github.com/ModelCloud/GPTQModel/pull/5959) · [Tracking issue #5954](https://github.com/ModelCloud/GPTQModel/issues/5954)
