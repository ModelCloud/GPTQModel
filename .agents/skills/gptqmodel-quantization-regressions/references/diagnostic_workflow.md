# Severe quantization regression workflow

## Reproduction contract

Capture these fields before comparing artifacts:

| Area | Required metadata |
|---|---|
| Model | source path/revision, architecture, parameter count, tied-weight setting |
| Quantization | method, bits, group size, symmetry, activation ordering, scale search, static groups, fallback |
| Calibration | dataset/revision, row selection and order, token count, text or token hash, batch/concat settings |
| Runtime | GPT-QModel, PyTorch, CUDA, Transformers, Safetensors, and evaluation versions |
| Hardware | GPU name, UUID, PCI bus, compute capability, SM count, memory |
| Evaluation | exact task/version, prompt rendering, tokenizer path, backend, seed, batch size, row count |

Establish a dense reference and an adapter-free quantized reference before testing EoRA/LoRA. A bad base invalidates an
adapter rank or algorithm sweep. When an adapter is generated during quantization, also test the joint checkpoint
with the exact adapter from that run. Report base-only and base-plus-matching-adapter results separately.

## Boundary matrix

| Boundary | Comparison | What a mismatch means |
|---|---|---|
| Pre-pack GPTQ | logged loss; `W` vs reconstructed `Wq` | quantization math, Hessian/error feedback, grouping, or scale search |
| Coupled processor | GPTQ `Wq` before adapter math vs the weight restored for packing | replay, shared state, or finalization lifecycle |
| Quant metadata | scale/zero/group index/code range and finiteness | quantizer output or serialization |
| Packed checkpoint | eager dequant vs independent manual unpack | packing, unpacking, signed shifts, masking, or zero convention |
| Direct reconstruction | saved scales + independently rounded codes vs packed dequant | pack/code disagreement or dtype-only rounding |
| Eager linear | dense matmul vs eager quantized matmul | quantized-linear reconstruction or application |
| Optimized backend | same checkpoint/input vs eager output and logits | backend selection, preprocessing, or kernel |
| End to end | exact token IDs, logits, generation, evaluation | tokenizer/prompt/evaluator or accumulated numerical error |

The first failing row owns the investigation. Later failures may be consequences.

## Layer and module comparisons

Parse matched `(layer, module role)` rows from each `quant_log.csv`.

```text
candidate/reference loss ratio = candidate module loss / reference module loss

normalized degradation =
    candidate/reference loss ratio
    / median(candidate/reference ratio over all matched modules)
```

The normalization separates expected global degradation at fewer bits from one module worsening abnormally. Also
calculate each module's fraction of total loss. A single module owning a large fraction of total loss is more
actionable than a high ratio produced by two tiny values.

Compare like module roles across depth (`mlp.down_proj` against other `mlp.down_proj` layers). Architectural depth
effects can naturally make late layers 100x larger than early layers, so a same-role ratio within only one snapshot
is a locator, not proof.

## Output-channel analysis

Quantizer-returned pre-pack scale tensors are shaped `[output_channels, groups]`; saved QuantLinear `.scales`
tensors are transposed to `[groups, output_channels]`. Reduce the group axis appropriate to the boundary being
inspected:

```text
channel_max[j] = max_g abs(scale[g, j])
```

Report the maximum channel, median and p99 of `channel_max`, non-finite counts, absolute threshold counts, and matched
candidate/reference ratios for every output channel. Then load only the suspicious dense and quantized module and
calculate:

```text
RMSE = sqrt(mean((Wq - W)^2))
relative RMSE = ||Wq - W||_2 / ||W||_2
cosine = dot(Wq, W) / (||Wq||_2 * ||W||_2)
```

Calculate whole-tensor and per-output-channel values. Whole-tensor cosine can remain apparently healthy when one
high-leverage output channel is destroyed.

### Example signature

In a Qwen3-8B investigation, layer 6 `mlp.down_proj` output channel 2276 showed:

| Width | Maximum saved scale | Relative weight RMSE |
|---:|---:|---:|
| 4-bit | 0.350 | 0.313 |
| 3-bit | 2.031 | 0.565 |
| 2-bit | 35.0 | 0.874 |

The 2-bit channel's GPTQ cosine versus BF16 was 0.4880, while independent symmetric RTN reached 0.8734. All layer-6
scales above 10 belonged to channel 2276. Because the layer's logged loss was already anomalous before packing and
manual unpack matched eager dequantization, the evidence pointed to sequential GPTQ error feedback amplifying an
outlier channel—not to the packer or Triton kernel.

## Independent controls

Use controls in this order:

1. Higher-bit snapshot with the identical calibration contract.
2. Native quantization without EoRA/LoRA, GAR, smoothing, or unrelated preprocessors.
3. For a coupled quantization-plus-adapter run, the saved base with its exact matching adapter.
4. CPU and GPU packers from the same reconstructed tensors; assert byte equality where supported.
5. Independent manual unpack, including masks, shifts, signed storage treatment, and logical zero convention.
6. Symmetric RTN on the suspicious module/channel with identical bits and groups.
7. `static_groups=True`, then a selective RTN fallback for only the suspicious module/channel.
8. Eager versus optimized backend on identical token IDs and inputs.

Never silently clamp or repair data only in a diagnostic. Assert legal code bounds at the quantization-to-packing
boundary, and compare the pre-clamp values when investigating saturation.

If two checkpoints have byte-identical scales, zero-points, and group indices but different `qweight`, do not assume
the packer changed identical inputs. First compare deterministic logical-code samples at three points: immediately
after quantization, immediately before packing after all coupled processors, and after unpacking the saved
checkpoint. Repeat one configuration unchanged to establish run-to-run variance. This separates quantizer
nondeterminism from adapter-state restoration and the packer itself.

The bundled analyzer's gated `--scan-codes` mode performs the saved-checkpoint side of this comparison. It processes
one packed module at a time, reports logical-code mismatch rate, mean and maximum absolute code delta, packed-word
match rate, and exact tensor counts for scales, zero-points, and group indices. Use it only for same-bit snapshots;
cross-bit integer codes have different ranges and are not a meaningful parity comparison.

During quantization, `quantization_diagnostics="channel"` performs the in-process side with a bounded sample. It
records up to 64 evenly spaced input positions by 64 output positions per module after GPTQ, recomputes the same
positions from the restored pre-pack weight, then extracts them from packed qweight. A post-GPTQ/pre-pack mismatch
localizes the fault to coupled processor state or finalization; pre-pack parity followed by a packed mismatch
localizes it to packing.

### Group-permutation invariants

Before interpreting a `static_groups`, activation-order, or GAR control, verify that its parameters still describe
the restored weight layout. Precomputed static quantizers normally start in original group order, while the
quantization loop may process groups in permutation order. The working quantizer sequence may be reordered for that
loop, but saved scales, zero-points, and `g_idx` must map back to the original columns together with `Q`.

Use a small deterministic test that forces a non-identity group permutation. Independently precompute each original
group's scale and zero-point, run quantization, and assert the returned parameters and `g_idx` match original group
order. A score from a control that accidentally pairs a group with another group's parameters does not measure the
intended quantization algorithm.

## Reporting template

Report a compact table with at least:

| Snapshot | Mean pre-pack loss | Worst layer/module | Worst loss | Max scale/channel | Weight cosine | Relative RMSE | Manual unpack parity | Backend parity | Eval score |
|---|---:|---|---:|---|---:|---:|---|---|---:|

State whether each check ran, skipped, or was blocked. Preserve raw JSON and exact commands. A controlled experiment
should change one factor and include its dense or healthy quantized reference.
