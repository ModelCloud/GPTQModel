<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Typed quantization configuration and representation contracts

**Status:** revision 0.14 design proposal, 2026-11-13. No new API below is implemented by this PR.

Recommendation: use a small, abstract `BaseQuantizeConfig`, typed weight specifications, shallow method-specific inheritance, and composed solver options. Separate the recipe used to produce weights from the specification required to load them and from the machine-specific execution settings.

This is a design proposal, not an implementation. The audit covers public **ModelCloud/GPTQModel**, at commit [`35987dfb7433d9b539e5ca9656621473fcd1fbe8`](https://github.com/ModelCloud/GPTQModel/tree/35987dfb7433d9b539e5ca9656621473fcd1fbe8), as a pinned source baseline. No runtime implementation or numerical tests accompany this proposal. This newer configuration inventory supplements the older lifecycle audit; it does not assert those additional methods exist at that audit's earlier revision.

Review verdict: retain the small config base and shared `ScaledWeight.scale_solver`, but enforce method-specific strategies and distinguish recipe, resolved plan, quantization result and saved representation. The correctness review in section 7 lists the holes addressed in this revision and the cases still requiring implementation evidence.


This is the recipe/representation layer of the [component quantization design](component-quantization-design.md). The [composer](quantization-composer-api.md) remains the preferred authoring API: `Language(GPTQConfig(...)).Use(data)` supplies one typed method configuration to the same method-owned compiler. Direct `model.quantize(recipe=..., data=..., devices=..., export=...)` calls lower to that compiler too.

| Concern | Single owner in the combined design |
|---|---|
| Weight format, partition and scale-fitting policy | `BaseQuantizeConfig` / concrete method configs with typed `weights` |
| Method procedure | Concrete `solver` config, such as `GPTQSolverConfig` |
| Preparation, recovery and staged refinement | Registered `Recipe.use(...)` / component `Use(...)` extensions; native required optimization stays in the method program |
| Target selection and module overrides | Component `target`, `Target.AUTO`, `DynamicModuleTargeting` and scoped `Rules` |
| Calibration and other data roles | Existing `QDataSet`, `CalibrationDataSet` and `DataBindings` |
| Placement and parallel execution | Existing action-local `Local`, `Default`, `MoE` and typed execution profiles |
| Output layout and container | Action-level `ExportConfig`, checked at compilation and finalization |
| Saved representation | `QuantizedModelSpec` inside the composition manifest; `ModelArtifact` pins the committed result |

No second calibration dataset hierarchy, device-selection API, or orchestration loop is introduced. Python method configs replace the earlier draft's short `GPTQ(bits=...)` / `AWQ(bits=...)` factories; no compatibility aliases are retained. Numeric descriptors keep `IntFormat` / `FloatFormat`, nested discriminators use `type`, and serialized method selection uses `method`.

## 1. Current methods and representations

There are 11 `METHOD` values and 12 concrete method configs when RTN is included. TT block-float is a loadable representation, but is explicitly excluded from the normal quantization API. A configured width or format does not imply that every backend can produce or execute it.

| Current config | Weight representation and segmentation | Distinct controls and support limits |
|---|---|---|
| `GPTQConfig` | Scalar integers, 2–8 bits; input-channel groups, default 128, `-1` means full input row | Hessian accumulation, damping, static groups, activation ordering, GPTAQ, FOEM, GSQ and staged GSQ. GPTQ/GPTQ-v2/planar layouts; 5/6/7-bit requests resolve to planar. Marlin is declared as a layout but rejected for new quantization by the public API. |
| `RTNConfig` | Scalar rounding into GPTQ-family or AWQ-family packing; same grouped-integer concepts | No Hessian calibration for the primary rounding pass; optional smoothing and GSQ. Internally reports `METHOD.GPTQ`; weight-only metadata and format distinguish it. Constructor compatibility can instead dispatch GGUF/FP8/BNB requests to their concrete configs. |
| `AWQConfig` | Scalar integer groups; configured widths 2/3/4/8 | Activation-based scale/clip search, chunked activation handling, GSQ/staged GSQ. Native GEMM/GEMV paths are 4-bit; other widths require compatible portable support. GEMM/GEMV/GEMV_FAST/LLM-AWQ plus declared Marlin/BitBLAS layouts. |
| `QQQConfig` | Fixed W4 weights with W4A8 execution; grouped scales | Inherits GPTQ options, but rejects activation reordering; QQQ layout. GSQ support is restricted and staged GSQ is rejected. |
| `ParoConfig` | Fixed 4-bit transformed-domain quantization with grouped weights | `krot`, rotation and fine-tuning stages, optimizer, samples, precision, implementation and scale-clamp settings; optional GSQ. ParoQuant layout. |
| `EXL3Config` | Trellis encoding; one integer depth 1–8 per module; no GPTQ group size | `head_bits`, codebook, output scales, tensor-storage/calibration metadata, module allowlist. Fields are typed as floats, but fractional per-module depths are rejected. Mixed integer-depth modules can produce a fractional average. |
| `VPTQConfig` | Vector codebooks, optional residual/outlier codebooks; `bits` is a derived rounded label | Vector lengths, centroid counts, outlier fraction, codebook partition by size/count, k-means, normalization/permutation, Hessian error propagation. `block_size` is a solver chunk, not a storage block. |
| `FP8Config` | E4M3/E5M2 variants; per-tensor, per-row or 2D block scales; inverse-scale convention | `weight_scale_method`, `weight_block_size`, `weight_scale_semantics`, GSQ and optional calibrated GSQ. `format` means a float dtype. E8M0 is accepted as a dtype where Torch provides it, but dense-to-E8M0 weight quantization is explicitly rejected. |
| `BitsAndBytesConfig` | Blockwise FP4/NF4 at 4 bits; direct vectorwise INT8 at 8 bits | `block_size`, `compress_statistics` apply to the four-bit representation; format and width are coupled. Four-bit block-size validation accepts 32/64/128/256/512/1024/2048/4096; the current INT8 producer uses vectorwise quantization with outlier routing disabled. Actual backend/operation support remains separate. |
| `GGUFConfig` | Named GGML block encodings; no user-controlled GPTQ group | Q/IQ/TQ, MXFP4 and NVFP4 families. `bits` accepts a string/object and is normalized into numeric width plus subtype `format`, with a third `_gguf_bits` runtime representation. Block geometry is fixed by the selected tensor encoding. |
| `NF12ADConfig` | BF16 encoded with a nominal 12-bit code plus escape data; no scale groups | Optional exponent base. Encoding preserves BF16 weights; a preceding cast to BF16 can be lossy. Escapes mean actual storage cost need not be exactly 12 bits per weight. |
| `TTBlockfloatConfig` | `bf4_b`/`bf8_b`, fixed shared exponent per 16 values | Both `block=16` and `group_size=16`; separate `embed_bits`. Checkpoint loading/inference only through this API; its producer is an external script. |

Sources: [enums and widths](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L114-L196), [concrete configs](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L4974-L6513), [public quantization preflight](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/models/base.py#L1650-L1760), [FP8 weight quantizer](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/fp8.py#L116-L160).

The GGUF config accepts these alias families: `q1_0`, `q1_0_g128`, `q2_0`, `q4_0/1`, `q5_0/1`, `q8_0/1`; `q2_k` and its s/m variants; `q3_k` and s/m/l variants; `q4_k` and s/m variants; `q5_k` and s/m variants; `q6_k`, `q8_k`; `iq1_s/m`, `iq2_xxs/xs/s`, `iq3_xxs/s`, `iq4_nl/xs`; `tq1_0`, `tq2_0`; `mxfp4`, `nvfp4`. Optional dependencies and backend capabilities further constrain operations.

Two additional surfaces matter:

- GPTQ W4A activation policies support FP8 and NVFP4 operands, attention overrides, selected MLP-layer promotion and calibrated global scales. These retain GPTQ INT4 checkpoint weights; an NVFP4 activation does not make the weights NVFP4. Current admission requires W4G128 symmetric INT32 packing, no descending activation order and no quantized lm-head.
- Native source-checkpoint detection recognizes FP8/NVFP4/MXFP4/MXFP8/NVFP8 and mixed source formats from metadata including ModelOpt and Quark. Reading or converting those inputs is a separate capability from exposing them as general quantization recipes. Protocol-to-config compilation currently covers GPTQ/AWQ/GGUF only; its input/output/KV fields are not proof of executable support for those targets.

Sources: [activation policy](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L1727-L2005), [activation admission](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L4368-L4382), [native source detection](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/models/loader.py#L202-L277), [protocol compiler](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/protocol.py#L339-L429).

## 2. Problems the redesign should solve

| Existing concern | Why it is misleading | Proposed treatment |
|---|---|---|
| Universal `bits` | Integer code width, GGUF encoding identity, trellis depth and rounded VPTQ cost are different quantities | Typed weight specifications; width exists only where meaningful |
| Universal `group_size` | GPTQ scale groups, BNB blocks, FP8 rectangles and VPTQ codebook partitions differ | Method-native segmentation objects; no universal size scalar |
| `format` | Means a checkpoint layout, float dtype, BNB codebook type or GGUF subtype | Separate encoding identity, numeric dtype, checkpoint layout and file container |
| `sym` and `desc_act` in the base | Many subclasses force them to constants, hide them or reject them | Define integer symmetry and solver ordering only on appropriate types |
| Mutable `method` plus subtype factory | Class, method, format, bits and weight-only metadata all help guess identity | Concrete Python classes and an explicit discriminator in serialized recipes |
| `QQQConfig`/`VPTQConfig` inheriting GPTQ | Reuses machinery but also exposes unsupported or duplicate options | Share Hessian/solver option components without inheriting all GPTQ behavior |
| Runtime resources in config | Construction can create a temporary directory; machine settings enter saved metadata | Pure config construction; execution context owns resources |
| Several normalization paths | Base config, dynamic clones, module constructors, protocol and readers can disagree | One resolved module specification passed to all consumers |
| Hand-maintained BPW estimates | Nominal width is confused with stored cost | Encoding descriptors plus shape-aware storage accounting |

For example, the GGUF tensor metadata gives Q4_K 144 bytes per 256 weights (4.5 bpw), Q6_K 210 bytes per 256 weights (6.5625 bpw), and MXFP4 17 bytes per 32 weights (4.25 bpw). The config's approximate Q6_K table currently reports 6.0. The two inventories can disagree because storage metadata is duplicated.

Also, current `q4_k`, `q4_k_s` and `q4_k_m` aliases map to the same `Q4_K` tensor type. They must not be presented as evidence that the implementation selects different per-tensor mixed-precision recipes. In the new API, a tensor encoding and a model-level quality preset should have separate identities. Implement a preset's selection rules explicitly before advertising that preset.

Sources: [current complex bits object](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L324-L681), [GGUF tensor metadata and aliases](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/gguf.py#L51-L147), [factory filtering and metaclass](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L3926-L3984), [base initialization and resource creation](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L4282-L4435).

## 3. Recommended public model

The reusable configuration is independent of component scope and machine resources:

| Object | Owns | Examples |
|---|---|---|
| `BaseQuantizeConfig` and concrete subclasses | Method recipe and semantic choices | Typed weights, activation policy, solver and native optimization settings |
| `Recipe` and its extensions | Ordered typed additions to a method's lifecycle | Preparation, EoRA, GSQ refinement; compatibility and required data roles |
| `Local` / `ExecutionLayout` | How each action uses the machine | Devices, P2P, offload, workspace budgets, telemetry and pack implementation |
| `ExportConfig` | Requested checkpoint layout and container | GPTQ-v2 versus planar, packing word dtype and sharding; safetensors by default |
| `QuantizedModelSpec` | Actual representation required for loading/inference | Per-owner formats, shapes, scales, transforms, layout versions and auxiliary payloads |

Inference backend, compilation and inference placement belong in `InferenceConfig`, which resolves through the same device-family/capability machinery. They must not be restored as authoritative host settings from another run's recipe.

`QuantizedModelSpec` is the representation portion of the composition manifest, derived from final packed state. It is not another run handle or a second editable copy of the recipe. `ModelArtifact` pins that manifest and its payload identity after a successful save. Inference-only formats can therefore be described and loaded without inventing a runnable quantizer.

### Small base and shallow inheritance

Proposed structure:

```text
BaseQuantizeConfig[WeightSpec]                 abstract
├── GroupedIntegerConfig                      shared weight contract
│   ├── GPTQConfig
│   ├── AWQConfig
│   ├── QQQConfig
│   ├── ParoConfig
│   └── RTNConfig
├── FP8Config
├── BitsAndBytesConfig
├── GGUFConfig
├── EXL3Config
├── VPTQConfig
└── NF12ADConfig

TT block-float: BlockWeights + TTBlockFloatFormat, checkpoint representation only
```

The base contains only typed `weights`, optional `activations`, and user annotations. Each concrete class declares its `method` as a class-level constant and owns its solver/native optimization settings. Target selection and module rules live on the containing action; preparation, recovery and staged refinement use the existing typed recipe-extension registry. `GroupedIntegerConfig` accepts `GroupWeights` and `ChannelWeights` with `IntFormat`, subject to concrete method constraints, so full-row scaling remains expressible. It does not acquire GPTQ damping or activation-order options. The new RTN recipe has the explicit `method="rtn"`; exporting a GPTQ layout does not change its producer identity.

`QQQConfig` and `VPTQConfig` compose shared Hessian/error-feedback controls rather than inheriting `GPTQConfig`. If a method truly supports every option and semantic invariant of another concrete config, inheritance is reasonable; code reuse alone is insufficient.

Keep familiar concrete names where useful. In particular, `GGUFConfig` can remain a convenient entry point for GGML block quantization, while the internal algorithm/encoding/container identities remain distinct.

### Typed weights instead of a universal bits wrapper

Proposed public examples, not existing APIs:

```python
GPTQConfig(
    weights=GroupWeights(
        format=IntFormat(bits=4, symmetric=True),
        axis="in_features",
        size=128,
        scale_solver=AutoScaleSolverConfig(),
    ),
    solver=GPTQSolverConfig(order="group_aware"),
)

GGUFConfig(weights=BlockWeights(format=GGMLFormat(qtype="q4_k")))

FP8Config(
    weights=BlockWeights(
        format=FloatFormat(dtype="float8_e4m3fn"),
        shape=(128, 128),
    )
)

BitsAndBytesConfig(
    weights=BlockWeights(
        format=BNBFormat(dtype="nf4", compress_statistics=True),
        shape=(64,),
    )
)

EXL3Config(weights=TrellisWeights(format=TrellisFormat(depth=3, codebook="mcg")))

VPTQConfig(
    weights=VectorWeights(
        format=VectorFormat(
            main=Codebook(vector_size=8, centroids=4096),
            residual=ResidualCodebook(centroids=16),
        ),
        vector_axis="out_features",
        partition=CodebookPartitionByCount(axis="in_features", count=1),
        scale_solver=None,
    ),
    solver=VPTQSolverConfig(error_feedback_tile=128),
)

NF12ADConfig(weights=NF12Weights(format=NF12Format(), scale_solver=None))
```

For the common GPTQ case, grouped integer weights can retain useful defaults, so `GPTQConfig()` remains concise. Default values should be centralized in the relevant spec rather than repeated in every leaf class.

### Complete GPTQ pipeline example

Use the existing composer/data/run contracts. This example requests symmetric INT4, groups of 128, no column reordering and optional weight-MSE scale search. All new imports and signatures below are proposed, not runnable against the current release.

```python
from gptqmodel import GPTQModel
from gptqmodel.quantization import (
    CalibrationDataSet,
    ExportConfig,
    GPTQConfig,
    GPTQSolverConfig,
    GroupWeights,
    IntFormat,
    Language,
    Local,
    LpScaleSolverConfig,
    MinMaxScaleSolverConfig,
    Quantize,
    TextDataSet,
)

model = GPTQModel.load("Qwen/Qwen2.5-0.5B")

text_data = (
    TextDataSet.from_hf(
        "Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        split="train",
    )
    .shuffle(seed=42)
    .limit(256)
)
calibration = CalibrationDataSet(text_data, max_length=2048)

use_mse = True
config = GPTQConfig(
    weights=GroupWeights(
        format=IntFormat(bits=4, symmetric=True),
        axis="in_features",
        size=128,
        scale_solver=(LpScaleSolverConfig(p=2) if use_mse else MinMaxScaleSolverConfig()),
    ),
    solver=GPTQSolverConfig(order="natural"),
)

job = Quantize(
    Language(config).Use(calibration),
    model=model,
    export=ExportConfig(layout="gptq_v2", container="safetensors"),
).Default(devices=Local(gpus="all"))

resolved = job.Compile()  # Metadata/capabilities only; no data consumption.
print(resolved.explain())
run = model.run(resolved)  # job.Run() is the convenience form.
artifact = run.save("./qwen-gptq-int4")
```

The direct spelling uses the same configuration and compiler:

```python
run = model.quantize(
    target="language",
    recipe=config,
    data=calibration,
    devices=Local(gpus="all"),
    export=ExportConfig(layout="gptq_v2", container="safetensors"),
)
artifact = run.save("./qwen-gptq-int4")
```

These are alternative ways to run the same initial quantization, not consecutive runs over an already quantized model. A later replacement of packed weights requires the existing explicit requantization/source contract.

`TextDataSet` owns source selection and immutable sampling provenance. `CalibrationDataSet(max_length=2048)` delegates that length constraint to the architecture preparation adapter; it neither tokenizes eagerly nor guarantees that every sample has 2,048 tokens. Validation stages check usable sample counts, masks and coverage. Model/dataset/tokenizer revisions and selected sample IDs are resolved and recorded; representative data and independent held-out evaluation remain necessary for quality claims.

`ExportConfig` is an optional output constraint on `Quantize`, advanced actions and direct operations. It is resolved before fitting; the selected layout must satisfy every affected owner and the assembled manifest. Each action's constraints are retained across composition, and incompatible demands fail rather than overriding one another. Saving revalidates actual payloads, untouched owners, aliases and mixed-format export support; it does not fit scales. External containers are explicit choices, with safetensors the default. `run.save()` returns the existing pinned `ModelArtifact`; optional artifact evaluation remains a separate action after durable save.

Keep these choices independent:

| Intent | Proposed setting | Meaning |
|---|---|---|
| Disable all column reordering | `GPTQSolverConfig(order="natural")` | Both existing ordering booleans false |
| Retain group-aware ordering with global descending order disabled | `GPTQSolverConfig(order="group_aware")` | `desc_act=False, act_group_aware=True` |
| Fit scales without clipping search | `MinMaxScaleSolverConfig()` | Scales remain enabled |
| Search scales for weight MSE | `LpScaleSolverConfig(p=2)` | Decoded-weight squared error in the method's fitting domain |
| Search scales for calibration-output MSE | Potential `OutputMSEScaleSolverConfig()` | Explicitly scoped output loss, requiring a supporting GPTQ adapter |

`LpScaleSolverConfig` is proposed configuration, not an existing class. The existing GPTQ setting is `mse=MSEConfig(norm=2.0)`; its scalar quantizer searches `|dequant(w)-w|^p`. Zero disables that search, not scale fitting. GPTQ still uses calibration-derived Hessian error feedback with natural order and min/max fitting, so there is no redundant `activation_aware=True` switch.

An output-MSE extension would evaluate the relevant `X @ W.T` versus `X @ W_quantized.T` objective, or a documented equivalent sufficient-statistics formulation, with the method adapter defining candidate scope and coordinates. Preflight rejects that strategy until implemented; it never silently falls back to weight MSE. Candidate search does not establish global optimality.

Proposed concise-default policy: `GPTQConfig()` uses symmetric INT4, `GroupWeights(axis="in_features", size=128)`, automatic native scale fitting and group-aware ordering. The explicit natural-order example overrides that choice. `IntFormat` defaults to symmetric integer codes, so AWQ examples explicitly request `symmetric=False`. Concrete method defaults live in one registered table; `GroupWeights` defaults to the input-feature axis and size 128 only where admitted. Required block/vector/codec geometry is not inferred from that group default. Other methods retain documented native defaults until their adapters are admitted.

All configuration-only scale policies use the `Config` suffix; `IntFormat` and `FloatFormat` remain numeric descriptor names.

### What `solver` means

`solver` is a typed configuration controlling how the selected method finds quantized weights and their numerical parameters. `GPTQSolverConfig` is proposed configuration data consumed by the GPTQ implementation. The example does not require a new universal solver engine. Use the `Config` suffix consistently for configuration classes, including method solvers and inference settings.

| Component | Responsibility | Example |
|---|---|---|
| `method` | Select the quantization procedure | `"gptq"`, fixed by `GPTQConfig` in Python |
| `weights` | Define the permitted weight representation and partition, plus its scale-fitting policy | Symmetric INT4, groups of 128 input features, a supported scale-fitting policy |
| `weights.scale_solver` | Select how scales and related affine parameters are fitted | Min/max, MSE clipping search, supported normalization or a method-native scale strategy |
| `solver` | Control the quantization procedure that uses those fitted parameters | Processing order, damping, static-group policy, supported error feedback |
| Execution settings | Existing action-local `Local` / execution profile | Compute devices, offload, memory budgets |
| Export settings | Select how the result is written | Packing layout and file container |

GPTQ uses calibration-derived Hessian information to quantize columns and compensate for quantization error in remaining columns. Its processing order and damping influence the resulting codes and scales even when the target weight format and group size stay the same.

The proposed `order` enum replaces the current mutually exclusive `desc_act` and `act_group_aware` booleans:

| Proposed order | Meaning | Current equivalent |
|---|---|---|
| `"natural"` | Process input columns in their original order | Both booleans false |
| `"activation_descending"` | Sort columns globally by descending Hessian diagonal | `desc_act=True`, `act_group_aware=False` |
| `"group_aware"` | Sort columns within each group, then reorder whole groups by importance | `desc_act=False`, `act_group_aware=True` |

The existing group-aware implementation scores groups by their maximum Hessian-diagonal value. This option preserves membership of the groups used by its ordering permutation. `weights.size` defines group size; `solver.order` defines processing order. Static versus adaptive scale estimation is a separate solver policy.

Damping magnitude/retry policy, static-group behavior and supported error-feedback variants can live in GPTQ solver options. Shared value types such as damping settings may be composed by multiple methods, while each method validates its own supported policies. VPTQ may expose k-means/error-feedback options; AWQ exposes its activation-aware transform search, while its local quantizer scale/clip objective belongs to the supported scale-fitting policy. A simple rounding method need not expose a method-solver object at all, although it still needs scale fitting. The common base should not force every method to accept GPTQ options.

Nesting makes ownership explicit and allows a solver preset to be reused with different valid weight specifications. Users can omit it to accept method defaults. `method` selects the procedure; the method-specific solver type validates its tuning options. It is unnecessary to repeat another method selector inside a solver whose type is already determined by the enclosing config.

The method solver decides when to invoke the weight's scale-fitting policy and how to interleave it with rounding, codebook fitting or error feedback. `weights.scale_solver` is the sole owner of local scale-fit settings such as the clipping objective; do not repeat those options inside the method solver. AWQ transform search, VPTQ centroid optimization and Paro joint optimization remain method-specific orchestration. A shared scale-solver configuration does not make those procedures interchangeable.

The saved artifact records the actual codes, scales, zero points, group mappings and any required permutations. Keep solver choices as recipe provenance; inference consumes the realized representation. Changing solver settings can change both numerical results and required metadata, so resolution must still validate the method, weight format and export layout together.

### Weight specifications

Use immutable weight recipe specifications with a typed `format` field. `GroupWeights` and `BlockWeights` describe how weights are partitioned; `format` holds an `IntFormat`, `FloatFormat`, `GGMLFormat`, `BNBFormat` or another applicable representation descriptor. Method configs narrow both axes to their supported combinations. `weights.format` always describes the weight representation; checkpoint layout and file container belong to export settings. These recipe objects contain scale-fitting policy; the saved weight descriptors contain realized metadata instead.

Proposed weight hierarchy:

```text
WeightSpec
└── ScaledWeight                  common scale-fitting capability
    ├── GroupWeights
    ├── BlockWeights
    ├── ChannelWeights
    ├── TensorWeights
    ├── VectorWeights
    ├── TrellisWeights
    └── NF12Weights               scale_solver fixed to None
```

There is no broad `FloatWeights` sibling: block-scaled floats belong to `BlockWeights`. Current FP8 also supports row and whole-tensor scales, which use `ChannelWeights` and `TensorWeights` with the same `FloatFormat`. The hierarchy separates partitioning from numeric representation. A float encoding alone does not imply a block partition, and integer encodings can also be block-based. BNB, GGML and TT block-float use `BlockWeights` with their corresponding format descriptors instead of introducing another weights class for each codec.

Every leaf, including `VectorWeights`, `TrellisWeights` and `NF12Weights`, must expose `format`. Vector/codebook topology belongs in `VectorFormat`; vector orientation and codebook partitioning belong in `VectorWeights`. Trellis depth/codebook identity belong in `TrellisFormat`; NF12 exponent-base policy belongs in `NF12Format`. The specialized weight classes constrain structure and admissible formats; they must not duplicate writable format fields. This is a set of structural contracts, not a proof that all representations reduce to one scalar partition or one scale tensor.

For example:

```python
BlockWeights(format=FloatFormat(dtype="float8_e4m3fn"), shape=(128, 128))
ChannelWeights(format=FloatFormat(dtype="float8_e4m3fn"), axis="out_features")
TensorWeights(format=FloatFormat(dtype="float8_e4m3fn"))
```

Internally a scalar group can share a geometry representation with a `1 x N` block when their traversal and assignment agree. Keep `GroupWeights` as the familiar constructor for per-output-row input-channel groups. BNB flat traversal, GGML nested block metadata and rectangular FP8 scale tiles still require explicit layout semantics; equal element counts do not make those encodings interchangeable. Backend checks consume resolved geometry, not just the weight class name.

For BNB4, `BlockWeights(shape=(64,))` means contiguous blocks of 64 values in the normalized matrix's row-major flattened order. FP8 `shape=(128, 128)` means rectangular scale-sharing tiles. The format descriptor validates the permitted shape/traversal combination. Fixed GGML and TT block geometries are derived from their format descriptors; conflicting user-supplied shapes are rejected. Backend shape limits remain separate from this semantic geometry.

`BNBFormat(dtype="nf4", compress_statistics=True)` owns the four-bit codebook identity and nested scale-statistics encoding. Width is derived from the selected dtype. Compression quantizes the scale statistics and changes both decoded values and saved metadata, so it is a representation choice rather than an execution resource setting. `BNBFormat` should construct a strict tagged NF4/FP4 or INT8 descriptor: the INT8 variant has no block-statistics compression field. The current BNB INT8 vectorwise producer uses `ChannelWeights(format=BNBFormat(dtype="int8"), axis="out_features")`. Thus the convenient constructor does not become another flat bag of irrelevant options.

### Shared `ScaledWeight` and `scale_solver`

A shared `ScaledWeight` base can own `scale_solver`, including for vector and trellis representations. Treat the base as a scale-fitting capability contract with an explicit absent state. It does not assert that every instance stores a separate scale tensor or runs a numerical scale search.

Not every supported representation performs separate scale fitting:

| Representation | Current scaling behavior |
|---|---|
| GPTQ / RTN / AWQ / QQQ / Paro | Scale parameters for scalar integer quantization; method-specific preparation and optimization remain distinct |
| FP8 | Tensor, row or rectangular-block scales, currently stored with inverse-scale semantics |
| BNB | Four-bit block statistics, optionally with nested scale quantization; INT8 uses vectorwise scales |
| GGML formats through GGUF | Codec-specific scales, including formats with nested scales, minima or shared exponents |
| EXL3 / `TrellisWeights` | Input factors include channel/global scaling; output factors can include optional output-channel scaling. Runtime reconstruction applies the saved factors around the transformed representation |
| VPTQ / `VectorWeights` | Optional affine normalization: stores `weight_scale` and `weight_bias` when `enable_norm=True`; axis follows `norm_dim`. It is disabled by default; centroid values can directly carry magnitudes without separate normalization scales |
| TT block-float | A shared exponent per 16 values represents a power-of-two scale; the exponent-selection/rounding contract is fixed by the format. The public config currently supports checkpoint loading/inference rather than a quantization producer |
| NF12 / `NF12Weights` | No separate quantization scale tensor. This is lossless BF16 coding with exponent-base metadata and escape data |

Proposed shared declaration, describing configuration rather than executing a solver:

```python
class ScaledWeight(WeightSpec):
    scale_solver: ScaleSolverConfig | None
```

`ScaleSolverConfig` is a typed policy describing how the producer fits the scale parameters supported by the representation. It may also fit the zero points or affine offsets associated with those scales. It must support representation-specific context, including multiple scale components, roles and axes. Do not reduce EXL3's input/output factors or VPTQ's affine normalization to a single scale number.

Define selection semantics explicitly:

- Where accepted, `AutoScaleSolverConfig()` selects the method/format's established policy, and resolves before execution to a concrete policy or `None`. It is a selection request, not a claim that every codec supports configurable scale fitting.
- `None` means no separate scale fitting. This is required for NF12 and permitted for VPTQ without affine normalization. It must not ambiguously mean "choose the default."
- Concrete policies such as `MinMaxScaleSolverConfig()` and `LpScaleSolverConfig(p=2)` are accepted only where the selected method and format implement them. The latter denotes a clipping search minimizing Lp error; `p=2` is MSE, while other positive exponents are not literally MSE. They are proposed APIs; this design does not establish new algorithm support.
- Disabling clipping search does not disable scales. Current `mse.norm=0` maps to the native non-search scale fit, not `scale_solver=None`. Keep method defaults in a documented defaults table; the explicit `AutoScaleSolverConfig()` example does not enable a new search by default.
- VPTQ defaults to `None` to preserve its current unnormalized behavior. NF12 defaults to and requires `None`; reject any non-None scale solver. Scaled formats that require a producer scale policy reject `None` for quantization.
- Format-mandated procedures, such as shared-exponent selection or a GGML codec's internal scale fitting, are represented by their native policy. Do not expose unsupported generic replacements. Loading an existing artifact needs its scale metadata, not a scale solver.

`format` continues to own how scale parameters are encoded: storage dtypes, inverse/direct conventions, nested-statistics compression and fixed codec structures. `scale_solver` owns how their values are obtained. `weights` owns logical partition geometry; the method `solver` owns the surrounding quantization procedure. Each configurable setting has one writable owner.

NF12 exponent-base selection optimizes codec representation/escape usage without changing decoded BF16 values. It remains a codec setting rather than being renamed a scale solver. Inheritance must not add scale arrays or transformations to its output.

This uses one shallow inheritance layer to deduplicate the policy field while retaining typed method-specific strategies through composition. A stricter alternative keeps NF12 directly under `WeightSpec`, with `ScaledWeight` reserved for scaling-capable families; the proposed common hierarchy instead favors one uniform, explicitly optional `scale_solver` property.

The shared field does not imply a universal `fit(weight) -> scale` function. A method adapter consumes `ScaleSolverConfig`, declares required inputs (weights, activations, Hessian, or transform state), and invokes the fit at the correct point in its own procedure. Each policy states its objective domain (decoded-weight error versus operator-output error), reference values, weighting and numerical candidate definition; a shared exponent alone does not identify the objective. Its typed result identifies each scale/offset/zero-point component, axis or assignment map, coordinate space, precision and convention. EXL3 and joint Paro optimization may delegate to a native method policy; do not insert an independent generic fit pass before them.

Refinement is ordered: primary quantization produces an initial representation; GSQ or another admitted stage may update its codes/scales; packing consumes the final validated result. Scale-fitting settings own the initial fitting policy, while a refinement stage owns its own update objective and learning rates. Both may affect scales at different stages. An inference artifact records final values and required transforms, with the full stage sequence retained as provenance. No consumer reruns the initial scale solver during load.

`bits` remains an integer on `IntFormat`. Fixed formats derive their nominal width. VPTQ derives index-stream cost from its vector/codebook geometry rather than accepting a rounded `bits` label. EXL3 exposes integer `depth` on `TrellisFormat`. Mixed-rate budget targets belong to an allocation config, for example `AllocationConfig(target_bpw=2.5)`, which resolves to concrete encodings per module.

Retire `BaseComplexBits`, the `QuantBits = GGUFBits` alias and numeric impersonation. An encoding should have exact identity: `q4_0 != q4_1`, even though both have nominal width 4. Do not make an encoding compare equal to the integer 4 or participate in arithmetic as though it were a width. Strings are accepted by explicit parsers/convenience constructors and become typed objects immediately.

### Segmentation: shared vocabulary, distinct semantics

| Concept | Proposed expression | Meaning |
|---|---|---|
| GPTQ/AWQ group | `GroupWeights(..., axis="in_features", size=128)` | Group cardinality and axis; actual membership follows the method's grouping/permutation policy |
| Full-row integer quantization | `ChannelWeights(..., axis="out_features")` | One scale set per output channel; replaces the `-1` size sentinel |
| FP8 scale rectangle | `BlockWeights(format=FloatFormat(dtype=...), shape=(128,128))` | Explicit 2D scale granularity |
| GGML packed block | Derived `weights.format.block_shape` / `block_bytes` | Fixed codec structure, including sub-block metadata; not freely editable |
| BNB four-bit block | `BlockWeights(format=BNBFormat(dtype="nf4", ...), shape=(64,))` | Contiguous flattened blocks sharing statistics; nested scale compression belongs to the format |
| VPTQ vector | `VectorFormat(main=Codebook(vector_size=8, ...), ...)` plus `VectorWeights.vector_axis` | Number and orientation of weight values represented by a centroid index |
| VPTQ codebook partition | Typed size-based or count-based partition | Which weights share codebooks; replaces two conflicting scalar knobs |
| VPTQ/GPTQ solver tile | `solver.error_feedback_tile=128` | Computation chunk; does not describe scale-sharing or encoded storage |
| NVFP4 activation block | Typed activation scale geometry | Independent from weight groups, with block scales and a global scale |

Use semantic weight axes in public specs, with a defined logical matrix orientation `(out_features, in_features)`. Resolve to physical axes after Conv1D/transpose normalization. For activations use token/feature semantics. A shared internal geometry representation may serve kernels, but a single `block_size` field cannot preserve these meanings.

`ChannelWeights.axis` names the axis indexing independent scales, not the reduction axis. For a logical `(out_features, in_features)` matrix, per-output-channel scales reduce over input features. VPTQ normalization `norm_dim=0` instead produces input-channel scale/bias vectors; a normalization policy must state that semantic axis explicitly.

Groups are not always consecutive in original column order. GPTQ with global activation ordering and dynamic groups quantizes groups in reordered coordinates, then restores weight columns while preserving a possibly non-monotonic `g_idx`. Its static-group policy uses original membership and different fitting timing. Keep those choices under one GPTQ group policy; do not also expose a contradictory writable group-order flag in `GroupWeights`. The resolved result must carry an implicit canonical assignment or an explicit assignment map, plus any required permutation/transform metadata. Reject a backend/export layout unable to represent that result.

Resolve scale partition, vector/codebook partition, codec storage geometry, solver tile and padding separately. Declare flattening order, row-boundary resets, original and padded extents, and partial-block handling. Default to rejecting unsupported tails; never infer zero padding from a size mismatch. Existing FP8 block production requires exact divisibility, while NF12 has its own matrix alignment constraints. Padding, where supported, is an explicit preparation/layout decision and must preserve valid extents and exclude artificial entries from scale fitting unless the method's documented math requires otherwise.

### Shared options and field ownership

| Current fields | New home / treatment |
|---|---|
| `bits`, `sym`, integer `group_size` | Integer encoding owns width/symmetry; `GroupWeights` owns group axis/size |
| `format`, `pack_dtype` | Typed `weights.format` for the weight representation; export settings for versioned layout and packing word dtype |
| `desc_act`, `act_group_aware`, `static_groups` | GPTQ solver ordering/group policy; replace mutually exclusive ordering booleans with one enum where applicable |
| `damp_percent`, `damp_auto_increment`, VPTQ `percdamp` | Shared damping value type, with method-specific solver policy and supported retry behavior |
| `hessian` | Shared Hessian accumulation options; execution-only staging budgets separated from semantic/numerical policy |
| `mse.norm` | Native non-search scale fit for zero; `weights.scale_solver=LpScaleSolverConfig(p=...)` for a supported positive-exponent search |
| MSE candidate grid, clipping range and search budget | Numerical search policy in `weights.scale_solver`; changes can alter the fitted scales |
| MSE/GSQ temporary tensor chunks and workspace limits | Execution settings only when scheduling the same candidates; pruning or approximate search remains a recorded numerical policy |
| Scale-fitting policy | Shared `ScaledWeight.scale_solver`, narrowed and validated by method/format; `None` explicitly denotes no separate scale fitting |
| `gptaq`, `foem` | Typed error-feedback settings on supported solvers; preserve distinct semantics and mutual exclusions |
| `gsq`, `gsq_training` | Separate typed refinement stages with a common integration point; do not merge their algorithms because their names overlap |
| Paro `krot` and `opt_*` | Rotation/optimization dataclasses; shared optimizer options only where definitions and defaults agree |
| VPTQ vector/centroid/partition fields | Vector/codebook topology in `weights.format`; orientation/partition in `weights`; k-means, damping and error-feedback chunking stay in VPTQ solver options |
| VPTQ `enable_norm`, `norm_dim` | `scale_solver=None`, or a supported `AffineNormalizationScaleSolverConfig(axis=...)`; do not retain an independent boolean that can contradict the solver |
| EXL3 `out_scales` | An option on its native scale strategy, with input/output/global component semantics; do not conflate it with tensor/block geometry |
| `preprocessors`, `smoother`, `smooth`, `rotation` | One ordered preparation pipeline. Fallback smoothing stays inside its fallback policy when it is conditional |
| `activation` | Typed activation policy; saved scales and transforms enter the artifact specification |
| `lm_head`, embedding mode, EXL3 head/include controls | Shared target-selection rules with explicit role-specific weight overrides; tied-weight conflicts validated |
| `true_sequential` | Quantization schedule policy, not a weight property |
| `fallback` | Calibration coverage/fallback policy; resolved per-module outcomes recorded in provenance |
| `adapter`, EoRA; norm tuning | Typed recovery stages with explicit support restrictions; persisted inference payloads described in the artifact spec |
| devices, P2P, VRAM pools, offload, GC, logging | Existing action-local execution profile and `Local` resource context |
| `compile`, inference backend/device | `InferenceConfig`, resolved through shared capability/device machinery |
| `meta` | User annotations plus structured provenance; no hidden behavioral fields |

Deduplicate shared concepts, not merely similar names. VPTQ's codebook normalization is not norm-layer tuning; BNB FP4 is not interchangeable with NVFP4; a shared damping value does not imply identical solver retry behavior. RCO's shape-aware cost accounting is an existing starting point for common encoding cost descriptors.

## 4. Resolution, serialization and extensibility

### One compiler, separate plan and artifact contracts

Build on existing `ResolvedModuleConfig`, `ModuleConstructorConfig` and `QuantizationRunPlan`; replace their flat universal fields with shared typed descriptors and one resolution pipeline. Do not make one immutable object stand for both requested settings and produced tensor state: actual scale values, codebook contents, permutations, escapes and packed sizes do not all exist before quantization.

Proposed flow:

```text
Concrete Python config OR versioned JSON/YAML
                  |
          explicit typed parser
                  |
  action scope + defaults + hard exclusions + priority-resolved recipe rule
                  |
      method + encoding + layout validation
                  |
 shape/device/backend/operation capability preflight
          |                       |
          |                       +--> clear unsupported/conflict error
          v
 immutable ResolvedPlan containing ResolvedModulePlan + ExecutionLayout
          |
 QuantizationRun: quantizer --> typed QuantizationResult --> refinement --> packer
                                                            |
                                   EncodedModuleSpec + tensor payloads
                                                            |
                           composition manifest / ModelArtifact --> loader --> inference backend
```

`ResolvedModulePlan` is a per-owner entry of the composer's `ResolvedPlan`, containing resolved policy and input requirements; `QuantizationResult` contains method-native realized state; `EncodedModuleSpec` describes the final persisted representation. `QuantizedModelSpec` collects the encoded module specs, tensor ownership and model-wide execution metadata. Share format/geometry/value types across phases, with explicit conversions and validation at boundaries. Runtime tensors and mutable optimizer state do not live in recipe dataclasses. A loader consumes `QuantizedModelSpec` directly without building a producer plan, importing a training dependency or requiring a calibration dataset.

Use immutable specs and nested tuples/frozen value objects so processors cannot mutate another module's config. Keep compiled regexes and caches in a resolver owned by the run. Replace a config to edit it; do not track arbitrary nested dictionary mutations. A frozen dataclass containing a mutable dict is not sufficient. Preserve typed extension payloads for plugin methods; do not recreate an unvalidated `options: dict[str, Any]` as the universal escape hatch.

Use the existing action-scoped `Rules` contract, with explicit selectors/actions instead of `+:`/`-:` dictionary keys. Enumerate checkpoint-owned logical modules before worker placement; resolve the action target; any matching preservation/exclusion is a hard write constraint. For remaining owners, a higher explicit priority wins between conflicting recipe overrides; equal-priority conflicting requests fail, while identical requests can coalesce. Insertion order is not precedence. This is distinct from the composer's field-level `Default`/`MoE` inheritance for execution/routing.

A recipe override atomically replaces the method configuration and attached extensions. `Recipe.configure(weights=..., solver=..., activations=...)` derives an immutable recipe: each supplied typed object replaces that object atomically, omitted fields retain the base, and existing extensions are revalidated. Explicit `None` is never an omitted value. There is no second nested config override engine. Method changes require a complete admitted replacement recipe. Preserve resolved-rule provenance and unmatched-selector diagnostics. Protocol inputs compile into this same model, with explicit capability checks for weight/activation/KV targets.

Resolve names before packing or fusion and keep logical-to-physical mappings. Detect shared/tied tensor ownership before applying contradictory rules: preserving the tie requires compatible representation and execution contracts; conflicting requests fail until an explicit supported untie operation is selected. Record quantized, dense, excluded and alias-owned tensors in the artifact. Fusion may combine only compatible resolved modules; incompatible choices retain separate modules unless fusion was mandatory, in which case preflight rejects the request. An allocator may only fill unpinned choices and must report an infeasible budget rather than overwrite an explicit module rule.

Calibration requirements come from the entire plan: the primary quantizer, scale solver, preparation, activation calibration, refinement and recovery may require different captured inputs. FP8 with calibrated GSQ and RTN with a calibration-dependent stage cannot be classified from their primary method alone. Replace the overloaded weight-only boolean with explicit operation/input requirements. Preparation declares its affected modules and coordinate transforms; a method's internal transform search is not also replayed as a preparation stage. Calibration captures must match the coordinate system in which each consumer operates.

Preflight covers every admitted fallback outcome, including its calibration needs and output representation. A fallback cannot silently change format, dtype, layout or export compatibility. Record the method actually used per module and the reason; a mixed-method receipt must not claim the original method ran everywhere. Data-dependent output invariants are checked again after quantization and before writing.

Native source decoding, dtype conversion and requantization are explicit plan stages. Preserve an existing supported quantized payload when the requested operation is load or lossless repacking; do not silently dequantize and requantize. For NF12, the losslessness claim applies to the BF16 tensor entering the codec; an earlier FP16/FP32-to-BF16 cast is recorded separately and must not be described as lossless source preservation.

### Complete artifact contract

The format catalog must specify the decoding equation, logical code range, symmetry/zero-point semantics, component dtypes and axes, scale direction, storage geometry and required payload roles. `IntFormat(bits=4, symmetric=True)` alone is not a complete loader contract. Layout adapters may change names and packing positions, but any change to effective numerical values, such as converting rounded inverse scales to direct scales, requires an explicitly admitted numerical conversion. Default export is layout-only; requantization is a separate explicit stage. Each format also specifies zero-range, saturation, rounding and non-finite behavior where relevant; a lossless NF12 codec and a finite-only FP8 producer must not inherit the same blanket input-cleanup rule.

| Encoded module information | Required meaning |
|---|---|
| Representation identity | Canonical format and codec version, logical numeric semantics, concrete parameters; no unresolved `Auto` selection |
| Geometry | Original logical shape, physical storage shape, orientation, valid extents, scale/codebook assignments, padding and any required permutations |
| Scale/affine state | Named components with dtype, axis or assignment, coordinate space and direct/inverse/exponent convention; explicit absence is valid |
| Codec payloads | Tensor references for codes, codebooks, residual/outlier indices, BNB nested statistics/offsets, EXL3 factors, NF12 escapes/base, or other required state |
| Layout and ownership | Packing layout/version, tensor names/dtypes/shapes, shard references, shared storage IDs and aliases; each required payload has one authoritative owner |
| Model execution semantics | Required transforms, biases, adapters/recovery payloads, dense retained tensors and supported activation policies; distinguish folded transforms from runtime transforms |

Use typed `TensorRef` records into checkpoint payloads rather than embedding arrays in JSON. Validate required payload presence, shapes, dtypes, index ranges and geometry consistency on load; reject incomplete or contradictory metadata. Two recipe inputs may produce the same representation, and one recipe may produce different content-dependent sizes, so recipe equality cannot serve as artifact identity. Bind manifests to their actual payloads and record source/model revision plus codec/producer versions.

Activation policies distinguish static calibrated parameters from runtime input-dependent scale rules. Only persistent calibration data belongs in the checkpoint; per-token/per-block runtime scales are computed during inference. A weight `scale_solver` must not be reused as an offline-only explanation of runtime activation quantization. Validate combined weight/activation/backend constraints, including QQQ and GPTQ activation modes, rather than admitting their components independently.

### Canonical schema

Proposed recipe fragment:

```json
{
  "schema_version": 1,
  "method": "gptq",
  "method_version": 1,
  "weights": {
    "type": "group",
    "format": {"type": "int", "bits": 4, "symmetric": true},
    "axis": "in_features",
    "size": 128,
    "scale_solver": {"type": "auto"}
  },
  "solver": {"order": "group_aware"}
}
```

`schema_version` versions the configuration structure; `method_version` versions the method descriptor and its defaults. Built-ins use short canonical method names such as `gptq`, with namespaced plugin names where needed. The compiler derives a pinned registry identity such as `gptqmodel/gptq@1` from those fields, rather than accepting a second writable method selector. Composer/plan schemas embed this same recipe object. Keep one canonical field for each concept. Use `method` as the sole method selector; remove its `algorithm`/`quant_method` aliases, legacy top-level `checkpoint_format`/`format`, `zero_point`/`sym` and `bnb_*` duplicate spellings from the internal schema. The new nested `weights.format` has one meaning: the typed weight representation. Explicit export adapters can emit whatever an external runtime requires, such as QQQ `wbits`; those fields do not become extra writable internal state.

Use `type` consistently for discriminators, including the descriptor inside `weights.format`. Direct Python construction uses `GPTQConfig`, `AWQConfig`, etc. A normal `parse_quantize_config()` reads the explicit discriminator. Retire the metaclass factory; `QuantizeConfig` may become a type union instead of a constructor with surprising dispatch. Unknown fields, contradictory types and explicitly unsupported settings fail. An explicit `Auto` request may resolve to a supported layout, but the selected result must be visible in the plan and saved artifact. Never silently repair a contradictory explicit method/layout pair.

The recipe and artifact have separately versioned schemas and dedicated parsers. A recipe round trip preserves user intent; an artifact round trip preserves decoding semantics. Materialize ordinary defaults in complete canonical config output; preserve explicit automatic-selection requests there and record their concrete outcomes in the run receipt. Partial updates must preserve which fields were supplied, without materializing defaults into omitted override fields. For `scale_solver`, omitted means the documented method/format default, explicit `null` means no fitting, and `{"type":"auto"}` requests default-policy resolution. Never merge null and omitted during overrides. Concrete Python class and serialized `method` must agree.

The enclosing `Recipe` serializer adds its registered `extensions` list alongside the method-config fields, as shown in the [plan schema](quantization-plan-api.md#7-reusable-fragments-and-configuration-files). Its strict parser validates that envelope and delegates the config fields to `parse_quantize_config()` and each extension to its registered decoder. A bare config parser does not silently consume extensions or unknown keys; both entry points share the same config schema and produce the same typed method value.

Validate integer widths and dimensions strictly (booleans are not widths), positive sizes/counts, finite numeric objectives, supported dtype identifiers, and unknown fields/versions with full error paths. Canonicalize tuple shapes as JSON arrays, enum values as stable strings, and imported aliases at an explicit adapter boundary. Unknown extension tags fail unless the matching lightweight schema/descriptor is registered; do not fall through to a nearby format.

Constructing a config performs pure validation only: no temporary directories, GPU probes, tensor allocation or optional backend imports. The execution context creates offload directories and runtime state after preflight, and owns their cleanup.

Keep numerically meaningful precision, rounding and seeds in the recipe or resolved numerical policy. For example, Hessian staging dtype can change values and must not disappear into an unrecorded execution hint. Pure placement, scratch capacity and telemetry belong in execution settings. Record resolved devices, implementations and dependency versions for reproducibility; changing a workspace budget must not silently select a different mathematical objective. JSON/schema checks cannot establish cross-device numerical determinism.

No backward Python API or native-schema compatibility layer is required. External checkpoint readers may translate their source formats into the new artifact spec at a single boundary. Whether to retain old GPT-QModel checkpoint readers is an explicit product choice; it should not force aliases back into the core model.

### Registry boundaries

Use three registries, each owning a different contract:

1. Method registration: config type, parser tag, processor/run-plan factory and method-specific admissibility.
2. Encoding/layout descriptors: canonical identity, fixed geometry, scale/zero semantics, versions, tensor schema and cost functions.
3. Backend declarations: operations, formats, shapes, dtypes, device families and dependency/architecture constraints.

Generate summary tables from those declarations. Keep backend discovery/capabilities authoritative for kernels; do not copy their support matrix into the method registry. A backend's subset of a method's encodings is a constraint intersection, not duplication to remove. Register lightweight descriptors without importing all optional engines.

Keep distinct capability flags/predicates for inspect, decode, quantize, pack, export and inference. This models TT load-only support and E8M0 decode-only weight support directly, without a global quantization blacklist or accidental admission. For production, compute the intersection of method, format, scale strategy, other stages, geometry, activations, layout and operation first; bind the available backend/device afterward. For inspect/decode/inference, validate the saved representation and consumer capabilities without requiring its original producer to be installed. A CPU packer need not be the inference backend. An explicitly selected unsupported backend fails; automatic selection may choose only a declared compatible alternative and must record the choice.

### Width and storage accounting

Expose separate queries:

- `nominal_bits`: meaningful coding width/depth when the representation has one; read-only for fixed encodings.
- `estimated_storage_bytes(shape)`: encoded payload plus scale/zero/codebook/index/escape/alignment costs, with assumptions stated.
- `actual_storage_bytes(packed_module)`: measured packed buffers, with shared tensors counted once.
- `target_bpw`: an allocation objective, not a property of one encoding.

Derive bpw from bytes and the logical number of weights. Distinguish per-module encoding cost, model-wide quantized-weight cost and full checkpoint size, which also includes unquantized tensors and container metadata. Bounds or estimates should remain labeled until content-dependent payloads such as NF12 escapes and codebooks are known.

Define the allocation budget domain explicitly. Recommended `target_bpw` default: total encoded bytes for the uniquely owned targeted weights, including their auxiliary payloads, divided by their logical weight count. Report full-model/checkpoint costs separately. Padding consumes bytes but does not increase the logical denominator; tied storage counts once. Compute vector index cost from actual packing/alignment rather than only `ceil(log2(centroids))`. A hard budget requires final measured bytes to satisfy it; an estimate is not proof of compliance.

## 5. Design alternatives

| Alternative | Benefits | Costs / limitations | Recommendation |
|---|---|---|---|
| Flat config with `bits: int \| str` and generic `block_size` | Smallest diff; familiar call sites | Preserves ambiguity, invalid combinations and inherited dead fields; validation keeps growing | Reject given no compatibility requirement |
| Separate independent config for every method | Excellent local readability; no irrelevant inherited knobs | Repeats targets, preprocessing, overrides, serialization and common solver options | Viable for a small library; expensive at current breadth |
| Deep inheritance / many mixins | Can centralize repeated fields | MRO and initialization order become fragile; unrelated algorithms inherit unsupported behavior | Avoid |
| Small base + typed specs + composed solver options | Familiar method entry points, semantic fields, reusable components, explicit extension contracts | Requires coordinated consumer migration and a real resolution step | **Recommended** |
| One generic `QuantizationRecipe(method, weights, stages, ...)` / protocol-first API | Most flexible mixed-method and staged orchestration; strong programmatic generation | Verbose for common use; schema and cross-product validation become a framework; current protocol compiler is narrower than its syntax | Useful as a lower-level model, with concrete configs as typed facades |

An independent implementation choice is dataclasses versus Pydantic. Prefer immutable standard-library dataclasses plus a strict tagged decoder to minimize dependency/import cost and match the current codebase. Pydantic is reasonable if generated JSON Schema, validation error paths and external API tooling justify the dependency. It does not resolve semantic ownership by itself.

## 6. Suggested implementation order and acceptance criteria

1. Introduce typed weight/geometry/layout descriptors and derive GGUF identity/cost metadata from one catalog. Establish strict parser and schema round trips.
2. Add the small config base and concrete classes, then migrate module resolution, constructor specs and the existing run plan together. Use one compiler with separate immutable plan metadata, typed result state and encoded artifact descriptors.
3. Migrate GPTQ/RTN/AWQ/QQQ/Paro, then scaled block/channel/tensor, codebook, trellis and NF12 representations. Represent TT as an artifact encoding until a quantizer exists. Use the existing action execution context and keep inference settings outside method recipes.
4. Move writers/loaders to the canonical artifact spec; put external conventions in adapters. Reuse the typed model for protocol compilation, RCO output and dynamic overrides.
5. Remove the factory metaclass, synonym maps, bit arithmetic wrappers, duplicate normalizers and compatibility state. Update examples and documentation in the same change series.

Acceptance should cover strict config/schema round trips for every concrete type; rejected irrelevant fields and ambiguous combinations; deterministic override resolution; operation-specific capability checks; mixed-width/planar GPTQ; typed scale geometry; storage cost agreement with packed buffers; save/load reconstruction for every representation; and actual supported-device numerical tests wherever adapting consumers can affect packing or execution. Numerical validation is necessary during implementation even though the proposed refactor does not intend to alter quantization math.

## 7. Correctness review: holes and required proofs

These are issues in the proposed abstraction, not newly discovered claims of defects in the current quantizers. The preceding sections incorporate the recommended contract changes. They have been source-reviewed, not implemented or numerically validated.

| Priority | Hole in the earlier draft | Consequence | Required design rule |
|---|---|---|---|
| High | One resolved object was expected to serve recipe, producer and loader | Future tensor state would be guessed, or a supposedly immutable plan mutated; loading could require quantization dependencies | Distinct plan/result/artifact objects sharing value types; loaders use realized metadata directly |
| High | Universal `scale_solver` lacked a context and lifecycle contract | A standalone pass could run in the wrong coordinates, ignore multi-component scales, or overwrite later refinement | Method adapters declare inputs and fitting points; native joint policies and ordered refinement remain explicit |
| High | Group cardinality was described as original-order contiguous membership | GPTQ activation ordering could load with the wrong scale-to-column assignment | Persist canonical assignment or explicit `g_idx` and required transforms; validate consumer support |
| High | Format/dtype plus a scale shape was not a sufficient decode contract | Wrong inverse/direct interpretation, zero-point conventions, nested statistics or missing auxiliary state could silently change outputs | Versioned decoding semantics and typed payload roles, with load-time structural validation |
| High | Capability validation did not fully cover the complete stage sequence | A weight-only method plus a calibrated scale/refinement stage could be admitted without inputs; load-only formats could acquire fictitious producers | Derive requirements from all stages, validate producer and consumer operations separately, and check fallback outcomes |
| Medium | Vector, trellis and NF12 examples still bypassed `format` | Codec properties would remain scattered and harder to serialize or override uniformly | Every weight leaf has a typed `format`; partition/orientation stay on weights |
| Medium | `None`, omitted, auto and disabled search could collapse together | Disabling clipping could accidentally disable required scales; an override could silently change defaults | Explicit selection semantics; `mse.norm=0` retains non-search scale fitting |
| Medium | Block/group/codec/tile geometry had no full tail and coordinate policy | Transposes, row-crossing blocks or implicit padding could change the quantization domain | Separate geometries and store original/physical shapes, valid extents, traversal and assignment |
| Medium | Override, tied-weight and fused-module conflicts were underspecified | Different consumers could resolve the same checkpoint weight differently | Deterministic rule precedence, atomic typed replacements, ownership validation and explicit logical/physical maps |
| Medium | Moving all implementation controls into execution could hide numerical policy | Precision or rounding choices could change results without appearing in the recipe | Preserve numerical dtype/objective/seed choices, record resolved defaults, and measure actual storage |

Source anchors for the concrete counterexamples: [GPTQ group assignments and permutation restoration](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/gptq.py#L2494-L2526), [MSE zero disables search rather than quantization scales](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py#L1657-L1687), [FP8 inverse scales and block divisibility](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/fp8.py#L135-L177), [VPTQ optional normalization payloads](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/vptq_torch.py#L246-L259), [GSQ can update scales](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/gsq_scalar.py#L218-L219), [NF12 ignores external quantization scales](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/nf12_ad.py#L267-L279).

Representative acceptance cases for the implementation:

| Case | Required result |
|---|---|
| GPTQ full-row integer scaling via `ChannelWeights` | Accepted where the concrete method/backend supports it; no `-1` size sentinel |
| Globally reordered GPTQ with dynamic groups | Saved scale assignments reproduce the same logical weight columns after reload |
| Static GPTQ groups versus adaptive groups | Correct membership and fitting timing; no contradictory grouping setting on weights |
| `scale_solver=None` for ordinary grouped INT4 production | Rejected; absence is not a synonym for native min/max fitting |
| VPTQ with and without affine normalization | Both supported paths preserve codebooks/indices and the correct presence or absence of scale/bias metadata |
| NF12 with a non-None scale solver | Rejected, including an unsupported auto request; BF16 codec round trip remains bit-exact without synthetic scales |
| BNB INT8 with four-bit statistics-compression fields | Rejected; NF4 compressed statistics retain both levels and offset metadata |
| FP8 block geometry with unsupported remainder | Rejected before production; no silent pad/crop or change to scale convention |
| Equal-sized BNB flat blocks and FP8 2D tiles | Remain distinct resolved geometries; identical element count does not imply compatibility |
| TT or E8M0 existing artifact on an admitted reader | Load/decode succeeds without producer/calibration dependencies; unsupported dense quantization fails |
| RTN or FP8 plus a calibration-dependent stage | Missing required inputs rejected during preflight |
| Refinement changes codes/scales | Packing and reload use final values, not the initial fit |
| Conflicting tied aliases or forced incompatible fusion | Clear preflight error; no silent duplication, untie or coercion |
| Recipe round trip with omitted/default/null/auto selections | Stable intent and field-specific semantics; artifact output has resolved values |
| Known format with missing payload or wrong tensor dtype/shape | Load fails with the module and field identified |
| Mixed quantized/dense/alias-owned model | Complete ownership coverage; measured byte accounting includes auxiliaries and counts shared storage once |

Before implementation is declared complete, instantiate a matrix of each current method's admitted weight types, scale strategies, default policy, calibration requirements and output layouts from the existing consumers. New strategy combinations remain unsupported until implemented and tested. The highest-risk prototypes are reordered GPTQ, compressed-statistics BNB, VPTQ normalization, EXL3 native scaling, and recipe-free NF12/TT loading. This is an implementation gate, not a claim that the new schema or these tests already exist.

## Source map

| Concern | Audited source |
|---|---|
| Base fields, factory, normalization, concrete configs, serialization | [quantization/config.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/config.py) |
| Resolved module constructor and pack/load normalization | [quantization/module_config.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/module_config.py) |
| Existing executable run plan | [models/quantization_run_plan.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/models/quantization_run_plan.py) |
| Public quantize and processor lifecycle | [models/base.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/models/base.py) |
| Typed protocol surface and limited config compiler | [quantization/protocol.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/protocol.py) |
| GGML block sizes, byte counts and alias-to-tensor mappings | [nn_modules/qlinear/gguf.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/gguf.py) |
| Shape-aware mixed-rate allocation | [quantization/rco_formats.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/rco_formats.py) |
| VPTQ codebook partition, damping and solver chunks | [quantization/vptq.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/quantization/vptq.py) |
| Backend contracts | [nn_modules/qlinear/__init__.py](https://github.com/ModelCloud/GPTQModel/blob/35987dfb7433d9b539e5ca9656621473fcd1fbe8/gptqmodel/nn_modules/qlinear/__init__.py) |
| Existing config regressions | [tests/qcfg](https://github.com/ModelCloud/GPTQModel/tree/35987dfb7433d9b539e5ca9656621473fcd1fbe8/tests/qcfg) |

## Appendix: exact current field inventory

These are declared dataclass fields, extracted from the audited source. Concrete configs also inherit fields from their parents. Redeclared fields are shown because their defaults, types or constructor visibility often differ. Private runtime state is identified separately. This inventory records current names, not the proposed API.

| Current base concern | Fields |
|---|---|
| Representation and method | `bits`, `group_size`, `sym`, `desc_act`, `method`, `format`, `pack_dtype`, `activation` |
| Target and recipe policy | `dynamic`, `true_sequential`, `lm_head`, `mse`, `adapter`, `rotation`, `fallback`, `moe`, `meta` |
| Execution | `device`, `telemetry`, `pack_impl`, `compile`, `offload_to_disk`, `offload_to_disk_path`, `keep_processor_logs`, `compute_device_filter`, `calibration_data_device`, `auto_forward_data_parallel`, `p2p`, `dense_vram_strategy`, `dense_vram_strategy_devices`, `moe_vram_strategy`, `moe_vram_strategy_devices`, `gc_mode`, `wait_for_submodule_finalizers` |
| Private state | `_offload_temp_dir` |
| `PreProcessorConfig` additions | `preprocessors`, `smoother`, `smooth` |

`QuantizeConfig` adds no dataclass fields; it is the compatibility factory. `PreProcessorConfig` inherits `BaseQuantizeConfig`. All listed concrete configs inherit `PreProcessorConfig` except QQQ and VPTQ, which inherit `GPTQConfig`.

| Concrete config | Fields declared or redeclared by the class |
|---|---|
| `GPTQConfig` | `damp_percent`, `damp_auto_increment`, `act_group_aware`, `static_groups`, `gptaq`, `foem`, `gsq`, `gsq_training`, `mock_quantization`, `hessian` |
| `AWQConfig` | `method`, `format`, `gsq`, `gsq_training`, `scale_search_chunked_activations` |
| `QQQConfig` | `method`, `format` |
| `RTNConfig` | `method`, `format`, `gsq` |
| `ParoConfig` | `method`, `format`, `gsq`, `krot`, `opt_rotation_epochs`, `opt_finetune_epochs`, `opt_train_samples`, `opt_validation_samples`, `opt_batch_size`, `opt_rotation_lr`, `opt_weight_lr`, `opt_quantizer_lr`, `opt_pair_ratio`, `opt_seed`, `opt_optimizer`, `opt_weight_decay`, `opt_betas`, `opt_eps`, `opt_amsgrad`, `opt_sgd_momentum`, `opt_sgd_dampening`, `opt_sgd_nesterov`, `opt_fused_rotation`, `opt_gradient_checkpointing`, `opt_stage_cudagraph`, `opt_amp`, `opt_best_state_dtype`, `opt_train_on_noisy_inputs`, `opt_scope`, `opt_stage_impl`, `opt_pair_impl`, `opt_quantizer_impl`, `opt_channel_scale_clamp_min`, `opt_channel_scale_clamp_max` |
| `VPTQConfig` | `method`, `format`, `bits`, `vector_lens`, `num_centroids`, `num_res_centroids`, `npercent`, `group_size`, `group_num`, `kmeans_mode`, `kmeans_iters`, `kmeans_tol`, `kmeans_seed`, `enable_norm`, `norm_dim`, `enable_perm`, `block_size`, `percdamp` |
| `FP8Config` | `gsq`, `gsq_calibration`, `bits`, `method`, `format`, `group_size`, `desc_act`, `sym`, `weight_scale_method`, `weight_block_size`, `weight_scale_semantics` |
| `BitsAndBytesConfig` | `bits`, `method`, `format`, `group_size`, `desc_act`, `sym`, `block_size`, `compress_statistics` |
| `GGUFConfig` | `bits`, `format`, `method`, `group_size`, `desc_act`, `sym`; private `_gguf_bits` |
| `EXL3Config` | `bits`, `method`, `format`, `group_size`, `desc_act`, `sym`, `head_bits`, `out_scales`, `codebook`, `tensor_storage`, `calibration`, `module_include` |
| `NF12ADConfig` | `bits`, `method`, `format`, `group_size`, `desc_act`, `sym`, `exp_base` |
| `TTBlockfloatConfig` | `bits`, `method`, `format`, `block`, `group_size`, `desc_act`, `sym`, `embed_bits` |

| Nested / related config | Declared fields |
|---|---|
| `HessianConfig` | `chunk_size`, `chunk_bytes`, `staging_dtype`, `dedup_shared_inputs` |
| `MSEConfig` | `norm`, `grid_budget`, `clip_grid_budget` |
| `GPTAQConfig` | `alpha`, `device` |
| `FOEMConfig` | `alpha`, `beta`, `device` |
| `GSQConfig` | `enabled`, `steps`, `candidates`, `seed`, `learning_rate`, `temperature_start`, `temperature_end`, `max_candidate_bytes`, `learn_scales`, `modules` |
| `GSQTrainingConfig` | `enabled`, `initializer`, `optimizer`, `seed`, `epochs`, `batch_size`, `microbatch_size`, `qk_steps`, `damp_percent`, `assignment_lr`, `scale_lr`, `weight_decay`, `betas`, `temperature`, `multiplier`, `warmup_steps`, `min_lr`, `decay`, `restore_best` |
| `ActivationConfig` | `mode`, `recipe`, `attention`, `mlp`, `global_scales` |
| `ActivationAttentionConfig` | `mode`, `recipe` |
| `ActivationMlpConfig` | `mode`, `layers` |
| `Fallback` | `strategy`, `threshold`, `smooth` |
| `WeightOnlyConfig` | `method`, `smooth` |
| `BasePreProcessorConfig` | `code` |
| `SmootherConfig` | `code`, `smooth` |
| `AutoModuleDecoderConfig` | `code`, `source_dtype`, `target_dtype`, `passthrough_forward_policy`, `passthrough_save_policy` |
| `TensorParallelPadderConfig` | `code` |
| `SmoothMethod` | `name`, `group_size_threshold` |
| Smoother variants | Percentile: `percentile`; asymmetric percentile: `low`, `high`; MAD: `k`; MSE: `steps`, `maxshrink`; outlier: `pct`; soft norm: `k`; log: `percentile`, `mu`; row/column: `axis` |
| `QuantizeEmbedConfig` | `embed_quant_mode`, `embed_only` |
| `NormConfig` | `teacher`, `types`, `steps`, `lr`, `reg`, `seed`, `skip` |
| `MoEConfig` | `routing` |
| Routing variants | Override: `num_experts_per_tok`; bypass: `batch_size` |
| `TelemetryConfig` | `device` |
| `P2PConfig` | `mode`, `probe`, `probe_mib`, `min_gbps`, `min_ratio`, `use_torch_replicate` |

Related derived/alias properties include `quant_method`, `checkpoint_format`, `runtime_bits`, activation mode/recipe/attention/MLP/global-scale accessors, and BNB `bnb_quant_type`, `bnb_block_size`, `bnb_compress_statistics`. The new model should retain useful read-only views while removing alternative writable names for the same setting.
