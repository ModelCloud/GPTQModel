# Real paired ParoQuant GSQ validation

Llama 3.2 1B layer-1 Q/K/V, seed7, W4/group128/krot8. Clean inputs are dense
FP32 layer-0 outputs; noisy inputs are canonical FP32 outputs from all seven
layer-0 projections of the locked F6 snapshot. Both use identical dense
embedding inputs. Model/source hashes and the 12 training / four initializer
validation / 32 held-out document split were preserved.

Both grouped optimizer paths use two rotation and two finetuning epochs,
followed by 100 GSQ steps. The actual pristine capture context and normal noisy
module hooks collect paired inputs, align batch/invocation IDs, and select
matching training rows. The asymmetric objective omits its candidate-independent
constant and uses noisy-teacher energy normalization. It is not clean-target
normalized MSE. Held-out data never selects a checkpoint.

| Scope | Baseline decoder MSE | Fixed GSQ decoder MSE | Change | Learned-scale GSQ |
|---|---:|---:|---:|---|
| compute_block | 3.30446329789889e-5 | 3.591732274017042e-5 | +8.6934% | Exact baseline payloads |
| layer | 3.240365115429397e-5 | 3.251543618002686e-5 | +0.3450% | Exact baseline payloads |

Fixed GSQ changes all three Q/K/V payloads in both scopes. No recovery or
promotion is claimed. The compute-block regression is substantial relative to
this metric; the smaller layer-scope change has no paired uncertainty estimate.
Final-logit KL/Top-K and task accuracy are not measured. Only selected QKV
modules are quantized in layer 1; this is not a complete-model export.

Both scopes pass all 288 native checks each through actual ParoLinear packing,
save, strict reload and installed native inference. Worst mean/max projection
drift: 0.0004994596 / 0.0064058304 (compute block) and
0.0004994778 / 0.0063552856 (layer), within 0.002 / 0.046875. Decoder MSE
compares native FP16 inference on noisy inputs against dense FP32 decoder
outputs on FP16-cast clean inputs. These propagated metrics are separate from
localized kernel gates.

Executed on leased SM80 PG506-230, physical GPU0, PCI DE:00.0. Evidence:
`artifacts/gsq-paro/real-paired-layer1-seed7-v2/`,
`artifacts/gsq-paro/real-paired-compute-block-seed7-v1/`, and
`artifacts/gsq-paro/real-paired-layer-scope-seed7-v1/`.

The recorded runs internally enabled the previously guarded setting, as stated
in their reports. After these checks, the obsolete configuration rejection was
removed and config round-trip tests passed. The runner now constructs that
public configuration directly. GSQ remains optional and disabled by default;
compatibility does not imply improved accuracy. These results do not verify
full-model capture orchestration, MoE routing alignment, or other GPU families.
