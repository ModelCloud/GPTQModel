# QVQ PGC16 handoff and TODOs

Status as of 2026-08-12: production QVQ accepts fixed `pgc16-v1` only. Learned `pgc16-v2` is retired after repeated
tests showed local/proxy error reductions without dependable held-out final-KLD improvement. The v2 evidence and
design notes below remain historical records; its implementation is isolated under `qvq_codecs/deprecated` and is
not selectable through configuration, lifecycle, loading, or inference. Continue from the latest draft PR #244 tip.

## V2B2-P32 bring-up (2026-08-15)

V2B2-P32 is now the first banked-V2 A/B arm. It uses the same one-byte-per-tile selector payload as V2B4-P64 but
spends it as eight binary P32 decisions. One complementary family is selected per module from the existing three
alternative graph mappings. The initial implementation used a Block-LDLQ/Torch reference recurrence. Commit
`9e420a89` added the exact native CUDA segmented-bank recurrence for both P32 and P64 quantization. The W3/W3.5
extension adds exact E=6/E=7 quantization and packed native CUDA inference; E=7 uses 16-bit traceback because four
banks times 128 predecessor prefixes require nine bits.

- **complete:** format/config/processor/QVQLinear integration, exact coupled P32 recurrence, binary packing,
  module-level alternative-family selection, strict serialization/reload, independent canonical-V2 rollback, and
  a matched four-layer comparison-driver arm;
- **run now:** `v2` versus `v2b2-p32`, W1--W3.5, real Llama 3.2 1B layers 0--3 Q/K/V/O, 64 full
  calibration rows and disjoint rows `[64,128)`, batch 1, no concatenation or length cap;
- **complete -- YAQA level 1:** generate/select P32 schedules from YAQA-corrected tiles, choose the module's
  complementary family under the complete Kronecker proxy, and retain an independently encoded canonical-V2 YAQA
  oracle. Apple quantization automatically dispatches corrected tiles to the native MLX segmented recurrence;
- **run both B2 YAQA controls:** `fixed_block_ldlq` freezes the Block-LDLQ family ID and searches only YAQA path/P32
  selectors; `reselect` evaluates all three complementary families and is the default combined-ceiling arm. Record
  selector churn, family-ID churn, selector entropy, independent V2+YAQA fallback count, and seed-to-seed spread;
- **factorial gate:** compare V2, B2-P32, and B4-P64 under matched Block-LDLQ and YAQA. Report whether the banked
  advantage grows under YAQA, but promote only when banked+YAQA beats independently encoded V2+YAQA on propagated,
  disjoint final-KL/Top-N/task gates;
- **P1 -- propagation:** use live-prefix candidate generation, disjoint replay search, and independent confirmation.
  This is the quality gate that can promote a low-rate selector map; it is not implemented in the base slice;
- **P2 -- native inference:** CUDA and MLX packed P32 decode are complete through W3.5; native MPS remains pending.
- **complete -- spectral P3 reference:** recover the truncated post-YAQA residual through triangular solves, use it
  only as a Viterbi rounding-target push, retain original-weight feedback and original-Kronecker acceptance, expose
  continuous-oracle/absorption/churn telemetry, and serialize only the exact ordinary V2B2-P32 payload;
- **complete -- spectral P3 gates:** the first W2 fixed-family rank-16 gate with `alpha={0.25,0.5,1.0}` used 512
  full calibration rows, disjoint evaluation rows `[512,1024)`, and disjoint YAQA rows `[1024,1536)`. Every one of
  the 16 Q/K/V/O modules rejected every pushed candidate under the original Kronecker objective. The accepted
  artifact consequently had zero selector churn and zero discrete absorption. Its final KL was `0.0575069`, Top-1
  `74.4203%`, Top-5 overlap `77.6291%`, and Top-10 overlap `78.4969%`; the independently executed B2-P32+YAQA
  baseline was `0.0575036`, `74.4098%`, `77.6175%`, and `78.4741%`. Differences at that scale are cross-run
  numerical noise, not spectral recovery.
- **complete -- rejected-candidate P3 sweep:** telemetry added in commit `07ce3e25` distinguishes an unchanged
  proposal from an alternate path rejected by exact rollback. A matched rerun swept ranks `{8,16,32}` and
  `alpha={0.5,1,2,4}`. All `192` module/candidate proposals changed selectors, with mean selector churn
  `49.65--50.02%` and mean state-path churn `99.63--99.96%`. Nevertheless, zero candidates improved the original
  YAQA proxy and zero candidates were selected. Mean original-proxy regressions were:

  | Rank | alpha 0.5 | alpha 1 | alpha 2 | alpha 4 |
  |---:|---:|---:|---:|---:|
  | 8 | +6.42% | +27.89% | +105.82% | +352.03% |
  | 16 | +11.39% | +46.46% | +174.96% | +558.00% |
  | 32 | +19.43% | +76.67% | +267.10% | +838.26% |

  The full 512-row result remained exactly at final KL `0.0575069`, Top-1 `74.4203%`, Top-5 overlap `77.6291%`,
  and Top-10 overlap `78.4969%`. P3 therefore failed because a dense global spectral push creates a near-total
  trellis path avalanche whose error is worse under the original objective, not because the push was too weak to
  cross discrete boundaries. Retire global P3 push as a promotion path. Preserve it as a default-off diagnostic and
  use the lesson to constrain any successor to sparse/localized proposals, explicit trust regions, or live-prefix
  propagation-aware candidate generation and scoring. Raw results:
  `artifacts/qvq_spectral_p3_gate/w2_rank16_push.json` and
  `artifacts/qvq_spectral_p3_gate/w2_r8_r16_r32_a05_a1_a2_a4.json`.

The module alternative ID is physically one serialized byte. Exact artifact accounting must include it even though
its amortized BPW is negligible for real projection matrices.

### W3/W3.5 native extension validation (2026-08-15)

The W3/W3.5 extension was measured on one idle PG506-230 A100-class `sm_80` GPU with CUDA 13.0 and FP16 inputs.
Quantization used one complete 128-step tile and compared the persistent native recurrence with the eager CUDA
recurrence. States and selectors were exact; the diagnostic FP32 objective differed by at most `3.82e-6` because
the native emission uses a different FP32 evaluation order.

| Rate | Format | Native quant | Eager quant | Speedup | States/selectors |
|---:|:---|---:|---:|---:|:---|
| W3 | V2B2-P32 | 6.465 ms | 52.891 ms | 8.18x | exact |
| W3 | V2B4-P64 | 12.253 ms | 48.689 ms | 3.97x | exact |
| W3.5 | V2B2-P32 | 6.491 ms | 48.358 ms | 7.45x | exact |
| W3.5 | V2B4-P64 | 12.109 ms | 54.411 ms | 4.49x | exact |

Packed inference used a representative 2048x2048 projection. The comparison below is against transient dense
reconstruction plus GEMM; a permanently cached dense matrix remains faster but consumes the full dense weight VRAM.
Across batch 1 and 16, max absolute error against an independently reconstructed FP32 dense weight was at most
`4.58e-5`, below the `2e-3` inference contract.

| Rate | Format | Batch | Native | Transient dense | Speedup | Max abs |
|---:|:---|---:|---:|---:|---:|---:|
| W3 | V2B2-P32 | 1 | 0.184 ms | 3.978 ms | 21.58x | 3.43e-5 |
| W3 | V2B2-P32 | 16 | 0.213 ms | 3.197 ms | 15.01x | 3.82e-5 |
| W3 | V2B4-P64 | 1 | 0.184 ms | 3.088 ms | 16.76x | 3.82e-5 |
| W3 | V2B4-P64 | 16 | 0.213 ms | 3.141 ms | 14.75x | 4.58e-5 |
| W3.5 | V2B2-P32 | 1 | 0.243 ms | 3.233 ms | 13.32x | 4.58e-5 |
| W3.5 | V2B2-P32 | 16 | 0.271 ms | 3.247 ms | 11.97x | 4.58e-5 |
| W3.5 | V2B4-P64 | 1 | 0.243 ms | 3.193 ms | 13.16x | 3.82e-5 |
| W3.5 | V2B4-P64 | 16 | 0.271 ms | 3.215 ms | 11.85x | 4.58e-5 |

## V2B4-P64 bring-up (2026-08-15; second banked arm)

The first V2B4-P64 checkpoint slice is intentionally a plain Block-LDLQ control. Do not delay its matched V2 test
for YAQA or propagation integration: adding either now would confound codec geometry with a different rounding or
acceptance objective.

- **P0 -- run now:** compare `v2` against `v2b4-p64` with real Llama 3.2 1B Instruct weights, decoder layers 0--3,
  all Q/K/V/O projections, W1--W3.5, 64 independent full calibration rows, and disjoint full rows `[64,128)`
  for evaluation. Use batch 1, no concatenation, no length cap, identical seeds/Hessians, and the Torch/reference
  reconstruction boundary. Record weight relative-L2/SQNR, Block-LDLQ proxy, local/live QKVO KL, every layer KL,
  final-logit KL, top-1/top-5/top-10, quantization time, effective BPW, selector occupancy/entropy, and repeat parity.
- **P0 safety gate:** bank zero must remain bit-exact V2; packed selectors must reload exactly; V2B4-P64's complete
  full-Hessian proxy must never exceed the independently quantized V2 artifact because the implementation restores
  V2 on ties, non-finite scores, or regressions. Model-level metrics are measurements, not implied by that proxy.
- **complete -- YAQA level 1:** the two-sided corrected tile enters the exact coupled P64 recurrence without resetting
  V2 state at segment boundaries. The complete candidate is compared with an independently encoded canonical-V2
  YAQA artifact under the full Kronecker proxy. Apple quantization uses the native MLX recurrence. Independent Fisher
  data and a disjoint downstream gate are still required before promotion; local Kronecker loss is insufficient.
- **P1 -- propagated bank selection:** start from the exact serialized local V2B4-P64 baseline, propose bounded
  fixed-trellis P64 bank changes, score them using live downstream replay, and confirm the selected map on a second
  disjoint prompt split. Any failure, non-finite output, confirmation regression, or replay omission serializes the
  exact local baseline. Later evaluate joint state re-encoding only if fixed-trellis bank refinement leaves a
  material quality gap.
- **P2 -- native inference:** CUDA and MLX packed P64 decode are complete through W3.5; native MPS remains pending.
  Native kernels match the serialized Torch reconstruction under the 2e-3 inference drift contract.

### P4 localized propagation status

- Implemented: exact fixed-entry/fixed-exit P32 search with at most one changed segment.
- Implemented: disjoint module-output search objective and independently supplied propagation acceptance callback.
- Implemented: callback sees the packed-and-decoded proposal; rejection, callback failure, or serialization mismatch
  restores the exact V2B2-P32+YAQA baseline.
- Implemented: configuration/processor plumbing is default-off and does not auto-split ordinary calibration rows for
  V2B2-P32.
- Validated on Apple P cores: boundary math across W1/W2/W3.5, changed-interior isolation, config roundtrip,
  serialization parity, explicit accept, explicit reject, and callback-error rollback.
- Pending: real-model disjoint search/confirmation gate, four-layer final KL and Top-1/5/10, multiple YAQA seeds,
  callback acceptance rate, and comparison against unchanged V2B2-P32+YAQA.

The first Apple P-core real-data microgate used Llama 3.2 1B layer-0 `q_proj`, its real `out[0:16],in[0:16]`
weight tile, cached YAQA512 factors, 602 valid-token search rows from dataset rows `[1536,1540)`, and 2,100
valid-token confirmation rows from `[1540,1544)`. At W2 the localized search proposed a changed state path without
selector churn. Confirmation MSE regressed from `2.7109018e-5` to `2.7253496e-5`, so the independent gate restored
the baseline exactly. This is a successful rollback/lifecycle check, not evidence of quality recovery and not a
substitute for the pending full-width live-prefix/final-logit gate.

The next Apple P-core gate exercised the complete 2,048x2,048 layer-0 `q_proj` inside a four-layer Llama 3.2 1B
shell at W2. It used the frozen YAQA512 factors from rows `[1024,1536)`, 285 valid search tokens from rows
`[1536,1538)`, independent confirmation rows `[1538,1540)`, and untouched evaluation rows `[1540,1542)`. The
localized search changed exactly one of 131,072 P32 selectors (`7.6293945e-6` churn) and completed in 150.38 seconds.

| Split | Arm | Final-logit KL | Top-1 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Confirmation | V2B2-P32+YAQA baseline | 0.0005097432 | 97.4334% | 97.0877% | 97.4719% |
| Confirmation | P4 proposal | 0.0005078066 | 98.0751% | 97.2160% | 97.4998% |
| Untouched evaluation | V2B2-P32+YAQA baseline | 0.0003955135 | 99.7409% | 98.7737% | 98.0749% |
| Untouched evaluation | confirmed P4 artifact | 0.0003901825 | 99.7409% | 98.6701% | 98.0663% |

Confirmation accepted the serialized proposal: KL improved by 0.38%, Top-1 by 0.64 percentage points, Top-5 by
0.13 points, and Top-10 by 0.03 points. On the untouched final split, KL improved by 1.35% and Top-1 was unchanged,
while Top-5 and Top-10 declined by 0.10 and 0.009 points. This is the first full-width evidence that a fixed-boundary
proposal can survive downstream final-logit confirmation without a path avalanche. It is not a promotion result:
the sample has only two rows per split, the live prefix before layer-0 `q_proj` is dense, and the Top-N evidence is
mixed. Next run multiple disjoint row blocks and YAQA seeds, then repeat after installing the complete live quantized
prefix.

Three follow-up gates tested whether that result survives prompt and YAQA-factor changes. All used W2, exact
serialized proposals, the same four-layer Llama shell, full-length independent rows, batch 1, and a fail-closed
confirmation callback. The two-row repeats used seed-0 YAQA512 factors. The larger matched gates used eight search,
eight confirmation, and eight untouched evaluation rows; promotion required lower confirmation KL and no more than
0.25 percentage points of regression in each Top-1/5/10 metric.

| Gate | Search / confirmation / evaluation rows | Confirmation KL delta | Untouched KL delta | Decision |
| --- | --- | ---: | ---: | --- |
| seed 0 repeat 1 | `[1542,1544)` / `[1544,1546)` / `[1546,1548)` | -0.95% | -0.85% | accepted |
| seed 0 repeat 2 | `[1548,1550)` / `[1550,1552)` / `[1552,1554)` | +0.65% | exact rollback | rejected |
| seed 0 larger gate | `[1554,1562)` / `[1562,1570)` / `[1570,1578)` | -0.76% | -0.77% | accepted |
| seed 1 larger gate | `[1554,1562)` / `[1562,1570)` / `[1570,1578)` | -0.88% | -0.19% | accepted |

The seed-0 larger gate changed one of 131,072 selectors. On untouched rows it changed Top-1/5/10 by -0.18, -0.009,
and -0.005 percentage points. The seed-1 gate changed the state path but not the selector map; its untouched Top-1,
Top-5, and Top-10 changes were -0.07, +0.04, and -0.06 points. These results establish repeatable small KL recovery,
not unconditional quality recovery: the rejected block and mixed Top-N deltas show why independent confirmation and
atomic rollback remain mandatory. The seed-1 Sketch-B cache used the same 512 rows `[1024,1536)` and contained
163,324 valid tokens, so only the categorical Fisher sampling seed changed in the matched seed comparison.

The first live-prefix gate then installed ordinary V2B2-P32+YAQA W2 reconstructions for all layer-0 Q/K/V/O
projections and refined layer-1 `q_proj`. Search rows `[1578,1586)` supplied 3,819 live student-input tokens while
the target remained the dense layer-1 `q_proj` output. Confirmation rows were `[1586,1594)` and untouched evaluation
rows were `[1594,1602)`. The seed-1 proposal changed the V2 state path without changing the P32 selector map and
passed confirmation.

| Split | Baseline final KL | Selected final KL | KL delta | Top-1 delta | Top-5 delta | Top-10 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| confirmation | 0.0107126643 | 0.0107104190 | -0.021% | +0.074 pp | -0.008 pp | -0.036 pp |
| untouched evaluation | 0.0106917837 | 0.0106884246 | -0.031% | +0.096 pp | -0.010 pp | +0.041 pp |

The live-prefix effect is small, but it is the first result in which untouched KL, Top-1, and Top-10 improve together
after upstream W2 quantization error is present. It supports continuing P4 as an opt-in, propagation-confirmed
candidate generator. It does not justify enabling P4 by default: only one target module, one rate, two YAQA seeds,
and one live prefix have passed, and no task benchmark has yet confirmed that these sub-percent logit changes recover
answers.

The research harness now supports a reusable packed prefix artifact so subsequent propagation gates do not need to
requantize the same upstream modules. The artifact is an atomic safetensors file containing only canonical QVQ
`trellis`, FP32 `SU`/`SV`, packed `bank_ids`, `bank_alt_id`, and an optional bias. Its versioned manifest records
format/rate/geometry/provenance plus a SHA-256 for every tensor. Loading rejects mismatched provenance, checksums,
dtypes, shapes, missing tensors, and unexpected tensors. Installation constructs and validates every `QVQLinear`
before replacing any dense module, so a later geometry failure leaves the model untouched.

The real seed-1 W2 layer-0 Q/K/V/O prefix required 410.65 seconds to quantize on the Apple host and produced a
2,721,532-byte artifact. A fresh four-layer shell loaded and installed all four packed modules in 0.0701 seconds;
the first full-row forward then took 0.1595 seconds. Against the same quantized result replayed through dense
reconstructed weights on two untouched rows, packed MPS inference had mean KL `6.8142e-6`, maximum logit difference
`0.0244141`, and 99.6933% Top-1/5/10 agreement. This is bounded FP16 kernel/accumulation drift rather than serialized
tensor drift: every saved tensor is checksum- and bit-validated before installation. The generated artifact remains
an untracked experiment output, not a production full-model checkpoint.

The first gate driven directly from that artifact targeted layer-1 `q_proj` at W2 with YAQA seed 1. It used eight
full search rows `[1602,1610)` (1,406 live-prefix input tokens), eight confirmation rows `[1610,1618)`, and eight
untouched evaluation rows `[1618,1626)`. All rows were batch 1, untruncated, and disjoint from ordinary calibration
and YAQA rows. The localized ranks `{8,16,32}` search took 335.53 seconds, changed the state path without changing
the P32 selector map, and passed the fail-closed confirmation gate.

| Split | Arm | Final-logit KL | Top-1 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| confirmation | rollback | 0.0090825312 | 90.6784% | 90.9517% | 91.2762% |
| confirmation | serialized proposal reconstruction | 0.0090807002 | 90.7760% | 90.9370% | 91.2811% |
| untouched evaluation | rollback | 0.0102990087 | 88.9154% | 89.7946% | 90.1529% |
| untouched evaluation | selected dense reconstruction | 0.0102970986 | 88.8199% | 89.7659% | 90.2198% |
| untouched evaluation | selected packed MPS | 0.0102969826 | 89.0110% | 89.7754% | 90.1768% |

Confirmation KL improved by 0.0202%, with Top-1 +0.0976 percentage points, Top-5 -0.0146 points, and Top-10
+0.0049 points. Untouched dense-reconstruction KL improved by 0.0185%, but Top-1 and Top-5 declined by 0.0956 and
0.0287 points while Top-10 improved by 0.0669 points. Packed MPS remained finite and closely matched the selected
dense reconstruction in KL, but accumulation-order drift moved Top-1/5/10 by +0.1911/+0.0096/-0.0430 points.
This is another accepted, reproducible small-KL proposal, not promotion evidence: the effect is tiny, selector churn
is zero, Top-N is mixed, and one target/split cannot establish task recovery. The validated driver is
`scripts/validate_qvq_p4_live_prefix.py`; raw JSON and the one-module selected artifact remain under the untracked
`artifacts/qvq_p4_promotion/` directory.

The driver now installs multiple compatible artifacts atomically, allowing conditional coordinate refinement rather
than independent module tests. It rejects duplicate modules, mixed codec contracts, source-model mismatches, and a
prefix that already contains the current target before mutating the model. The first sequential gate installed both
the layer-0 Q/K/V/O prefix and the accepted layer-1 `q_proj`, then refined layer-1 `k_proj` on fresh eight-row splits:
search `[1626,1634)`, confirmation `[1634,1642)`, and untouched evaluation `[1642,1650)`. Search captured 2,954 live
input tokens and completed in 80.63 seconds. The proposal changed the state path without selector or family churn.

| Split | Arm | Final-logit KL | Top-1 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| confirmation | rollback | 0.0124357109 | 86.2589% | 88.7390% | 89.4512% |
| confirmation | serialized proposal reconstruction | 0.0124295838 | 86.3008% | 88.7306% | 89.4344% |
| untouched evaluation | rollback | 0.0112255992 | 89.0890% | 90.0353% | 90.3566% |
| untouched evaluation | selected dense reconstruction | 0.0112241216 | 88.9831% | 90.1059% | 90.3566% |
| untouched evaluation | selected packed MPS | 0.0112184444 | 88.9477% | 90.0141% | 90.3566% |

Confirmation accepted a 0.0493% KL improvement with Top-1 +0.0419 percentage points and Top-5/10
-0.0084/-0.0168 points. On untouched rows the selected dense reconstruction improved KL by 0.0132%, decreased
Top-1 by 0.1059 points, increased Top-5 by 0.0706 points, and left Top-10 unchanged. Native packed execution moved
KL another -0.0506% relative to the selected dense reconstruction while moving Top-1/5 by -0.0353/-0.0918 points.
This is useful conditional evidence but still below a promotion threshold: both accepted layer-1 proposals improve
KL by only hundredths of a percent, neither changes the selector map, and their untouched Top-N changes are mixed.
The next sequential arm should test `v_proj` on another fresh split, but the complete Q/K/V/O chain must ultimately
beat the original live-prefix baseline on a common locked confirmation set before any accumulated map is retained.

That `v_proj` arm installed layer-0 Q/K/V/O plus the accepted layer-1 Q/K artifacts, then used search rows
`[1650,1658)` (2,743 live inputs), confirmation `[1658,1666)`, and untouched evaluation `[1666,1674)`. The
three-rank localized search took 81.71 seconds and again changed only the state path, with zero selector/family churn.

| Split | Arm | Final-logit KL | Top-1 | Top-5 | Top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| confirmation | rollback | 0.0149397256 | 87.0996% | 87.9947% | 88.6090% |
| confirmation | serialized proposal reconstruction | 0.0149329457 | 87.1435% | 88.0562% | 88.6134% |
| untouched evaluation | rollback | 0.0131516776 | 87.0335% | 89.2853% | 89.9114% |
| untouched evaluation | selected dense reconstruction | 0.0131537210 | 87.0335% | 89.2347% | 89.9051% |
| untouched evaluation | selected packed MPS | 0.0131574372 | 87.0335% | 89.2094% | 89.8735% |

Confirmation accepted a 0.0454% KL improvement and Top-1/5/10 gains of +0.0439/+0.0614/+0.0044 percentage
points. The independent split reversed that result: dense-reconstruction KL regressed 0.0155%, packed KL regressed
0.0438% versus rollback, and packed Top-5/10 fell 0.0759/0.0380 points while Top-1 was unchanged. This is direct
evidence that an eight-row confirmation gate can overfit even when every confirmation metric improves. Preserve the
raw selected artifact for diagnosis, but do not include it in the sequential prefix. The later `o_proj` arm must
condition on accepted Q/K only, and the rows used here are now development evidence rather than a final locked test.

The conditional `o_proj` arm therefore installed layer-0 Q/K/V/O plus only the accepted layer-1 Q/K artifacts; the
rejected layer-1 V artifact was deliberately excluded. It used search rows `[1674,1682)`, confirmation rows
`[1682,1690)`, and untouched evaluation rows `[1690,1698)`, all batch 1, untruncated, and disjoint from ordinary
calibration and YAQA rows. Search captured 2,397 valid live-prefix input tokens and the rank `{8,16,32}` localized
search completed in 307.19 seconds on Apple P cores.

This gate found no alternate candidate: every localized spectral proposal reconstructed the exact current state path,
P32 selector map, and module family (`proposed=false`, selector churn zero). The fail-closed confirmation callback was
therefore not invoked, and the independently serialized Q/K-conditioned rollback remained selected. Untouched
evaluation streamed 2,621 aligned output positions from 2,629 valid source tokens:

| Evaluation arm | Final-logit KL | Top-1 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rollback dense reconstruction | 0.0168956196 | 84.0900% | 85.9748% | 87.0088% |
| selected dense reconstruction | 0.0168956196 | 84.0900% | 85.9748% | 87.0088% |
| selected packed MPS | 0.0168964521 | 84.0519% | 85.9748% | 87.0317% |

Packed MPS changed KL by only `+0.00493%` relative, Top-1 by `-0.0382` percentage points, Top-5 by less than the
reported precision, and Top-10 by `+0.0229` points. Those are backend accumulation-order measurements, not an
`o_proj` refinement effect. The result excludes layer-1 `o_proj` as a useful candidate under this localized search
contract and split. It also reinforces the architectural limit exposed by the accepted Q/K gates: fixed-boundary P4
can find tiny alternate state paths in some projections, but its present spectral generator does not reliably expose
a new discrete basin for every module. Do not force selector churn or weaken rollback. The next decision should be a
common locked evaluation of the retained Q/K map against independently encoded ordinary-YAQA Q/K, with layer-0-only
kept as a separate precision-cost control, followed by a task-like gate. Do not continue accumulating projection
candidates merely because fresh rows remain available.

The locked comparison harness now streams one full row at a time through one dense teacher and two packed-prefix
students. This avoids caching a complete vocabulary-logit corpus and makes larger batch-1, untruncated gates practical
on Apple. An explicit `--baseline-only` mode in the P4 driver serializes ordinary V2B2-P32+YAQA without entering
localized spectral generation; it must be used instead of an invalid zero-alpha spectral proposal.

The first 64-row control used rows `[1698,1762)` (23,093 valid tokens) and compared the layer-0-only prefix with the
same prefix plus the retained layer-1 Q/K artifacts. Adding two W2 modules raised KL from `0.0084199104` to
`0.0102976438` (+22.30%) and changed Top-1/5/10 by `-0.9510/-0.7608/-0.8342` percentage points. This measures the
expected precision cost of replacing dense Q/K with W2 Q/K; it does **not** measure the P4 refinement and must not be
used to reject P4.

For the matched P4 control, canonical layer-1 Q and K artifacts were independently generated with ordinary
V2B2-P32+YAQA. Canonical K was conditioned on canonical Q, while refined K remained conditioned on refined Q. Both
complete coordinate paths therefore quantize the exact same six modules at W2 and differ only in their P4-selected
Q/K state paths. On the same 64 locked rows:

| Four-layer arm | Final KL | JSD | Top-1 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary YAQA Q/K | 0.0103102394 | 0.0025641023 | 88.8011% | 90.0013% | 90.4112% |
| P4-refined Q/K | 0.0102976438 | 0.0025609845 | 88.8228% | 89.9961% | 90.4182% |
| P4 minus ordinary | -0.1222% | -0.1216% | +0.0217 pp | -0.0052 pp | +0.0069 pp |

The matched map passes the predeclared distribution guard: KL improves and every Top-N change is far inside the
0.25-point non-regression margin. The signal is nevertheless too small for promotion from this shell alone.

A second locked gate then used the complete 16-layer model and fresh rows `[1762,1826)` (21,862 valid tokens). It
kept the same two exact packed coordinate paths and changed only the downstream propagation horizon:

| Full-model arm | Final KL | JSD | Top-1 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary YAQA Q/K | 0.0029644251 | 0.0007304349 | 97.9035% | 96.8144% | 96.7960% |
| P4-refined Q/K | 0.0029592371 | 0.0007290677 | 97.9310% | 96.8043% | 96.7850% |
| P4 minus ordinary | -0.1750% | -0.1872% | +0.0275 pp | -0.0101 pp | -0.0110 pp |

The tiny KL/JSD and Top-1 gains survive all 16 decoder layers; Top-5/10 move slightly backward but remain well inside
the guardrail. This is positive propagated evidence, not task recovery. Retain Q/K as an experimental paired map,
keep V rejected and O unchanged, and advance only to a paired task-like gate using the exact ordinary-versus-refined
artifacts. Do not spend further quantization time accumulating layer-1 modules before that decision.

The paired ARC-Challenge gate is now complete over all 1,172 test examples. It used the repository evaluation
contract (`Question: {question}\nAnswer:`, Llama chat rendering, leading-space choice continuations), one question per
step, all choices batched only within that question, and the complete 16-layer FP16 model on MPS. The two students
quantized the same six W2 modules: layer-0 Q/K/V/O plus layer-1 Q/K. The only difference was ordinary YAQA versus the
retained P4-refined layer-1 Q/K paths. The run was split into rows `[0,256)` and `[256,1172)` without overlap; their
token-contract hashes are `12605a2d10e0a5eef0456aaa483a7431f9bf84f7a3602737208e36698b71aa94` and
`c5e7a141e76481d8ce0f30b59ca9a9183d948da0826c00eaf5517e4e6838ae7d`.

| Full ARC arm | Raw correct | Raw accuracy | Normalized correct | Normalized accuracy |
| --- | ---: | ---: | ---: | ---: |
| dense FP16 | 374/1172 | 31.9113% | 414/1172 | 35.3242% |
| ordinary YAQA Q/K | 367/1172 | 31.3140% | 402/1172 | 34.3003% |
| P4-refined Q/K | 368/1172 | 31.3993% | 402/1172 | 34.3003% |
| P4 minus ordinary | +1 | +0.0853 pp | 0 | 0.0000 pp |

The raw paired comparison changed two predictions: one wrong-to-correct and one wrong-to-different-wrong, with zero
correct-to-wrong flips. The normalized comparison changed one prediction, wrong-to-different-wrong, and therefore
had zero net flips. Mean raw gold margin moved slightly backward (`-4.3896833` to `-4.3898933`), while normalized
margin moved slightly forward (`-1.1053791` to `-1.1051739`). This passes a task non-regression gate and supplies one
exact rescue, but the effect is too small to establish a statistically meaningful task improvement or justify a
default. Preserve the P4 Q/K map as an opt-in candidate; the next useful test is a distinct reasoning/task split or
seed replication, not further selection on ARC.

A first distinct-task gate then evaluated GSM8K rows `[0,64)` with the repository's fixed eight-shot
`gsm8k-cot` prompt, Llama chat rendering, deterministic batch-1 generation, and at most 256 new tokens. It used the
same complete 16-layer dense, ordinary-YAQA, and P4-refined models as the ARC gate. Rows `[0,1)`, `[1,16)`, and
`[16,64)` were executed as non-overlapping shards with token-contract hashes
`8df83cb701c9018ea338b325292d4fd6c3ebb1fd0f8b27a91c528ba25de22b20` and
`f3e4bffe7bf8b7b0d552e48895c0520c22cbe9f45f008b52c97e0dd248662310`, and
`c14612dccbdca6ad3a1a6b4693a0025f188161c112911b35ab3fcdc712a2266c`.

| GSM8K diagnostic arm | Strict correct | Flexible numeric correct | Strict invalid | Flexible invalid |
| --- | ---: | ---: | ---: | ---: |
| dense FP16 | 6/64 (9.375%) | 12/64 (18.750%) | 46 | 28 |
| ordinary YAQA Q/K | 1/64 (1.5625%) | 4/64 (6.2500%) | 55 | 42 |
| P4-refined Q/K | 1/64 (1.5625%) | 5/64 (7.8125%) | 54 | 42 |

P4 changed four flexible extracted answers: one wrong-to-correct rescue, zero correct-to-wrong regressions, and three
wrong-to-different-wrong changes. Strict extraction changed once but produced no correctness flip. The original
16-row rescue survived; the 48-row extension added no correctness flip. This agrees in direction with the one-net-
rescue ARC result, but one rescue in 64 examples and the high invalid-extraction rate remain unsuitable for promotion
or percentage-recovery claims. Preserve the exact generations for paired diagnosis. A larger run with this partial
W2 prefix has poor information return: the next decisive experiment is an independent YAQA/P4 seed replication,
followed by the same locked ARC/GSM8K gates, rather than further prompt or candidate selection on these development
rows.

#### P4 target-Fisher seed replication

The next gate changed the layer-1 Q/K YAQA Sketch-B cache from Fisher seed 1 to seed 0 while holding the exact
layer-0 W2 packed prefix, model snapshot, RHT/quantization seed, row splits, and P4 rank/alpha search fixed. This is
an intentionally controlled **target-factor seed replication**, not a whole-prefix seed replication: rebuilding
layer 0 would change both the upstream quantized activations and the target factors and would therefore confound the
question being tested. Both caches contain 512 independent `neuralmagic/calibration` rows from `[1024,1536)` and
163,324 valid tokens; only the Sketch-B sampling seed differs.

Layer-1 `q_proj` ordinary YAQA took 310.65 seconds. Its P4 search over ranks 8/16/32, alphas 0.25/0.5/1.0, and at
most eight localized P32 segments took 339.62 seconds and accepted a one-state proposal with selector churn
`7.6294e-6`. Layer-1 `k_proj`, conditioned on the selected Q artifact for each coordinate path, took 71.90 seconds
for ordinary YAQA and 79.07 seconds for P4. K's proposal had zero selector churn and failed confirmation, so its
exact ordinary-YAQA rollback artifact was serialized. This is evidence that P4's candidate generator frequently
remains in the same discrete basin; acceptance must not be inferred merely from running the spectral search.

The first locked comparison used the same fresh rows `[1698,1762)` as the seed-1 gate: 64 full, untruncated,
batch-1 rows containing 23,093 valid tokens. Both paths quantized the same six modules; only the seed-0 Q proposal
and the K coordinate re-encoded under its corresponding Q differed.

| Seed-0 four-layer arm | Final KL | Top-1 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: |
| ordinary YAQA Q/K | 0.0103252763 | 89.1268% | 89.9518% | 90.3582% |
| P4 coordinate path | 0.0103286730 | 89.1441% | 89.9587% | 90.3717% |
| P4 minus ordinary | +0.0329% | +0.0174 pp | +0.0069 pp | +0.0135 pp |

The complete 16-layer comparison then used disjoint rows `[1762,1826)` and the same two serialized coordinate
paths. It produced `KL -4.3817e-7`, Top-1 `-0.0184` points, Top-5 `-0.0138` points, and Top-10 `-0.0009` points for
P4 versus ordinary YAQA. Thus the four-layer KL sign reverses after full propagation while the Top-N signs reverse
in the other direction. All changes are minute.

The seed-1 four-layer gate had improved KL by `1.2596e-5`; seed 0 worsened it by `3.3967e-6`. At the full-model
horizon seed 1 improved KL by `5.1879e-6`, while seed 0 improved it by only `4.3817e-7`, and neither seed produced a
consistent Top-1/5/10 direction. The effect is therefore seed-sensitive and near the measurement/decision boundary,
not a replicated recovery signal. Together with the one-net-rescue ARC result and one-net-rescue 64-row flexible
GSM8K diagnostic, this closes the current P4 promotion gate as **neutral/negative evidence**. Keep P4 available as a
diagnostic candidate generator, but do not enable it by default or spend a full task sweep on this formulation.
Future spectral work must first demonstrate materially larger selector/path churn and a predeclared, seed-stable
held-out gain before task evaluation.

#### P5 multi-change fixed-boundary refinement

P5 removed P4's hard experimental cap of one changed P32 segment while preserving the same checkpoint and inference
format. Candidate segments keep their exact entry and exit V2 states. The implementation greedily recomputes each
remaining candidate's conditional gain after every accepted replacement, forbids selecting the same tile/segment
twice, and retains the independently encoded V2B2-P32+YAQA artifact as the atomic rollback oracle. The default is
still one change; `yaqa.spectral_localized_max_changes` only widens an explicitly enabled experiment.

The first W2 Llama 3.2 1B test used the fixed seed-1 layer-0 QKVO prefix, seed-0 512-sequence YAQA factors, ranks
8/16/32, alphas 0.25/0.5/1.0, at most 32 screened segments, and at most four composed changes. Search,
confirmation, and evaluation used the same disjoint full-row splits as the P4 seed replication. Q selected three
fixed-boundary replacements and K selected four. This proves that the former one-change cap was suppressing real
path diversity rather than merely counting duplicate proposals.

Short four-layer selection was unsafe: the combined Q/K coordinate path improved KL by `8.7987e-6` on rows
`[1698,1762)` but worsened complete 16-layer KL by `1.6306e-5` on fresh rows `[1762,1826)`. A second gate therefore
moved proposal confirmation itself to the complete 16-layer horizon. Both modules then failed closed:

| Target | Proposed changes | Full-horizon baseline KL | Full-horizon proposal KL | Decision |
| --- | ---: | ---: | ---: | :--- |
| layer-1 `q_proj` | 3 | 0.0014115776 | 0.0014153905 | reject; serialize exact baseline |
| layer-1 `k_proj` | 4 | 0.0038981916 | 0.0039030055 | reject; serialize exact baseline |

Q's proposal improved confirmation Top-1 but regressed Top-5/10; K improved Top-1/5/10 but still regressed KL.
Neither pattern satisfies the predeclared strict-KL gate. Because both selected artifacts are the independently
encoded rollback baselines, no separate downstream task sweep is warranted. The result is useful negative evidence:
multi-change composition is algebraically valid and generates more distinct paths, but short-horizon acceptance
cannot be trusted. Keep the wider mode default-off and require a complete downstream confirmation horizon for any
future low-rate promotion.

#### P6 bounded full-horizon candidate reranking

P5 showed that full-model confirmation can reject a bundle chosen by module-output search, but confirmation occurs
too late to recover another candidate from the same portfolio. P6 keeps spectral generation and conditional module
loss as a cheap breadth screen, then scores only a bounded top-K shortlist through the complete live quantized model.
The search and confirmation prompt sets remain disjoint. The implementation scores exact packed/decoded candidate
weights, composes winners conditionally, and falls back atomically on callback failure or non-finite baseline score.

The format and inference contract are unchanged. Quantization cost is bounded by
`1 + replay_candidates * max_changes` full-model search forwards plus the existing independent confirmation. The
initial causal gate uses one changed segment and four replay candidates on layer-1 Q, followed by K only if Q passes
confirmation. Promotion still requires strict confirmation KL improvement with bounded Top-1/5/10 regressions; a
better search-split rank alone is not sufficient.

The first 8-row search/8-row confirmation run selected one Q segment. Search KL improved from `0.0035915290` to
`0.0035741710`; confirmation KL changed from `0.0014115776` to `0.0014115307`, and the permissive sign-only gate
accepted it. K found a search winner but failed independent confirmation and restored its exact baseline. On 64
fresh full rows `[1762,1826)`, the resulting Q-plus-rollback-K coordinate regressed KL by `5.1892e-6`, Top-1 by
`0.0138` points, and Top-5 by `0.0046` points while improving Top-10 by `0.0037` points. The 8-row Q confirmation
delta was only `0.0033%` of baseline KL and did not generalize.

This is evidence that full-horizon ranking fixes the objective horizon but not sampling noise. The validation gate
now requires at least 0.1% relative KL improvement by default, so the same Q proposal fails. The next experiment
must use a materially larger disjoint confirmation set; no task evaluation is justified from the 8-row result.

A larger layer-1 Q gate used 32 search rows `[1602,1634)`, 64 confirmation rows `[1634,1698)`, and 64 evaluation
rows `[1826,1890)`, with respectively 10,567, 19,039, and 23,265 valid tokens. Rank `{8,16,32}`, alpha
`{0.25,0.5,1}`, four full-horizon replay candidates, and one changed segment selected `r16_a1_t19_s5` on search:

| Horizon | Baseline KL | Proposal KL | Decision |
| --- | ---: | ---: | :--- |
| 32-row search | 0.0028417924 | 0.0028383114 | provisional winner |
| 64-row confirmation | 0.0028031558 | 0.0028038107 | reject; exact baseline serialized |

The independently evaluated rollback reconstruction had KL `0.0017517403`; the selected packed artifact measured
`0.0017531236`, a packing/backend numerical comparison rather than an accepted quantization change. This larger
split again shows that the pooled search winner is not stable enough to promote. The next bounded experiment uses
two round-robin search folds and the minimax relative score

```text
S(Q) = max_f KL_f(Q) / KL_f(Q0).
```

It reuses the same search prompts and forwards, but rejects any candidate that regresses either fold. This makes the
search criterion more robust at no model-format or inference cost. It does not turn either fold into confirmation;
the disjoint 64-row confirmation gate remains authoritative.

The matched two-fold run selected the same `r16_a1_t19_s5` proposal. It improved both round-robin search folds, but
still failed the independent confirmation horizon:

| Horizon | Baseline KL | Proposal KL | Relative/worst-fold result |
| --- | ---: | ---: | ---: |
| search fold 0 | 0.0024100594 | 0.0024068049 | -0.1350% |
| search fold 1 | 0.0033453477 | 0.0033416025 | -0.1119% |
| pooled search | 0.0028417924 | 0.0028383114 | -0.1225% |
| independent confirmation | 0.0028031558 | 0.0028038107 | +0.0234%; reject |

Confirmation Top-1/5/10 point estimates improved, but the predeclared KL authority regressed, so the exact baseline
was serialized. Quantization took `548.67` seconds. The untouched packed evaluation artifact measured final KL
`0.0017531236`, Top-1 `98.2113%`, Top-5 overlap `97.5648%`, and Top-10 overlap `97.3217%`; these are baseline
metrics because the proposal was rejected. The result excludes a simple two-fold-consensus fix: the candidate's
tiny improvement is consistent across the 32-row search split yet does not transfer to the later 64-row split.
Further spectral P4/P6 promotion work should stop until candidate generation changes materially. The next candidate
generator should search direct bank/path alternatives under live-prefix propagation rather than rescore more
variants of the same EoRA-derived local direction.

One reusable P6 defect was fixed before that next generator: replay mode still required `local_loss < current_loss`
both when forming the shortlist and when validating the composed result. That silently excluded locally worse but
downstream-helpful error directions, contradicting the sub-W3 propagation design. With a full-horizon scorer, finite
local loss now orders the bounded shortlist only; the exact full-horizon score against the immutable baseline is the
selection authority. The strict local gate remains unchanged when replay is disabled. A focused adversarial test
uses a zero-loss local target that rejects every changed path locally and proves that an independently better
full-horizon candidate is still replayed and selected. Independent confirmation remains mandatory and can still
restore the exact baseline.

The matched real-Llama rerun at commit `8c4b6127` produced the exact same four replay evaluations, selected
`r16_a1_t19_s5`, and failed confirmation with the same metrics as the pre-fix run. Quantization took `954.56`
seconds on this Apple run, but the host spent substantial time waiting on Metal; do not interpret the wall-time
ratio as algorithm cost. The unchanged portfolio proves that K=4 was already filled by locally improving candidates,
so removing the veto did not expose a new direction in this configuration. The fix remains required for correctness
and future portfolios, but it does not rescue P6 by itself. The next experiment must diversify candidate generation
or reserve shortlist capacity for direct bank/path alternatives; increasing confirmation rows or repeating the same
local-loss-ordered K=4 portfolio has no new causal value.

#### P7 direct fixed-boundary replay portfolio

P7 implements that materially different generator behind
`yaqa.spectral_localized_direct_replay_candidates`. It ranks P32 segments by the remaining dense-weight residual,
re-encodes each chosen segment directly toward the dense transformed weight under its exact accepted entry/exit
states, removes candidates identical to an existing spectral path, and reserves `D` of the existing `K` replay slots
for the best distinct direct candidates. This does not add replay forwards, checkpoint bytes, or inference work:

```text
K full-horizon replay slots
  |- D direct dense-reencode paths
  `- K-D best remaining local/spectral paths
```

The local score orders candidates but cannot veto them. Full-horizon search, minimum-effect confirmation, and exact
baseline rollback remain unchanged. The first matched gate should repeat the W2 layer-1 Q 32/64/64 experiment with
`K=4`, `D=1`, and two search folds. Compare candidate identities and fold scores against the recorded P6 portfolio
before running another module.

### Completed four-layer V2/V2B2-P32/V2B4-P64 comparison

The V2/V2B4-P64 result was captured from commit `393114880c7032ed9a0c8dd7e938a3d6ca77a96c`. The matched V2B2-P32
result was captured after the exact native CUDA quantization path landed in commit `9e420a89`. All V2B2-P32
quality fields were byte-for-byte equal to its pre-native reference run; only timing changed.

V2 is the baseline at each rate for every delta below. For error/KL metrics, a negative delta is an improvement. For
Top-N agreement, each cell reports relative percent followed by the absolute percentage-point change; positive is
an improvement. V2B2-P32 improves local metrics at every rate, but W1 is the decisive quality regression: layer KL
rises 9.10%, final KL rises 13.55%, and Top-1/5/10 all lose more
than 3.4 points. At W1.5--W2.5 it improves final KL and Top-N, with the strongest layer-KL result at W2.5. V2B4-P64
retains its known W2 counterexample, where local metrics improve while layer and final KL regress. These results
reinforce that local/Hessian selection is not a sufficient unconditional acceptance rule below W3.

| Rate | Arm | EBPW | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Time |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | V2 | 1.00000 | 0.659996 | 0.042760 | 0.074643 | 0.263296 | 0.733819 | 34.18% | 36.83% | 36.74% | 33.46s |
| W1 | V2B2-P32 | 1.03125 | 0.645997 | 0.037820 | 0.070428 | 0.287250 | 0.833278 | 30.69% | 33.40% | 33.27% | 231.25s |
| W1 | V2B4-P64 | 1.03125 | 0.646150 | 0.036563 | 0.065726 | 0.262046 | 0.706435 | 35.00% | 37.78% | 37.63% | 435.46s |
| W1.5 | V2 | 1.50000 | 0.477342 | 0.007191 | 0.017653 | 0.096077 | 0.310906 | 49.63% | 54.11% | 54.49% | 30.41s |
| W1.5 | V2B2-P32 | 1.53125 | 0.466209 | 0.006186 | 0.017011 | 0.088885 | 0.286538 | 50.86% | 55.38% | 56.10% | 254.07s |
| W1.5 | V2B4-P64 | 1.53125 | 0.465909 | 0.006031 | 0.015189 | 0.094804 | 0.283900 | 50.96% | 55.80% | 56.38% | 418.19s |
| W2 | V2 | 2.00000 | 0.343255 | 0.002360 | 0.006637 | 0.029056 | 0.130507 | 64.23% | 67.98% | 68.94% | 33.87s |
| W2 | V2B2-P32 | 2.03125 | 0.334570 | 0.002198 | 0.006596 | 0.026981 | 0.121946 | 64.68% | 68.91% | 69.76% | 222.15s |
| W2 | V2B4-P64 | 2.03125 | 0.334940 | 0.002250 | 0.006151 | 0.032317 | 0.130783 | 63.64% | 67.79% | 68.78% | 410.22s |
| W2.5 | V2 | 2.50000 | 0.245716 | 0.001032 | 0.003082 | 0.016850 | 0.063957 | 72.79% | 76.24% | 77.33% | 31.42s |
| W2.5 | V2B2-P32 | 2.53125 | 0.239230 | 0.000991 | 0.002694 | 0.014070 | 0.060997 | 73.34% | 76.94% | 77.93% | 224.90s |
| W2.5 | V2B4-P64 | 2.53125 | 0.238943 | 0.000992 | 0.002831 | 0.016822 | 0.063667 | 73.22% | 76.47% | 77.35% | 409.71s |

Matched deltas relative to V2 at the same rate:

| Rate | Arm | EBPW | Weight MSE | Proxy loss | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Time |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | V2B2-P32 | +3.12% | -4.29% | -8.98% | -2.12% | -11.55% | -5.65% | +9.10% | +13.55% | -10.22% (-3.49pp) | -9.30% (-3.43pp) | -9.44% (-3.47pp) | 6.91x |
| W1 | V2B4-P64 | +3.12% | -4.27% | -9.20% | -2.10% | -14.49% | -11.95% | -0.47% | -3.73% | +2.40% (+0.82pp) | +2.58% (+0.95pp) | +2.41% (+0.89pp) | 13.02x* |
| W1.5 | V2B2-P32 | +2.08% | -4.67% | -6.50% | -2.33% | -13.97% | -3.64% | -7.49% | -7.84% | +2.47% (+1.23pp) | +2.34% (+1.27pp) | +2.95% (+1.61pp) | 8.35x |
| W1.5 | V2B4-P64 | +2.08% | -4.81% | -6.98% | -2.40% | -16.13% | -13.96% | -1.32% | -8.69% | +2.68% (+1.33pp) | +3.13% (+1.69pp) | +3.46% (+1.88pp) | 13.75x* |
| W2 | V2B2-P32 | +1.56% | -5.05% | -5.80% | -2.53% | -6.88% | -0.61% | -7.14% | -6.56% | +0.71% (+0.46pp) | +1.37% (+0.93pp) | +1.19% (+0.82pp) | 6.56x |
| W2 | V2B4-P64 | +1.56% | -4.79% | -5.51% | -2.42% | -4.68% | -7.32% | +11.22% | +0.21% | -0.91% (-0.58pp) | -0.28% (-0.19pp) | -0.23% (-0.16pp) | 12.11x* |
| W2.5 | V2B2-P32 | +1.25% | -5.31% | -5.58% | -2.64% | -3.99% | -12.58% | -16.49% | -4.63% | +0.75% (+0.55pp) | +0.91% (+0.70pp) | +0.78% (+0.61pp) | 7.16x |
| W2.5 | V2B4-P64 | +1.25% | -5.54% | -5.86% | -2.76% | -3.89% | -8.13% | -0.16% | -0.45% | +0.59% (+0.43pp) | +0.30% (+0.23pp) | +0.03% (+0.03pp) | 13.04x* |

`*` V2B4-P64 timing is from its older Torch-reference quantization run and must not be compared directly with the
native V2B2-P32 timing. V2B2-P32's native recurrence reduced its own matched pre-native wall time by 3.38--3.82x
without changing any quality or selector field.

Selector usage confirms that the mixed arm did not collapse to canonical bank zero:

| Rate | Arm | Nonzero selectors | Entropy (bits) | Bank histogram 0 / 1 / 2 / 3 | Module alt IDs 0 / 1 / 2 / 3 |
|---:|:---|---:|---:|:---|:---|
| W1 | V2B2-P32 | 50.08% | 0.999998 | 654341 / 656379 / 0 / 0 | 0 / 2 / 8 / 6 |
| W1 | V2B4-P64 | 73.21% | 1.996100 | 175570 / 152492 / 151130 / 176168 | n/a |
| W1.5 | V2B2-P32 | 50.04% | 1.000000 | 654850 / 655870 / 0 / 0 | 0 / 6 / 4 / 6 |
| W1.5 | V2B4-P64 | 72.68% | 1.993749 | 179069 / 179071 / 149378 / 147842 | n/a |
| W2 | V2B2-P32 | 49.94% | 0.999999 | 656138 / 654582 / 0 / 0 | 0 / 5 / 3 / 8 |
| W2 | V2B4-P64 | 66.47% | 1.973751 | 219761 / 145758 / 145371 / 144470 | n/a |
| W2.5 | V2B2-P32 | 50.06% | 0.999999 | 654513 / 656207 / 0 / 0 | 0 / 5 / 8 / 3 |
| W2.5 | V2B4-P64 | 71.87% | 1.989493 | 184323 / 182805 / 145263 / 142969 | n/a |

Run contract:

- real Llama 3.2 1B Instruct; decoder layers 0--3; all 16 Q/K/V/O projections;
- 64 independent full calibration rows with 27,455 valid tokens;
- disjoint evaluation rows 64--127 with 20,384 valid tokens;
- batch 1, no concatenation, no length limit, Block-LDLQ, and YAQA/propagation disabled;
- Torch 2.13.0, CUDA 13.0, A100-class SM80 physical GPUs 4--7;
- V2B4-P64 focused reference tests passed 15/15; native segmented-bank CUDA tests passed 31/31 and V2B2/V2B4
  lifecycle/packing tests passed 30/30; all sweep workers exited successfully.

Original artifacts:

- `gpt-qmodel-ultra-v2b4-p64-run/artifacts/qvq_v2b4_p64_4layer/w1_gpu4.json`
- `gpt-qmodel-ultra-v2b4-p64-run/artifacts/qvq_v2b4_p64_4layer/w1p5_gpu5.json`
- `gpt-qmodel-ultra-v2b4-p64-run/artifacts/qvq_v2b4_p64_4layer/w2_gpu6.json`
- `gpt-qmodel-ultra-v2b4-p64-run/artifacts/qvq_v2b4_p64_4layer/w2p5_gpu7.json`
- `/root/qvq-benchmark-artifacts/v2b2-p32-4layer-9e420a89-native/w1_gpu4.json`
- `/root/qvq-benchmark-artifacts/v2b2-p32-4layer-9e420a89-native/w1p5_gpu5.json`
- `/root/qvq-benchmark-artifacts/v2b2-p32-4layer-9e420a89-native/w2_gpu6.json`
- `/root/qvq-benchmark-artifacts/v2b2-p32-4layer-9e420a89-native/w2p5_gpu7.json`

The comparison driver `scripts/compare_qvq_codecs_llama_qkvo.py` now defaults to only `v2` and `v2b2-p32` so the
base result is available quickly. Its other defaults encode the P0 contract above: four layers, rates W1--W2.5,
64 calibration rows, 64 evaluation rows at offset 64, batch 1, and full row lengths. Dual-V2, V4, and L18/V4 remain
explicit optional arms. A CUDA host can launch the matched gate with:

```bash
python scripts/compare_qvq_codecs_llama_qkvo.py \
  --model /path/to/Llama-3.2-1B-Instruct \
  --dataset neuralmagic/calibration \
  --output artifacts/qvq_v2_vs_v2b2_p32.json \
  --device cuda
```

The default `--module-scope qkvo` preserves historical comparability. Use `--module-scope all-linear` to quantize
all seven decoder projections per layer: Q/K/V/O plus MLP gate/up/down. The expanded scope excludes embeddings and
the LM head, records the scope in the report, uses generic `local_modules`/`live_modules` metrics, and creates a
distinct YAQA factor-cache contract. For all 16 Llama 3.2 1B layers, add `--layers 16 --module-scope all-linear`.

## Sketch-B scaling plan for large dense and MoE models

Sketch-B is exact enough for the current Llama 3.2 1B experiments, but its collection architecture does not scale to
200B+ dense or sparsely routed MoE models. For every independent sequence and target linear, the current collector
forms the FP32 per-sequence weight gradient

```text
G_s = gradient_s^T @ activation_s                         [out_features, in_features]
H_I = sum_s(G_s^T @ G_s) / (sequences * out_features)   [in_features, in_features]
H_O = sum_s(G_s @ G_s^T) / (sequences * in_features)    [out_features, out_features]
```

It currently materializes `G` for the whole Sketch-B batch and batched input/output Gram tensors, keeps every
module's dense FP32 factors resident on the GPU for the complete collection pass, performs repeated finite checks
that can synchronize the host, and runs a full teacher forward/backward with activation checkpoint recomputation.
Batching improves throughput, but increases the `G` transient linearly and does not reduce persistent factor memory.
For an `[8192, 2048]` MLP projection at batch 8, `G` alone is 512 MiB; its dense input and output factors add 16 MiB
and 256 MiB. Dense factors across all seven linears of all 16 Llama 3.2 1B layers are already roughly 14--15 GiB.
Applying the same design independently to hundreds of MoE experts can require hundreds of GiB or more.

The work should proceed in accuracy-first phases. Each phase must retain the existing collector as an A/B oracle
until its numerical and downstream gates pass.

### P0: telemetry and reproducible baselines

- Report forward, backward, per-sequence-gradient, Gram-accumulation, transfer, and factor-finalization time.
- Report peak transient VRAM separately from persistent accumulator VRAM, host RAM, transferred bytes, cache size,
  valid token counts, independent sequence counts, and per-expert routed sequence/token counts.
- Add NVTX ranges around collection stages and record factor convergence at configurable row checkpoints.
- Benchmark collection throughput and quality at Sketch-B batches 1, 2, 4, and 8 rather than assuming the largest
  batch is optimal for every module shape.

### P1: exact, workspace-bounded single-GPU collection

- Replace materialized `[batch, in, in]` and `[batch, out, out]` Gram batches with direct FP32 accumulation into the
  two output factors. Use a fused CUDA operator or workspace-bounded cuBLAS schedule that consumes sequence-gradient
  tiles and updates both factors without retaining all intermediate Grams.
- Bound and reuse the `G` workspace. Stream sequence or output tiles when a complete `[batch, out, in]` allocation
  exceeds the allocator-aware budget; never select batch size from a fixed constant alone.
- Fuse padding exclusion, FP32 conversion, and layout preparation where profitable. Padding must contribute exactly
  zero, and one gradient sample must still represent one independent unpadded sequence.
- Replace per-module `.all().item()` finite checks with device-side error flags and at most one host synchronization
  per collection batch.
- Bucket independent full-length sequences by length and use asynchronous pinned-memory transfers. Do not concatenate
  rows or truncate sequence lengths to improve utilization.
- Avoid materializing full `[batch, tokens, vocabulary]` logits when only the teacher loss is needed. Evaluate a
  fused or streamed exact cross-entropy/log-softmax path before accepting an approximation.
- Make activation checkpointing and saved-tensor policy memory-budget-aware. Benchmark recomputation against a
  bounded layer window and CUDA graphs per stable length bucket.

These changes should preserve the current FP32 estimator and accumulation contract first. Any changed reduction
order must be tested against the reference factors and against final serialized QVQ states/selectors, not only text
generation.

### P2: exact distributed and streamed collection

- Accumulate factors where each tensor-parallel, pipeline-parallel, or expert-parallel module is owned. Do not gather
  full per-sequence weight gradients to one device.
- Data-parallel workers should process disjoint independent sequences and reduce only finalized FP32 factor sums and
  integer counts. Perform one deterministic final normalization.
- Keep only a bounded module or layer window resident. Asynchronously spill completed factors to pinned CPU memory or
  NVMe and overlap collection, stabilization, quantization, and eviction.
- Make factor caches self-describing: model revision, tokenizer, ordered dataset rows, valid-token masks, seed,
  module scope, module geometry, dtype, shard topology, route counts, and collector version must all participate in
  the cache contract.

### P3: sparse-MoE correctness and efficiency

The current dense-model invariant that every target module executes once per batch is invalid for routed experts.
The MoE collector must:

- accumulate only the experts actually selected by the router and maintain independent per-expert sequence/token
  counts;
- avoid allocating dense workspaces or factors for inactive experts in a batch;
- define minimum effective-sample and convergence thresholds, with targeted or stratified calibration for rare
  experts;
- support a fail-closed shared or cluster factor for experts that cannot meet the threshold;
- keep shared experts and routed experts separate, and reduce route-aware statistics without changing sequence
  weighting semantics.

### P4: opt-in structured factors for 200B+ MoE

Exact dense factors scale as `O(sum(in_features^2 + out_features^2))` and eventually become mathematically
impractical regardless of kernel speed. Add an explicit scalable mode that evaluates:

- 16x16-aligned block-diagonal factors;
- block-diagonal plus low-rank corrections;
- diagonal plus low-rank factors, `H ~= D + U @ U.T`, with measured ranks such as 32--256;
- one shared factor per expert cluster plus a small expert-specific correction;
- online randomized sketches or Frequent-Directions-style compression that never materializes a dense expert factor.

Structured modes must remain opt-in until they pass factor/subspace error, YAQA proxy, serialized round-trip,
held-out layer/final-logit KL, Top-1/5/10, selector/family churn, and rollback gates. Full dense factors remain the
reference mode for small and medium models.

### P5: adaptive sample allocation

Replace a universal fixed row count with deterministic convergence criteria: relative factor change, principal-
subspace angle, YAQA proxy stability, selector/family Hamming churn, and serialized-state stability. Stop collection
for stable modules and redirect rows to unstable or rarely routed experts. Report the stopping reason and effective
sample count for every factor.

The intended user-visible modes are:

| Mode | Target | Contract |
|:---|:---|:---|
| `exact` | Small/medium models | Current dense FP32 estimator with workspace and synchronization optimizations |
| `distributed_exact` | Large dense models | Same estimator, sharded accumulation, bounded residency, deterministic reduction |
| `scalable_moe` | 200B+ MoE | Route-aware structured factors with explicit quality gates and exact fallback where feasible |

Before merging any performance implementation, record a full A/B table containing wall time, sequences and valid
tokens per second, peak VRAM, host RAM, transfer volume, cache size, factor error, YAQA proxy, exact state/selector
parity where required, final KL, and Top-1/5/10. The immediate highest-return implementation is the exact fused,
workspace-bounded `G` plus dual-Gram accumulation and device-side finite flag; the architectural follow-up is
route-aware distributed accumulation. Structured factors are required, not optional optimization polish, for the
eventual 200B+ MoE target.

## Already validated and pushed

- QVQ is the QTIP-derived quantizer plus this repository's planar PGC16 and backend upgrades. `pgc16-v1` uses the
  fixed Gaussian table; `pgc16-v2` uses one learned model-level table. HYB-Q9 and uniform remain explicit
  regression/debug/benchmark references, not checkpoint formats.
- The software identity is QVQ-only across config, checkpoints, Python, native operators, scripts, and tests. The
  former software identifier is intentionally unsupported; QTIP remains only as attribution to the source paper.
- The deterministic v2 fitter uses Gaussian initialization, Hessian-weighted Lloyd centroids, weighted PAVA,
  strict FP16 freezing, reassignment, and fail-closed weighted-error acceptance.
- V2 stores 256 exact unsigned FP16 patterns in quantization metadata. It adds no module tensor, does not change the
  planar payload, and replaces rather than supplements the shared 512-byte runtime table.
- PGC16 has exactly 65,536 unique two-value vectors.
- Planar W1, W1.5, ..., W8 payloads use one whole transition code per pair and are exactly `rate / 8` bytes per
  weight. Earlier scalar-split streams are intentionally unsupported.
- No codebook tensor is serialized or registered as a module buffer.
- The synthetic distortion gate passes W2 non-regression versus HYB and has no W5--W8 plateau.
- Torch, MPS, and MLX reconstruction parity passes every half-step W1--W8. The current Apple validation also gates
  MSE, relative L2, cosine similarity, forward KLD, top-1, ordered top-5, and repeated-launch determinism.
- The QVQ-only identity migration rerun passes `844` tests with `101` expected CUDA/EXL3 skips on this Apple host;
  the renamed CUDA suite collects all `738` cases, with native compilation and execution still pending on CUDA.
- Learned-table Torch/MPS/MLX tests pass W2/W5/W8 over three seeds and M=1/4, including forward KL `< 2e-5`, exact
  top-1, exact ordered top-5, and ten bitwise-identical repeated launches per Apple backend case.
- QVQ-v2 versus v1 Apple latency passes all 16 measured W2--W5, M=1/4, 2048x2048 MPS/MLX rows. Median v2/v1 is
  `0.995`, range `0.921--1.029`; detailed distributions are generated CI artifacts.
- Rate-specialized, multi-row MPS/MLX PGC16 kernels pass the W2--W5 latency gate across all 168 combinations of the
  three declared shapes, W2--W8, M=1/4/16/32, and both runtimes. W8 is still reported separately against Pangolin.
- CUDA GEMV/WMMA and deterministic split-K inference now have E2--E16 whole-edge planar source and a collected
  half-step test matrix. The preceding scalar-split/integer-rate implementation passed 400 tests on one `sm_80` GPU,
  but the format-breaking E2--E16 kernels still require fresh runtime validation on a CUDA host.
- Fresh CUDA-event measurements cover W2--W8 at M=1/2/4/8/16/32 with K=N=4096. All measured rows retained top-1 and
  top-5 agreement of 1.0 against independent dense PGC16 reconstruction. The DeepSeek V4 Flash W4 shape set also
  passed all 48 rows with maximum MSE `2.39644e-05`, maximum forward KLD `1.92373e-07`, and minimum top-1 `1.0`.
- The reference quantizer accepts `--qvq-device cuda`, `auto` prefers CUDA when available, and unavailable explicit
  accelerators fail closed. The persistent native Viterbi operator improves W2--W8 production throughput by
  5.50--7.94x over a freshly forced eager baseline on the same tip. States, reconstructed values, and reported error
  are bitwise exact across the expanded eager-oracle matrix.
- The `L16/V2` format now supports a quantization-only tail-biting overlap candidate list. Candidate 1 is the exact
  historical two-pass result; wider lists retain it as a tie-preserving fallback. A full small-trellis list matches an
  exhaustive circular-path oracle and a deterministic W2 regression case improves with four candidates. Payload and
  inference are unchanged. Model-level 1/4/8-candidate KLD/top-k/timing evidence remains pending.
- Clean-room YAQA v3 core math is implemented: per-sequence Sketch-B Gram factors, the exact Kronecker proxy,
  input/output RHT Hessian transforms, and two-sided anti-diagonal PGC16 rounding. `H_O=I` is state-for-state identical
  to BlockLDLQ, a fixed-point oracle passes, and a synthetic downstream fixture improves the coupled proxy and KLD
  without top-1/top-5-overlap regression. The diagnostic now collects valid per-sequence real-Fisher Sketch-B factors
  with a full source-model backward while restricting quantization targets to the first N layers; its factors match an
  independent per-sequence autograd oracle and reject averaged gradients. Real-model CUDA promotion evidence and the
  processor lifecycle are still pending. Forward activations alone remain invalid substitutes for `H_O`.
- The first real-model W1/W1.5 diagnostic exposed an invalid 25-sequence Sketch-B population plus a 100x
  paper/official-CLI regularization mismatch. Those pre-fix numbers are not promotion evidence. Production now defaults
  to the v3 paper's `1e-4 * trace(H) / n`, requires at least 2,000 independent sequences, and forms each sequence score
  by summing its valid-token scores. Keep `0.01` only as an explicitly labeled author-code control in future sweeps.
- The corrected 256-sequence memory/quality gate ran on Llama-3.2-1B-Instruct at commits `2c264687`/`3c80d9fc` with
  Python 3.14.6 free-threaded, 256 unpacked `neuralmagic/calibration:LLM` rows `[0, 256)`, 94,939 valid tokens, zero
  padding, and disjoint held-out rows `[256, 384)`. Non-reentrant checkpointing covered all 16 decoder layers while
  collecting FP32 factors for the first two layers' 14 projections. Two independent runs recorded 9.797 GiB peak
  allocated and 12.340 GiB peak reserved CUDA memory; the full Fisher pass took 76.61--77.97 seconds. The matched
  driver-memory peak was 15.318 GiB, about 4.96x below the earlier approximately 76 GiB uncheckpointed observation.
  CPU and CUDA oracles are bit-exact across checkpointed/uncheckpointed factors, three CUDA seeds, repeated launches,
  and stochastic eval modules with RNG preservation.
- Exact diagnostic reduction now has an opt-out native CPU path that bounds temporary memory to one FP32 absolute-error
  scratch plus 16M-element distribution chunks. It preserves the complete MAE/RMSE/relative-L2/SQNR/cosine/Pearson,
  exact interpolated p50/p95/p99, forward/reverse KLD, JSD, TV, Hellinger, entropy, cross-entropy, and top-k surface.
  On the dual-socket EPYC 7V13 host, an 8,388,608-value raw-logit microgate fell from 1.3717 to 0.3662 seconds
  (3.75x); the largest reference/native delta across normal, constant, zero, tied, noncontiguous, and odd-width cases
  was `4.7684e-7`. Python 3.14.6 free-threaded validation passes 35/35 native-metric and extension API tests, including
  concurrent calls. The real 788M-value timing gate remains pending before this is called a production-scale win.

  | W1 method | payload/effective BPW | weight rel-L2 | local KLD | live KLD | final KLD | top-1 | top-5 | quant seconds |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | BlockLDLQ | 1.00000 / 1.02317 | 0.63925 | 0.045479 | 0.153354 | 0.475084 | 0.7134 | 0.6867 | 98.66 |
  | YAQA paper `1e-4` | 1.00000 / 1.02317 | 0.87890 | 0.132828 | 0.264470 | 1.201018 | 0.5187 | 0.5335 | 196.23 |
  | YAQA author `1e-2` | 1.00000 / 1.02317 | 0.75360 | 0.082554 | 0.194324 | 0.489866 | 0.7083 | 0.6805 | 194.02 |
  | YAQA `2e-2` | 1.00000 / 1.02317 | 0.72135 | 0.076346 | 0.185311 | 0.445371 | 0.6966 | 0.6953 | 202.09 |
  | YAQA `3e-2` | 1.00000 / 1.02317 | 0.70146 | 0.073583 | 0.179794 | 0.463098 | 0.6919 | 0.6917 | 203.19 |

  `2e-2` improves final KLD by 6.25% but loses 1.68 top-1 points and regresses weight/local/live error versus
  BlockLDLQ. `3e-2` improves the local metrics versus `2e-2` but gives back final KLD and top-1. YAQA therefore remains
  opt-in: checkpointing is a safe memory optimization, but no tested W1 regularization is an accuracy-uniform default.
  The matched W1.5 gate also rejects paper `1e-4`: final KLD rose from 0.209820 to 0.258883 and top-1 fell from 0.8317
  to 0.7760. Final-logit comparisons use the combined full-16-layer run; a standalone BlockLDLQ diagnostic truncates
  to the requested target depth and its final logits are not cross-depth evidence.

### Matched 512/1024-sequence YAQA regularization sweep (2026-08-12)

Commit `b97a3b78` was measured with Python 3.14.6 free-threaded (`PYTHON_GIL=0`) on isolated PG506-230 `sm_80`
GPUs. Every arm quantizes the first two Llama-3.2-1B-Instruct decoder layers (14 projections) while loading and
evaluating all 16 layers. Calibration is unpacked, batch size one, and uses either rows `[0, 512)` (188,256 valid
tokens) or `[0, 1024)` (375,715 valid tokens). All arms use the same disjoint held-out rows `[1152, 1280)` with
6,133 valid tokens and 11 padding positions excluded. The final metrics therefore reduce the same 786,594,048 raw
vocabulary logits and can be compared directly across calibration sizes. YAQA remains opt-in.

| Rate | Calibration | Arm | Fwd KLD | JSD | RMSE | Rel-L2 | Top-1 | Top-5 | Quant s | Infer s | Metric s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 512 | BlockLDLQ | 0.465712 | 0.094010 | 1.210902 | 0.377589 | 0.7023 | 0.6934 | 86.41 | 3.15 | 48.73 |
| W1 | 512 | YAQA `4e-2` | 0.408032 | 0.087293 | 1.124899 | 0.350771 | 0.7235 | 0.6998 | 187.44 | 8.12 | 49.78 |
| W1 | 512 | YAQA `6e-2` | 0.448965 | 0.095010 | 1.218080 | 0.379827 | 0.7215 | 0.6877 | 210.05 | 3.47 | 54.02 |
| W1 | 512 | YAQA `8e-2` | 0.376143 | 0.080521 | 1.130377 | 0.352479 | 0.7158 | 0.7170 | 182.21 | 2.81 | 47.32 |
| W1 | 512 | YAQA `10e-2` | 0.387382 | 0.082592 | 1.097018 | 0.342077 | 0.7001 | 0.7159 | 183.80 | 3.78 | 49.85 |
| W1 | 1024 | BlockLDLQ | 0.454664 | 0.095325 | 1.163811 | 0.362905 | 0.6913 | 0.6890 | 79.56 | 4.50 | 59.19 |
| W1 | 1024 | YAQA `4e-2` | 0.421831 | 0.089778 | 1.128325 | 0.351840 | 0.6848 | 0.7070 | 183.72 | 4.38 | 53.62 |
| W1 | 1024 | YAQA `6e-2` | 0.431145 | 0.093552 | 1.136460 | 0.354376 | 0.7274 | 0.6901 | 192.69 | 3.42 | 55.98 |
| W1 | 1024 | YAQA `8e-2` | 0.390948 | 0.085316 | 1.093929 | 0.341114 | 0.7098 | 0.7109 | 191.09 | 3.50 | 53.69 |
| W1 | 1024 | YAQA `10e-2` | 0.436659 | 0.093145 | 1.142395 | 0.356227 | 0.7045 | 0.6996 | 192.13 | 3.08 | 54.94 |
| W1.5 | 512 | BlockLDLQ | 0.204621 | 0.043625 | 0.777080 | 0.242313 | 0.7895 | 0.7945 | 84.32 | 2.99 | 48.67 |
| W1.5 | 512 | YAQA `4e-2` | 0.155682 | 0.035631 | 0.727987 | 0.227004 | 0.8203 | 0.8059 | 180.48 | 3.42 | 48.49 |
| W1.5 | 512 | YAQA `6e-2` | 0.165372 | 0.036729 | 0.747690 | 0.233148 | 0.8273 | 0.8005 | 208.43 | 3.47 | 53.62 |
| W1.5 | 512 | YAQA `8e-2` | 0.188876 | 0.042684 | 0.807178 | 0.251698 | 0.7707 | 0.7961 | 189.21 | 2.94 | 56.21 |
| W1.5 | 512 | YAQA `10e-2` | 0.159724 | 0.035846 | 0.738067 | 0.230147 | 0.8048 | 0.8093 | 187.81 | 2.96 | 55.25 |
| W1.5 | 1024 | BlockLDLQ | 0.177956 | 0.040593 | 0.714123 | 0.222681 | 0.7880 | 0.8015 | 78.06 | 2.84 | 49.92 |
| W1.5 | 1024 | YAQA `4e-2` | 0.157901 | 0.035435 | 0.736676 | 0.229714 | 0.7844 | 0.8116 | 180.62 | 3.49 | 57.52 |
| W1.5 | 1024 | YAQA `6e-2` | 0.150444 | 0.034291 | 0.709522 | 0.221246 | 0.8003 | 0.8133 | 181.60 | 3.02 | 50.50 |
| W1.5 | 1024 | YAQA `8e-2` | 0.144920 | 0.033020 | 0.801129 | 0.249812 | 0.8280 | 0.8232 | 180.10 | 3.03 | 49.99 |
| W1.5 | 1024 | YAQA `10e-2` | 0.154336 | 0.034863 | 0.694458 | 0.216549 | 0.8260 | 0.8055 | 181.06 | 3.17 | 54.54 |

At W1, `8e-2` is the best final-distribution setting for both calibration sizes, but increasing calibration from
512 to 1,024 sequences does not improve it: KLD rises from 0.376143 to 0.390948 and top-1 falls 0.60 points. `6e-2`
is the only W1 YAQA arm that improves both KLD and top-1 with more calibration, and it still trails the 512-sequence
`8e-2` KLD. At W1.5, 1,024 sequences materially help `6e-2` and `8e-2`; `8e-2` reaches the best final KLD/JSD/top-k
of the sweep, while `10e-2` has the lowest RMSE/relative-L2. The disagreement again shows that raw-logit norm and
final-distribution quality must remain separate gates. There is no monotonic "more rows is better" result and no
single regularization is accuracy-uniform across W1 and W1.5, so neither YAQA nor a regularization value is promoted
to a default.

The native exact CPU metric path processed each 786.6M-logit arm in 47.32--59.19 seconds. This closes the earlier
production-scale timing gap while preserving the complete metric definitions; post-quant inference itself remained
2.81--8.12 seconds and is reported separately.

### Full-model W1.5 GSM8K quality failure (2026-08-12)

Commit `1db90c0b` fixed FP16 overflow in the delayed-normalization Hadamard and commit `3275281d` aligned the lifecycle
validator with QVQ's native FP16 inference contract. The five existing 512-row, full-16-layer Llama-3.2-1B-Instruct
W1.5 snapshots were then screened on the same first 32 GSM8K-Platinum CoT rows with eight few-shots, the Llama chat
template, greedy 256-token generation, and the format-insensitive numeric scorer. These are diagnostic slices, not
full benchmark estimates.

| Arm | Dtype | Attention | Batch | Correct | Invalid | Accuracy |
|---|---|---|---:|---:|---:|---:|
| Dense | FP16 | SDPA | 16 | 19/32 | 0 | 0.5938 |
| Dense | FP16 | FlashAttention 2 | 16 | 20/32 | 0 | 0.6250 |
| BlockLDLQ | FP16 | FlashAttention 2 | 16 | 2/32 | 0 | 0.0625 |
| BlockLDLQ | FP16 | SDPA | 16 | 2/32 | 0 | 0.0625 |
| BlockLDLQ | FP16 | FlashAttention 2 | 1 | 2/32 | 0 | 0.0625 |
| BlockLDLQ | BF16 | FlashAttention 2 | 16 | 2/32 | 0 | 0.0625 |
| YAQA `4e-2` | FP16 | FlashAttention 2 | 16 | 0/32 | 0 | 0.0000 |
| YAQA `6e-2` | FP16 | FlashAttention 2 | 16 | 0/32 | 0 | 0.0000 |
| YAQA `8e-2` | FP16 | FlashAttention 2 | 16 | 1/32 | 0 | 0.0312 |
| YAQA `10e-2` | FP16 | FlashAttention 2 | 16 | 0/32 | 0 | 0.0000 |

The saved BlockLDLQ checkpoint's CUDA decoder is bit-exact against an independently decoded FP16 weight matmul for
all 112 QVQ modules and for final logits. A direct left-padding check across 16 prompts and 731 padding tokens is also
bit-exact between batched and one-prompt inference: KLD is numerical zero, relative-L2 is zero, and top-1 agreement is
1.0. Dense FlashAttention 2 scores one answer above dense SDPA. Decoder/packing, padding masks, batching, attention,
and FP16-versus-BF16 therefore do not explain the collapse.

The checkpoint already failed a meaningful output-quality gate at creation. Its four-prompt lifecycle artifact
recorded forward KLD `0.720624`, relative-L2 `0.556995`, and top-1 agreement `0.567568`, but the diagnostic invocation
used `max_forward_kld=100` and `min_top1_agreement=0`, so serialization parity was incorrectly treated as model-quality
acceptance. On an 838-token GSM8K prompt, all-token dense-versus-QVQ KLD is `0.696374` mean, `0.308554` p50,
`2.425545` p95, and `7.352506` maximum; top-1 agreement is `0.713604`. Hidden-state error accumulates to layer-15
relative-L2 `0.589314` and final-token relative-L2 `0.889869`. Full-model W1.5 is therefore not accuracy-valid with
these settings even though the checkpoint is coherent and reloads exactly. Do not spend full Evalution suites on
these five snapshots or promote their YAQA settings. Future full-model snapshots must pass a nontrivial held-out
KLD/top-k gate before task evaluation; two-layer W1.5 results do not establish full-model viability.

A dense-suffix causal sweep further rules out one catastrophically broken decoder layer. Starting with a dense model,
the sweep retained QVQ layers `[0, N)` and replaced every layer from `N` onward with its dense FP16 counterpart. On
the same 838-token prompt, mean KLD rises smoothly from `0.020718` at `N=1`, through `0.203551`/`0.412986` at
`N=5`/`N=8`, to `0.696466` at `N=16`; top-1 agreement falls from `0.9547` to `0.8461`, `0.7936`, and finally
`0.7112`. The largest adjacent KLD increments are layers 7 and 8 (`+0.073507` and `+0.096021`), but no layer accounts
for the collapse. This is distributed, cumulative W1.5 error. Improving isolated CUDA decode or packing code cannot
recover it; the next viable experiment must improve the offline objective/error propagation or raise the rate, then
re-run a held-out full-model gate before Evalution.

### Full-model W4 sanity control (2026-08-12)

Commit `3b2052a6` was quantized end to end as a W4 BlockLDLQ/PGC16-v1 control using all 16
Llama-3.2-1B-Instruct decoder layers and all 112 linear modules. Calibration used 512 unpacked
`neuralmagic/calibration:LLM` rows `[0,512)`, batch size 1, 188,256 valid tokens, no padding, FP16 inference, Python
3.14.6 free threading, Torch 2.13.0+cu130, and one PG506-230 GPU. Quantization took 463.890 seconds; save and reload
took 1.150 and 5.739 seconds. The checkpoint is retained at
`/monster/data/model/Llama-3.2-1B-Instruct-QVQ-W4-block-cal512-3b2052a6`.

An exact full-vocabulary gate used the disjoint 128-row slice `[512,640)`, a 48-token cap, 6,125 valid tokens, and
excluded all 19 padding tokens. Metrics compare the reloaded W4 model directly with dense FP16 logits.

| MAE | RMSE | Relative L2 | SQNR dB | Cosine | KLD | JSD | Top-1 | Top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.270084 | 0.385760 | 0.120815 | 18.3576 | 0.992748 | 0.026425 | 0.006547 | 0.891592 | 0.918498 |

The full 1,209-row GSM8K-Platinum CoT suite used the Llama chat template, eight few-shots, greedy 256-token
generation, FP16, FlashAttention 2, and batch size 32. Dense and QVQ consumed the same row identities. QVQ was
scheduled as eight disjoint row-index shards on eight identical PG506 GPUs; summing the exact per-row scores is
equivalent to one sequential suite. QVQ consumed 2,260.145 aggregate GPU-seconds and 305.135 seconds of parallel
wall time; dense consumed 236.998 GPU/wall seconds on one GPU.

| Arm | Correct | Invalid | Accuracy | Dense retention |
|---|---:|---:|---:|---:|
| Dense FP16 | 593/1,209 | 0 | 0.490488 | 1.000000 |
| QVQ W4 | 573/1,209 | 0 | 0.473945 | 0.966273 |

W4 retains 96.63% of dense GSM8K accuracy and passes the predeclared 80% sanity threshold. Together with low KLD,
89.16% token top-1 agreement, exact live/reload parity, finite outputs, and zero invalid GSM8K extractions, this rules
out a general catastrophic QVQ lifecycle, packing, or inference failure. The W1.5 failure above is specific to
cumulative ultra-low-rate quantization quality. The W4 control also exposes a separate inference-performance issue:
QVQ generation used 9.54x dense GPU-seconds per sample, which does not affect this accuracy conclusion.

Primary evidence:

- `docs/qvq.md`
- `docs/qvq_cuda.md`
- `docs/qvq_compander_two_layer_comparison.csv`
- `docs/qvq_two_layer_w2_comparison.csv`
- `docs/qvq_two_layer_w3_comparison.csv`
- `docs/qvq_two_layer_w4_comparison.csv`
- `docs/qvq_two_layer_w5_comparison.csv`
- `docs/qvq_two_layer_w6_comparison.csv`
- `docs/qvq_two_layer_w7_comparison.csv`
- `docs/qvq_two_layer_w8_comparison.csv`
- commits `39ca1381`, `9b0c71c9`, `03dc6a16`, `3f0bf39b`, and `edad9e1a`

## Pending: validate the whole-edge half-step CUDA format

On a CUDA host, build the changed native extension and run `tests/test_qvq_cuda.py` across all 15 public rates. Gate
E2--E16 against Torch reconstruction for FP16/BF16, structured and random path codes, constrained/unconstrained
Viterbi, KLD, top-1, top-5, determinism, non-default streams, split-K, WMMA, and concurrent devices. Record compiler,
PyTorch/CUDA, compute capability, SM count, shapes, latency, and throughput. Until that passes, the earlier integer-rate
CUDA evidence remains historical and must not be treated as validation of the new serialized layout.

## Completed: validate the PGC16 CUDA inference port

The inference-port validation above is complete. Preserve its test and benchmark coverage when changing the kernel.
The exact environment, commands, full W2--W8 table, and real projection-shape results are in `docs/qvq_cuda.md`.

Future CUDA changes must continue to preserve:

- the native operator schema now receives a process-cached 256-element level tensor, not a `(512, 2)` module LUT;
- exact xor-MAD-xor unsigned 16-bit mixing parity with Torch;
- BF16 conversion of the frozen FP16 canonical levels;
- the M>=32 Ampere WMMA path and FP32 accumulation;
- deterministic split-K reduction for narrow projections;
- no `tlut`, `lut`, `levels`, or `codebook` tensor in a QVQ layer state dict or checkpoint.

## Completed: make CUDA available to the reference quantizer

The diagnostic CLI now accepts `cuda`; `auto` resolves CUDA, then MPS, then CPU. Explicit unavailable accelerators are
rejected before model quantization. The native CUDA batch map is now independently measured for every half-step from
W1 through W8. The CUDA benchmark and full throughput/VRAM table are in `scripts/benchmark_qvq_viterbi.py` and
`docs/qvq.md`.

The persistent kernel preserves trellis math and exactly reproduces eager FP32 arithmetic. CUDA tests executed on
PG506-230 `sm_80` GPUs 6 and 7 under Python 3.14.6 free-threaded. The direct oracle gate now spans W1--W8,
three seeds, constrained/unconstrained search, production tail biting, and ten repeated launches. It requires bitwise
states, values, and `squared_error`; the two-device and non-default-stream tests also pass.

The current MPS defaults are W2/W3=32, W4/W5/W6=64, W7=256, and W8=512. They are scheduling choices, not part of
`pgc16-v1`.

The low-rate CUDA work raised the exact W1/W1.5 scheduling defaults from 16 to 496 after matched sweeps on two
PG506-230 `sm_80` GPUs. W1 replaced its 64-KiB shared suffix array with ping-pong FP32 costs, and both rates now store
their transient 2/3-bit predecessor indices losslessly in `uint8`. W1 reaches 11,527.7 tiles/s (7.15x over the original
1,611.9) with 1.21 GiB peak transient allocation; W1.5 reaches 11,902.3 tiles/s (10.58x over 1,124.8) with 0.60 GiB.
Full selected batches are bitwise identical to batch-16 chunks for states, reconstructed values, and squared error
under Python 3.14.6 with `PYTHON_GIL=0`; W1 and W1.5 also pass concurrent two-device replay. Next, fuse provisional
overlap discovery without changing quantization math and use expert waves to keep 496-sequence launches full.

The W2.5--W8 follow-up is complete on the same two `sm_80` GPUs. Rate-specific batches add 1.02--1.09x over the prior
defaults at W2.5--W7.5 with exact paths and zero measured loss delta. W2.5's lossless `uint8` backpointers cut batch-496
workspace from about 619 MiB to 251 MiB. W8 now exploits its zero-overlap recurrence: it carries one scalar survivor
cost and one selected state per step instead of 65,536 costs and backpointers. This reaches 95,383.8 tiles/s at batch
2,976, 3.86x over the immediately preceding CUDA kernel/default and 22.49x over eager, while a compiled old/new A/B is
bitwise exact. Production PGC16 long-path tests cover weighted constrained/unconstrained search and remain exact under
Python 3.14.6 with `PYTHON_GIL=0`.

The subsequent W2.5--W7.5 fused recurrence removes the redundant 65,536-entry global cost round trip at every step.
Exact suffix minima are ping-ponged in shared memory while predecessor storage, FP32 addition order, and tie rules stay
unchanged. A compiled old/new gate is bitwise exact across every rate and constrained mode. At batch 3,968 the rates
reach 41,774.5--48,059.6 tiles/s, a 3.13--3.62x gain over the preceding CUDA implementation and 21.18--27.96x over
available integer-rate eager anchors. Peak workspace ranges from 27.2 MiB at W7.5 to 1,991.8 MiB at W3.

The warm-kernel one-layer Llama-3.2-1B integration run used all seven projections and the full 128-row calibration
protocol. Quantization took 46.6845 seconds at W1 and 46.7440 seconds at W1.5; module validation took 0.1354/0.1487
seconds and replay 0.0143/0.0154 seconds. The disjoint-row smoke metrics remained finite (KLD 0.22330/0.04223,
top-1 0.8750/0.9375). Use the existing 128-row quality matrix for promotion decisions; this run is phase-timing and
lifecycle evidence.

## Completed: regenerate matched two-layer W2--W8 quality data

Use two Llama 3.2 1B decoder layers and all 14 q/k/v/o/gate/up/down projections for the interim gate. Run each bit as
a separate job so every completed rate can be validated, committed, and pushed independently:

```bash
python scripts/analyze_gptq_low_bit_grid.py \
  --model <local-llama-3.2-1b-instruct> \
  --layers 2 \
  --method both \
  --bits <2-through-8-one-at-a-time> \
  --symmetry both \
  --qvq-device cuda \
  --threads <measured-host-setting> \
  --json-out /tmp/qvq_two_layer_w<bits>_comparison.json \
  --csv-out docs/qvq_two_layer_w<bits>_comparison.csv
```

Calibration must use the first 128 rows from `/monster/data/model/dataset/nm-calibration`, configuration `LLM`, split
`train`, through the production preparation path with concat size 2,048. The current preparation yields 25 batches,
49,725 valid tokens, and 1,475 padded slots. Masked positions must contribute neither Hessian terms nor sample count,
and module/layer/final-logit metrics must contain only valid positions. The calibration forward uses the decoder
backbone and must not collect LM-head logits.

Each result must compare the same dense baseline and evaluation prompts across:

- GPTQ symmetric;
- GPTQ adjacent-asymmetric;
- PGC16 QVQ.

Preserve per-module local and live metrics, per-layer metrics, and final raw-logit metrics. At minimum report MAE,
RMSE, relative L2, SQNR, cosine, Pearson, forward/reverse KL, Jensen--Shannon, total variation, Hellinger,
entropy/cross-entropy, top-1, top-5 overlap/exact agreement, and both top-1-in-top-5 directions. Combine the seven
per-bit summaries into one table only after checking settings and dense-baseline identity.

The previous W2--W8 reports were invalid because they used four synthetic calibration strings and included padding in
the Hessian and metric populations. All seven artifacts have now been overwritten under the replacement contract.
Their 21 arms are finite. W2 reports raw-logit KLD `7.326883`/`1.056947`/`0.104966`; W3 reports
`0.205386`/`0.199719`/`0.015648`; W4 reports `0.024610`/`0.022460`/`0.003754`; W5 reports
`0.005283`/`0.005315`/`0.001204`; W6 reports `0.006068`/`0.001835`/`0.000360`; W7 reports
`0.001391`/`0.000757`/`0.000114`; W8 reports `0.000292`/`0.000227`/`0.000040` for symmetric GPTQ,
adjacent-asymmetric GPTQ, and PGC16 QVQ respectively. W3 symmetric/asymmetric top-1 is `0.9545`/`0.9773`, W4
symmetric is `0.9773`, and W5 asymmetric is `0.9773`; PGC16 retains top-1 `1.0` at every rate.

## P1: implement the Ultra lifecycle

QVQ is integrated into `GPTQModel.quantize()`. The list below is retained as its implementation checklist. Fixed-v1
BlockLDLQ is complete, and dense-model YAQA now has an opt-in full-model Fisher prepass for W1/W1.5 testing.

1. Add a dedicated `QVQProcessor`; never let QVQ fall through to `GPTQProcessor`.
2. Move the diagnostic's validated calibration collectors into the processor. Collect each linear module's input
   covariance/Hessian. For `rounding="yaqa"`, keep the complete source model for the real-Fisher backward, retain the
   sequence axis through `H_I=E[G.T@G]/out_features` and `H_O=E[G@G.T]/in_features`, ignore dataset labels, and disable
   CUDA TF32. Never truncate downstream layers or square an averaged minibatch gradient.
3. Prefer layer replay so later Hessians observe the intended quantization state; document any dense-only capture
   approximation.
4. Call `quantize_qvq_linear` and replace eligible `nn.Linear` modules with the appropriate QVQ QuantLinear.
5. For `pgc16-v2`, collect one model-level transformed-vector population with diagonal Hessian importance, fit/freeze
   one shared compander, and set `compander_bits` before module quantization. Do not fit per-module tables.
6. Persist only `trellis`, `SU`, `SV`, and optional bias plus the PGC16 version metadata and, for v2, model-level
   `compander_bits`. Never persist a codebook tensor.
7. Integrate backend selection for Torch, MPS, CUDA, and MLX where applicable, with explicit capability checks and
   portable fallback behavior.
8. Implement save, reload, inference, resharding compatibility, and state-dict validation.
9. Add a tiny quantize/save/load/reload/inference test and a two-layer Llama 3.2 1B integration test.
10. Only then remove `METHOD.QVQ` from `QUANTIZE_BLACK_LIST` and add QVQ to required lifecycle routing.

## QVQ W2 scale-optimization roadmap

Scale optimization has three materially different storage and runtime contracts. Do not call all three "group-scale
optimization." Treat the format-compatible variants as independently switchable quantization behavior, not as
`pgc16-v3`, `pgc16-v4`, or another decoder-table version. `pgc16-v1` and `pgc16-v2` continue to identify only the exact
PGC16 decoder table contract. Every experiment must retain a baseline arm with all scale toggles disabled.

### Level 0: module scale, current format

`SV[out]` currently contains the output signs multiplied by one shared module scalar. For a fixed decoded dense matrix
`Q`, optimize that scalar with
`alpha = trace(Q H W^T) / trace(Q H Q^T)`. If `Q` already includes the current scale, the result is a multiplicative
correction; if `Q` is the unit-scale reconstruction, it is the absolute scale. State which convention an implementation
uses. This changes no checkpoint tensor, inference operation, or kernel contract, but its single degree of freedom is
likely to provide only a small W2 gain.

Implemented as the default-off `module_scale_search` control. The implementation uses the multiplicative convention:
`Q` already includes the baseline module scale. It fits the scalar against the damped input Hessian, reconstructs and
scores the exact checkpoint-dtype `SV` against the original undamped Hessian, tries one complete re-encoding at the
proposed scale, and retains the best of the original, fixed-trellis, and re-encoded candidates. It stores only the
existing `SV`, so payload, auxiliary bytes, reconstruction kernels, and inference FLOPs are unchanged. It is rejected
with YAQA until selection uses YAQA's full Kronecker objective rather than only the input Hessian.

### Level 1: output-channel scales in `SV`, current format

The runtime applies `matmul_hadU(output) * SV` after the left Hadamard, so `SV[out]` can hold an independent signed scale
for each output channel without adding storage or an inference operation. For fixed decoded row `Q_r`, solve
`alpha_r = (Q_r H W_r^T) / (Q_r H Q_r^T)` independently for every output row, retain the existing sign in `SV[r]`, and
guard non-finite values, non-positive denominators, and non-positive scales. The closed-form update is exact only while
`Q` is fixed. Because changing `SV` changes the weights presented to the encoder, joint optimization must re-encode or
perform a bounded alternating search. The current `rht_preprocess_weight()` helper assumes `SV` contains only signs and
multiplies by it because a sign is its own inverse. A scale-aware encoder must instead apply the reciprocal signed
output scale before the inverse left transform (or add an explicit inverse helper); passing arbitrary magnitudes through
the existing sign-only operation would square the scale on reconstruction. The fixed-trellis portion is now implemented
as the default-off `output_channel_scale_optimization` experiment. It solves a positive multiplicative correction for
each existing `SV` entry, rejects every row whose original-Hessian proxy does not improve, and retains the exact
pre-optimization reconstruction as a final fallback.

For W1 and W1.5, the opt-in path now applies half-strength shrinkage before exact checkpoint-dtype reconstruction:

```text
alpha_full = (Q_r H W_r^T) / (Q_r H Q_r^T)
alpha_low_rate = 1 + 0.5 * (alpha_full - 1)
```

W2 and higher retain the full closed-form correction. Shrinking toward one is quantization-only regularization; the
accepted value is still folded into the existing FP32 `SV`, so checkpoint bytes, inference VRAM, decoder operations,
and kernel dispatch are unchanged. The public optimizer keeps `correction_strength=1` as its exact mathematical
default. Only `quantize_qvq_linear(..., output_channel_scale_optimization=True)` selects `0.5` at W1/W1.5, and the
feature remains default-off.

The 2026-08-12 P-core synthetic screen used 64x64 modules, correlated/lognormal activations, row-scale heterogeneity,
periodic weight outliers, 4,096 independent held-out rows, and 12 seeds per cell. The final confirmation below ran the
actual production-shaped implementation rather than a post-hoc formula:

| Rate | Calibration rows | Baseline KLD | Shrunk-`SV` KLD | Paired KLD delta (95% CI) | Top-1 delta points (95% CI) | Top-5 delta points (95% CI) | KLD wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 8 | 0.283716 | 0.262428 | -0.021288 (-0.027109, -0.015468) | +0.98 (+0.65, +1.31) | +2.06 (+1.77, +2.35) | 12/12 |
| W1 | 1,024 | 0.181173 | 0.146986 | -0.034187 (-0.039874, -0.028500) | +2.61 (+2.31, +2.90) | +2.75 (+2.46, +3.04) | 12/12 |
| W1.5 | 8 | 0.128900 | 0.123590 | -0.005310 (-0.006284, -0.004336) | +0.17 (-0.07, +0.41) | +0.57 (+0.46, +0.69) | 12/12 |
| W1.5 | 1,024 | 0.071543 | 0.062636 | -0.008906 (-0.010254, -0.007558) | +1.05 (+0.86, +1.24) | +1.16 (+1.02, +1.29) | 12/12 |

The W1.5/eight-row top-1 interval includes zero, so this is not model-promotion evidence. Keep the control opt-in until
a matched two-layer Llama W1/W1.5 run confirms raw-logit KLD and top-k, especially on rank-poor MLP-down modules.
Disabled W1/W1.5/W2 paths are bitwise identical to the historical default.

Default-off format-compatible experiments, in implementation order:

1. `module_scale_search` for the Level 0 control and independent validation of the scalar math (implemented,
   default-off).
2. `output_channel_scale_optimization` for Level 1 per-output scales (implemented).
3. `scale_grid_search` to evaluate a bounded candidate grid around each closed-form point, always including the current
   scale as a non-regression control.
4. `error_feedback_aware_scale_search` to evaluate each candidate after replaying the exact BlockLDLQ correction from
   already quantized blocks. It depends on `scale_grid_search` and must preserve block ordering.

The first implemented Hessian-aware encoder control is `viterbi_objective="hessian_diagonal"`. BlockLDLQ already
factors `H = L D L^T`; this control weights each additive PGC16 emission by the corresponding diagonal entry of the
conditioned block metric `D`. It deliberately does not pretend that dense off-diagonal terms are additive Viterbi
costs. The complete Euclidean encoding remains a candidate, and the diagonal-metric path is selected only when its
reconstruction has strictly lower loss under the original full Hessian. Both controls preserve checkpoint bytes and
runtime inference operations.

### Level 2: input-group scales, new QVQ format revision

The current checkpoint has no input-column `scale[group]` tensor. `SU[in]` is applied before the right Hadamard and
`SV[out]` is applied after the left Hadamard. A different scale for every 128 input columns cannot commute through those
transforms or be folded into either vector. An exact GPTQ-style input-group implementation must add a group-scale buffer
consumed during decoded GEMM and update Torch/MPS/MLX/CUDA reconstruction, save/reload, resharding, and format metadata.
This is a QVQ checkpoint-format revision, separate from the `pgc16-v1`/`pgc16-v2` decoder-table version, and must never
be hidden behind a quantization-only toggle for the current format.

Required gates for each format-compatible toggle: exact disabled-control parity for both PGC16 tables, closed-form
agreement with an independent FP64 reference, singular/zero denominator fallback, positive scales, W2/W3
normal/outlier/correlated Hessians, exact pack/reload/backend reconstruction, held-out KLD/top-1/top-5 non-regression,
and quantization-time reporting. A Level 2 format revision additionally requires byte-accounting, peak-VRAM, backend
compatibility, and latency gates so its new scale tensor is never presented as free.

### Large paired W1/W2 module-scale gate (2026-08-12)

The host-side synthetic gate uses 64x64 modules, 1,024 calibration activation rows and 4,096 independent held-out rows
per seed, 20 seeds, and 5,242,880 evaluated output values per arm per rate. Weights include log-normal row scales and
periodic outlier columns; activations include log-normal channel scales and short-range correlations. Positive KLD,
JSD, and RMSE deltas are regressions; positive top-1/top-5 deltas are improvements. Confidence intervals are paired
normal 95% intervals across seeds. The complete settings, per-seed rows, and token distributions are stored in
the generated JSON artifact; rerun `scripts/benchmark_qvq_module_scale.py` with
`--json-out /tmp/qvq_module_scale_w1_w2_synthetic.json` to reproduce it without adding generated data to `docs/`.

| Rate | Proxy ratio | KLD delta (95% CI) | JSD delta (95% CI) | RMSE delta (95% CI) | Top-1 delta | Top-5 delta | Quant-time ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 0.991067 | +0.000868 (-0.000869, +0.002604) | +0.000238 (-0.000194, +0.000669) | -0.002300 (-0.003668, -0.000931) | -0.001379 | +0.001082 | 2.0018x |
| W2 | 0.996542 | -0.000344 (-0.000560, -0.000129) | -0.000097 (-0.000152, -0.000041) | -0.000744 (-0.001234, -0.000254) | +0.000134 | +0.000449 | 1.9926x |

Every calibration proxy improved. W1 nevertheless regressed held-out KLD in 14/20 seeds and therefore fails
promotion despite lower RMSE; this is another direct example of reconstruction loss not being a sufficient model
quality gate. W2 improved KLD/JSD in 15/20 seeds with confidence intervals excluding zero, while top-1/top-5 were
mostly tied and their intervals included zero. This is a useful W2 hypothesis, not a production promotion: keep the
control default-off until a matched two-layer Llama held-out run confirms final-logit KLD and task behavior. The
roughly 2x quantization time is expected from the guarded second encoding and has no inference-time counterpart.

Implementation coverage milestone (2026-08-11): all changed executable Python lines for the two current controls are
covered (151/151 quantizer, 11/11 config, 10/10 CUDA wrapper, 2/2 diagnostic CLI). Python 3.14.6 free-threaded tests pass
with 263 host tests and 326 CUDA tests; weighted native Viterbi matches the eager reference for constrained and
unconstrained W2--W8 paths and is deterministic with one caller per physical GPU. Model-level W2 held-out A/B remains
the next acceptance gate; neither control should be enabled by default before that evidence is recorded.

### W2 accuracy-control A/B (2026-08-11)

The first model A/B used fixed `pgc16-v1`, two Llama 3.2 1B Instruct decoder layers, all 14 projections, the first 128
`nm-calibration/LLM` rows, 49,725 valid calibration tokens, and 1,475 excluded padding tokens. Each arm ran under Python
3.14.6 free-threaded on one physical PG506-230 `sm_80` GPU after a three-sample idle gate. The current-code control was
rerun because the previously committed W2 artifact predated the latest v1 normalization correction.

| Arm | Seconds | Mean weight rel-L2 | Local KLD | Live KLD | Layer KLD | Logit KLD | JSD | Logit RMSE | Logit rel-L2 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current control | 87.19 | 0.336754 | 0.010968 | 0.036420 | 0.093610 | 0.103540 | 0.021766 | 0.726232 | 0.355106 | 1.0000 | 0.8636 |
| Damped output `SV` only | 90.65 | 0.336790 | 0.010825 | 0.036718 | 0.102752 | 0.114848 | 0.022901 | 0.728508 | 0.356219 | 1.0000 | 0.8682 |
| Hessian diagonal only | 150.29 | 0.336889 | 0.010604 | 0.038671 | 0.061086 | 0.063769 | 0.014666 | 0.723017 | 0.353534 | 1.0000 | 0.8636 |
| Combined damped `SV` + diagonal | 143.74 | 0.336879 | 0.010535 | 0.039434 | 0.072299 | 0.071714 | 0.016197 | 0.723796 | 0.353915 | 1.0000 | 0.8773 |

The initial raw-Hessian output-scale fit exposed severe overfitting in rank-poor MLP-down activations. The current
implementation therefore solves against the same damped Hessian used by BlockLDLQ while still accepting rows under the
original undamped objective. Damping cuts the preliminary logit-KLD regression from 22.65% to 10.92%, but does not
eliminate it; `output_channel_scale_optimization` remains experimental and must not be enabled by default. The
conditioned-Hessian diagonal encoder selects 13/14 modules and reduces held-out logit KLD by 38.41%, JSD by 32.62%,
layer KLD by 34.74%, and RMSE by 0.44%; its cost is 72.36% more quantization time because the guarded implementation
retains and evaluates the complete Euclidean encoding. Combining the controls improves logit KLD by 30.74% versus the
control but remains worse than the Hessian-diagonal arm alone even though its calibration proxy is lowest (38.84317).
This is direct evidence that the calibration proxy cannot be the only acceptance metric. Next: hold output-SV behind
its default-off gate, validate the Hessian-diagonal gain at larger scope, and remove the redundant second full encoding
without weakening exact baseline retention.

The later fixed-trellis W1/W1.5 gate on 128 disjoint held-out prompts confirmed the default-off decision: both full and
half corrections reduced local/live/layer errors while worsening final KLD and JSD; W1 top-1 also regressed with a
95% interval wholly below zero. Do not infer promotion from a local proxy. W3--W8 may be screened independently because
the historical higher-rate behavior transferred local improvements more often, but each rate requires a held-out
final-KLD/JSD/top-k gate before it can be considered for enablement.

### Four-layer promotion gate (2026-08-11)

The matched four-layer rerun rejects promotion of the Hessian-diagonal objective. Calibration, padding exclusion,
runtime, codec, and physical-GPU isolation were identical to the two-layer experiment; the scope doubled to all 28
projections in the first four decoder layers.

| Arm | Seconds | Mean weight rel-L2 | Local KLD | Live KLD | Layer KLD | Logit KLD | JSD | Logit RMSE | Logit rel-L2 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current control | 175.68 | 0.328718 | 0.012341 | 0.084596 | 0.158862 | 0.395669 | 0.080162 | 0.966227 | 0.493852 | 0.7500 | 0.6545 |
| Hessian diagonal | 280.52 | 0.328784 | 0.012101 | 0.076159 | 0.140377 | 0.402224 | 0.082638 | 0.987174 | 0.504558 | 0.7273 | 0.6318 |

The diagonal objective selects 25/28 modules and improves local KLD by 1.95%, live KLD by 9.97%, and layer KLD by
11.64%, but those intermediate gains do not survive replay: final logit KLD regresses 1.66%, JSD regresses 3.09%, RMSE
regresses 2.17%, top-1 falls from 0.7500 to 0.7273, and top-5 falls from 0.6545 to 0.6318. Quantization time increases
59.67%. Therefore `viterbi_objective="hessian_diagonal"` remains default-off research instrumentation, not an accepted
accuracy upgrade. Do not remove the Euclidean control or enable either new toggle from these results. A future attempt
needs layer-replay-aware acceptance or a better conditioned objective, followed by a fresh four-layer gate.

### Authoritative QTIP implementation audit (2026-08-12)

The public paper implementation at `Cornell-RelaxML/qtip` was audited through current `main` (`e90c668`), its complete
commit history, issues, and open pull requests. Every closed public PR is merged into that tip; PR #20 is the only
public unmerged head. Per project policy, open PR #20 is treated as a candidate author correction
until model evidence disproves it. The audit separates the core BlockLDLQ equations from the complete published model
recipe; reproducing only Algorithm 5 is not equivalent to reproducing the reported QTIP quality.

Corrections after the initial public release that affect correctness or faithful reproduction are:

- `05fd393` fixes packed trellis shape;
- `9781dc7` stores the Hessian as the raw second moment `X.T @ X / N`; the previous covariance plus restored mean
  outer product is mathematically equivalent;
- `e703526` uses the active CUDA device and stream in the extension;
- `f430911` performs a warm forward before fine-tuning so tensor-parallel state is materialized;
- `69589b4` and `192f4b7` complete per-module skip handling;
- open PR #20 (`1f7eaa3c`) restores the differentiable decoded-weight path when training the HYB lookup table. Trellis
  path selection remains discrete. The PR reproducer reports identical forward/input/SU/SV gradients to the fixed
  lookup path plus a nonzero lookup gradient. Importantly, the PR author reproduced the paper's Llama-2-7B W2
  perplexity with the public fixed-lookup path (`5.88` WikiText and `7.73` C4 versus paper `5.86`/`7.73`). Merely
  enabling the corrected lookup gradient produced `5.92`/`7.72` without retuning. The paper-matching control is
  therefore the published fixed-lookup behavior; PR #20 is a separate intended-math arm, not an assumed accuracy win.

QVQ's low-rate numerical core now has an independent regression against the author implementation. Commit
`0c438ab9` reproduces the author's block-Cholesky normalization and reverse Algorithm 5 recurrence, then verifies
identical W1, W1.5, and W2 reconstructed weights and trellis states. Commit `0dc9a278` separately proves the production
lifecycle's propagation contract: a later subset observes the earlier subset's reconstructed output and the final
layer replay handed downstream contains every reconstructed subset. These checks rule out an LDL transposition,
reverse-order, or missing subset/layer replay defect as the source of the W2 full-model cliff.

The material remaining difference is optimization scope. The official W2 example uses `scale_override=0.9`, five
decoder-layer fine-tuning epochs after each installed projection, and four end-to-end epochs with lookup-table
training. Decoder-layer fine-tuning optimizes the mixed layer's continuous parameters against dense layer outputs;
for a quantized projection this includes continuous `SU` and `SV`, while not-yet-quantized dense projections may also
move before their later encoding. The order is `v, q, k, o, up, gate, down`. The end-to-end phase uses dense soft
targets. QVQ currently performs exact BlockLDLQ with reconstructed subset and layer replay but does not perform either
model-output alignment phase. That missing offline alignment—not the already validated Algorithm 5 recurrence—is the
leading explanation for why two/four-layer micro-metrics remain coherent while W2 error compounds across all 16
layers and collapses GSM8K.

The scale conventions differ. QTIP computes
`Wscale = rms(Wr) / (rms(codebook) * scale_override)`, whereas QVQ multiplies its RMS scale by a factor. The exact QVQ
equivalent of the paper's `0.9` is therefore `1 / 0.9 = 1.111111...`, not `0.9`. This must remain an empirical,
held-out-logit gate: the author's issue #31 states that the `0.9682458` lookup scaling constant was selected by a
Gaussian MSE sweep, and QVQ's PGC16 codebook is not the same HYB lookup. The matched W2 PGC16 scale sweep is in
progress; no scale or alignment feature is promoted without KLD/JSD/top-k, save/reload, and task-level evidence.

The paper also reports an architecture-specific failure mode that must be controlled directly: quantizing layer-0
`v_proj` catastrophically degrades Llama 3 70B zero-shot quality unless prompts receive a special prefix/chat
template or that projection remains dense. Ultra's GSM8K evaluation already applies the Llama chat template, so the
remaining paper-authoritative control is to leave exactly layer-0 V dense at W1.5/W2. The real-model validator now
supports asserted exact-path exclusions, natural un-concatenated rows, and unsorted calibration order for this A/B.
The paper publishes Llama-3.2-1B only at W4, not W1.5/W2, so its successful W4 result supports Ultra's W4 sanity
control but does not establish an expected task score for this ultra-low-rate 1B experiment.

Two source-faithful controls remain deliberately separate from production defaults. First, QTIP's W2
`scale_override=0.9` corresponds to the exact QVQ multiplier `1 / 0.9`; the completed `1.05` and `1.20` full-model
arms bracket that value but cannot rule it out because a discrete trellis path can change non-monotonically at the
intermediate point. Second, the author worker seeds one CUDA random-sign stream with the layer index and consumes it
in `v,q,k,o,up,gate,down` order, whereas QVQ derives one deterministic sign seed from each exact module path. Random
signs are part of incoherence processing rather than learned parameters, so this is not evidence of a correctness
defect; it is a seed-sensitivity A/B that must compare held-out logits and downstream tasks before any configuration
surface is added. The output-alignment attachment already uses module-tree semantic tags to reproduce the author's
projection order when enabled.

### Exact frozen-compander rerun (2026-08-11)

The learned `pgc16-v2` arms were rerun after the fitter began scoring the exact frozen FP16 table and rejecting every
positive objective regression. The retained `pgc16-v1` arms did not traverse the changed fitter and were not rerun.
Both v2 jobs used the same two Llama 3.2 1B layers, 14 projections, 128 `nm-calibration/LLM` rows, 49,725 valid tokens,
and 1,475 excluded padding tokens. W2--W4 ran on physical GPU 6 while W5--W8 ran on physical GPU 7 under Python 3.14.6
free-threaded. Times are included for completeness but are not a promotion signal because the two jobs shared host
resources and JIT startup.

| W | Arm | Seconds | Weight rel-L2 | Logit KLD | JSD | RMSE | Logit rel-L2 | Top-1 | Top-5 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | v1 baseline | 89.43 | 0.336754 | 0.104966 | 0.021898 | 0.725617 | 0.354806 | 1.0000 | 0.8636 |
| 2 | v2 fixed | 88.76 | 0.336724 | 0.060530 | 0.013271 | 0.704611 | 0.344534 | 1.0000 | 0.8955 |
| 3 | v1 baseline | 108.45 | 0.173287 | 0.015648 | 0.003637 | 0.370942 | 0.181380 | 1.0000 | 0.9409 |
| 3 | v2 fixed | 80.37 | 0.173042 | 0.016310 | 0.003897 | 0.375119 | 0.183422 | 1.0000 | 0.9182 |
| 4 | v1 baseline | 106.30 | 0.090464 | 0.003754 | 0.000931 | 0.192320 | 0.094039 | 1.0000 | 0.9727 |
| 4 | v2 fixed | 78.60 | 0.090214 | 0.004170 | 0.001042 | 0.189198 | 0.092512 | 1.0000 | 0.9682 |
| 5 | v1 baseline | 89.77 | 0.048687 | 0.001204 | 0.000302 | 0.105578 | 0.051625 | 1.0000 | 0.9864 |
| 5 | v2 fixed | 96.36 | 0.048455 | 0.000848 | 0.000211 | 0.099594 | 0.048699 | 1.0000 | 0.9909 |
| 6 | v1 baseline | 103.49 | 0.027454 | 0.000360 | 0.000090 | 0.059077 | 0.028887 | 1.0000 | 0.9909 |
| 6 | v2 fixed | 98.33 | 0.027375 | 0.000421 | 0.000105 | 0.057531 | 0.028131 | 1.0000 | 0.9955 |
| 7 | v1 baseline | 90.43 | 0.015485 | 0.000114 | 0.000028 | 0.031894 | 0.015595 | 1.0000 | 1.0000 |
| 7 | v2 fixed | 83.20 | 0.015469 | 0.000126 | 0.000032 | 0.032330 | 0.015808 | 1.0000 | 1.0000 |
| 8 | v1 baseline | 71.47 | 0.009165 | 0.000040 | 0.000010 | 0.019016 | 0.009298 | 1.0000 | 0.9955 |
| 8 | v2 fixed | 66.11 | 0.009184 | 0.000031 | 0.000008 | 0.018873 | 0.009228 | 1.0000 | 1.0000 |

V2 is a clear held-out win at W2 and W5, but W3, W4, W6, W7, and W8 still contain at least one regression in KLD,
RMSE, relative L2, or top-5. Exact fitter-objective non-regression is therefore necessary but not sufficient for model
quality. Keep `pgc16-v1` as the accuracy-safe default.

### Matched GPTQ-symmetric and EXL3 comparison (2026-08-11)

The same dense baseline, calibration Hessians, 44 valid evaluation tokens, and final-logit metric path now compare
four implementations. GPTQ uses `sym=True`, group size 128, and activation scale search. EXL3 uses its production
`mcg` codebook, automatic output scales, `sigma_reg=0.025`, seed 787, and exact returned reconstruction. EXL3's
effective bpw is computed from its actual `trellis`, `suh`, `svh`, and codebook-marker tensors; bias is common to all
arms and excluded. Dashes mean the older artifact did not record serialized storage bytes, not zero overhead.

| W | Arm | Arm wall-sec | EXL3 bpw | Weight rel-L2 | KLD | JSD | RMSE | Logit rel-L2 | Top-1 | Top-5 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | GPTQ sym | 57.42 | - | 0.409464 | 7.326883 | 0.525739 | 2.746706 | 1.343060 | 0.2273 | 0.1364 |
| 2 | QVQ v1 | 377.78 | - | 0.336754 | 0.104966 | 0.021898 | 0.725617 | 0.354806 | 1.0000 | 0.8636 |
| 2 | QVQ v2 | 88.76 | - | 0.336724 | 0.060530 | 0.013271 | 0.704611 | 0.344534 | 1.0000 | 0.8955 |
| 2 | EXL3 mcg | 54.84 | 2.0198 | 0.320269 | 0.066761 | 0.014759 | 0.743057 | 0.363333 | 0.9773 | 0.8773 |
| 3 | GPTQ sym | 64.90 | - | 0.215255 | 0.205386 | 0.038246 | 0.765105 | 0.374114 | 0.9545 | 0.8591 |
| 3 | QVQ v1 | 354.56 | - | 0.173287 | 0.015648 | 0.003637 | 0.370942 | 0.181380 | 1.0000 | 0.9409 |
| 3 | QVQ v2 | 80.37 | - | 0.173042 | 0.016310 | 0.003897 | 0.375119 | 0.183422 | 1.0000 | 0.9182 |
| 3 | EXL3 mcg | 50.64 | 3.0198 | 0.161923 | 0.015402 | 0.003740 | 0.377859 | 0.184762 | 1.0000 | 0.9318 |
| 4 | GPTQ sym | 70.88 | - | 0.112744 | 0.024610 | 0.005937 | 0.404732 | 0.197902 | 0.9773 | 0.9227 |
| 4 | QVQ v1 | 334.04 | - | 0.090464 | 0.003754 | 0.000931 | 0.192320 | 0.094039 | 1.0000 | 0.9727 |
| 4 | QVQ v2 | 78.60 | - | 0.090214 | 0.004170 | 0.001042 | 0.189198 | 0.092512 | 1.0000 | 0.9682 |
| 4 | EXL3 mcg | 48.54 | 4.0198 | 0.082774 | 0.004030 | 0.001004 | 0.195184 | 0.095439 | 1.0000 | 0.9591 |
| 5 | GPTQ sym | 80.96 | - | 0.058092 | 0.005283 | 0.001322 | 0.190818 | 0.093304 | 1.0000 | 0.9500 |
| 5 | QVQ v1 | 344.92 | - | 0.048687 | 0.001204 | 0.000302 | 0.105578 | 0.051625 | 1.0000 | 0.9864 |
| 5 | QVQ v2 | 96.36 | - | 0.048455 | 0.000848 | 0.000211 | 0.099594 | 0.048699 | 1.0000 | 0.9909 |
| 5 | EXL3 mcg | 49.33 | 5.0198 | 0.042617 | 0.001326 | 0.000331 | 0.097694 | 0.047769 | 1.0000 | 0.9909 |
| 6 | GPTQ sym | 86.09 | - | 0.030523 | 0.006068 | 0.001581 | 0.112101 | 0.054814 | 1.0000 | 0.9500 |
| 6 | QVQ v1 | 351.37 | - | 0.027454 | 0.000360 | 0.000090 | 0.059077 | 0.028887 | 1.0000 | 0.9909 |
| 6 | QVQ v2 | 98.33 | - | 0.027375 | 0.000421 | 0.000105 | 0.057531 | 0.028131 | 1.0000 | 0.9955 |
| 6 | EXL3 mcg | 50.17 | 6.0198 | 0.022151 | 0.000294 | 0.000073 | 0.051812 | 0.025335 | 1.0000 | 0.9955 |
| 7 | GPTQ sym | 89.03 | - | 0.016437 | 0.001391 | 0.000357 | 0.066359 | 0.032447 | 1.0000 | 0.9682 |
| 7 | QVQ v1 | 399.06 | - | 0.015485 | 0.000114 | 0.000028 | 0.031894 | 0.015595 | 1.0000 | 1.0000 |
| 7 | QVQ v2 | 83.20 | - | 0.015469 | 0.000126 | 0.000032 | 0.032330 | 0.015808 | 1.0000 | 1.0000 |
| 7 | EXL3 mcg | 46.39 | 7.0198 | 0.011654 | 0.000078 | 0.000020 | 0.026521 | 0.012968 | 1.0000 | 0.9955 |
| 8 | GPTQ sym | 88.51 | - | 0.008903 | 0.000292 | 0.000073 | 0.040374 | 0.019742 | 1.0000 | 0.9773 |
| 8 | QVQ v1 | 319.30 | - | 0.009165 | 0.000040 | 0.000010 | 0.019016 | 0.009298 | 1.0000 | 0.9955 |
| 8 | QVQ v2 | 66.11 | - | 0.009184 | 0.000031 | 0.000008 | 0.018873 | 0.009228 | 1.0000 | 1.0000 |
| 8 | EXL3 mcg | 46.87 | 8.0198 | 0.006214 | 0.000020 | 0.000005 | 0.013771 | 0.006734 | 1.0000 | 1.0000 |

EXL3 has the lowest weight relative L2 at every rate and wins held-out KLD at W3 and W6--W8. QVQ-v2 still wins W2
and W5 KLD, while QVQ-v1 wins W4 KLD. At W2, EXL3 improves weight relative L2 by 4.89% versus QVQ-v2 but regresses
KLD by 10.29% and top-1 from 1.0 to 0.9773. This is another direct demonstration that weight error, and even a
Hessian proxy, cannot replace replayed held-out logits as the promotion gate.

Transferable EXL3 ideas worth isolating in QVQ experiments are its automatic output-scale decision, global-scale grid
search, and stronger Hessian regularization around block LDLQ. Test each behind an independent toggle and retain exact
QVQ controls: importing all three together would not reveal which mechanism helps, and EXL3's W2 result shows that a
lower reconstruction norm alone can still damage model outputs. Historical timing rows span different QVQ commits and
shared-host conditions, so they are diagnostic only; do not claim implementation speedups from this table.

### Expanded W2 held-out overlap-candidate sweep (2026-08-11)

The production Llama-3.2-1B-Instruct comparison was expanded to 128 `nm-calibration/LLM` calibration rows and a
strictly disjoint 128-row evaluation range (`[128, 256)`). Calibration used 49,725 valid tokens with 1,475 padding
positions excluded. Evaluation used 6,106 valid tokens with 38 padding positions excluded. Metrics cover both decoder
layers and all 14 q/k/v/o/gate/up/down projections; final-logit metrics reduce approximately 783 million vocabulary
values without sampling. GPTQ is the usable adjacent-asymmetric W2 control. The prior symmetric W2 arm is excluded
from the active comparison because its KLD and top-k agreement make it unusable at this rate.

The latest widened tail-biting search exposed a real native-CUDA interface defect: selecting one column from the
candidate-overlap matrix produced a strided `int64` tensor, while the CUDA Viterbi operator requires a contiguous
overlap vector. Passing `overlaps[:, candidate_index].contiguous()` fixes the failure. A real-CUDA regression test
executes candidate 4 twice and checks deterministic output, circular transition validity, and non-regression against
the historical candidate-1 path.

| W2 arm | Candidates | Arm wall-sec | Weight rel-L2 | KLD | JSD | RMSE | Logit rel-L2 | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPTQ adjacent-asym | - | 194.27 | 0.389458 | 1.128488 | 0.129775 | 1.457915 | 0.705236 | 0.8560 | 0.6919 |
| QVQ v1 | 1 | 203.50 | 0.336754 | 0.120011 | 0.024028 | 0.717056 | 0.346861 | 0.9148 | 0.8424 |
| QVQ v1 | 4 | 461.24 | 0.336404 | 0.113285 | 0.022822 | 0.716213 | 0.346453 | 0.9125 | 0.8384 |
| QVQ v1 | 8 | 537.32 | 0.336362 | 0.108201 | 0.022107 | 0.718697 | 0.347655 | 0.9155 | 0.8398 |
| QVQ v2 | 1 | 199.20 | 0.336724 | 0.111881 | 0.022899 | 0.717980 | 0.347308 | 0.9189 | 0.8352 |
| QVQ v2 | 4 | 424.42 | 0.336343 | 0.103996 | 0.021484 | 0.721665 | 0.349091 | 0.9198 | 0.8410 |
| QVQ v2 | 8 | 531.19 | 0.336319 | 0.109424 | 0.022561 | 0.711323 | 0.344088 | 0.9132 | 0.8382 |
| EXL3 mcg | - | 156.73 | 0.320269 | 0.132914 | 0.026173 | 0.730421 | 0.353326 | 0.9142 | 0.8336 |

QVQ-v2 with four candidates is the probability-quality winner: it lowers KLD by 7.05% versus QVQ-v2 candidate 1,
by 4.04% versus the best QVQ-v1 arm, and by 21.76% versus EXL3, while attaining the highest top-1 agreement (0.9198).
EXL3 retains 4.78% lower raw weight relative L2, reinforcing that Euclidean weight fidelity and downstream
distribution fidelity are distinct objectives. Candidate 8 produces the best QVQ-v2 RMSE and logit relative L2 but
regresses KLD and top-1 versus candidate 4, so a wider search is not monotonically better under held-out replay.
Reported seconds include concurrent model evaluation and exact metric reductions on a shared host and are diagnostic,
not clean throughput claims. YAQA is not included in this historical table because its real-Fisher Sketch-B collector
landed afterward. The harness now obtains the required per-sequence full-model gradients; a YAQA arm must use those
measured output Hessians and must never fabricate `H_O`.

### W1 codec, model-quality, and timing gate (2026-08-11)

QVQ now supports W1 without a format fork: each scalar contributes one trellis bit, every 16x16 tile stores eight
`int32` words, and the existing `SU`/`SV` vectors and PGC16 decoder remain unchanged. `gptq_p` still starts at W2.
Torch reconstruction, packing, config, quantization, YAQA, CUDA GEMV/WMMA/split-K, and the CUDA Viterbi oracle now
cover W1. The complete CUDA suite passes 400 tests plus two expected one-visible-device skips under Python 3.14.6
free-threaded; a separate two-GPU test launches W1 Viterbi concurrently on both physical devices.

The model gate used the same two Llama 3.2 1B decoder layers, all 14 projections, first 128 calibration rows, 49,725
valid calibration tokens, and disjoint 128-row held-out range as the expanded W2 comparison. All masked positions were
excluded. CUDA synchronization closes every timing boundary. `Quant` is offline QVQ encoding, `Validate` is the
per-module dense reconstruction check/install, `Replay` is one post-quant model forward, and `Metrics` is exact
reduction of approximately 783 million held-out vocabulary logits.

```text
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
| Arm      | BPW | Eff BPW | Wall-s | Quant-s | Validate | Replay-s | Metric-s | Weight L2  | Fwd KLD  | JSD      | RMSE    | Logit L2  | Top-1  | Top-5  |
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
| W1 v1    | 1.0 | 1.02317 | 690.14 | 360.13  | 77.34    | 2.64     | 250.03   | 0.647528   | 0.699739 | 0.089050 | 1.50119 | 0.726170  | 0.8652 | 0.7054 |
| W1 v2    | 1.0 | 1.02320 | 708.02 | 399.10  | 75.06    | 9.61     | 224.25   | 0.635690   | 1.042415 | 0.112460 | 1.44801 | 0.700450  | 0.8626 | 0.7028 |
| EXL3 W1  | 1.0 | 1.01159 | 324.44 | 66.65   | 56.10    | 2.41     | 199.27   | 0.619463   | 1.061750 | 0.115881 | 1.50655 | 0.728761  | 0.8608 | 0.6893 |
| W2 v1 c1 | 2.0 | 2.02317 | 494.15 | 113.50  | 80.99    | 2.79     | 296.86   | 0.336754   | 0.120011 | 0.024028 | 0.71706 | 0.346861  | 0.9148 | 0.8424 |
| W2 v2 c4 | 2.0 | 2.02320 | 673.38 | 352.88  | 73.19    | 2.48     | 244.82   | 0.336343   | 0.103996 | 0.021484 | 0.72166 | 0.349091  | 0.9198 | 0.8410 |
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
```

W1-v2 lowers weight relative L2 by 1.83%, logit RMSE by 3.54%, and logit relative L2 by 3.54% versus W1-v1, but it
raises forward KLD by 48.97%, raises JSD by 26.29%, and slightly lowers top-1/top-5. `BPW` is trellis payload rate;
`Eff BPW` is total trellis plus every serialized auxiliary, divided by the total weights across all 14 matrices. It
includes QVQ `SU`/`SV`, the single 512-byte model-level v2 compander table, and EXL3's actual returned auxiliary
tensors; common bias is excluded. EXL3 W1 uses its production `mcg` codebook and measures 1.01159 effective
bits/weight. It lowers raw weight
relative L2 by 4.33% versus W1-v1 and quantizes 5.40x faster, but its held-out forward KLD is 51.73% higher, top-1 is
0.44 percentage points lower, and top-5 overlap is 1.61 points lower. Therefore fixed `pgc16-v1` remains the
accuracy-safe default at W1 as well as the general QVQ default. W1 is necessarily a substantial quality step down
from W2: versus W2-v1 candidate 1, W1-v1 has 5.83x KLD and 4.96 percentage points lower top-1 agreement. The timing
rows ran concurrently and are phase-attribution evidence, not clean throughput comparisons; notably, exact metric
reduction dominates W2-v1 wall time while the smaller W1 Viterbi branch factor does not offset its 16,384 suffixes
and 64 KiB shared-memory reduction.

### CUDA BF16 compute-range gate (2026-08-11)

An unconditional switch from FP16 to BF16 activation compute preserves BF16 range but loses three mantissa bits for
ordinary values. The W2--W8, M=1/2/4/8/16/32 rerun used learned v2 tables, three seeds, 256x256 layers, and an
independent FP32 decoded-weight/RHT oracle. Across 126 normal-range cases, native BF16 raised mean MSE from
`0.00861430` to `0.13068792` (15.17x) and mean relative L2 from `0.00193478` to `0.00753987` (3.90x). The range-safe
hybrid is bit-identical to the FP16 path for every normal case and retries in native BF16 only after non-finite output.
All 21 W2--W8 extreme-basis cases overflow in the old FP16 path; native BF16 and the hybrid remain finite, retain
top-1 `1.0`, and have worst FP32-oracle relative L2 `0.00608718`.

| Path | M | Mean ms over W2--W8 | Ratio vs FP16 |
|---|---:|---:|---:|
| FP16 compute | 1 | 1.497600 | 1.0000x |
| Native BF16 | 1 | 1.442889 | 0.9635x |
| Range-safe hybrid | 1 | 1.616311 | 1.0793x |
| FP16 compute | 2 | 1.470464 | 1.0000x |
| Native BF16 | 2 | 1.409170 | 0.9583x |
| Range-safe hybrid | 2 | 1.594807 | 1.0846x |
| FP16 compute | 4 | 1.474706 | 1.0000x |
| Native BF16 | 4 | 1.418386 | 0.9618x |
| Range-safe hybrid | 4 | 1.599634 | 1.0847x |
| FP16 compute | 8 | 1.482825 | 1.0000x |
| Native BF16 | 8 | 1.427675 | 0.9628x |
| Range-safe hybrid | 8 | 1.616018 | 1.0898x |
| FP16 compute | 16 | 1.483557 | 1.0000x |
| Native BF16 | 16 | 1.423945 | 0.9598x |
| Range-safe hybrid | 16 | 1.610825 | 1.0858x |
| FP16 compute | 32 | 1.477047 | 1.0000x |
| Native BF16 | 32 | 1.421312 | 0.9623x |
| Range-safe hybrid | 32 | 1.614994 | 1.0934x |

The hybrid costs 8.63% on average in this small-layer benchmark because the finite-output decision synchronizes the
host. Accuracy takes precedence; remove that cost later with an asynchronous kernel-owned range signal, not by
returning to unconditional lower-mantissa BF16 arithmetic or dropping the extreme-range fallback.

### Half-step QVQ v1/v2 model-quality sweep (2026-08-11)

The W1.5/W2.5/W3.5 comparison uses the current large-sample Llama-3.2-1B-Instruct protocol: the first two decoder
layers and all 14 projections, 128 `nm-calibration/LLM` calibration rows with 49,725 valid tokens, and the disjoint
held-out range `[128, 256)` with 6,106 valid tokens. The harness excluded 1,475 calibration padding positions and 38
held-out padding positions from Hessians and metrics. Quantization math, source weights, and dense references are
FP32; each arm used candidate count 1 and its measured CUDA rate default. Six independent Python 3.14.6
free-threaded processes ran concurrently on physical PG506 GPUs 0--5 after three consecutive 0 MiB/0% idle samples.

`BPW` is trellis payload rate. `Eff BPW` includes planar trellis data, `SU`/`SV`, and the single 512-byte v2 learned
table where applicable. `Quant` is offline QVQ encoding, `Validate` checks and installs dense reconstruction,
`Replay` is post-quant inference, and `Metrics` exactly reduces approximately 783 million held-out vocabulary logits.
Concurrent shared-host timings are phase-attribution evidence, not clean speed comparisons.

```text
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
| Arm      | BPW | Eff BPW | Wall-s | Quant-s | Validate | Replay-s | Metric-s | Weight L2  | Fwd KLD  | JSD      | RMSE    | Logit L2  | Top-1  | Top-5  |
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
| W1.5 v1  | 1.5 | 1.52317 | 743.34 | 500.17  | 55.92    | 2.47     | 184.78   | 0.468199   | 0.294294 | 0.049026 | 1.03148 | 0.498958  | 0.8950 | 0.7800 |
| W1.5 v2  | 1.5 | 1.52320 | 766.76 | 529.21  | 54.08    | 2.70     | 180.77   | 0.467477   | 0.333620 | 0.053469 | 1.01967 | 0.493245  | 0.8899 | 0.7728 |
| W2.5 v1  | 2.5 | 2.52317 | 369.13 | 123.19  | 55.71    | 2.36     | 187.86   | 0.241287   | 0.051356 | 0.011536 | 0.50743 | 0.245457  | 0.9396 | 0.8790 |
| W2.5 v2  | 2.5 | 2.52320 | 372.06 | 97.09   | 57.94    | 2.86     | 214.17   | 0.241067   | 0.045013 | 0.010184 | 0.50544 | 0.244494  | 0.9409 | 0.8812 |
| W3.5 v1  | 3.5 | 3.52317 | 366.18 | 118.66  | 51.43    | 2.47     | 193.62   | 0.124739   | 0.010064 | 0.002461 | 0.26054 | 0.126031  | 0.9681 | 0.9326 |
| W3.5 v2  | 3.5 | 3.52320 | 334.36 | 82.26   | 53.98    | 2.48     | 195.63   | 0.124510   | 0.009678 | 0.002381 | 0.25973 | 0.125641  | 0.9692 | 0.9373 |
+----------+-----+---------+--------+---------+----------+----------+----------+------------+----------+----------+---------+-----------+--------+--------+
```

V2 is not uniformly better at fractional rates. At W1.5 it lowers weight relative L2 by 0.15%, RMSE by 1.15%, and
logit relative L2 by 1.15%, but raises KLD by 13.36%, raises JSD by 9.06%, and loses 0.51/0.71 percentage points of
top-1/top-5 agreement. V1 therefore remains the W1.5 accuracy winner. At W2.5, v2 lowers KLD by 12.35% and JSD by
11.73% while improving every reported accuracy metric. At W3.5, v2 lowers KLD by 3.83% and JSD by 3.25%, and also
improves every reported accuracy metric. The learned compander's crossover from harmful to useful therefore occurs
somewhere above W1.5 for this model and dataset; weight fidelity alone still cannot select the W1.5 codec.

EXL3 has no half-step arm in this diagnostic, so integer EXL3 results must bracket rather than directly match these
rates. The large-sample EXL3 W1 and W2 anchors above use the same 128-row held-out range and are directly comparable
at their endpoints. Historical EXL3 W3/W4 rows used only 44 valid held-out tokens; use them as trend context only and
do not treat their absolute KLD/top-k values as matched comparisons with this half-step sweep.

### Ultra lifecycle and Evalution E2E gate (2026-08-11)

QVQ now owns the complete Ultra lifecycle: calibration input-Hessian capture, padding exclusion, offline encoding,
dense replay, `QVQLinear` replacement, checkpoint save, `.trellis`-driven reload, inference, and Evalution scoring.
The real-model gate used `/monster/data/model/Llama-3.2-1B-Instruct`, physical PG506-230 GPU 6, Python 3.14.6
free-threaded (`PYTHON_GIL=0`), QVQ W2 PGC16-v1, and decoder layers 0--1 (all 14 q/k/v/o/gate/up/down
projections). It consumed the first 128 `nm-calibration/LLM` rows with `concat_size=2048`, batch 1, 49,725 valid
tokens, and 1,475 excluded padding tokens. The partial checkpoint is intentionally hybrid: layers 0--1 are QVQ and
layers 2--15 remain dense. It was saved to a new 2.2 GB path and did not overwrite the source model.

The E2E run exposed and fixed two lifecycle bugs that isolated module tests could not reveal:

1. `layer_scope` writes `False` dynamic overrides, but the dynamic cache assumed every value was a dictionary. The
   resolver now treats `False` as an explicit exclusion and has a pre-fix regression test.
2. `StageLayer` clears transient processor tasks before its free-threaded finalize drain. QVQ finalization therefore
   raced with task-map cleanup. Immutable decoder metadata now travels with each `NamedModule` payload, and the unit
   test explicitly clears processor tasks before replacement.

Successful quantization command:

```bash
PYTHON_GIL=0 CUDA_VISIBLE_DEVICES=6 python scripts/validate_qvq_lifecycle.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /monster/data/model/Llama-3.2-1B-Instruct-QVQ-W2-L2-lifecycle-0811 \
  --results /tmp/qvq_lifecycle_llama32_w2_l2.json \
  --rows 128 --concat-size 2048 --layers 2 --bits 2 --batch-size 1 --device cuda:0 \
  --max-forward-kld 0.04 --min-top1-agreement 0.88
```

The successful run used the already-built CUDA extension. The preceding first kernel use spent 34 seconds in the
one-time JIT build, which is not included in the successful quantization time.

### Lifecycle correctness audit (2026-08-12)

| Finding | Resolution | Regression evidence | Commit |
|---|---|---|---|
| Scoped W2.5/W3.5 metadata was truncated with `int(module.bits)` | Preserve the installed public rate exactly | Config serialize/parse, reload allocation, payload load, and bitwise output parity; explicit truncated W2.5 size-mismatch reproducer | `4f6b4615` |
| QVQ Hessian and staged payload survived processor exceptions | Always release capture; drain/remove partial staging and restore the original dense weight without masking the primary error | Encode failure, partial host-stage failure, sync-cleanup failure, restore-cleanup failure, missing Hessian, and zero samples | `eb2e80b5`, `64dfe61a`, `dc27c415` |
| Transformers `Conv1D` lifecycle orientation was unproven | Exercise canonical transpose, dense replay, serialized tensors, and reload inference | Dense replay close and live/reload bitwise equality | `e8cd095f` |
| Learned-v2 lifecycle ownership was ambiguous | Run one bounded model-level fitting prepass before encoding; also accept one explicit pre-fitted table and reject mixed-rate automatic fitting | Exact 256-entry table propagation, deterministic bounded sampling, configurable Hessian damping, failure cleanup, and fail-closed quantization ordering | `4b6cd015`, `762fb51e`, `72f88cb2`, `dc5a0f03`, `9a975949` |
| Validator could hide live-versus-reload drift and lacked the declared metric gate | Compare live/reloaded logits directly; report MAE/MSE/RMSE/relative-L2/SQNR/cosine/forward+reverse KLD/JSD/top-1/top-5/max error; require KLD/top-1 thresholds | Exact/deviating/non-finite cases and direct serialization-drift rejection | `faa72c38` |

Coverage is deliberately scoped rather than reported as whole-repository coverage. The new validation utility measures
100% line and branch coverage (54 statements, 18 branches). The new QVQ failure-cleanup block has zero missing lines or
branches across its success/failure cleanup decisions. The fractional-rate recorder line is executed for W2.5 and W3.5,
and the pre-fix `int()` behavior is retained only in a negative reproducer that proves the 20-word versus 16-word load
mismatch. Behavioral tests use Python 3.14.6 with `PYTHON_GIL=0`; coverage instrumentation uses the stable Python 3.12
review environment because coverage under the free-threaded 3.14 environment cannot safely re-import NumPy/Torch.

```text
+----------------------+----------+-----------------------------------------+
| Phase                | Seconds  | Context                                 |
+----------------------+----------+-----------------------------------------+
| Dense oracle load    |   0.9905 | Transformers BF16                      |
| QVQ lifecycle total  |  67.1134 | capture + 14 projections + replay      |
| Layer 0 wall         |  32.1230 | QVQ process 31.374s                     |
| Layer 1 wall         |  31.4850 | QVQ process 30.720s                     |
| QVQ process total    |  62.0940 | 14 projection encodes                  |
| Checkpoint save      |   1.7078 | hybrid QVQ/dense safetensors           |
| Checkpoint reload    |   8.0192 | 14 `.trellis` modules found as QVQ     |
+----------------------+----------+-----------------------------------------+
```

Reloaded-logit comparison uses four fixed prompts and reduces only the 37 non-padding token positions. The dense
oracle is an independent Transformers BF16 load; the candidate is the saved-and-reloaded QVQ checkpoint.

```text
+--------------------+------------+
| Metric             | QVQ W2 L0-1 |
+--------------------+------------+
| MSE                |   0.186313 |
| Relative L2        |   0.164260 |
| Forward KLD        |   0.032801 |
| Top-1 agreement    |   0.891892 |
| Maximum abs error  |   5.505859 |
+--------------------+------------+
```

Evalution ran all 1,172 ARC-Challenge test samples with the Llama chat template and batch 32:

```bash
PYTHONPATH=/root/repos/GPT-QModel-Ultra PYTHON_GIL=0 CUDA_VISIBLE_DEVICES=6 \
  python scripts/eval_model.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct-QVQ-W2-L2-lifecycle-0811 \
  --backend qvq --tasks arc_challenge --chat-template-tasks arc_challenge \
  --batch-size 32 --dtype bfloat16 --model-arg device=cuda:0 \
  --output /tmp/qvq_lifecycle_llama32_w2_l2_arc.json --table-format grid
```

The dense numbers are the repository's recorded full-dataset Llama-3.2-1B-Instruct BF16 baseline from
`tests/models/test_llama3_2.py`; the QVQ numbers are the new matched task/configuration run. This is a lifecycle gate
for a two-layer partial checkpoint, not a claim about full-model W2 quality.

```text
+---------------------------+------------+------------+------------+
| ARC-Challenge metric      | Dense BF16 | QVQ W2 L0-1 | Abs delta  |
+---------------------------+------------+------------+------------+
| Accuracy, loglikelihood   |   0.324232 |   0.304608 |  -0.019625 |
| Accuracy, normalized LL   |   0.351536 |   0.350683 |  -0.000853 |
+---------------------------+------------+------------+------------+
```

## Fresh matched QVQ-v1/v2 versus EXL3 matrix (2026-08-12)

This rerun uses PR tip `43dfc156` for the measured code. Eight isolated Python 3.14.6 free-threaded processes ran one
rate each on physical PG506 `sm_80` GPUs 0--7. Every arm uses the same two Llama-3.2-1B-Instruct decoder layers and
all 14 projections. Calibration is rows `[0, 128)` of `nm-calibration/LLM`, producing 49,725 valid tokens with 1,475
padding positions excluded. Evaluation is the disjoint rows `[128, 256)`, producing 6,106 valid tokens with 38
padding positions excluded. Final metrics reduce the complete held-out vocabulary logits without sampling. `Infer s`
is post-quant model inference; `Metric s` is the exact CPU reduction and is reported separately from quantization.

| Rate | Arm | Eff BPW | Weight rel-L2 | Fwd KLD | JSD | RMSE | Top-1 | Top-5 | Quant s | Infer s | Metric s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | QVQ v1 | 1.02317 | 0.647576 | 0.816873 | 0.098755 | 1.459798 | 0.8654 | 0.7073 | 333.63 | 2.57 | 192.25 |
| W1 | QVQ v2 | 1.02320 | 0.635803 | 1.075508 | 0.117785 | 1.465395 | 0.8601 | 0.6965 | 327.71 | 3.06 | 212.86 |
| W1 | EXL3 mcg | 1.01159 | 0.619274 | 1.168127 | 0.120476 | 1.462633 | 0.8343 | 0.6926 | 68.46 | 3.25 | 228.09 |
| W2 | QVQ v1 | 2.02317 | 0.336747 | 0.108171 | 0.021890 | 0.717728 | 0.9196 | 0.8403 | 88.90 | 3.61 | 193.03 |
| W2 | QVQ v2 | 2.02320 | 0.336746 | 0.115274 | 0.023332 | 0.718547 | 0.9209 | 0.8449 | 88.16 | 3.07 | 177.34 |
| W2 | EXL3 mcg | 2.01159 | 0.320358 | 0.129555 | 0.025668 | 0.733193 | 0.9094 | 0.8243 | 55.70 | 3.29 | 172.80 |
| W3 | QVQ v1 | 3.02317 | 0.173289 | 0.022869 | 0.005399 | 0.361037 | 0.9535 | 0.9128 | 77.93 | 4.06 | 199.43 |
| W3 | QVQ v2 | 3.02320 | 0.173060 | 0.023147 | 0.005486 | 0.361346 | 0.9566 | 0.9140 | 80.16 | 3.63 | 191.70 |
| W3 | EXL3 mcg | 3.01159 | 0.161910 | 0.022254 | 0.005309 | 0.364564 | 0.9546 | 0.9144 | 52.08 | 3.18 | 197.76 |
| W4 | QVQ v1 | 4.02317 | 0.090482 | 0.005289 | 0.001308 | 0.190335 | 0.9771 | 0.9500 | 80.14 | 3.25 | 189.51 |
| W4 | QVQ v2 | 4.02320 | 0.090230 | 0.005589 | 0.001375 | 0.189776 | 0.9763 | 0.9529 | 77.65 | 2.69 | 173.64 |
| W4 | EXL3 mcg | 4.01159 | 0.082787 | 0.005266 | 0.001296 | 0.185034 | 0.9800 | 0.9555 | 44.36 | 2.93 | 179.63 |
| W5 | QVQ v1 | 5.02317 | 0.048686 | 0.001503 | 0.000374 | 0.101085 | 0.9882 | 0.9734 | 91.94 | 3.64 | 219.04 |
| W5 | QVQ v2 | 5.02320 | 0.048457 | 0.001420 | 0.000354 | 0.100599 | 0.9871 | 0.9747 | 94.94 | 3.44 | 188.59 |
| W5 | EXL3 mcg | 5.01159 | 0.042614 | 0.001349 | 0.000336 | 0.094719 | 0.9895 | 0.9706 | 44.98 | 3.22 | 178.07 |
| W6 | QVQ v1 | 6.02317 | 0.027454 | 0.000475 | 0.000118 | 0.057339 | 0.9938 | 0.9840 | 95.31 | 3.04 | 195.70 |
| W6 | QVQ v2 | 6.02320 | 0.027382 | 0.000483 | 0.000120 | 0.057212 | 0.9946 | 0.9847 | 97.32 | 2.95 | 181.48 |
| W6 | EXL3 mcg | 6.01159 | 0.022145 | 0.000360 | 0.000090 | 0.049722 | 0.9949 | 0.9865 | 48.27 | 3.86 | 194.60 |
| W7 | QVQ v1 | 7.02317 | 0.015483 | 0.000142 | 0.000036 | 0.032093 | 0.9954 | 0.9905 | 83.71 | 2.79 | 218.26 |
| W7 | QVQ v2 | 7.02320 | 0.015466 | 0.000145 | 0.000036 | 0.031856 | 0.9966 | 0.9917 | 85.13 | 3.33 | 211.01 |
| W7 | EXL3 mcg | 7.01159 | 0.011653 | 0.000105 | 0.000026 | 0.025655 | 0.9969 | 0.9927 | 46.56 | 2.93 | 212.12 |
| W8 | QVQ v1 | 8.02317 | 0.009166 | 0.000048 | 0.000012 | 0.018402 | 0.9974 | 0.9951 | 68.24 | 3.16 | 203.44 |
| W8 | QVQ v2 | 8.02320 | 0.009185 | 0.000049 | 0.000012 | 0.018567 | 0.9975 | 0.9953 | 68.64 | 3.07 | 183.48 |
| W8 | EXL3 mcg | 8.01159 | 0.006214 | 0.000027 | 0.000007 | 0.013599 | 0.9980 | 0.9963 | 47.26 | 2.99 | 184.63 |

Fixed QVQ-v1 remains the accuracy-safe QVQ default. V2 improves weight relative-L2 at W1--W7 but regresses final
KLD at W1--W4 and W6--W8; its only KLD win is W5, where top-1 regresses. EXL3 wins final KLD at W3--W8 and is
1.45--4.87x faster to quantize, but QVQ-v1 remains materially better at W1/W2 final KLD and top-k. Thus the current
low-rate priority is improving QVQ W1/W2 quantization speed without replacing its downstream-accuracy advantage with
weight-MSE-only optimization.

### Matched native W1.5 QVQ-v1 versus allocated EXL3 (2026-08-12)

The measured harness and quantizer commit is `e018b6bc`. This rerun keeps the preceding two-layer model, calibration,
held-out evaluation, padding exclusion, exact-vocabulary reduction, Python 3.14.6 free-threaded runtime, and PG506
`sm_80` protocol unchanged. It ran in one isolated process on physical GPU 6. Raw generated results are
`/tmp/qvq_v1_exl3_w15_e018b6bc/w1.5.{json,csv,log}` and are not checked into `docs/`.

EXL3 has an integer rate per tensor. Its production fractional-rate policy therefore floors every tensor to W1 and
spends the remaining weight-counted budget on whole architecture groups: attention before MLP, q/k/v together, and
gate/up together, with groups nearest the ends of the decoder stack winning ties. The harness reproduces that policy
instead of truncating `1.5` to W1 or interpolating W1/W2 measurements. Both arms have exactly 1.5 payload BPW.

| Arm | Payload BPW | Effective BPW | Weight rel-L2 | Fwd KLD | JSD | RMSE | Top-1 | Top-5 | Quant s | Infer s | Metric s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QVQ-v1 native W1.5 | 1.50000 | 1.52317 | 0.468118 | 0.292119 | 0.047980 | 1.031417 | 0.8932 | 0.7802 | 458.15 | 2.63 | 171.32 |
| EXL3 MCG allocated W1.5 | 1.50000 | 1.51159 | 0.367591 | 0.676646 | 0.090159 | 1.212520 | 0.8470 | 0.7319 | 57.72 | 3.01 | 188.80 |

QVQ-v1 lowers final KLD by 56.83%, lowers JSD by 46.79%, and improves top-1/top-5 by 4.62/4.83 percentage points.
EXL3 lowers mean weight relative-L2 by 21.48% and quantizes 7.94x faster. This is strong evidence that Euclidean
weight reconstruction alone ranks the low-rate formats incorrectly.

The exact EXL3 allocation explains much of the W1.5 gap:

| Allocated rate | Weight share | Modules |
|---:|---:|---|
| W3 | 5.17% | layer-0 q/k/v |
| W2 | 39.66% | layer-0 o/gate/up and layer-1 q/k/v/o |
| W1 | 55.17% | both down projections and layer-1 gate/up |

The concentrated W1 MLP errors dominate the gains purchased for attention. Against native QVQ W1.5, EXL3's local
KLD is 7.65x higher for layer-1 up, 2.94x higher for layer-1 down, 2.39x higher for layer-1 gate, and 2.66x higher for
layer-0 down. QVQ instead spends an exact three transition bits per reconstructed pair in every tensor, distributing
the 1.5-BPW constraint without forcing an entire large projection over the W1 cliff.

The integer-rate matrix narrows the broader crossover: QVQ-v1 is materially better in downstream KLD at W1 and W2;
W1.5 now confirms the same low-rate region. At W3 EXL3 already wins KLD narrowly (`0.022254` versus `0.022869`), so
QVQ should not be described as better through W3. From W4--W8 EXL3's lower Euclidean reconstruction error increasingly
translates into lower KLD. Its weight-relative-L2 advantage over QVQ grows from 8.50% at W4 to 32.21% at W8, where
quantization noise is small enough that EXL3's per-channel normalization, automatic output-scale decision, sampled
global-scale search, and procedural MCG reconstruction dominate. At W1--W2, by contrast, codebook/path geometry and
the direction of residual error matter more than total error norm; QVQ's fixed Gaussian pair manifold and BlockLDLQ
feedback produce a larger raw norm but a less damaging model-output perturbation. This last attribution is supported
by the matched weight/output metrics and module hotspots, but separating codebook geometry from normalization and
scale search still requires controlled one-feature A/B arms.

### Full-model W2 scale and fixed-trellis alignment audit (2026-08-12)

The official QTIP source and its open author PR were audited against Ultra's low-rate recurrence. Ultra's BlockLDLQ
recurrence is exact at W1/W1.5/W2, padding is excluded from the Hessian statistic, and QTIP's global Hessian scaling
difference cancels during LDL normalization. QTIP's W2 recipe also applies a `0.9` scale override (equivalent to an
Ultra PGC16 multiplier of `1 / 0.9`), but its codebook is not PGC16. A matched full 16-layer scale test therefore
kept the trellis codec unchanged and changed only the PGC16 scale multiplier. All arms used 512 unpacked calibration
rows, batch 1, FP16 QVQ execution, Python 3.14.6 free-threaded, and the same four-prompt full-vocabulary lifecycle
validation. Live and reloaded logits were bitwise identical in every arm.

| PGC16 scale multiplier | Quant s | Fwd KLD | JSD | Top-1 | Top-5 | Rel-L2 | SQNR dB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 402.16 | 0.209691 | 0.049366 | 0.7027 | 0.7297 | 0.43781 | 7.17 |
| 1.05 | 479.18 | 0.238997 | 0.056863 | 0.6757 | 0.6865 | 0.41669 | 7.60 |
| 1.20 | 440.11 | 0.283519 | 0.068645 | 0.7568 | 0.6919 | 0.48853 | 6.22 |

The two-layer scale sweep suggested lower KLD around `1.05`--`1.20`, but the full-model gate rejects both. The
`1.05` arm improves logit relative-L2 while worsening KLD, JSD, top-1, and top-5; `1.20` worsens every distributional
metric and relative-L2. This is another concrete example of local or norm-only criteria selecting the wrong QVQ
change. The production PGC16-v1 scale remains unchanged.

QTIP's material difference is its offline model-output alignment: decoder-layer fine-tuning after each installed
projection plus end-to-end fine-tuning. A fixed-trellis experiment isolated that mechanism without changing the
checkpoint or decoder. It froze all trellis payloads in the full W2 checkpoint, optimized only the existing FP32
`SU`/`SV` vectors for one epoch against dense soft targets, and evaluated on 128 disjoint rows `[512, 640)`. This
diagnostic cached decoded inner weights only for training and is not a permissible production memory strategy.

| Alignment calibration | LR | Fwd KLD | JSD | Top-1 | Top-5 | Rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| None | -- | 0.351263 | 0.075425 | 0.7488 | 0.7132 | 0.36995 |
| 32 rows `[0, 32)` | 3e-6 | 0.317262 | 0.068498 | 0.7560 | 0.7235 | 0.36272 |
| 32 rows `[0, 32)` | 1e-5 | 0.292833 | 0.063099 | 0.7645 | 0.7332 | 0.35634 |
| 32 rows `[256, 288)` | 1e-5 | 0.293681 | 0.063667 | 0.7628 | 0.7330 | 0.35648 |
| 64 rows `[128, 192)` | 1e-5 | 0.276263 | 0.060638 | 0.7683 | 0.7380 | 0.35151 |

The gain replicates across a disjoint training slice and strengthens with 64 rows: relative to the prepared
unaligned baseline, the 64-row arm lowers KLD by 21.35%, lowers JSD by 19.60%, raises top-1 by 1.95 percentage points,
raises top-5 by 2.48 points, and lowers relative-L2 by 4.98%. This validates missing output alignment as a material
cause of the W2 cliff. Production work must stream one decoder layer at a time, keep trellis fixed initially, use a
disjoint validation split with best-state rollback, restore all state transactionally on failure, and add no
persistent decoded-weight cache or inference metadata.

### Transactional final-layer alignment W1 rejection (2026-08-12)

Commit `ea295e34` integrated the format-preserving fixed-trellis experiment as an explicit, default-disabled lifecycle
attachment. It optimizes FP32 `SU`/`SV` only, keeps decoded weights within one layer scope, uses disjoint internal
training/validation batches, requires both dense-reconstruction and real-QVQ-runtime validation MSE to improve, and
rolls every weight and auxiliary tensor back on rejection or exception. Its implementation has 100% line and branch
coverage (330 statements, 118 branches), including a real Python 3.14.6 `PYTHON_GIL=0` launch on ThreadX's sole CUDA
owner for physical PG506 GPU 6.

The external acceptance set is a stronger, fully disjoint gate: 128 `neuralmagic/calibration:LLM` rows `[128, 256)`,
6,106 non-padding tokens, complete 128,256-way final logits, and decoder-layer hidden states. Calibration uses rows
`[0, 128)`. Both arms quantize the first two Llama-3.2-1B-Instruct decoder layers at W1 and execute in FP16 on PG506
`sm_80`. Live and reloaded outputs are bitwise identical. The baseline timing includes a one-time 118-second native
kernel build and is therefore not a steady-state speed comparison.

```text
+--------------------+---------+---------+----------+----------+--------+--------+----------+---------+
| Arm                | Quant s | Infer s | Fwd KLD  | JSD      | Top-1  | Top-5  | RMSE     | Rel-L2  |
+--------------------+---------+---------+----------+----------+--------+--------+----------+---------+
| Fixed trellis      |  182.99*|   1.457 | 0.468343 | 0.096169 | 0.7162 | 0.6962 | 1.218632 | 0.38222 |
| Final SU/SV align  |   75.99 |   1.471 | 0.511680 | 0.104293 | 0.6729 | 0.6921 | 1.225725 | 0.38444 |
+--------------------+---------+---------+----------+----------+--------+--------+----------+---------+
| Aligned delta      |    n/a  |  +0.96% |   +9.25% |   +8.45% | -4.32pp| -0.41pp|   +0.58% |  +0.58% |
+--------------------+---------+---------+----------+----------+--------+--------+----------+---------+
* Includes the one-time 118-second CUDA extension build.
```

The layer surrogate explains why the internal acceptance gate was insufficient:

```text
+---------+---------+----------+----------+--------+--------+----------+---------+
| Layer   | Arm     | Norm KLD | Norm JSD | Top-1  | Top-5  | RMSE     | Rel-L2  |
+---------+---------+----------+----------+--------+--------+----------+---------+
| Layer 0 | Fixed   | 0.226010 | 0.051292 | 0.7801 | 0.7777 | 0.037190 | 0.30284 |
| Layer 0 | Aligned | 0.241032 | 0.054334 | 0.7728 | 0.7797 | 0.034625 | 0.28195 |
| Layer 1 | Fixed   | 0.486638 | 0.105527 | 0.6821 | 0.6971 | 0.261433 | 0.10325 |
| Layer 1 | Aligned | 0.485834 | 0.105517 | 0.7250 | 0.7015 | 0.207191 | 0.08183 |
+---------+---------+----------+----------+--------+--------+----------+---------+
```

Layer-1 relative-L2 improves by 20.75%, yet final-logit KLD and top-1 both regress materially. The W1 recipe is
therefore rejected and remains default-disabled. Local decoder MSE cannot promote this feature.

The authoritative QTIP implementation also shows that this final-only experiment is not the paper recipe. QTIP
quantizes one projection, replaces it immediately, then optimizes `mixed_layer.parameters()` before quantizing the
next projection. That parameter set includes every already-installed quantizer's `SU`/`SV`, every as-yet-unquantized
dense projection, and layer-level parameters. Ultra's experiment quantized the complete layer first and only then
optimized `SU`/`SV`. QTIP also provides a later end-to-end cross-entropy phase. Finally, acknowledged open QTIP PR
`#20` repairs missing gradients into the continuous HYB lookup table during this fine-tuning. PGC16-v1 has a fixed
global decoder table, so that PR is authoritative evidence for the offline objective but cannot be copied without an
explicit codec extension. The next controlled arm is sequential projection installation plus transactional adaptation
of the still-dense layer parameters; it must pass external KLD/JSD/top-k before any default changes.

### Sequential QTIP-order W1 alignment rejection (2026-08-12)

Commit `55296117` implements the missing projection-at-a-time blockwise phase behind the same default-disabled
attachment. Each completed projection receives its own transactional alignment pass. The trainable set matches the
official QTIP blockwise implementation: already-installed fixed-trellis `SU`/`SV`, still-dense projections, and
decoder-layer parameters. The ordering is derived only from module-tree semantic tags and matches the authoritative
Llama recipe: V, Q, K, attention output, UP, GATE, DOWN. No projection-name matching is used. The differentiable
surrogate also mirrors production FP16 stable-Hadamard ordering while retaining FP32 inner accumulation; this fixed a
reproducible non-finite gate-projection surrogate even though the real QVQ runtime remained finite. All live and
reloaded outputs remain bitwise identical.

The controlled W1 test used Llama-3.2-1B-Instruct, the first two decoder layers (14 linears), 128 calibration rows
`[0, 128)`, `concat_size=2048`, batch 1, FP16, Python 3.14.6 with `PYTHON_GIL=0`, and PG506 `sm_80`. External
acceptance used 128 disjoint rows `[128, 256)`, 6,106 valid tokens, complete 128,256-way final logits, and both layer
hidden states. The alignment arms used one epoch at `1e-5`; the official blockwise `3e-6`, five-epoch recipe was also
run through the former failure point and demonstrated that the non-finite surrogate was operation-ordering drift,
not excessive learning rate. The baseline quantization time includes a one-time 118-second native extension build.

```text
+-----------------------+---------+---------+----------+----------+--------+--------+----------+---------+
| Arm                   | Quant s | Infer s | Fwd KLD  | JSD      | Top-1  | Top-5  | RMSE     | Rel-L2  |
+-----------------------+---------+---------+----------+----------+--------+--------+----------+---------+
| Fixed trellis         |  182.99*|   1.457 | 0.468343 | 0.096169 | 0.7162 | 0.6962 | 1.218632 | 0.38222 |
| Final-only align      |   75.99 |   1.471 | 0.511680 | 0.104293 | 0.6729 | 0.6921 | 1.225725 | 0.38444 |
| Sequential tree-order |   94.36 |   1.586 | 0.487341 | 0.098570 | 0.7008 | 0.6967 | 1.220173 | 0.38270 |
| Sequential QTIP-order |  100.70 |   1.471 | 0.472772 | 0.092463 | 0.7304 | 0.6876 | 1.191721 | 0.37378 |
+-----------------------+---------+---------+----------+----------+--------+--------+----------+---------+
* Includes the one-time 118-second CUDA extension build.
```

Authoritative ordering matters: versus tree order it improves every reported final metric except top-5. Versus the
unaligned fixed-trellis control it improves JSD by 3.85%, top-1 by 1.42 percentage points, RMSE by 2.21%, and
relative-L2 by 2.21%, but forward KLD is still 0.95% worse and top-5 is 0.86 percentage points worse. Both aligned
layers improve strongly (layer-1 KLD falls from `0.486638` to `0.314943`), yet the final distribution remains mixed.
This confirms that local decoder MSE does not preserve downstream error correlations. The feature remains
default-disabled. The next controlled phase is the official end-to-end objective after all selected projections are
installed; it must gate on disjoint final-logit KLD/JSD/top-k, not layer MSE alone.

Commit `66c4ef5b` later found that this sequential arm still differed from the author implementation in its layer
input population. QTIP's parent process forwards the untouched dense model and gives every blockwise worker the
dense input and dense output for that layer. Ultra's generic quantization stream instead contains the reconstructed
outputs of preceding QVQ layers; the attachment was using that noisy input both to create its pristine target and to
train the mixed layer. The table above remains valid evidence for noisy-input alignment, but it is not evidence for
the official clean-input/clean-target blockwise phase.

The correction uses the lifecycle's existing clean-stream attachment hooks. QVQ now owns only the current and next
layer's pristine CPU inputs, transfers the current clean input into its already bounded replay state, and leaves the
normal noisy Hessian/error-replay stream unchanged. A pre-fix regression supplies distinct clean and noisy layer-1
inputs and proves that training captures the clean value without aliasing. The new statements and branches have
measured 100% line/branch coverage under Python 3.14.6 with `PYTHON_GIL=0`. Output alignment remains explicit and
default-disabled; its W1/W1.5/W2 accuracy rows must be rerun before interpreting the corrected blockwise recipe.

### Fixed-trellis end-to-end W1 alignment acceptance (2026-08-12)

The follow-up isolates QTIP's end-to-end soft-target cross-entropy objective while preserving the QVQ codec. It
trains only the existing `SU`/`SV` auxiliaries for the 14 QVQ modules; trellises, dense weights, embeddings,
normalization weights, and the language-model head remain fixed. Thirty-two rows `[0, 32)` train for one epoch,
16 disjoint rows `[96, 112)` select the best state by forward KLD, and 128 rows `[128, 256)` form the external gate.
Metrics exclude padding and the first token of each row because next-token logits are compared. The gate contains
5,978 valid next-token distributions. Parameters use their native FP16 runtime precision under FP16 autocast, and
two batch gradients are averaged before each optimizer update.

An initial exact-QTIP control (`quant_model.float()` and summed, unnormalized gradients across update frequency)
produced a non-finite training loss. A forward trace showed this was not a property of QTIP's objective: CUDA
autocast narrowed the differentiable surrogate's nominally FP32 inner matmul back to FP16. The factorized inner
result can exceed `65504` before the output Hadamard and `SV` restore the completed linear's range. Commit
`6103a316` disables autocast around that reduction and adds a CUDA overflow regression. With that correction, the
source-faithful FP32/summed-gradient arm is finite at both W1.5 and W2.

```text
+-------------------------+----------+----------+--------+--------+----------+
| Arm                     | Fwd KLD  | JSD      | Top-1  | Top-5  | Total s  |
+-------------------------+----------+----------+--------+--------+----------+
| Fixed baseline          | 0.470614 | 0.096482 | 0.7143 | 0.6958 |      --  |
| Fixed + E2E native      | 0.426681 | 0.087817 | 0.7253 | 0.7001 |   25.69  |
| QTIP-order baseline     | 0.476563 | 0.092991 | 0.7516 | 0.6871 |      --  |
| QTIP-order + E2E native | 0.397001 | 0.083030 | 0.7591 | 0.7009 |   25.82  |
+-------------------------+----------+----------+--------+--------+----------+
```

End-to-end alignment improves every distributional gate from either starting point. From QTIP-order it lowers KLD
16.70%, lowers JSD 10.71%, raises top-1 0.75 percentage points, and raises top-5 1.38 points. Against untouched QVQ,
the combined blockwise-plus-E2E result lowers KLD 15.64%, lowers JSD 13.94%, raises top-1 4.48 points, and raises
top-5 0.51 points. A second full-logit evaluator (6,106 non-padding hidden/logit positions, including each row's
first position) confirms KLD `0.395273`, JSD `0.082716`, top-1 `0.7371`, top-5 `0.7015`, RMSE `1.136785`, and
relative-L2 `0.35655`, versus the untouched values `0.468343`, `0.096169`, `0.7162`, `0.6962`, `1.218632`, and
`0.38222`.

The experiment also exposed two loaded-checkpoint writer defects. Rebuilding a QVQ shell first converted every
dense module outside the partial two-layer scope into an empty QVQ module, producing NaN logits. Filtering by source
trellis fixed corruption but silently discarded live `SU`/`SV` updates. Commit `06955608` makes loaded QVQ save its
already-serialized live runtime tensors directly, as EXL3 does. The corrected checkpoint retains 14 trellises and
132 dense weights, reloads finite, and matches live held-out metrics exactly. The experiment additionally requires an
exact SHA-256 match over disjoint non-padding FP16 logits before reporting a saved candidate. Production integration
remains default-disabled until this objective is attached transactionally and validated on full-model W1/W1.5/W2.

The same stable control was then applied to the preserved full-model W1.5 checkpoint with all 112 projections
quantized. This is the first test in this series that crosses every decoder layer, rather than relying on a partial
model whose later dense layers can compensate for the quantized prefix.

```text
+-----------------------+----------+----------+--------+--------+----------+
| Full-model W1.5 arm   | Fwd KLD  | JSD      | Top-1  | Top-5  | Total s  |
+-----------------------+----------+----------+--------+--------+----------+
| Fixed baseline        | 0.944199 | 0.178817 | 0.5458 | 0.5811 |      --  |
| Fixed + E2E native    | 0.785362 | 0.152337 | 0.5843 | 0.6117 |    81.43 |
| SU/SV + QTIP math     | 0.785332 | 0.152337 | 0.5845 | 0.6115 |    69.30 |
| All params + QTIP     | 0.772983 | 0.149206 | 0.5843 | 0.6134 |    66.21 |
+-----------------------+----------+----------+--------+--------+----------+
```

Across the same 5,978-token external gate, KLD falls 16.82%, JSD falls 14.81%, top-1 rises 3.85 percentage points,
and top-5 rises 3.06 points. The saved checkpoint retains all 112 QVQ trellises and its reloaded evaluation metrics
are exactly equal to the live metrics; its live and reloaded validation-logit SHA-256 values are both
`bb9da3466112f348a6c1e5ef08628f3f845db314a68d20c7cb1f5086015862df`. After the autocast correction, QTIP math
produces effectively the same W1.5 result: KLD differs by only `0.0000306`, while top-1 is 0.017 percentage points
higher and top-5 is 0.017 points lower. Its exact reload digest is
`acf927f19410d2f4193a939f477c67c3961c68e86494e9eae648ecc907b8e227`.

The full-model W2 control confirms the improvement is not specific to W1.5:

```text
+-----------------------+----------+----------+--------+--------+----------+
| Full-model W2 arm     | Fwd KLD  | JSD      | Top-1  | Top-5  | Total s  |
+-----------------------+----------+----------+--------+--------+----------+
| Fixed baseline        | 0.429785 | 0.089487 | 0.6932 | 0.6916 |      --  |
| Fixed + E2E native    | 0.344952 | 0.073401 | 0.7153 | 0.7229 |    69.76 |
| SU/SV + QTIP math     | 0.345295 | 0.073256 | 0.7374 | 0.7232 |    69.69 |
| All params + QTIP     | 0.335304 | 0.071180 | 0.7166 | 0.7245 |    69.40 |
+-----------------------+----------+----------+--------+--------+----------+
```

QTIP's optimizer actually receives `quant_model.parameters()`, not only QVQ's `SU`/`SV`. That distinction is now an
explicit experimental axis. The SU/SV-only QTIP-math arm lowers W2 KLD 19.66%, lowers JSD 18.14%, raises top-1 4.42
percentage points, and raises top-5 3.15 points. It is substantially better than native/mean on top-1 (+2.21
points), while native/mean has only a `0.000343` KLD advantage. The exact live/reload digest for this arm is
`f57471390d9808671afc4db1bd5dc5c13be463f17c679be295c3ba8f8898609d`.

The official all-parameter scope trains 263,440,384 parameters in this checkpoint: QVQ `SU`/`SV` plus the remaining
dense embeddings, norms, biases, and tied language-model head. At W1.5 it improves KLD, JSD, and top-5 beyond the
SU/SV-only arm without changing top-1 materially. At W2 it improves KLD by another 2.89% and JSD by another 2.83%,
but top-1 is 2.07 points below SU/SV-only. Neither scope dominates every external metric, so both remain research
arms and default-disabled. The all-parameter W1.5 and W2 save/reload digests are respectively
`34ce8d8f2e8f0de34975e209a5fed7b77ad88c75f14c9aa3163d1a45e4fc5623` and
`d908da24cb495783e981e5d4edccd940dfa624f8dc907fe0a998cff47ae45ea5`.

### Full-length 256-row W1.5 alignment replication (2026-08-13)

Commit `bf48a3f1` repeated the full-model all-parameter QTIP-math arm without fixed-length packing or truncating
calibration rows. The 256 training rows `[0, 256)` retain their natural lengths up to 1,993 tokens under a 2,048-token
cap and batch 1. Validation uses disjoint rows `[1024, 1088)` (20,614 next-token distributions), and the external
gate uses disjoint rows `[1152, 1280)` (40,824 distributions). Per-batch trimming removes only columns that are
padding for every row in that batch; the valid-token masks and logits are bitwise unchanged by the trim.

```text
+----------------------+----------+----------+--------+--------+--------------+
| Full-length W1.5 arm | Fwd KLD  | JSD      | Top-1  | Top-5  | Valid tokens |
+----------------------+----------+----------+--------+--------+--------------+
| Validation baseline  | 0.865197 | 0.169856 | 0.6606 | 0.5932 |       20,614 |
| Validation aligned   | 0.618398 | 0.123491 | 0.7097 | 0.6327 |       20,614 |
| Evaluation baseline  | 0.796109 | 0.156563 | 0.6725 | 0.5964 |       40,824 |
| Evaluation aligned   | 0.566696 | 0.113185 | 0.7268 | 0.6384 |       40,824 |
+----------------------+----------+----------+--------+--------+--------------+
```

On the external gate, alignment lowers KLD by 28.82%, lowers JSD by 27.71%, raises top-1 by 5.44 percentage points,
and raises top-5 by 4.20 points. The run trained the same 263,440,384-parameter official scope for one epoch at
`1e-5`, used summed gradients across an update frequency of two, took 461.97 seconds on one PG506-230, and ran under
Python 3.14.6 with `PYTHON_GIL=0`. It accepted on the disjoint validation gate, retained all 112 fixed trellises, and
saved `/monster/data/model/Llama-3.2-1B-Instruct-QVQ-W1.5-block-cal512-e2e-qtip-all-256full`.

Live and reloaded evaluation metrics are exactly equal. The live and reloaded non-padding validation-logit SHA-256 is
`c1a956f603669cdf83ae6c8d01be46a5698585c83c38e192a3df5569a11c4826`. This is stronger evidence than the 32-row
pilot, but the absolute metrics are not directly comparable because this run deliberately uses different, much
larger validation/evaluation slices. Task-level Evalution remains the promotion gate, and the feature remains
default-disabled while that suite runs.

The saved parameter deltas also constrain interpretation of this result. They were measured directly against the
unaligned W1.5 checkpoint, streaming each common safetensors entry through an FP64 sum-of-squares accumulator. The
tied embedding/language-model-head tensor dominates the official all-parameter scope:

```text
+----------------+------------------+------------------+------------------+
| Parameter class| SU/SV-only 32-row| All-param 32-row | All-param 256-row|
+----------------+------------------+------------------+------------------+
| Tied embed/head|       0.000000017|          0.002834|          0.010772|
| QVQ SU         |       0.000051024|          0.000050|          0.000162|
| QVQ SV         |       0.002537580|          0.002475|          0.008065|
| Norms          |       0.000000000|          0.000033|          0.000310|
+----------------+------------------+------------------+------------------+
```

Values are relative L2 deltas, not output metrics. The 256-row improvement therefore includes a 1.08% update to the
262,668,288-parameter tied embedding/head, not only correction of QVQ's 704,512 auxiliary values. That is faithful to
the official `Adam(quant_model.parameters())` implementation, but it is dense-model adaptation rather than a pure
codec improvement. SU/SV-only and all-parameter results must remain separately labeled, and task quality must decide
whether the larger head update generalizes.

This arm is faithful to the official end-to-end optimizer scope and soft-target objective, but it is not a faithful
reproduction of the complete QTIP training recipe. The current public source defaults to
`8192 * 4096 = 33,554,432` full-length Hessian tokens. The paper instead specifies `4096 * 8192` for Llama 3/3.1,
which has the same token count but materially different sequence geometry; its Llama-2 recipe is `6144 * 2048`.
The source then aligns every newly installed projection for five epochs
on 256 training sequences of length 4,096 (`5,242,880` token exposures per projection), in
`v,q,k,o,up,gate,down` order. Only after that blockwise phase does it run four end-to-end epochs on 512 training
sequences of length 4,096 (`8,388,608` token exposures), with batch 2 and update frequency 4. The example requests
lookup-table training, but public main does not propagate a lookup gradient; unmerged author PR #20 corrects that
path and must be tested separately.
The author data loader selects individual documents that already reach the requested context length; it does not use
the separate concatenating sampler for these two scripts.

The QVQ arm above instead starts from a checkpoint with no blockwise output alignment, uses 512 natural calibration
rows containing 188,256 Hessian tokens, and performs one end-to-end epoch over 256 natural rows containing 87,124
valid next-token positions. Its Hessian population is therefore 178.24 times smaller than the author default, and
its end-to-end token exposure is 96.28 times smaller than the author example. Fixed PGC16 has no trainable HYB lookup
table, matching the effective published path but not the intended PR #20 correction. The exact comparison is:

```text
+--------------------------+----------------------+----------------------+------------------+
| Offline phase            | Official QTIP example| Current QVQ arm      | Exposure ratio   |
+--------------------------+----------------------+----------------------+------------------+
| Input Hessian (Llama 3)  | 33,554,432 tokens    | 188,256 tokens       | 178.24x          |
| Blockwise per projection | 5,242,880 tokens     | 0 tokens             | absent           |
| End-to-end               | 8,388,608 tokens     | 87,124 tokens        | 96.28x           |
| Lookup behavior          | fixed on public main | fixed PGC16 table    | matched class    |
+--------------------------+----------------------+----------------------+------------------+
```

Consequently, a downstream task rejection of this checkpoint rejects the current fixed-trellis retrofit. It does
not by itself disprove the paper's much larger two-stage recipe. A complete control must add the blockwise phase,
increase Hessian and optimization exposure, retain disjoint validation/evaluation, and still pass QVQ runtime
save/reload and task-level gates. The exact original RedPajama source is currently unavailable upstream (author issue
`#33`), so any such control must explicitly identify its substitute dataset rather than imply bitwise reproduction.

### Author-control and clean-input discriminators (2026-08-13)

Commit `9be919c4` audited the complete public QTIP repository at `e90c6688`, including every pull-request ref. All
closed pull requests are merged into public main; PR #20 is the only unmerged author change. It fixes the intended
differentiable HYB lookup path and adds an alternative inference kernel, but does not change the published BlockLDLQ
recurrence. Public main seeds one CUDA random-sign stream per layer, uses the W2 `scale_override=0.9` example, and
runs the sequential blockwise phase in `v,q,k,o,up,gate,down` order. QVQ's semantic module-tree ordering matches that
order when the output-alignment attachment is enabled. Its fixed PGC16 table and per-module deterministic sign seed
remain deliberate codec differences rather than hidden source fixes.

The released `relaxml/Llama-3.1-8b-Instruct-QTIP-2Bit` checkpoint provides a useful independent control on what the
author training actually changes. A range read of its safetensors auxiliary region shows that the trained HYB lookup
has RMS `0.968332`, matching the author's documented Gaussian-MSE constant. Most BF16 `SU` magnitudes remain exactly
one, while `SV` becomes an output-channel scale: sampled `SV` absolute-value coefficients of variation range from
1.2% to 5.9%, with layer-0 `v_proj` at 13.8%. The author blockwise/end-to-end recovery is therefore dominated by
output-side scaling and later mixed-layer adaptation, not a large input-sign amplitude change.

On one fixed 128-step standard-Gaussian W2 sequence, using the same QVQ tail-biting solver, PGC16-v1 has relative
MSE `0.0699920`; the released trained HYB table has `0.0673762`, a 3.9% relative distortion advantage. That is real
but too small to explain the full-model task cliff by itself. Applying the QTIP scale factor blindly to PGC16 is also
not justified: the matched PGC16 sweep gives `0.0680156` at factor `0.9`, `0.0699920` at `1.0`, and `0.0712389` at
`1/0.9`. PGC16 and trained HYB have different reconstruction geometry, so the author's HYB scale cannot be treated
as a codec-independent constant. Full-module held-out controls remain the authority.

The clean-input correction from `66c4ef5b` was then tested on two full Llama-3.2 decoder layers using 128 natural
calibration rows and 128 disjoint evaluation rows (`[128, 256)`, 6,106 valid tokens). Both arms run FP16 on one
PG506-230 under Python 3.14.6 with `PYTHON_GIL=0`; the aligned checkpoint reload is exactly equal to its live model.

```text
+--------------------+----------+----------+--------+--------+--------+----------+----------+
| W2 arm             | Fwd KLD  | JSD      | Top-1  | Top-5  | Rel-L2 | L0 KLD   | L1 KLD   |
+--------------------+----------+----------+--------+--------+--------+----------+----------+
| Fixed baseline     | 0.091728 | 0.020802 | 0.8885 | 0.8608 | 0.1827 | 0.034640 | 0.076192 |
| Clean-input align  | 0.084574 | 0.019285 | 0.8501 | 0.8624 | 0.1613 | 0.028218 | 0.067069 |
+--------------------+----------+----------+--------+--------+--------+----------+----------+
```

Clean-input alignment lowers KLD 7.8%, JSD 7.3%, and relative-L2 11.7%, but loses 3.83 top-1 percentage points.
All 14 projection-local MSE gates accepted, demonstrating that local clean-target improvement is not a sufficient
promotion criterion. The attachment remains default-disabled. The next decisive controls are the full-layer
layer-0 `v_proj` exclusion and exact held-out PGC16 scale arms. A code-and-test audit also confirmed that the existing
attachment already implements the author blockwise state transition: every later pass reconstructs and re-optimizes
all earlier fixed-trellis SU/SV payloads, while accepted future dense-weight and layernorm updates persist into their
later encoding. The two-pass regression verifies that both trellises remain fixed, the earlier SV changes again, the
new SV changes, and both live replay weights exactly match their committed payloads.

The exact scale controls were also run through two complete decoder layers and the same disjoint held-out slice.
The factor is folded into `SV`, so live/reloaded inference remains exact without adding checkpoint metadata; these
are diagnostic snapshots rather than supported configuration.

```text
+--------------------+----------+----------+--------+--------+--------+
| W2 scale/arm       | Fwd KLD  | JSD      | Top-1  | Top-5  | Rel-L2 |
+--------------------+----------+----------+--------+--------+--------+
| Production 1.0000  | 0.091728 | 0.020802 | 0.8885 | 0.8608 | 0.1827 |
| PGC micro 0.9000   | 0.122238 | 0.027056 | 0.8551 | 0.8391 | 0.1998 |
| QTIP 1/0.9         | 0.095380 | 0.021722 | 0.8634 | 0.8412 | 0.1699 |
+--------------------+----------+----------+--------+--------+--------+
```

Both alternatives are rejected. In particular, the exact author-derived scale lowers relative-L2 but worsens KLD,
top-1, and top-5. This closes the fixed global-scale hypothesis for PGC16 and provides another concrete case where
reconstruction error and model-output accuracy disagree.

Three deterministic per-module sign-seed salts were tested without changing the calibration rows, solver, scale,
packing, or inference. Seed choice materially moves both proxy and final-output metrics, but none dominates the
existing CRC-derived stream:

```text
+------------+----------+----------+--------+--------+--------+
| 2-layer W2 | Fwd KLD  | JSD      | Top-1  | Top-5  | Rel-L2 |
+------------+----------+----------+--------+--------+--------+
| CRC base   | 0.091728 | 0.020802 | 0.8885 | 0.8608 | 0.1827 |
| Salt 1357  | 0.093291 | 0.021097 | 0.8662 | 0.8535 | 0.1780 |
| Salt 2468  | 0.087252 | 0.019995 | 0.8497 | 0.8586 | 0.1728 |
| Salt 5a5a  | 0.091384 | 0.020569 | 0.8485 | 0.8530 | 0.1659 |
+------------+----------+----------+--------+--------+--------+
```

All alternate streams lower relative-L2, two lower KLD, and every one loses at least 2.23 top-1 percentage points.
This rules out replacing the production seed stream from proxy evidence. A future seed search would require a
disjoint final-output gate and would multiply quantization cost; it is not an accuracy-safe default mechanism.

The exact QTIP-inspired layer-0 `v_proj` exclusion was tested across all 16 Llama-3.2 decoder layers. The validator
asserted exactly 111 QVQ modules and one exact exclusion, then reload matched live output exactly. Comparison uses
the same natural calibration population and held-out `[1152, 1280)` slice as the 112-module baseline.

```text
+----------------------+----------+----------+--------+--------+--------+
| Full W2, 16 layers   | Fwd KLD  | JSD      | Top-1  | Top-5  | Rel-L2 |
+----------------------+----------+----------+--------+--------+--------+
| All 112 projections  | 0.423678 | 0.089536 | 0.7093 | 0.6877 | 0.3709 |
| Dense layer-0 v_proj | 0.401941 | 0.090020 | 0.7068 | 0.6968 | 0.3697 |
+----------------------+----------+----------+--------+--------+--------+
```

The exclusion lowers KLD 5.13% and raises top-5 0.90 points, but worsens JSD and top-1. Layer-0 hidden-state KLD
falls from `0.032324` to `0.026019`, yet the final mixed result is not Pareto-safe and remains far from recovering
the task cliff. This control is rejected as a general default; model-specific dense exceptions still require an
independent task gate.

### Full-model uniform-rate task sweep (2026-08-13)

The current production-default uniform-rate sweep quantizes all 16 Llama-3.2-1B-Instruct decoder layers (112
projections) from 512 natural `neuralmagic/calibration:LLM` rows: 188,256 valid tokens, zero padding, no packing,
batch 1, FP16 execution, fixed PGC16-v1, Euclidean Viterbi, and one tail-biting candidate. W2.5/W3/W4 snapshots
were produced at `dfd9cbc1`; W3.5/W4.5 snapshots were produced at `f1c7a738`. Every completed snapshot has finite
output and bit-exact live-after-save and reload-versus-live logits.

```text
+------+----------+----------+----------+----------+--------+--------+------------------------------+
| Rate | Quant s  | Eff BPW  | Fwd KLD  | JSD      | Top-1  | Top-5  | Snapshot                     |
+------+----------+----------+----------+----------+--------+--------+------------------------------+
| W2.5 | 898.712  | 2.52317  | 0.134099 | 0.029030 | 0.7838 | 0.7405 | ...-W2.5-ALL-dfd9cbc1        |
| W3   | 927.504  | 3.02317  | 0.044396 | 0.010430 | 0.9189 | 0.8162 | ...-W3-ALL-dfd9cbc1          |
| W3.5 | 582.831  | 3.52317  | 0.029498 | 0.007610 | 0.9189 | 0.8649 | ...-W3.5-ALL-f1c7a738        |
| W4   | 931.116  | 4.02317  | 0.009817 | 0.002410 | 1.0000 | 0.9243 | ...-W4-ALL-dfd9cbc1          |
| W4.5 | 469.554  | 4.52317  | 0.008746 | 0.002197 | 0.9730 | 0.9297 | ...-W4.5-ALL-f1c7a738        |
+------+----------+----------+----------+----------+--------+--------+------------------------------+
```

The four-prompt lifecycle diagnostics are not promotion gates; the full-row tasks below are authoritative. The W4
GSM8K result is 1.24 points above the stored dense reference, which is treated as noise-sized/slightly favorable and
not as evidence that quantization improves the model. A matching dense FP16 MMLU-History baseline was subsequently
collected across all 930 rows at `cf824d53`, batch 16, without a chat template.

Each parenthesized percentage is a dense-relative score, computed as `quantized / dense * 100`; it is not the fraction
of the quantization-induced gap recovered. `Macro dense-relative` is the unweighted macro mean across normalized ARC,
GSM8K, MMLU-STEM, and MMLU-History. Raw ARC remains visible but is not included because counting it alongside
normalized ARC would double-weight the same examples. Coverage is explicit for incomplete rows. A value above 100%
means the quantized run exceeded this dense run; it is not automatically a significant model-quality gain.

```text
+------+--------------------+--------------------+--------------------+--------------------+--------------------+------------------+
| Rate | ARC acc (dense-rel)| ARC norm (dense-rel)| GSM8K (dense-rel)  | STEM (dense-rel)   | History (dense-rel)| Macro dense-rel |
+------+--------------------+--------------------+--------------------+--------------------+--------------------+------------------+
| Dense| 0.324232 (100.00%) | 0.351536 (100.00%) | 0.472291 (100.00%) | 0.394200 (100.00%) | 0.546237 (100.00%) | 100.00% (4/4)   |
| W2.5 | 0.298635 ( 92.11%) | 0.331058 ( 94.17%) | 0.314309 ( 66.55%) | 0.385347 ( 97.75%) | 0.465591 ( 85.24%) |  85.93% (4/4)   |
| W3   | 0.290102 ( 89.47%) | 0.337031 ( 95.87%) | 0.401985 ( 85.11%) | 0.374564 ( 95.02%) | 0.507527 ( 92.91%) |  92.23% (4/4)   |
| W3.5 | 0.308874 ( 95.26%) | 0.335324 ( 95.39%) | 0.479735 (101.58%) | 0.403108 (102.26%) | 0.523656 ( 95.87%) |  98.77% (4/4)   |
| W4   | 0.312287 ( 96.32%) | 0.348976 ( 99.27%) | 0.484698 (102.63%) | 0.398351 (101.05%) | 0.534409 ( 97.83%) | 100.20% (4/4)   |
| W4.5 | 0.313140 ( 96.58%) | 0.353242 (100.49%) | 0.471464 ( 99.82%) | 0.394545 (100.09%) | 0.540860 ( 99.02%) |  99.85% (4/4)   |
+------+--------------------+--------------------+--------------------+--------------------+--------------------+------------------+
```

Matched YAQA arms run at W2.5, W3, and W3.5 with seed 0 and regularization `1e-4`. The YAQA512 snapshots deliberately
set `minimum_sequences=512` to isolate rounding on the same 512-row population. They use fewer sequences than the
paper's smallest reported 2K-sequence ablation and much shorter natural rows than its approximately 2,048-token
sequences, so sequence count alone is not a matched Fisher budget. They are not production-default evidence. W3 and
W3.5 passed exact reload parity. The completed YAQA512 snapshots reused ordinary rows `[0,512)` and are overlap
controls; strict disjoint YAQA slices are documented in `docs/qvq.md`.

Absolute YAQA512 point estimates received for this running campaign are below. The task row counts match the uniform
baselines: ARC has 1,172 rows, GSM8K Platinum has 1,209, STEM has 3,153, and History has 930. Snapshot paths, exact
quantization/evaluator commits, per-arm valid-token totals, and the final reload/evaluator status still need to be
copied from the remote runner before this table is archival evidence.

```text
+------+----------+----------+----------+----------+----------+
| Rate | ARC acc  | ARC norm | GSM8K   | STEM     | History  |
+------+----------+----------+----------+----------+----------+
| W2.5 | 0.291809 | 0.329352 | 0.381307 | running  | queued   |
| W3   | 0.308020 | 0.341297 | 0.435897 | 0.387885 | running  |
| W3.5 | 0.313993 | 0.344710 | 0.470637 | running  | queued   |
+------+----------+----------+----------+----------+----------+
```

### YAQA overlap-control evidence and next recovery gate

The following table compares each completed YAQA512 score with its same-rate ordinary-QVQ score. `Raw delta` is the
benchmark accuracy change in percentage points. `Dense-relative delta` is the change in `score / dense_score * 100`
and must not be mislabeled as raw accuracy. Incomplete task columns are not averaged with completed ones.

```text
+------+--------------------+--------------------+--------------------+--------------------+
| Rate | ARC delta          | ARC norm delta     | GSM8K delta        | STEM delta         |
+------+--------------------+--------------------+--------------------+--------------------+
| W2.5 | -0.683 / -2.11     | -0.171 / -0.48     | +6.700 / +14.19    | pending            |
| W3   | +1.792 / +5.53     | +0.427 / +1.22     | +3.391 /  +7.18    | +1.332 / +3.38     |
| W3.5 | +0.512 / +1.58     | +0.939 / +2.67     | -0.910 /  -1.93    | pending            |
+------+--------------------+--------------------+--------------------+--------------------+
| Each cell: raw accuracy percentage points / dense-relative percentage points               |
+---------------------------------------------------------------------------------------------+
```

The point-estimate reading is narrow; paired outputs and uncertainty intervals have not yet been captured:

- W3 has higher point estimates on every completed task metric and is the highest-priority strict-split validation
  arm. ARC accuracy and normalized accuracy score the same 1,172 examples and are not independent evidence.
- W2.5 has the largest observed GSM8K point-estimate gain but lower ARC point estimates, making it the secondary
  compression-frontier validation arm rather than a default candidate.
- W3.5 is mixed and incomplete. Its result does not establish a YAQA saturation point.
- Across completed W2.5/W3 columns, GSM8K has the largest dense gap and the largest YAQA gain. This does not yet prove
  that loss is generally concentrated in multi-step reasoning.
- These are same-population, 512-sequence overlap controls below the paper's smallest reported 2K-sequence ablation.
  They motivate a strict-split experiment; they do not select a production rounding or mixed-rate policy.

The immediate decision gate is:

- [ ] Record each YAQA snapshot path, quantization commit, evaluator commit, hardware/runtime/prompt contract, exact
  ordinary and Fisher slices, valid-token/context-length statistics, and reload-parity result beside the score table.
- [ ] Finish pending task columns and retain per-example IDs, predictions, exact-match state, and generated text.
- [ ] Complete at least one matched, strictly disjoint YAQA arm at the paper's smallest reported 2,000-sequence
  ablation size; prioritize W3, then W2.5 if resources permit. Keep ordinary calibration, runtime, seed, and evaluator
  fixed. Record valid tokens and the context-length distribution, and gate on held-out proxy convergence rather than
  treating row count as a universal minimum.
- [ ] Report raw deltas, dense-relative scores, correct-to-wrong and wrong-to-correct flips, 95% paired-bootstrap
  intervals, exact McNemar tests, and normalized Kronecker-product/eigenspace plus held-out-proxy convergence;
  individual `H_I`/`H_O` scales are non-identifiable. Do not compare macro means unless the exact same task set is
  complete.

If the strict W3 arm confirms a meaningful low-rate gain, run the propagation-aware precision-rescue protocol in
`docs/qvq.md`. The ARC/GSM8K/STEM/History results above have already influenced experiment selection and are now
development evidence; later scores on the same rows are descriptive, not an unbiased final evaluation. Use a new
allocation-search set, a locked confirmation set or cross-fitting, and a preregistered external final suite/split.

Use this hierarchy:

1. Screen decoder bands under full live execution.
2. Within positive bands, screen module-tree semantic groups: attention Q/K/V, attention output, MLP gate/up, and MLP
   down. These are hypotheses to test, not presumed sensitivity rankings.
3. Within positive groups, screen exact modules only where the lifecycle can preserve all transform and recovery
   semantics.
4. Recompute gains after every accepted promotion and inspect shortlist pairs. A module's value is conditional on the
   already-promoted set because omitted cross-module curvature and live-input drift create interactions.
5. For groups proven propagation-sensitive, generate a small set of same-rate QVQ candidates from controlled
   rounding, tail-biting, scale, or seed choices and rerank them on the allocation-search set. A same-rate rescue
   is preferable to extra bits when it survives in-loop replay and final gates.
6. Re-quantize from dense with the winning candidate choices and dynamic rate overrides. Post-hoc replacement is only
   a cheap sensitivity screen and cannot reverse errors already introduced by sequential quantization/recovery.

For current accepted action set `A` and positive-byte action `a`, rank the search by conditional held-out loss
reduction per actual byte:

```text
S_a(A) = [L(A) - L(A union {a})] / [B(A union {a}) - B(A)].
```

`L` must be a predeclared, normalized, lower-is-better scalar used only for search ordering; retain the complete vector
of full-model teacher KL/JSD, token/sequence margins and top-k, and task-like paired behavior. GSM8K also requires
deterministic generated-answer and exact-match changes; a teacher-forced answer-token margin alone does not capture
autoregressive divergence. `B` includes planar codes and every auxiliary tensor/required-metadata byte; report logical
payload, effective BPW, and actual sharded checkpoint size separately. Handle zero-byte candidate changes first as
Pareto improvements rather than dividing by zero. Preserve the quality-versus-BPW Pareto frontier instead of choosing
a default from one opaque composite score.

Promote a policy only after in-loop re-quantization, exact save/reload parity, backend reconstruction gates, positive
paired evidence on the locked confirmation contract, predeclared guardrail margins, multiple calibration/Fisher
seeds, matched-BPW uniform and random-promotion controls, and one preregistered external benchmark evaluation. If
high-gain rescues are sparse, retain mixed precision. If the residual loss is diffuse, first improve independent YAQA
sample coverage and factor convergence; richer factor models are a later algorithm project because YAQA's efficient
solver assumes one Kronecker product.

## Remaining acceptance gates

### Queued full-model mixed-rate task gate

- [x] Quantize every Llama-3.2-1B-Instruct decoder layer with attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`
  at W3 and every other eligible projection at W1.5. Resolve the four attention roles from the model module tree's
  semantic tags, never path substring heuristics. Use the current production defaults: fixed PGC16-v1, Euclidean
  Viterbi, one tail-biting candidate, no scale override/search, no output alignment, no output-scale shrinkage, no
  exclusions, and the deterministic production seed. Keep unconditional FP32 accumulation, stable-Hadamard,
  serialization, and reload correctness fixes enabled.
- [x] Preserve the complete all-layer checkpoint and assert live-versus-reload equality before evaluation.
- [ ] Run every row of ARC-Challenge, GSM8K Platinum, MMLU-STEM, and the repository's MMLU-History subset using the
  Llama-3.2 evaluation/chat-template contract from `tests/models/test_llama3_2.py`, FP16 QVQ inference, and the
  repository evaluator's established batch controls.
- [ ] Record quantization, save, reload, and per-task inference wall times; exact task scores; effective BPW including
  auxiliary tensors; hardware; dtype; Python/PyTorch/CUDA versions; calibration slice; commit; and snapshot path in
  this file. Keep aggregate accept/reject separate from task-specific signals and label noise-sized changes using the
  agreed KLD/JSD, top-k, and confidence-interval rules.

The production-default run at quantization commit `4e90ea13` used Python 3.14.6 free-threaded, PyTorch 2.13.0+cu130,
one PG506-230 `sm_80` GPU, FP16 inference, batch 1, and 512 natural `neuralmagic/calibration:LLM` rows (188,256 valid
tokens, zero padding). Semantic module-tree tags selected exactly 64 Q/K/V/O projections at W3; the other 48
eligible MLP projections remained W1.5. The preserved snapshot is
`/monster/data/model/Llama-3.2-1B-Instruct-QVQ-QKVO-W3-REST-W1.5-4e90ea13`. Live-after-save and reload-versus-live
logits are bit-exact. Evalution commit `4a85ef8a` runs one full task per GPU. ARC and GSM8K completed; MMLU-STEM and
MMLU-History were deliberately stopped after those two critical-task gates established a catastrophic regression.

```text
+--------------------------+-------------+-------------+-------------+-------------+-------------+
| Lifecycle/storage        | Load dense  | Quantize    | Save        | Reload      | Value       |
+--------------------------+-------------+-------------+-------------+-------------+-------------+
| Mixed W3/W1.5            | 1.239 s     | 408.713 s   | 0.930 s     | 10.396 s    |             |
| Payload BPW              |             |             |             |             | 1.758621    |
| Effective BPW (SU/SV)    |             |             |             |             | 1.781789    |
| Dense forward KLD / JSD  |             |             |             |             | .517834/.119753 |
| Dense top-1 / top-5      |             |             |             |             | .6757/.6216 |
+--------------------------+-------------+-------------+-------------+-------------+-------------+
```

```text
+------------------------+--------+------------+-------------+------------+----------+
| Full-row task          | Rows   | Mixed score| Dense ref   | Abs delta  | Eval s   |
+------------------------+--------+------------+-------------+------------+----------+
| ARC accuracy           | 1,172  | 0.269625   | 0.324232    | -0.054608  | 145.538  |
| ARC normalized         | 1,172  | 0.294369   | 0.351536    | -0.057167  | 145.538  |
| GSM8K Platinum         | 1,209  | 0.082713   | 0.472291    | -0.389578  | 985.015  |
| MMLU-STEM              | 3,153  | stopped    | 0.394200    | n/a        | n/a      |
| MMLU-History           |   930  | stopped    | unavailable | n/a        | n/a      |
+------------------------+--------+------------+-------------+------------+----------+
```

ARC regresses by 5.46/5.72 absolute points versus the repository's full Llama-3.2 reference, which is a meaningful
critical-task loss. GSM8K regresses by 38.96 absolute points after all 1,209 rows and is catastrophic. Keeping Q/K/V/O
at W3 therefore does not rescue the cumulative full-model W1.5 MLP error. The mixed policy is rejected. The two MMLU
jobs were stopped rather than spending additional GPU time characterizing a policy that had already failed both
critical-task gates; the replacement gate keeps Q/K/V/O at W3 and raises all remaining eligible projections to W2.

The replacement run at commit `c66dd620` changes only the other 48 eligible projections from W1.5 to W2. It uses the
same model, semantic module selection, 512 natural calibration rows, 188,256 valid tokens, zero padding, batch 1,
FP16 execution, and production QVQ controls as the rejected run. The preserved snapshot is
`/monster/data/model/Llama-3.2-1B-Instruct-QVQ-QKVO-W3-REST-W2-c66dd620`. It contains exactly 64 W3 attention
projections and 48 W2 MLP projections; live-after-save and reload-versus-live logits are bit-exact.

```text
+--------------------------+-------------+-------------+-------------+-------------+-------------+
| Lifecycle/storage        | Load dense  | Quantize    | Save        | Reload      | Value       |
+--------------------------+-------------+-------------+-------------+-------------+-------------+
| Mixed W3/W2              | 1.298 s     | 428.143 s   | 0.953 s     | 17.578 s    |             |
| Payload BPW              |             |             |             |             | 2.172414    |
| Effective BPW            |             |             |             |             | 2.195582    |
| Payload bytes            |             |             |             |             | 264,241,152 |
| Auxiliary bytes          |             |             |             |             | 2,818,048   |
+--------------------------+-------------+-------------+-------------+-------------+-------------+
```

```text
+----------------------+-------------+-------------+-------------+-------------+-------------+
| Four-prompt metric   | W3/W1.5     | W3/W2       | Abs change  | Rel change  | Direction   |
+----------------------+-------------+-------------+-------------+-------------+-------------+
| Forward KLD          | 0.517834    | 0.241229    | -0.276605   | -53.41%     | better      |
| JSD                  | 0.119753    | 0.055504    | -0.064250   | -53.65%     | better      |
| Top-1 agreement      | 0.675676    | 0.675676    |  0.000000   |   0.00%     | unchanged   |
| Top-5 overlap        | 0.621622    | 0.702703    | +0.081081   | +13.04%     | better      |
| Relative L2          | 0.504557    | 0.452482    | -0.052074   | -10.32%     | diagnostic  |
+----------------------+-------------+-------------+-------------+-------------+-------------+
```

```text
+------------------------+--------+------------+-------------+------------+----------+
| Full-row task          | Rows   | Mixed score| Dense ref   | Abs delta  | Eval s   |
+------------------------+--------+------------+-------------+------------+----------+
| ARC accuracy           | 1,172  | 0.303754   | 0.324232    | -0.020478  | 116.345  |
| ARC normalized         | 1,172  | 0.333618   | 0.351536    | -0.017918  | 116.345  |
| GSM8K Platinum         | 1,209  | 0.273780   | 0.472291    | -0.198511  | 1,107.155|
| MMLU-STEM              | 3,153  | 0.317158   | 0.394200    | -0.077042  | 1,988.874|
| MMLU-History           |   930  | 0.401075   | unavailable | n/a        | 1,649.435|
+------------------------+--------+------------+-------------+------------+----------+
```

Raising only the MLP rate to W2 recovers about 63% of the W1.5 ARC-accuracy loss and 69% of its normalized-accuracy
loss. GSM8K also rises from `0.082713` to `0.273780`, recovering 49.0% of the W1.5-to-dense deficit, but its remaining
19.85-point loss is still catastrophic. MMLU-STEM loses 7.70 absolute points versus dense. The additional 0.414
effective BPW materially improves every available critical-task signal but does not make the mixed policy viable.

- CUDA reconstruction and QuantLinear parity: passed on `sm_80` for W1--W8 FP16/BF16.
- Fresh PGC16 CUDA inference latency/KLD/top-k results: passed and recorded in `docs/qvq_cuda.md`.
- CUDA reference-quantizer device selection, scheduling, path parity, throughput, and peak VRAM: passed.
- Matched two-layer W2--W8 GPTQ symmetric/asymmetric versus PGC16 QVQ: passed with the first 128
  `nm-calibration/LLM` rows and padding excluded from Hessians and all metrics.
- No W2 model-quality regression versus the retained HYB reference: pending model-level evidence.
- Learned model-level QVQ-v2 versus fixed v1 two-layer KLD/top-k/error matrix: completed on CUDA for W2--W8. V2
  improves its fitting objective at every rate but regresses at least one held-out weight relative-L2, KLD, RMSE, or
  top-5 metric at five of seven rates, so fixed v1 remains the accuracy-safe default.
- CUDA v2 learned-table reconstruction/QuantLinear matrix: passed on PG506-230 `sm_80` under Python 3.14.6
  free-threaded; the complete CUDA file reports 302 passed and two expected multi-device skips in the one-GPU run.
- Four-layer KLD/top-k comparison: deferred until the quantizer is faster or a suitable CUDA host is used.
- Full processor, replacement, save/load, reload inference, and Evalution lifecycle: passed for a two-layer hybrid
  Llama-3.2-1B checkpoint on `sm_80` under Python 3.14.6 free-threaded.
- Apple 2048x2048, 2048x8192, and 8192x2048 latency matrices: passed and recorded.

Commit and push every validated CUDA rate or lifecycle milestone so parallel hosts can synchronize without repeating
long work. Update `docs/qvq.md`, this handoff, and draft PR #244 whenever an acceptance status changes.

## Full Llama-3.2-1B QVQ/EXL3 evaluation sweep (`ef7cce80`)

FP16 full-task evaluation results. Percentages in parentheses are recovery versus the dense reference. Mean is the
average of ARC normalized, GSM8K, STEM, and History recovery; raw ARC accuracy is displayed but excluded from mean.
`O` is overlapping YAQA and `S` is strict-disjoint YAQA.

```text
| Arm              | GB     | EBPW   | ARC acc          | ARC norm         | GSM8K            | STEM             | History          | Mean % |
| Dense            |   —    |   —    | 0.3242(100.0000%)| 0.3515(100.0000%)| 0.4723(100.0000%)| 0.3942(100.0000%)| 0.5462(100.0000%)| 100.0000 |
| QVQ W2.5         | 0.8497 | 2.5232 | 0.2986(92.1100%) | 0.3311(94.1700%) | 0.3143(66.5500%) | 0.3853(97.7500%) | 0.4656(85.2400%) | 85.9300 |
| QVQ W3           | 0.9105 | 3.0232 | 0.2901(89.4700%) | 0.3370(95.8700%) | 0.4020(85.1100%) | 0.3746(95.0200%) | 0.5075(92.9100%) | 92.2300 |
| QVQ W3.5         | 0.9713 | 3.5232 | 0.3089(95.2600%) | 0.3353(95.3900%) | 0.4797(101.5800%)| 0.4031(102.2600%)| 0.5237(95.8700%) | 98.7700 |
| QVQ W4           | 1.0321 | 4.0232 | 0.3123(96.3200%) | 0.3490(99.2700%) | 0.4847(102.6300%)| 0.3984(101.0500%)| 0.5344(97.8300%) | 100.2000 |
| QVQ W4.5         | 1.0929 | 4.5232 | 0.3131(96.5800%) | 0.3532(100.4900%)| 0.4715(99.8200%) | 0.3945(100.0900%)| 0.5409(99.0200%) | 99.8500 |
| W2 Y512 O        | 0.7888 | 2.0232 | 0.2594(80.0000%) | 0.3063(87.1400%) | 0.2043(43.2600%) | 0.3010(76.3500%) | 0.3430(62.8000%) | 67.3900 |
| W2 Y1024 O       | 0.7888 | 2.0232 | 0.2756(85.0000%) | 0.3157(89.8100%) | 0.2382(50.4400%) | 0.3102(78.6900%) | 0.3441(62.9900%) | 70.4800 |
| W2 Y2048 O       | 0.7888 | 2.0232 | 0.2722(83.9500%) | 0.3148(89.5600%) | 0.2308(48.8600%) | 0.3441(87.3000%) | 0.4022(73.6200%) | 74.8400 |
| W2.5 Y512 O      | 0.8497 | 2.5232 | 0.2918(90.0000%) | 0.3294(93.6900%) | 0.3813(80.7400%) | 0.3616(91.7200%) | 0.4699(86.0200%) | 88.0400 |
| W2.5 Y1024 O     | 0.8497 | 2.5232 | 0.2969(91.5800%) | 0.3268(92.9600%) | 0.4069(86.1600%) | 0.3635(92.2000%) | 0.4430(81.1000%) | 88.1100 |
| W2.5 Y2048 O     | 0.8497 | 2.5232 | 0.2961(91.3200%) | 0.3353(95.3900%) | 0.3838(81.2600%) | 0.3603(91.4000%) | 0.4495(82.2800%) | 87.5800 |
| W3 Y512 O        | 0.9105 | 3.0232 | 0.3080(95.0000%) | 0.3413(97.0900%) | 0.4359(92.2900%) | 0.3879(98.4000%) | 0.5000(91.5400%) | 94.8300 |
| W3 Y1024 O       | 0.9105 | 3.0232 | 0.3046(93.9500%) | 0.3439(97.8200%) | 0.4500(95.2700%) | 0.3863(98.0000%) | 0.4935(90.3500%) | 95.3600 |
| W3.5 Y512 O      | 0.9713 | 3.5232 | 0.3140(96.8400%) | 0.3447(98.0600%) | 0.4706(99.6500%) | 0.3914(99.2800%) | 0.5247(96.0600%) | 98.2600 |
| W3.5 Y1024 S     | 0.9713 | 3.5232 | 0.3012(92.9000%) | 0.3447(98.0600%) | 0.4731(100.1800%)| 0.3895(98.8000%) | 0.5312(97.2400%) | 98.5700 |
| W2.5 Y2048 S     | 0.8497 | 2.5232 | 0.3131(96.5800%) | 0.3379(96.1200%) | 0.3747(79.3300%) | 0.3555(90.1900%) | 0.4516(82.6800%) | 87.0800 |
| W3 Y2048 S       | 0.9105 | 3.0232 | 0.3020(93.1600%) | 0.3302(93.9300%) | 0.4409(93.3500%) | 0.3819(96.8700%) | 0.5129(93.8976%) | 94.5128 |
| W3.5 Y2048 S     | 0.9713 | 3.5232 | 0.3072(94.7400%) | 0.3379(96.1200%) | 0.4773(101.0500%)| 0.3892(98.7200%) | 0.5290(96.8504%) | 98.1880 |
| EXL3 W2          | 0.7876 | 2.0116 | 0.2901(89.4700%) | 0.3183(90.5300%) | 0.2109(44.6600%) | 0.3194(81.0200%) | 0.3946(72.2400%) | 72.1100 |
| EXL3 W3          | 0.9093 | 3.0116 | 0.3063(94.4700%) | 0.3498(99.5100%) | 0.3978(84.2400%) | 0.3819(96.8700%) | 0.5022(91.9300%) | 93.1400 |
| EXL3 W4          | 1.0309 | 4.0116 | 0.3191(98.4200%) | 0.3473(98.7900%) | 0.4748(100.5300%)| 0.4009(101.7000%)| 0.5409(99.0200%) | 100.0100 |
```

All rows are complete. The strict-disjoint W3 and W3.5 History evaluations finished at 0.5129 and 0.5290.
The legacy W3 checkpoints use `format=qvq, vector_size=2`; they do not exercise the experimental V4 CUDA GEMV path
added by `ef7cce80`.
