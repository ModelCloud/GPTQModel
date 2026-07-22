# EoRA test model paths

- GPU 6: PCI bus `DE:00.0`; checkpoint: `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512`
- GPU 7: PCI bus `E4:00.0`; checkpoint: `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512`
- GPU 6 fixed EoRA adapter: `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora-rank128-eighfix`
- GPU 7 fixed EoRA adapter: `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/eora-rank128-eighfix`

## EoRA-active evaluation baseline

| Model | Checkpoint | ARC raw / normalized | GSM8K Platinum | Status |
|---|---|---:|---:|---|
| GPU 6 — GPTQ 4-bit, g128, activation scale search, GAR, EoRA r128 | `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512` | 53.33% / 54.44% | 89.00% | Passed |
| GPU 7 — GPTQ 3-bit, g64, EoRA r128 | `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512` | 22.70% / 21.93% | 0.00% | Failed quality |

## GPU 6 EoRA eigensolver fix

The INT4 EoRA adapter was regenerated from the unchanged dense/quantized model pair using the same first 512 calibration rows (181,796 tokens), descending-length order, batch size 1, BF16 factors, and rank 128. The original adapter and quantized checkpoint were not rewritten.

| GPU 6 EoRA mode | Adapter path | ARC raw / normalized | GSM8K Platinum | Status |
|---|---|---:|---:|---|
| Original unsafe adapter | `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora-rank128` | 53.33% / 54.44% | 89.00% | Passed, but contained ill-conditioned factors |
| Fixed truncated-pseudoinverse adapter | `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora-rank128-eighfix` | 55.38% / 55.55% | 90.98% | Passed |
| Fixed minus original | — | +2.0478 / +1.1092 pp | +1.9851 pp | Improved |

Exact correct counts improved from 625 to 649/1,172 for ARC raw, from 638 to 651/1,172 for ARC normalized, and from 1,076 to 1,100/1,209 for GSM8K Platinum. Both GSM8K runs had zero invalid outputs.

The fixed generation exercised two unsafe fallback cases:

- Layer 2 `mlp.down_proj`: retained 434/12,288 supported covariance directions and discarded 11,854 at or below `4.214e-1` (`123` negative eigenvalues).
- Layer 13 `self_attn.o_proj`: retained 2,112/4,096 directions and discarded 1,984 at or below `5.361e-1` (`1` negative eigenvalue).

Across the adapter, maximum factor magnitude fell from 1,826,816 to 23.125. For layer-2 `mlp.down_proj`, `lora_A` max-abs fell from 1,826,816 to 0.4414 and the reconstructed correction norm fell from 14,280,652 to 1.9215. On the diagnostic prompt, fixed layer-2 hidden-state cosine to dense BF16 was 0.99924 and last-token logit cosine was 0.99637.

ARC was evaluated with `GPTQ_TORCH` in 73.18 seconds. The first fixed-adapter GSM8K task used the safe Marlin fused-tail fallback and completed in 2,525.75 seconds, versus 2,756.73 seconds for the original `GPTQ_TORCH` run. After the cooperative workspace repair, the full Marlin rerun completed in 1,437.50 seconds: 43.09% less wall time and 1.757x faster than the safe fallback, or 47.85% less wall time than the original Torch run. Evalution reported 1,430.709 seconds generation, 5.184 seconds dataset loading, and 1.091 seconds scoring. ARC remains a strict same-backend adapter comparison; the GSM8K results span backend and runtime changes.

The current full evaluation used `GPTQMODEL_MARLIN_LORA_COOPERATIVE=1`, `GPTQMODEL_MARLIN_LORA_FUSED_MAX_M=16`, and `GPTQMODEL_MARLIN_LORA_CUDA_UP_ADD=1`. All 252 adapters had prepared cooperative state and 192-word workspaces; the preceding `M=1` smoke left zero dirty Marlin lock-prefix words. The rerun scored 1,100/1,209 (90.9843%) with zero invalid outputs, versus 1,095/1,209 (90.5707%) on the earlier fallback. With identical prompts, targets, and sample order, the new output had 10 wins and 5 losses, 38 changed extracted answers, and 384 changed prediction strings. The small +0.4136 percentage-point score movement is not a strict kernel-only comparison because Evalution also changed from 0.0.8 to 0.0.9.

### M=1 mega-kernel to M=12 lock-lifetime fix

The reported `M=12` hang was a route-transition bug, not a standalone `M=12` cooperative-tail defect. The runtime smoke first exercised the specialized `M=1`, 4096-by-4096, rank-128 Marlin+LoRA mega-kernel. That kernel stored its 128 BF16 LoRA-down values in persistent Marlin lock words 32 through 95 and left all 64 packed words populated. Repeated `M=1` tests did not expose the problem because that specialization did not interpret those words as locks. The next ordinary `M=12` Marlin launch did, so its reduction protocol spun on stale nonzero state. Before the fix, 72 square attention modules each retained 64 dirty workspace words after the smoke generation, and the following `M=12` call reproduced the 100%-utilization hang.

The retained fix gives the two protocols disjoint ownership in a 192-word persistent workspace. Ordinary Marlin keeps words 0 through 127, including the mega-kernel phase counters at 96 and 97; the packed 128-value LoRA-down payload moves to words 128 through 191. Adapter-enabled Marlin modules request that larger workspace, while the mega-kernel launcher requires all 192 words and safely falls back to ordinary Marlin plus the LoRA tail for legacy 128-word prepared callers. This adds 256 bytes per adapter module and leaves CPU, non-adapter, non-target-GPU, and unsupported-shape routes unchanged.

On PCI-ordered GPU 6 (`DE:00.0`, `sm_80`, 124 SMs, 102,191,202,304 bytes; Torch 2.13.0+cu130), the fixed real checkpoint had zero dirty Marlin lock-prefix words across all 252 quantized modules after the same 24-token `M=1` smoke, and every adapter module had the 192-word workspace. The exact post-smoke 12-row GSM8K Platinum Evalution batch then exercised `M=12` and completed all 12 rows in 31.073 seconds (23.868 seconds generation), scoring 10/12 with zero invalid outputs.

Focused CUDA-event measurements compare the repaired cooperative route with the safe CUDA-up-add fallback. `stream_us` is the median sustained per-call time over five 500-call passes; both routes use the same packed Marlin weights and FP32 dense LoRA reference.

| Dtype | M | K | N | Rank | Cooperative p50 / stream | CUDA-up-add p50 / stream | Stream speedup | Peak | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 1 | 4096 | 4096 | 128 | 30.72 / 20.70 us | 160.77 / 145.08 us | 7.008x | 24 KiB | 0.004135 |
| BF16 | 1 | 4096 | 4096 | 128 | 30.72 / 20.23 us | 161.79 / 156.75 us | 7.748x | 24 KiB | 0.01007 |
| FP16 | 12 | 4096 | 4096 | 128 | 87.04 / 74.05 us | 165.89 / 162.69 us | 2.197x | 352 KiB | 0.005394 |
| BF16 | 12 | 4096 | 4096 | 128 | 88.06 / 73.00 us | 156.67 / 146.10 us | 2.001x | 352 KiB | 0.01127 |

At `M=12`, “cooperative” is ordinary Marlin followed by the cooperative LoRA tail; the one-launch mega-kernel remains narrowly gated to `M=1`. The transition regression is what connects the two routes.

The matched broken-source `M=1` baseline measured 31.74/20.45 us p50/stream for FP16 and 29.70/19.52 us for BF16. The retained ownership-isolation fix changes sustained time by +1.22% for FP16 and +3.64% for BF16, while p50 changes by -3.21% and +3.43%, respectively. A correct intermediate clear-on-exit repair was rejected because it moved BF16 sustained time to 21.50 us; isolating the payload retains the downstream lock invariant without that cleanup on the critical path.

Validation commands and results:

| Scope | Command | Result |
|---|---|---|
| EoRA eigensolver and merge | `pytest -q tests/test_eora_cholesky.py tests/test_eora_merge.py` | 9 passed, 16 warnings |
| Merged integrated Marlin LoRA | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 pytest -q tests/test_marlin_jit.py -k 'integrated_lora or marlin_lora_attention' -vv` | 7 passed, 26 deselected, 16 warnings |
| FP16/BF16 `M=1 -> M=12` transition and legacy fallback | `pytest -q tests/test_marlin_jit.py -k 'mega_kernel_matches_dense_update_and_releases_locks'` | 2 passed; dense references matched, the first 128 words stayed zero before `M=12`, and a forced 128-word workspace selected the correct fallback |
| Non-default-stream transition stress | 100 consecutive `M=1 -> M=12` transitions per dtype | Passed in 10.528 ms FP16 and 10.912 ms BF16; zero dirty lock-prefix words; FP16 max abs 0.004105/0.006042 and BF16 0.008953/0.013295 for `M=1`/`M=12` |
| BF16 synchronization checking | `compute-sanitizer --tool synccheck ... releases_locks[dtype1]` | 1 passed; 0 sanitizer errors |
| FP16 memory checking | `compute-sanitizer --tool memcheck ... releases_locks[dtype0]` | 1 passed; 0 sanitizer errors |
| Post-smoke real-checkpoint acceptance | `M=1` smoke, lock-prefix audit, then 12-row GSM8K Platinum Evalution | 0/252 dirty lock-prefix modules; all workspaces 192 words; 12/12 completed in 31.073 seconds; 83.33% |
| Safe Marlin fused tail | `GPTQMODEL_MARLIN_LORA_COOPERATIVE=0 GPTQMODEL_MARLIN_LORA_FUSED_MAX_M=16 GPTQMODEL_MARLIN_LORA_CUDA_UP_ADD=1 pytest -q tests/test_marlin_lora_fused.py -k 'addmm_tail or cuda_up_add'` | 4 passed, 67 deselected, 16 warnings |
| Changed Python paths | `ruff check ...` and `python -m py_compile ...` | Passed |
| Patch hygiene | `git diff --check` | Passed |

Fixed-adapter artifacts:

- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora_eighfix_validation.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora_eighfix_eval_manifest.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evaluation_complete_eora_eighfix.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evalution_eora_eighfix/arc_challenge.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evalution_eora_eighfix/gsm8k_platinum_cot.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/eora_eighfix_cooperative_megafix_eval_manifest.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evaluation_complete_eora_eighfix_cooperative_megafix.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evalution_eora_eighfix_cooperative_megafix/gsm8k_platinum_cot.json`

## GPU 7 EoRA eigensolver fix

The unsafe eigensolver fallback was replaced with a numerical-rank-aware truncated pseudoinverse. The fixed EoRA adapter was regenerated from the same dense/INT3 pair and the same first 512 calibration rows, without rewriting the original checkpoint or adapter.

| GPU 7 EoRA mode | Adapter path | ARC raw / normalized | GSM8K Platinum | Status |
|---|---|---:|---:|---|
| Original unsafe adapter | `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/eora-rank128` | 22.70% / 21.93% | 0.00% | EoRA failure |
| Fixed truncated-pseudoinverse adapter | `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/eora-rank128-eighfix` | 33.53% / 34.73% | 26.05% | EoRA fixed; INT3 base remains low quality |
| Fixed minus original | — | +10.8362 / +12.7986 pp | +26.0546 pp | — |

The real layer-2 `mlp.down_proj` covariance retained 316/12,288 numerically supported directions. The repair reduced `lora_A` max-abs from 3,817,472 to 0.527, and reduced the reconstructed correction norm from 37,079,004 to 10.944. On the diagnostic prompt, layer-2 hidden-state norm fell from 60,001.9 to 100.4 and cosine to dense BF16 rose from 0.1052 to 0.9910. Last-token logit cosine to dense rose from -0.0731 to 0.9566.

This fixes the catastrophic EoRA amplification. The remaining score gap is attributable to the separately measured GPTQ INT3 base degradation; the fixed adapter improves it substantially but does not make that base checkpoint production quality.

Fixed-adapter artifacts:

- `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/eora_eighfix_validation.json`
- `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/evaluation_complete_eora_eighfix.json`
- `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/evalution_eora_eighfix/arc_challenge.json`
- `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/evalution_eora_eighfix/gsm8k_platinum_cot.json`

## GPU 6 EoRA ablation

The same GPU 6 checkpoint was reloaded through the `GPTQ_TORCH` backend without loading its EoRA adapter. All 252 quantized linear modules were present, and all 252 adapter slots were verified inactive before evaluation.

| EoRA mode | Active adapters | ARC raw / normalized | GSM8K Platinum | Status |
|---|---:|---:|---:|---|
| Active (baseline) | 252 | 53.33% / 54.44% | 89.00% | Passed |
| Disabled / not loaded | 0 | 53.33% / 54.52% | 88.25% | Passed |
| Disabled minus active | — | +0.0000 / +0.0853 pp | -0.7444 pp | — |

Exact paired-result comparison:

- ARC raw: 625/1,172 correct in both modes. The selected choice changed on 64 items, with 23 wins and 23 losses when EoRA was disabled.
- ARC normalized: 638/1,172 active versus 639/1,172 disabled. The selected choice changed on 71 items, with 27 wins and 26 losses when EoRA was disabled.
- GSM8K Platinum: 1,076/1,209 active versus 1,067/1,209 disabled. The extracted numeric answer changed on 145 items; disabling EoRA produced 42 wins and 51 losses, for a net loss of 9 correct answers. Both modes had zero invalid outputs.
- The no-EoRA runtime smoke test remained coherent and matched the dense model's top-1 token. Its last-token logit cosine to dense BF16 was 0.995079 versus 0.994328 with EoRA active.

The ablation ran on PCI-ordered GPU 6 (`00000000:DE:00.0`, NVIDIA PG506-230, compute capability 8.0, 124 SMs, 97,458 MiB, driver 610.43.02) using PyTorch 2.13.0+cu130, CUDA 13.0, GPTQModel 7.2.0+ultra, Transformers 5.14.1, Evalution 0.0.8, BF16 model outputs, batch size 16, and seed 898. No local CUDA extension was built for this run; the explicit backend was `GPTQ_TORCH`.

Reproducibility artifacts:

- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/no_eora_eval_manifest.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evaluation_complete_no_eora.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evalution_no_eora/arc_challenge.json`
- `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512/evalution_no_eora/gsm8k_platinum_cot.json`

## GPU 7 quality diagnosis

The failure is produced during quantization/EoRA construction, not by GPU selection or the inference backend. The saved INT3 base is already badly degraded, and one numerically unstable EoRA correction then causes a separate activation blow-up.

| Control (`The capital city of France is`) | Decoder layer 2 cosine / norm (dense norm 116.43) | Last-token logit cosine to dense |
|---|---:|---:|
| Saved INT3 + all EoRA adapters | 0.1052 / 60,001.9 | -0.0731 |
| Saved INT3 + only `model.layers.2.mlp.down_proj` EoRA disabled | 0.6170 / 106.7 | -0.4736 |
| Saved INT3 + all EoRA disabled | 0.6726 / 108.4 | -0.3948 |
| In-memory symmetric INT3/group-64 RTN control, no EoRA | 0.9911 / 109.0 | 0.8410 |

Key localization evidence:

- All 75 non-quantized weights are bit-exact to the dense model. The first packed INT3 `q_proj` produces bit-exact eager and Triton dequantizations, and the focused INT3 packing/zero-offset suite passed 4/4 tests. This rules out the PCI mapping, unquantized weights, and a generic inference pack/dequant mismatch.
- Across all 252 saved quantized weights, base INT3 cosine to dense is 0.9103 mean, 0.9115 median, and 0.8555 minimum. At layer 2, saved GPTQ cosine is 0.9110 for `gate_proj`, 0.9099 for `up_proj`, and 0.8555 for `down_proj`; direct symmetric INT3/group-64 RTN controls for the same dense weights are 0.9741, 0.9740, and 0.9711. The full-model RTN control is imperfect, but it remains coherent (`0.8410` last-logit cosine) while the saved GPTQ base is collapsed (`-0.3948`).
- The first sharp base-model nonlinear failure is the layer-2 SwiGLU product feeding `down_proj`: the individual INT3 gate/up outputs have cosine 0.9768/0.9426, but their product has cosine 0.0557 and norm 106.08 versus dense 11.78.
- Quantization loss grows through the checkpoint: layer-0 mean is `4.27e-6`, layer 6 reaches `3.44e-3` with `mlp.down_proj=2.27e-2`, and layers 29-35 rise from `2.20e-3` to `8.25e-3` mean. Dense/quant decoder hybrids confirm that error accumulates across later quantized blocks rather than originating in one inference kernel.
- EoRA reported that layer-2 `mlp.down_proj` had a non-positive-definite covariance (`cholesky_ex info=1556`) and fell back to the eigensolver. The saved factor is an extreme outlier: `lora_A` max-abs 3,817,472 and norm 44,931,084; `B @ A` norm is about 37.08 million. On the actual INT3 layer-2 input, this adapter alone emits a correction with norm 59,987.6 and causes the 60,001.9 hidden-state norm.
- The same layer is ill-conditioned in the INT4 adapter (`lora_A` max-abs 1,826,816), but the accurate INT4 path does not excite that near-null direction: its actual adapter-output norm is only 0.469 and its layer-2 cosine remains 0.9987.

The concrete EoRA bug is the unsafe non-PD fallback in `gptqmodel/eora/eora.py`: negative covariance eigenvalues are replaced with the smallest positive eigenvalue, which can be effectively zero, and the code then forms `1 / sqrt(eigenvalue)`. There is no eigenvalue floor/condition threshold, and `EoraProcessor` applies and saves `B @ A` without a factor-norm or held-out activation check. The existing Cholesky tests pass (7/7 with merge tests) but cover a well-conditioned fallback whose minimum positive eigenvalue is 0.5, not this near-singular case.

Repair status and remaining follow-up:

1. Implemented: use a float32-precision numerical-rank cutoff and a truncated pseudoinverse for unsupported covariance directions.
2. Implemented: regressions cover a non-PD covariance with a near-zero positive eigenvalue and a covariance with no positive support.
3. Remaining: add a post-quant dense/RTN control so a GPTQ result this much worse than RTN cannot be saved silently; the current RTN fallback is triggered by sample coverage or numerical failure, not by observed quality.

Primary diagnostic artifact: `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512/int3_layerwise_diagnostic.json`.
