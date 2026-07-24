# Qwen3-8B-Base post-quantized embedding and LM head

Run date: 2026-07-24 UTC

## Result

GPT-QModel Ultra built and validated post-quantized variants of the untied `model.embed_tokens` and
`lm_head` modules in Qwen3-8B-Base GPTQ checkpoints:

- W4/group-128 decoder with GPTQ W8/group-128 targets
- W4/group-128 decoder with GPTQ W4/group-32 targets
- W3/group-128 decoder with GPTQ W4/group-32 targets
- W3/group-32 decoder with BF16 endpoints
- W3/group-32 decoder with GPTQ W4/group-32 targets

All target variants use `sym=True`, `desc_act=False`, activation scale search, and group-aware reordering
(GAR). The existing 252 decoder quantized-linear modules retain their source checkpoint contract: GPTQ
W4/group-128 for the first two post-quantized snapshots, GPTQ W3/group-128 for the third, and GPTQ
W3/group-32 for the final snapshot.

Evalution 0.0.9 ran the full `arc_challenge` and `gsm8k_platinum_cot` datasets with seed 898, BF16 compute,
batch size 32, and no chat template. It also completed all 3,153 `mmlu_stem` questions for dense and the three
W4-decoder rows. The W3 MMLU-STEM entries are starred 128-question prefixes because the user stopped their
full runs after they exposed a severe regression signal; they are not full-corpus aggregates. MMLU-STEM used
batch size 32 for dense and Base W4, and batch size 16 for the remaining rows. W4 decoder models used Marlin.
W3/group-128 originally used automatic fallback selection; the W3/group-32 comparison used the dedicated
native CUDA `TrilinLinear` selected by AUTO. The dense model used the automatic dense backend.
Model sizes are decimal MB from total serialized tensor bytes divided by 1,000,000.

| ARC/GSM GPU | MMLU GPU | Snapshot | Decoder bit rate | Embedding / LM-head bit rate | Model tensors (MB) | ARC raw | ARC normalized | GSM8K Platinum | MMLU-STEM | GSM invalid |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | Dense Qwen3-8B-Base | BF16 | BF16 | 16,383.568 | 630/1,172 (53.75%) | 644/1,172 (54.95%) | 1,090/1,209 (90.16%) | 2,476/3,153 (78.53%) | 0 |
| 0 | 1 | Base W4 | GPTQ W4/g128 | BF16 | 6,103.917 | 632/1,172 (53.92%) | 665/1,172 (56.74%) | 1,072/1,209 (88.67%) | 2,429/3,153 (77.04%) | 0 |
| 1 | 2 | Base W4 + endpoints W8 | GPTQ W4/g128 | GPTQ W8/g128 | 4,889.053 | 645/1,172 (55.03%) | 660/1,172 (56.31%) | 1,060/1,209 (87.68%) | 2,431/3,153 (77.10%) | 0 |
| 3 | 3 | Base W4 + endpoints W4 | GPTQ W4/g128 | GPTQ W4/g32 | 4,334.790 | 630/1,172 (53.75%) | 653/1,172 (55.72%) | 1,074/1,209 (88.83%) | 2,420/3,153 (76.75%) | 0 |
| 2 | 2 | Base W3 + endpoints W4 | GPTQ W3/g128 | GPTQ W4/g32 | 3,459.657 | 543/1,172 (46.33%) | 584/1,172 (49.83%) | 1,031/1,209 (85.28%) | 76/128 (59.38%)* | 0 |
| 2 | 2 | Base W3/g32 | GPTQ W3/g32 | BF16 | 5,615.543 | 594/1,172 (50.68%) | 629/1,172 (53.67%) | 1,051/1,209 (86.93%) | 72/128 (56.25%)* | 0 |
| 3 | 2 | Base W3/g32 + endpoints W4 | GPTQ W3/g32 | GPTQ W4/g32 | 3,846.416 | 577/1,172 (49.23%) | 601/1,172 (51.28%) | 1,059/1,209 (87.59%) | 77/128 (60.16%)* | 0 |

\* Starred MMLU-STEM values are independent first-128 prefixes. The stopped full runs reached 9,450/12,612
choice likelihoods for W3/g128 + W4/g32 and 9,072/12,612 for W3/g32 + BF16 without emitting aggregate
accuracy. The partitioned W3/g32 + W4/g32 run was stopped during model setup. No incomplete run is reported as
a full score. On the same first 128 questions, dense scored 85/128 (66.41%), Base W4 81/128 (63.28%),
W4 + W8 endpoints 80/128 (62.50%), and W4 + W4 endpoints 78/128 (60.94%).

Snapshot paths used by the table:

- Dense: `/monster/data/model/Qwen3-8B-Base`
- Base W4:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-cal512`
- Base W4 + endpoints W8:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-postquant-embed-lmhead-8bit-g128`
- Base W4 + endpoints W4:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-postquant-embed-lmhead-4bit-g32`
- Base W3 + endpoints W4:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g128-activation-GAR-postquant-embed-lmhead-4bit-g32`
- Base W3/g32:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g32-activation-GAR-cal512`
- Base W3/g32 + endpoints W4:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g32-activation-GAR-postquant-embed-lmhead-4bit-g32`

The W8/g128 snapshot is 70.16% smaller than dense BF16 tensor bytes. The W4-decoder/W4-endpoint snapshot is
73.54% smaller than dense and 9.07% smaller than W8/g128. The W3-decoder/W4-endpoint snapshot is 78.88%
smaller than dense and 20.19% smaller than the W4-decoder/W4-endpoint snapshot.
The W3/group-32 decoder with BF16 endpoints is 65.72% smaller than dense. Quantizing its embedding and
LM-head to W4/group-32 makes the final snapshot 76.52% smaller than dense and 31.50% smaller than its
BF16-endpoint source.

Relative to the W4-decoder/W4-endpoint snapshot, the W3-decoder/W4-endpoint snapshot changes ARC raw by
-87 correct samples (-7.42 percentage points), ARC normalized by -69 (-5.89 points), and GSM8K Platinum by
-43 (-3.56 points). A full W3-decoder/BF16-endpoint baseline was not run, so the aggregate evaluation does
not by itself isolate decoder-width loss from endpoint quantization. The post-quant diagnostic against the
W3 source had logit cosine 0.9999529.

Within the matched W3/group-32 comparison, endpoint W4/group-32 changes ARC raw by -17 correct samples
(-1.45 points), ARC normalized by -28 (-2.39 points), and GSM8K Platinum by +8 (+0.66 points). The serialized
snapshot is 31.50% smaller than the W3/group-32 decoder with BF16 endpoints.

Paired pre-to-post sample changes were:

| Target variant | Metric | Correct to wrong | Wrong to correct | Net |
|---|---|---:|---:|---:|
| W8/g128 | ARC raw | 16 | 29 | +13 |
| W8/g128 | ARC normalized | 23 | 18 | -5 |
| W8/g128 | GSM8K Platinum | 42 | 30 | -12 |
| W4/g32 | ARC raw | 25 | 23 | -2 |
| W4/g32 | ARC normalized | 29 | 17 | -12 |
| W4/g32 | GSM8K Platinum | 35 | 37 | +2 |
| W3/g32 + W4/g32 endpoints | ARC raw | 26 | 9 | -17 |
| W3/g32 + W4/g32 endpoints | ARC normalized | 33 | 5 | -28 |
| W3/g32 + W4/g32 endpoints | GSM8K Platinum | 22 | 30 | +8 |

## Base lineage

Every primary result above derives from the pretraining-only Base model:

- Dense source: `/monster/data/model/Qwen3-8B-Base`
- Its model card identifies training stage `Pretraining`.
- W4 decoder checkpoint:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-cal512`
- W3/group-128 decoder checkpoint:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g128-activation-GAR-cal512`
- W3/group-32 decoder checkpoint:
  `/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g32-activation-GAR-cal512`
- All decoder checkpoint manifests record `base_model=/monster/data/model/Qwen3-8B-Base`.
- Neither an instruction checkpoint nor an EoRA adapter was loaded.

The distinct `/monster/data/model/Qwen3-8B` model card identifies training stage
`Pretraining & Post-training` and documents chat inference. It is not the source of the Base results.

## Base decoder checkpoints

The W4/group-128, W3/group-128, and W3/group-32 Base decoder checkpoints use `sym=True`, `desc_act=False`,
activation scale search, and GAR. All used the same Base lineage and calibration contract:

- Calibration: 512 NeMo-calibration rows, 181,796 non-padding tokens, concatenation length 2,048
- Runtime: physical GPUs 0 and 1, `PYTHON_GIL=0`, disabled GIL confirmed
- W4 quantization/save wall time: 1,082.90/7.89 seconds
- W4 exact indexed model size: 6,103,917,280 bytes
- W3 quantization/save wall time: 1,183.34/4.79 seconds
- W3 exact indexed model size: 5,228,913,160 bytes
- W3/group-32 quantization plus save/validation wall time: 1,247.4 seconds; save 5.2 seconds
- W3/group-32 exact serialized tensor size: 5,615,543,328 bytes

## Post-quantized checkpoints

W8/group-128 target checkpoint:
`/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-postquant-embed-lmhead-8bit-g128`

- Quantization/save/reload wall time: 218.47/4.70/29.52 seconds
- Peak allocated CUDA memory: GPU 0 12,638,091,264 bytes; GPU 1 7,817,508,864 bytes
- Reloaded embedding sampled-weight cosine/RMSE versus BF16: 0.9999571/0.0001821
- Reloaded LM-head sampled-weight cosine/RMSE versus BF16: 0.9999548/0.0002338
- Reloaded last-token logit cosine/RMSE versus pre: 0.9999821/0.04614, with identical top-1 token
- Exact indexed model size: 4,889,052,992 bytes

W4/group-32 target checkpoint:
`/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-postquant-embed-lmhead-4bit-g32`

- Quantization/save/reload wall time: 241.74/4.23/31.26 seconds
- Peak allocated CUDA memory: GPU 0 12,754,806,784 bytes; GPU 1 7,821,155,840 bytes
- Reloaded last-token logit cosine/RMSE versus pre: 0.9999532/0.12514
- The diagnostic prompt's top-1 token changed; the full quality table above is the authoritative quality check
- Exact indexed model size: 4,334,790,304 bytes

W3/group-128 decoder with W4/group-32 target checkpoint:
`/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g128-activation-GAR-postquant-embed-lmhead-4bit-g32`

- Quantization/save/reload wall time: 265.96/4.31/21.34 seconds
- Peak allocated CUDA memory: GPU 0 11,920,623,616 bytes; GPU 1 7,821,155,840 bytes
- Reloaded last-token logit cosine/RMSE versus pre: 0.9999529/0.13444, with identical top-1 token
- Exact serialized tensor size: 3,459,657,216 bytes

W3/group-32 decoder with W4/group-32 target checkpoint:
`/monster/data/model/Qwen3-8B-Base-GPTQ-3bit-g32-activation-GAR-postquant-embed-lmhead-4bit-g32`

- Quantization/save/reload wall time: 296.46/4.51/26.43 seconds
- Peak allocated CUDA memory: GPU 0 12,303,591,424 bytes; GPU 1 7,821,155,840 bytes
- Reloaded last-token logit cosine/RMSE versus pre: 0.9999448/0.13371, with identical top-1 token
- Exact serialized tensor size: 3,846,416,384 bytes
- Exact full snapshot size including tokenizer and metadata: 3,857,850,292 bytes

All checkpoints passed clean save/reload validation. Their reloaded target contracts match the requested
width/group settings, all 252 decoder modules retain their source W4/group-128, W3/group-128, or W3/group-32
contract, and sampled first/last decoder `qweight` SHA-256 values are unchanged before, directly after, and
after reload. The reloaded checkpoint is the supported inference boundary.

## Multi-GPU lifecycle

The post-quantization and evaluation runs exposed and fixed six lifecycle issues:

1. An Accelerate-sharded load left Qwen RMSNorm on GPU 1 while its hidden state reached GPU 0 during LM-head
   replay. The driver now loads one authoritative compact W4 tree on GPU 0; GPT-QModel owns later replication.
2. Input capture temporarily stages the first decoder shell on CPU, but an input embedding being quantized
   intentionally remains on GPU. `StageInputsCapture` now resolves the embedding module's actual device and
   executes `input_ids` there. Output-only embedding quantization retains the ordinary fallback device.
3. Mixed W3-decoder/W4-endpoint packing converted GPTQ-v2 qzeros back to GPTQ-v1 using the global decoder
   bit width. Format conversion now derives bit width and packing dtype from each packed module, so mixed-width
   checkpoints round-trip using each module's own contract.
4. Automatic backend discovery treated `TorchQuantEmbeddings` as a general QuantLinear candidate and could
   install it in decoder projections. Role-only kernels can now opt out of backend discovery and are substituted
   only for their declared input-embedding role.
5. A single global class selection could not represent mixed W3 decoder and W4 endpoint contracts. AUTO now
   validates and selects each module against its effective dynamic contract. A real post-snapshot load selected
   252 `TrilinLinear` decoder modules, one W4 `TritonV2Linear` LM-head, and one `TorchQuantEmbeddings` input.
6. Trilin had been embedded as helper dispatch inside TritonV2. It is now a distinct native-CUDA backend:
   `BACKEND.TRILIN`, `TrilinLinear` for GPTQ layout, and `AwqTrilinLinear` for AWQ GEMM layout. TritonV2 remains
   a separate Triton backend. BF16-loaded scales are normalized to the native Trilin FP16 scale ABI.

For true batch data parallelism, the post-quantization driver uses:

- `auto_forward_data_parallel=True`
- `calibration_data_device="balanced"`
- dense strategy `EXCLUSIVE` with no fixed placement map

This lets `ForwardExecutor` replicate each active module to `cuda:0` and `cuda:1` and route balanced-resident
calibration batches to both devices. Dense `BALANCED` is a module-placement strategy and deliberately serializes
subset forward, so it is not used for this data-parallel replay. Durable logs contain
`Forward: replicate to 2 devices` and successful staging on both devices. Both runs fail fast unless two GPUs
are visible, `PYTHON_GIL=0` is set before interpreter startup, and the runtime confirms the GIL is disabled.

The lifecycle and mixed-width fixes were pushed in commits:

- `bbb202cc` (`Fix embedding input capture device lifecycle`)
- `b5200bb7` (`Enable data-parallel embedding replay`)
- `3f8b1c1d` (`Preserve source decoder width in endpoint quantization`)
- `d8decfa5` (`Use module width for mixed GPTQ format conversion`)
- `9529e3c7` (`Exclude embedding kernels from backend auto selection`)
- `c1278c9f` (`Allow mixed precision with Trilin 3-bit kernels`)
- `28975a56` (`Split Trilin into dedicated GPTQ and AWQ kernels`)

The complete upstream inference port, including `TorchQuantEmbeddings`, endpoint checkpoint discovery,
role-only AUTO exclusion, and module-local mixed-width conversion, is
[ModelCloud/GPTQModel#2971](https://github.com/ModelCloud/GPTQModel/pull/2971) at commit `c4434a2`.

Interrupted diagnostic attempts did not serialize snapshots:

- Accelerate layer-sharded replay failed on the RMSNorm cross-device mismatch.
- The first authoritative-tree attempt exposed the CPU-input/GPU-embedding capture bug.
- A run using dense `BALANCED` placement was stopped after telemetry showed GPU 1 resident but effectively idle.

## Evaluation hardware

| Physical GPU | UUID | Device | Evaluation |
|---:|---|---|---|
| 0 | `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2` | NVIDIA PG506-230, sm_80, 124 SMs | Pre W4/g128 + BF16 targets |
| 1 | `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855` | NVIDIA PG506-232, sm_80, 124 SMs | Post W8/g128 targets |
| 2 | `GPU-8be4c651-4058-83df-154b-291f1b86add8` | NVIDIA PG506-230, sm_80, 124 SMs | Dense BF16 Base; W3/g128 + W4/g32; W3/g32 + BF16 |
| 3 | `GPU-471ecdd7-a171-4d5c-d61f-a1802dc76e4c` | NVIDIA PG506-230, sm_80, 124 SMs | W4/g128 + W4/g32; W3/g32 + W4/g32 |

Physical GPUs 0 and 1 performed all post-quantization runs. NVIDIA driver 610.43.02, PyTorch 2.13.0+cu130,
CUDA 13.0, Transformers 5.14.1, and Evalution 0.0.9 were used.

ARC/GSM evaluation wall times were 658.99 seconds dense, 767.07 seconds pre, 829.60 seconds W8/g128,
783.23 seconds W4/g32, 1,110.61 seconds W3/g128 + W4/g32, 1,427.03 seconds W3/g32 + BF16 endpoints, and
1,519.12 seconds W3/g32 + W4/g32 endpoints. Complete MMLU-STEM wall times were 619.88 seconds dense,
773.81 seconds Base W4, 792.30 seconds W8/g128 endpoints, and 766.51 seconds W4/g32 endpoints. These
concurrent, different-device wall times are not a controlled performance comparison.

## Post-trained control

An earlier control used the post-trained `/monster/data/model/Qwen3-8B` lineage, with no EoRA adapter loaded:

| Post-trained control | Model tensors (MB) | ARC raw | ARC normalized | GSM8K Platinum |
|---|---:|---:|---:|---:|
| Dense BF16 size reference | 16,381.47 | — | — | — |
| Pre: W4/g128 decoder, BF16 embedding/head | 6,103.92 | 635/1,172 (54.18%) | 637/1,172 (54.35%) | 1,069/1,209 (88.42%) |
| Post: W4/g128 decoder, W8/g128 embedding/head | 4,889.05 | 631/1,172 (53.84%) | 636/1,172 (54.27%) | 1,083/1,209 (89.58%) |

This control is retained only to document the initial variant-selection finding. It is not evidence about
Qwen3-8B-Base quality.

## Artifacts and checks

Run artifacts:
`scripts/post_quantization/results/qwen3_8b_base_embed_lmhead_8bit_g128_gar/20260724T041000Z`

- `pytest -q tests/test_calibration_data_device.py -k
  'resolves_quantized_embedding_device or keeps_fallback_for_output_only_quantization or
  stage_inputs_capture_detects'`: 3 passed
- `pytest -q tests/test_named_module.py -k 'parameterless or register_and_state_locking'`: 2 passed
- `pytest -q tests/test_qzero_offsets.py`: 17 passed
- `pytest -q tests/kernels/test_selection.py -k
  'auto_select_excludes_embedding_only_kernel or TorchLinear-gptq'`: 4 passed
- Ruff on all changed Python paths: passed
- Python compile checks for both drivers: passed
- `git diff --check`: passed
- Trilin backend-selection focused tests: 29 passed
- Native Trilin GPTQ/AWQ wrapper, quality, and cache tests: 24 passed
- BF16-loaded Trilin scale regression tests: 2 passed
- Trilin LoRA/QKV/SwiGLU focused tests: 59 passed
- Real W3/group-32 Base load: 252/252 decoder projections selected native `TrilinLinear`; finite BF16 logits
- Base decoder quantize/save validation: passed on physical GPUs 0 and 1
- W8/g128, W4-decoder/W4-endpoint, and W3-decoder/W4-endpoint
  post-quantize/save/reload/inference validation: passed on physical GPUs 0 and 1
- Full dense/pre/W8/g128/W4/g32/W3+W4/W3g32 ARC/GSM Evalution runs: passed on physical GPUs
  2/0/1/3/2/2/3
- Full dense/Base-W4/W8-endpoint/W4-endpoint MMLU-STEM runs: passed on physical GPUs 0/1/2/3
- W3 MMLU-STEM regression prefixes: completed at 128 questions on physical GPU 2; longer runs stopped by user
- Safe GPU preflight/task/prefix/subset evaluation driver commits: `b5f18680`, `40ddd493`, `0573a20c`
- GPU-testing skill validation: passed; workflow commit `d40960c5`
