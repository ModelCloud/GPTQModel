# Qwen3-8B dual GPTQ + EoRA run

Run date: 2026-07-22 UTC
Repository commit: `644ac5cdf863a03c9ae6ce489d1c7b651feb471e`
Base model: `/monster/data/model/Qwen3-8B`

## Outcome

The GPU 6 INT4 checkpoint passed structural, runtime, dense-reference, ARC Challenge, and GSM8K Platinum validation.

The GPU 7 INT3 checkpoint is structurally complete and reloadable, but failed quality validation. It generated malformed text, had strongly divergent logits from the BF16 model, scored near random on ARC Challenge, and scored zero on all 1,209 GSM8K Platinum examples. Do not treat the INT3 artifact as deployable.

| PCI-ordered GPU | Quantization | ARC raw | ARC normalized | GSM8K Platinum | Result |
|---|---|---:|---:|---:|---|
| 6 | GPTQ INT4, group 128, activation scale search, GAR, EoRA r128 | 0.533276 | 0.544369 | 0.889992 | PASS |
| 7 | GPTQ INT3, group 64, EoRA r128 | 0.226962 | 0.219283 | 0.000000 | FAIL quality |

Evalution used the complete `arc_challenge` test split (1,172 samples / 4,687 continuations) and complete `gsm8k_platinum_cot` test split (1,209 samples), batch size 16, deterministic generation, and no row cap.

## Device contract

`CUDA_DEVICE_ORDER=PCI_BUS_ID` and a single visible device were enforced in each process. The harness rejected UUID or PCI-bus mismatches.

| GPU | PCI bus | UUID | Device | Compute capability | SMs | Memory |
|---|---|---|---|---:|---:|---:|
| 6 | `00000000:DE:00.0` | `GPU-737e2423-874a-23a4-1126-dfbe3e77c294` | NVIDIA PG506-230 | 8.0 | 124 | 102,191,202,304 bytes |
| 7 | `00000000:E4:00.0` | `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28` | NVIDIA PG506-230 | 8.0 | 124 | 102,191,202,304 bytes |

Software: Python 3.14.5t, PyTorch 2.13.0+cu130, CUDA 13.0, Transformers 5.14.1, GPT-QModel 7.2.0+ultra, Evalution 0.0.8.

## Calibration and quantization

Calibration source: `/monster/data/model/dataset/nm-calibration`, configuration `LLM`, train split. The first 512 of 10,000 rows were selected. The selected `text` digest is `a89c5ed40152f435d5102657166cb72399603042596c57dd2f1095724ad8f59d`; tokenization used the `messages` chat template and produced 181,796 non-padding tokens. Quantization used batch size 1 and descending length sort.

Common settings: GPTQ format, symmetric quantization, `desc_act=false`, CPU packing, EoRA rank 128 on 252 quantized linear modules, synchronous submodule finalization, and stage-end garbage collection.

- GPU 6: 4 bits, group size 128, activation scale search (`mse=2.0`), activation-group-aware quantization enabled. Quantization wall time 1,971.716 seconds; save time 6.292 seconds.
- GPU 7: 3 bits, group size 64. Scale search and GAR were disabled because they were specified only for GPU 6. Quantization wall time 2,148.570 seconds; save time 5.336 seconds.

Each adapter contains 504 BF16 tensors, 349,175,808 elements, observed rank 128, and is 698,420,736 bytes.

## Saved artifacts

- INT4: `/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512`
- INT3: `/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512`

Each directory contains the model shards, `eora-rank128/`, quantization/runtime manifests, dense-reference results, per-task Evalution JSON, and `evaluation_complete.json`.

## Runtime and numerical validation

| Check | INT4 + EoRA | INT3 + EoRA |
|---|---:|---:|
| Reloaded quantized modules / adapters | 252 / 252 | 252 / 252 |
| Finite BF16 logits | yes | yes |
| Last-token cosine versus dense BF16 | 0.994328 | -0.073113 |
| MAE versus dense BF16 | 0.391938 | 4.134137 |
| RMSE versus dense BF16 | 0.492337 | 4.903020 |
| Top-1 token agrees | yes | no |
| Top-5 overlap | 4/5 | 0/5 |

The INT4 model passed a Marlin reload, forward, and generation smoke test. Marlin then stalled on the first batched ARC continuation, so full Evalution used the supported `GPTQ_TORCH` fallback without changing the checkpoint. INT3 evaluation used `GPTQ_TORCH`; registered Marlin/BitBLAS/TritonV2 kernels do not support this INT3 + EoRA contract.

## INT3 failure isolation

The first layer's default Triton and eager Torch INT3 dequantizations were bit-exact (`MAE=RMSE=max=0`). Disabling EoRA made end-to-end logits worse, showing that the adapter helps but cannot recover this checkpoint. The checked first `q_proj` unpacked weight had cosine 0.919507 versus dense; end-to-end logits nevertheless collapsed to cosine -0.073113.

Focused CPU regression tests covered INT3/group-64 pack consistency, Torch repacking/dequantization, and GPTQ v1/v2 zero-offset conversion: 4 passed. This rules out a simple generic pack/dequant mismatch and leaves the produced INT3 quantization quality as the failure.

## Checks and evidence

- `ruff check` on `run_quant_eval.py` and `diagnose_int3.py`: passed.
- `git diff --check`: passed.
- Focused pytest: 4 passed, 16 warnings in 29.46 seconds.
- Main harness: `run_quant_eval.py`.
- INT3 diagnostic: `diagnose_int3.py` and the saved model's `int3_diagnostic.json`.
- GPU 6 fallback evaluation log: `gpu6.torch_eval.log`.
- GPU 7 full evaluation log: `gpu7.full_eval.log`.
- Preserved Marlin ARC stall log: `gpu6.marlin_arc_stall.log`.
