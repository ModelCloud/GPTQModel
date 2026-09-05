# Qwen3.8 YAQA Fisher 2x follow-up

## Scope

This follow-up starts from merged `origin/main` revision `23dadb60` (PR #130). It retains the rank-256 FP32
Gaussian streaming estimator, exact FP32 Fisher diagonal, diagonal congruence correction, and eager attention used by
the PR #130 H200 harness. Exact dense-factor collection and devices below 128 GiB retain the prior batch/checkpoint
policy.

The source-diagonal reduction used by the congruence correction is now computed once on the accumulator device,
validated there, retained with the compact factor, and reused during materialization. Previously every compact source
was scanned on pageable CPU memory during construction and reduced again on the quantization device. This is exact
value reuse; it does not reassociate the projection or Fisher-diagonal contractions.

For compact streaming collection on CUDA devices with at least 128 GiB, automatic execution uses batch 16 and disables
decoder checkpoint recomputation when each prepared batch is within a 1,024-token envelope. Longer prepared batches
are split to that envelope. Explicit `batch_size` integers and `activation_checkpointing` booleans remain authoritative.

## H200 performance

Both runs passed the strict idle gate on physical GPU 0, PCI `00000000:1C:00.0`, UUID
`GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`. The workload uses the local Qwen3.5-27B checkpoint as the same
geometry proxy declared by PR #130: four rows, sequence length 64, four decoder layers, 25 targets, rank 256, BF16
model execution, FP32 factors, and CUDA accumulation.

```text
+----------------------------+-------------+-----------+------------+----------------+----------------+
| Arm                        | Batch / AC  | Wall (s)  | CUDA (s)   | Throughput     | Peak VRAM GiB  |
+----------------------------+-------------+-----------+------------+----------------+----------------+
| merged main                | 1 / on      | 6.6012    | 6.1562     | 38.7806 tok/s | 51.73          |
| optimized high-memory path | 4 / off     | 2.3199    | 2.1117     | 110.3517 tok/s | 63.00          |
+----------------------------+-------------+-----------+------------+----------------+----------------+
| Speedup / reduction        |             | 2.846x    | 2.915x     | 2.846x         | +11.27 GiB     |
+----------------------------+-------------+-----------+------------+----------------+----------------+
```

Baseline command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
python scripts/benchmark_qvq_yaqa_qwen38.py \
  --rows 4 --batch-size 1 --sequence-length 64 \
  --all-targets --layer-count 4 --arms streaming_256 \
  --output /tmp/qvq_yaqa_speed2x_4layer_baseline.json
```

Candidate command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
python scripts/benchmark_qvq_yaqa_qwen38.py \
  --rows 4 --batch-size 4 --sequence-length 64 \
  --all-targets --layer-count 4 --arms streaming_256 \
  --no-activation-checkpointing \
  --output /tmp/qvq_yaqa_speed2x_e8f163cd.json
```

The benchmark's four-row population only permits batch 4. The lifecycle auto policy uses up to batch 16 when the
configured calibration population is large enough. A preceding candidate repeat measured 2.7333 seconds, so the
observed end-to-end speedup range is 2.415x--2.846x; the slower repeat still clears the 2x target.

## Correctness

- Cached and legacy congruence materialization were bit-exact on the Qwen MLP input geometry
  (`5120 x 256` source): maximum/mean absolute drift `0 / 0`.
- The exact Fisher diagonal remains independently accumulated in FP32. The added cached source diagonal is metadata
  for the existing congruence scale and increases compact storage by one FP32 value per feature.
- `72 passed, 105 deselected` in the YAQA diagnostic/lifecycle regression selection.
- `35 passed, 891 deselected` in YAQA configuration validation.
- Python compilation and `git diff --check` passed. Repository Ruff still reports pre-existing whole-file import and
  modernization findings; the changed code has no syntax or undefined-name failures after adding its required
  `Sequence` import.

No custom CUDA/C++ source, template, launch geometry, or compiler flag changed. The optimization changes PyTorch
operation scheduling and removes repeated reductions, so the PR #130 custom-kernel SASS evidence remains unchanged;
the synchronized end-to-end harness above is the promotion measurement.
