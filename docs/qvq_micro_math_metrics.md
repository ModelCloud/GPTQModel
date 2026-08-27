# Micro-math post-quantization metrics

Every newly discovered checkpoint with a passing strict disjointness contract
is evaluated on `dataset/micro_math_llama3.2_1b.jsonl` before it is promoted by
the monitor.  The set contains 128 deterministic rows from the local GSM8K
main **train** split.  It is excluded from the calibration streams, both D300
manifests, and the local GSM8K test split using normalized user-question
hashes.  The JSON dataset and this contract are SHA-256 bound.

The monitor runs 64 rows as a fast screening slice.  It reports:

* `mini_math_exact_answer_accuracy`: numeric answer equivalence after a fixed
  48-token greedy rollout (EOS is treated as an ordinary token);
* `reasoning_delta_ce` and `reasoning_delta_kl`: quantized minus dense loss on
  the reference reasoning answer tokens;
* answer-token log-probability and margin retention for the suffix after
  `####` (both per-token deltas and retention ratios);
* numeric/operator critical-token Top-1 agreement and target accuracy;
* dense and quantized short-rollout invalid-answer rates.

The result also records paired rollout transitions (`dense wrong → quantized
right` and the reverse), so a small accuracy change can be audited rather than
treated as an unpaired percentage.  `mini_math_exact_answer_accuracy` and
`short_rollout_semantic_success` are numeric-answer equivalence, not exact text
or dense-token agreement.

These are screening proxies, not replacements for full GSM8K Platinum.  They
are intentionally task-matched and should be used with D300 as a paired,
prompt-level health check.  No micro-math row may be used to choose a
quantization candidate when the checkpoint's strict disjointness manifest
does not pass.
