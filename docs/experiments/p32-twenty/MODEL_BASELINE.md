# Initial full-model baseline

This bounded baseline uses the existing F6 seed-7 snapshot read-only. It does not complete an experiment
or establish preservation of a post-quantization advantage on downstream tasks.

| Arm | GPU | C4 perplexity |
| --- | ---: | ---: |
| Original BF16 Llama 3.2 1B Instruct | 0 | 25.3876977933493 |
| Canonical FP32 P32 reconstruction/forward | 1 | 26.99102051460363 |
| Window FP32 P32 reconstruction/forward | 2 | 26.99102051460363 |
| Production QVQ, FP16 endpoint | 3 | 26.989785799529503 |

Inputs: 16 ordinary C4 training documents, 256 tokens per document, 4080 scored next-token positions.
These are evaluation inputs, not a replacement for historical teacher calibration. Exact normalized text
matches against the historical Fisher messages were excluded; this is not semantic decontamination.
The ordered token IDs, original document hashes and source shard hash are in
[inputs.json](results/model-baseline/inputs.json). No candidate fitting uses these rows.

Canonical/window models load all saved dense parameters (including embeddings/norms) and reconstruct every
quantized projection from the snapshot. The RHT input scale→Hadamard→FP32 GEMM→Hadamard→output scale
composition is explicit; TF32 and reduced-precision matmul reduction modes are disabled. Ordinary W4 QVQ
projections remain canonically reconstructed in both arms. Weight reconstruction is cached for reference
execution, so these arm timings do not represent runtime trellis-decoding performance.

Every canonical/window saved logit is exactly equal across all 16 documents. Against the P32 teacher,
BF16 has mean KL 0.08989235769123081 and top-1/5/10 overlap 0.8544921875 / 0.8474609375 / 0.8533447265625.
These agreement metrics include all 256 input positions per document; perplexity scores only 255 next tokens.
P32 perplexity is worse on this small sample. The snapshot’s downstream advantage must be assessed on the
specified downstream tasks, not inferred from matching P32 or from this small perplexity slice.

Per-arm reports record all nine requested prefill lengths, three warmups and ten CUDA-event samples per
length, plus 32 growing-cache decode steps after a 128-token prompt. This is eager full-model inference,
not isolated linear-layer timing or a CUDA-graph serving benchmark. GPU models/configurations differ across
arms; **do not form speedup ratios across these runs**. Matched-device baseline/candidate timing is required.
Initial idle gates and prefill exclusivity rechecks ran; decode still needs a dedicated warmed timing protocol.

Artifacts: [BF16](results/model-baseline/bf16.json),
[canonical](results/model-baseline/canonical.json), [window](results/model-baseline/window.json),
[logit comparison](results/model-baseline/comparison.json). Production status is recorded in its external
`/root/p32-model-baseline/production/report.json`; only a complete report may enter the result ledger.
Full saved logits remain outside the checkpoint under `/root/p32-model-baseline/{arm}/logits-*.pt`.

Scripts: `prepare_baseline_inputs.py`, `model_baseline.py`, `compare_baselines.py` in `scripts/p32_twenty/`.
The obsolete fresh-quantization launcher/config/verifier have been removed. The authoritative saved
configuration is in [F6_SEED7_SNAPSHOT.md](F6_SEED7_SNAPSHOT.md).

Outstanding: larger disjoint evidence, paired uncertainty, downstream tasks, full localized layer gates,
matched-device performance, instruction profiling, total runtime storage/BPW, and all candidate sweeps.
