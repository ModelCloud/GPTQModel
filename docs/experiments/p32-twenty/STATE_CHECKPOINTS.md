# Experiments 12/23: stored independent decoder states

Store the exact 16-bit incoming state every 8/16/32/64 transitions. Each GPU CTA
processes independent blocks across 32 trellis streams, with no cross-block state
dependency. The existing circular-window payload remains unchanged; state metadata
is an additive sidecar. This is exact, not re-quantization. The original snapshot
is read only. Raw metadata overhead is 8/B BPW because each transition emits two
weights; 32-byte versioned per-projection headers are also counted.

All four spacings, each at 4 and 8 warps, decoded every tile of all 94 P32 projections
exactly. The stored files' magic/version, geometry, length and SHA256 were audited.
The measured kernel consumed the in-memory checkpoint arrays also serialized to
those files; loading files directly into the kernel is not yet independently tested.
Files stay external under /root/p32-state-checkpoints.

Median direct-window/candidate latency ratios are about 0.36–0.37x: these unfused
checkpoint decoders are 2.7–2.8x slower than direct extraction in this launch setup.
Only 32 lanes carry distinct streams, so a smaller warp-count follow-up is running.
No fused GEMM or model-performance conclusion follows from these state-only runs.

The storage summary includes all existing quantized tensor buffers (SU/SV, banks,
packed words and ordinary QVQ layers) plus actual state file bytes. The whole-file
figure additionally includes dense weights, saved configs/tokenizer/index and
shard overhead under the original inventory conventions. Runtime repack and state
output scratch are separate from serialized BPW and remain reported in raw runs.
There is no additional transition LUT. A materialized derivative model checkpoint
and full operator/model accuracy/performance scorecard remain outstanding.

[Storage and timing](results/state-checkpoints/storage-and-timing.json), with raw
per-module samples and exact-state assertions in the same directory.

The smaller-launch follow-up completed for B=32 at 1 and 2 warps across all 94
projections, again exactly matching states. Median direct/candidate ratio improves
to 0.5001x, but the checkpoint kernel is still about 2x slower. This confirms that
idle warp allocation explained part, not all, of the initial slowdown. Raw samples
are in results/state-checkpoints/steps32-warp12.json.
