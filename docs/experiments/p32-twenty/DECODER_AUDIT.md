# Real F6 seed-7 decoder audit

Status: experiment 10 correctness prerequisite passed; performance and common scorecard remain incomplete.

Four GPU workers partitioned all 94 P32 projections in the fixed read-only F6 seed-7 snapshot.
All 94 passed bitwise planar→window→planar round-trip, exact reconstructed FP32 inner-weight value equality
including bank IDs/alternate bank selection, and finite-value checks. Payload byte counts are unchanged.
All 17 snapshot shards retained identical before/after SHA-256 on every worker. The remaining 18 ordinary-QVQ
projections are not P32 and were excluded explicitly, not treated as passing P32 cases.

Raw device, Torch, source revision, module geometry/rate, bank and checksum evidence:
[GPU 0](results/repack-audit/gpu0.json), [GPU 1](results/repack-audit/gpu1.json),
[GPU 2](results/repack-audit/gpu2.json), [GPU 3](results/repack-audit/gpu3.json).

Runner: `scripts/p32_twenty/audit_snapshot.py --worker I --workers 4 --uuid UUID --output OUTSIDE_SNAPSHOT.json`.
Uses CPU safetensor reads and copies to GPU; no checkpoint writes. Initial strict three-sample idle gates passed
with an explicit 8 MiB driver allowance. This is a correctness run, not a timing or model-quality result.

## Actual inference dependency

`repack_p32_planar_to_window` encodes one circular 128-transition history with unchanged word count.
`unpack_p32_window_states` extracts each 16-bit state independently. The CUDA
`window_state_pair64` uses two word loads and a funnel shift/mask for state recovery.
Transitions use 4–7 bits for the snapshot’s W2–W3.5 P32 modules. There is no need for a runtime
serial traversal to recover these states in this implementation.

Thus the illustrative 16-state/four-bit assumption is not the current P32 state domain: a general
16-bit state-function table would use 65536*16 bits = 128 KiB per function before outputs.
Experiments 21–26 must compare their work/storage against direct circular extraction, not a hypothetical
128-step serial baseline. Bank/PGC lookup and MMA still require measured decomposition.

This audit does not establish local full-operator numerical gates, propagated logits, perplexity,
downstream quality, or an end-to-end speedup. No experiment is marked complete or promoted yet.
