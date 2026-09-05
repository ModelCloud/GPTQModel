# Additional P32 algebraic experiments: 21–30

Status: queued, not executed. These extend experiments 1–20 in the same PR.
Use the existing [F6 seed-7 snapshot and complete saved config](F6_SEED7_SNAPSHOT.md)
as the fixed teacher. Preserve its checkpoint and verified historical quantization datasets.
Use the [common scorecard](README.md#common-scorecard), including original BF16 comparison,
held-out model quality, all nine M values, real Llama prefill/decode, executed-instruction profiling,
and complete effective BPW. Target at least 2× end-to-end linear speedup, stretch 4×.
No synthetic fixture or decoder-only speedup establishes model-quality recovery or advancement.

## Prerequisite: derive the actual decoder contract

Inspect the deployed checkpoint codec and kernel before implementing the queue. Establish the actual
state domain, transition width at each mixed rate, weights emitted per transition, bank/PGC mapping,
tile boundaries, and tail-biting semantics. Distinguish quantization-time Viterbi dependencies from
inference-time packed-window extraction; do not assume inference follows a serial recurrence.
The proposed 16-state/four-bit examples are hypotheses, not established P32 properties.
For S states, an uncompressed function table requires S*ceil(log2(S)) bits. Account for composition
cost, state expansion, and output selection as well as scan depth. Compare against the existing
parallel/window decoder, including exact reconstructed values and unchanged serialized input bits.

## Queue

| ID | Direction | Accuracy contract | Required implementation and sweep | Relationship to original queue |
| --- | --- | --- | --- | --- |
| 21 | Finite-state associative scan | Exact | Encode transition functions; compose h[s]=g[f[s]] with ordered warp/shared prefix scans. Sweep sequence/block length and packed-table layout. Measure work and depth. | New schedule; combine with 5/13 |
| 22 | Hierarchical all-start-state decode | Exact | Decode blocks from every legal initial state, scan block transfers, then select valid outputs. Sweep B, state count cost, registers/shared storage and stitching. Match canonical tail-biting state selection when multiple fixed points exist. | Extends 3/11 |
| 23 | Sparse exact state checkpoints | Exact if original states are stored losslessly | Store boundary state every 8/16/32/64 transitions; independent CTAs and MMA-aligned K blocks. Prove reconstruction parity before timing. Requantized independent states are a separate approximate arm. | Extends 12; distinct from requantized 11 |
| 24 | Multi-step super-symbol automaton | Exact | Combine 2/4/6/8 transitions; store end state and emitted outputs. Sweep LUT residency, factoring, packed half2/fragment output; compose four-step functions using 21. | Extends 5/6/13 with six-step arm |
| 25 | Bit-sliced Boolean decoder | Exact | Derive and minimize Boolean state/index circuits; pack 32 independent streams per bit plane. Sweep stream/channel interleaving, LOP3/XOR/shift mappings and transpose overhead. Exhaustively verify legal truth tables where feasible. | New Boolean implementation of decode |
| 26 | GF(2) affine jump-ahead | Exact only where affine structure is proven | Test s'=A(c)s XOR b(c); prove for all legal states/symbols. Compose augmented affine maps by prefix scan; isolate nonlinear bank/PGC lookup. Record non-affine cases rather than approximating silently. | Alternative to 21 function tables |
| 27 | Tensor-product pair codebook | Requantized unless exact parity proven | Learn additive/multiplicative or sum_r A_r[i]B_r[j] dictionaries, R=1/2/4. Compare post-hoc factorization with direct constrained quantization. Include all dictionary/index/scale costs. | Extends 15 |
| 28 | Signed/ternary tile basis | Requantized/approximate | Sweep binary and ternary planes, rank, scales and tile shape; compare sum_r alpha_r X S_r with low-bit U and dense dictionary V. Include unpack and reduction latency. | Extends 16 from pairs to tiles |
| 29 | Native base plus structured residual | Approximate with measured recovery | Jointly allocate native codes, activation-weighted AB, sparse S and retained P32 blocks under BPW/latency budgets. Sweep ranks 16/32/64/128, sparse density and retained-block fraction. Ablate each residual component. | Combines 17–20 |
| 30 | Sparse Walsh/Hadamard spectrum | Requantized/approximate | Measure spectral compressibility before implementing sparse execution. Sweep coefficient budgets, sparse/block-sparse storage, and local block widths 16/32/64. Compare direct spectrum quantization with post-hoc pruning. | New transform-domain representation |

For 23, metadata overhead is state_bits/(B*weights_per_transition) BPW before padding,
indices and layout costs. Never equate bits per transition with bits per weight.
For 24, symbol combinations depend on actual encoded symbol width: 2^(k*symbol_bits),
multiplied by legal states, banks and output storage. The two-bit-symbol example is not universal.
FP16/BF16 packed outputs in exact arms must represent the canonical levels exactly; matching the
codebook indices alone does not establish equality after conversion or accumulation.
For 30, state normalization explicitly: W=H^T C H assumes the matching orthonormal transform convention.
Include both transforms and sparse indexing in full-operator timing and storage.

## Priority and combinations

First implementation order for the added queue: **21 → 24 → 22 → 25 → 23**.
This implements the requested associative scan → super-symbol → all-start-state → Boolean → checkpoint order.
Experiment 26 can follow the transition-contract audit where an affine proof is available.
Experiments 27–30 follow exact schedule investigations; they cannot inherit an exactness claim.
The original twenty experiments remain in the ledger; overlapping work shares baselines and artifacts,
while each named experiment retains its own measured result and commit.

1. **Exact combined kernel:** four-step super-symbols + associative scan + bit-sliced composition +
   warp-specialized MMA (21/24/25 plus original 4). Compare components and the combination, including
   synchronization, decoding, GEMM, transforms and epilogue.
2. **P32-P:** exact state checkpoints every 32/64 transitions, four-step super-symbols, interleaving
   32 independent output streams, packed vector output and MMA-aligned K blocks (23/24/25).
   Version any new layout and verify lossless round-trip before treating it as exact.
3. **Native three-part recovery:** native base + AB + S, with an optional small retained P32 block tail (29).
   Fit residuals from actual deployed outputs Z=Y_P32(X)-Y_native(X), including activation quantization,
   scales and kernel arithmetic. Refit using the remaining output residual after each component to
   avoid double counting. Compare with the same-BPW rank-only and sparse-only controls.

Native target hardware: INT4 on Ampere, FP8 on Hopper, NVFP4 on Blackwell. Probe the actual device
before scheduling; do not carry the preceding session's GPU identity into a new host.

## Completion record required for every experiment

Commit and push each completed experiment to PR #137 with configuration, exact source revision,
checkpoint/data hashes, raw common-scorecard metrics, profiler/SASS paths, full byte inventory,
and a measured decision (advance, escalate, reject, or unsupported). All entries currently remain queued.
Declare quality gates and uncertainty handling before candidate selection; real-model inference and
held-out model quality control advancement. Unavailable hardware or metrics remain explicitly missing.
