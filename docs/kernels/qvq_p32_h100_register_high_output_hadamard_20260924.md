# H100 M960 register-local output Hadamard (2026-09-24)

## Change and scope

The base-only P32 output-recovery path for `M=960`, `N=2048` and `N=8192` now assigns each thread the same pair index across 256-wide slots. The low Hadamard stages keep the established FP16 operation order: adjacent-pair arithmetic, five warp-shuffle stages, then shared-memory stages at pair distances 32, 64 and 128. Higher stages (256 and above) exchange values only among pairs owned by the same thread, so they use registers instead of CTA-wide shared-memory exchanges and barriers. FP16 rounding after every butterfly, normalization position, and SV scaling are unchanged. Other shapes and Rank-8 correction paths retain their previous dispatch.

No dense-weight cache, model-weight change or approximation is introduced. The production serving profile has Rank-8 off for prefill and on for decode.

## Correctness and isolated kernel evidence

The baseline and candidate shared libraries passed bitwise FP16 output comparison at `M=960`, `N=2048/8192`, for both `normalize_first=0/1`, including five changed-input CUDA-graph replays for each case. Median warm CUDA-event timing (`100` launches, `7` repeats, `normalize_first=1`):

| Shape | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| M960 N2048 | 10.876 µs | 6.538 µs | 1.66× |
| M960 N8192 | 39.026 µs | 21.625 µs | 1.80× |

NCU's paired N8192 measurement changed from 45.184 to 27.264 µs. Executed warp instructions fell from 25,950,720 to 12,526,080; barriers from 69,120 to 38,400; shared loads from 1,105,920 to 614,400; shared stores from 1,228,800 to 614,400. Shared store bank conflicts fell from 161,297 to 686. Registers/thread increased from 22 to 32 with **zero spills**, and achieved active warps remained approximately 77–78%. These are kernel-level measurements, not end-to-end speedups.

## Full-suite matched gate

One H100, GPU UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, GPU-local CPU affinity, B128, M960, FA2, 544-page KV pool, 0.45 BFC fraction, concurrent prefill. Same ZML runner executable SHA-256 `a21a2d2d536068014c9eafca9ecbdae672fba36d09366e96079069500a8ac6ab`; only the QVQ shared library changes between arms. Its baseline SHA-256 is `b109d8ffbaaef7e87e5f02f0ef1097c19f019c48e5de47ce65fe16b01d1bf879`; candidate SHA-256 is `402ed52eab4af839b108e9826c2c8b737d1498fa792e2b9d7b3f9d400706a86f`.

| Run order | Useful prefill tok/s | Padded prefill tok/s | Useful decode tok/s | Padded decode tok/s | Padded decode per stream | Wall s | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate | 75,558 | 87,149 | 10,553 | 12,294 | 96.04 | 24.880 | 543/1,209 |
| Matched control | 73,528 | 84,807 | 10,575 | 12,320 | 96.25 | 25.224 | 543/1,209 |
| Candidate repeat | **76,232** | **87,927** | **10,641** | **12,397** | **96.85** | **24.659** | **543/1,209** |

All three runs had zero invalid outputs and 1,209/1,209 token streams identical to the fixed reference. The prefill token numerator is the same in each run: 1,006,268 useful and 1,160,640 padded slots. Both candidate runs exceed the control by 2.76–3.68% in prefill throughput. Decode differences are small in this comparison; there is no observed sustained decode regression. This remains below the 120,000 padded prefill tok/s objective.

Model: `qvq-f6-p32-r8-seed7-20260908`, Llama 3.2 1B Instruct. Dataset SHA-256 `b4c541a3b63d3d5045acc16dc64370b411384b2eb994b1d7bafa30d677fe4720`; fixed reference SHA-256 `bc1f47aa974032dab8c4b8a40a8d053fd645a78558f96df794da301f9812208b`. The `/monster/data/model` mount is read-only, so full evaluation JSON/logs and NCU binary reports are stored outside the repository at `/root/work/qvq-bench-results/p32-register-high-20260924/` and `/var/tmp/qvq-register-high-*-n8192-20260924.ncu-rep`; this compact report records their key data without committing large artifacts.

The gate ran from Inference-Ultra's `zml_qvq/run_gsm8k_b128_pinned.sh` with pinned GPU-local CPU affinity and the same model, dataset, reference, server options and executable for all three arms. The QVQ library was switched by Bazel `--override_repository=qvq=<control-or-candidate-worktree>` and verified by the hashes above.
