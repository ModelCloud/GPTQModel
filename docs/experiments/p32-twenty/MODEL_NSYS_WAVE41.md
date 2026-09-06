# Full-model Nsight wave 41

Wave 41 captured four production-window controls and four BM64/BN32 runs with
Nsight Systems on eight distinct A100-class GPU UUIDs. The trace covered the
full bounded model scorecard: C4-style quality, the nine prefill sizes, and
decode.

The pooled M=2048 prefill medians were approximately 291.7 ms for production
window and 205.6 ms for BM64/BN32, or 1.42×. The profile also showed the
composition of the full run:

| Kernel family share of traced GPU kernel time | Window | BM64/BN32 |
|---|---:|---:|
| Production window kernels | 63.5% | 0% |
| Direct `_gemm` | 0% | 10.4% |
| Hadamard kernels | 10.4% | 6.2% |
| QVQ GEMV kernels | 0.6% | 69.7% |

The percentages are whole-run aggregates across all prefill sizes and decode,
not an isolated M=2048 breakdown. The BM64 arm in this wave used the old
dense fallback below the 512-row threshold, so its low-M timing and QVQ GEMV
share are not valid dispatcher claims. Wave 42 corrected that fallback and is
the authoritative full-model timing record.

The useful profiling conclusion remains that BM64 removes the production
window kernel from the large-prefill path, while Hadamard work is a secondary
boundary cost. The trace did not show a 3× path; deeper decode/MMA fusion or a
native residual operator is still required. Machine-readable kernel/API
summaries and per-arm reports are in [raw wave 41 results](results/model-nsys-wave41/).
The original `.nsys-rep` captures remain on the experiment host at
`/root/p32-model-nsys-wave41/`.
