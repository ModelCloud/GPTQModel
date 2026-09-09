# QQQ native saturation audit (2026-09-09)

The local correction clamps the biased FP16 FMA result to [1024, 1279]
before extracting INT8 bytes. The committed exhaustive arithmetic regression
covers all 507,888 signed-nibble / positive-finite-FP16-scale combinations.
The old kernel failed the saturation endpoint fixture; the corrected kernel
passes all five native grouped tests. Four GSQ quantize/pack/reload cases
(M=1/17, K=256, N=128, group=-1/128) pass with zero eager drift and three
changed-input graph replays on a non-default stream each. The four tests pass
again after profiling. These are synthetic correctness checks, not model-quality
evidence. Allocator pressure and concurrent workspace misuse are not tested here.

## Source and reproduction

Base revision: `2505ec01b83c25ded0da9a39872c7507555f6af4`.
`source.patch` binds the local source/test changes. Corrected CUDA source SHA256:
`91ecb7f3d70c35563dfc5555cbf4bba07d083343f0df864a1a6fd057b6407664`.
Old and corrected JIT fingerprints are `df9dafd6a9487e88` and
`52fb614b41598281`; binary SHA256s are recorded in the identity lines of the logs.
Both use the same build configuration and unchanged specialization/launch geometry.

GPU: physical 0, PCI DE:00.0, UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, SM80, 124 SMs.
Torch 2.15.0.dev20260817+cu130, CUDA runtime 13.0. Each successful capture
used an exclusive allocator lease and three idle preflight samples.
The first unsuccessful capture used the wrong `QQQ` kernel filter; the
successful captures use `Marlin`.

The archived profile script pins the requested binary in a separate process,
then uses the public QQQ layer and actual packer with identical ordinary inputs
that both binaries must reproduce exactly. It warms 20 calls and brackets one
M=17, K=256, N=128, FP16, W4/group128 launch with CUDA profiler markers.
Run through `python -m gpu_allocator.cli run -n 1 --style uuid --`:

```sh
ncu --profile-from-start off --kernel-name regex:Marlin --launch-count 1 \
  --section SpeedOfLight --section InstructionStats --section LaunchStats \
  --section Occupancy --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis -o REPORT \
  python profile-qqq-saturation.py BINARY
ncu --import REPORT.ncu-rep --page source --print-source cuda,sass --csv
ncu --import REPORT.ncu-rep --page raw --csv
```

## Measured results

Raw reports, source-correlated SASS, CSV metrics, scripts and correctness logs
are gzip archived with uncompressed SHA256 bindings in `manifest.json`.
`audit.json` contains extracted metrics/opcode counts and all timing samples.

| Metric | Previous | Corrected |
|---|---:|---:|
| Executed warp instructions | 197089 | 198088 |
| Registers/thread | 254 | 254 |
| Reported local/shared spilling requests | 0/0 | 0/0 |
| Active occupancy (%) | 12.394149 | 12.399050 |
| Eligible warps/cycle | 0.188635 | 0.207902 |
| HFMA2 executed | 768 | 768 |
| HMNMX2 executed | 0 | 1024 |
| PRMT executed | 272 | 256 |
| NCU instrumented duration (us) | 10.720 | 10.848 |
| Public-layer event median (us) | 166.4 | 169.0 |
| Public-layer event p10–p90 (us) | 163.840–177.152 | 162.816–181.248 |

The SASS preserves fused multiplication/rounding and adds the intended half2
min/max clamp. LOP3 (11892), matrix instructions (512), scale/decode FMA and
principal load/store counts remain unchanged. No extra address computation or
load/store round-trip is introduced for the clamp. Small control/polling count
differences are runtime dependent, not algebraic optimization claims. The dominant
reported stall ratio remains immediate-constant-cache miss (7.808560 to 7.242185
per issue-active); a single tiny launch is insufficient for a scheduler conclusion.

Timing uses 100 warmed event pairs without NCU and verifies correctness afterward.
It measures the complete public layer including launch gaps, not isolated GEMM
latency. Distributions overlap; no speedup or stable slowdown is established.
No fusion or overlap optimization is proposed by this correctness audit.

## Final verification and remaining scope

Separate captures explicitly requested
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` with the same marker,
filter and workload. The raw CSVs report shared-load conflicts 72/72 and
shared-store conflicts 3/0 (previous/corrected). This one-launch observation
is not a performance claim. All nine native tests pass after these captures.

An allocator waiter stalled after GPU activity ended because the condition was
only rechecked on a lease event. With no active leases, the stranded request
was terminated and the daemon restarted with periodic FIFO reconsideration.
Two regressions cover finite and indefinite waits with no TTL janitor. The
focused allocator suite passes 35 tests. One intermediate rerun exposed an
existing test that confused worker resume order with assignment order; its
assertion now records the actual assignment order under the allocator lock.
The failed and passing logs are both archived.

This completes the scoped SM80 correctness/profile audit, not real-model QQQ
quality validation or all-shape/all-architecture performance qualification.
Real-model QQQ validation and the broader compatible GSQ adapters remain open.
