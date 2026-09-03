# Phase 48: H100 register-prefetched P32 decode

Phase 48 separates the next decoded W2.5/W3 fragment from the WGMMA source
fragment that is still live. This allows the level-table shared-memory loads
to execute before the WGMMA dependency wait. The optimization is promoted
only for the Llama 3.2 1B grouped gate/up launch, where both the isolated
kernel and complete MLP improve. Ordered-split QKV, split-16 down, W2, and
W3.5 retain the Phase-47 schedule.

## Dependency problem

For fragment depth four, fragment slot `j mod 4` cannot be overwritten until
the oldest outstanding WGMMA has stopped consuming it. The Phase-47 source
placed the wait immediately before the level lookup, but the generated SASS
placed the dependency barrier before most of the state-to-level chain:

```text
state extraction -> PGC -> WGMMA dependency wait -> eight level loads
```

That left the shared-memory level fetch on the critical path even though it
does not depend on the old fragment values.

Phase 48 creates an independent register fragment `D`:

\[
D_k = L\left(\operatorname{PGC}(S_k, B_k)\right),\quad k=0,\ldots,7
\]

and schedules:

```text
state extraction -> PGC -> eight level loads into D
                 -> WGMMA dependency wait -> copy D into reusable F
```

`D` and `F` contain the same eight FP16 values. The change does not alter
P32 decoding, K order, WGMMA order, child-local split reduction, FP16
rounding, or the output tensor. It changes register lifetime only.

## Why more instructions are faster

The compiler materializes the independent fragment with additional register
packing/copy instructions. Nsight Compute nevertheless shows that the added
work removes a longer exposed dependency stall.

| W3 M1 grouped gate/up metric | Phase 47 | Phase 48 | Change |
|:--|--:|--:|--:|
| Duration | 27.136 us | 25.152 us | **-7.31%** |
| Executed instructions | 10,631,478 | 11,529,740 | +8.45% |
| Shared-load instructions | 2,228,224 | 2,228,224 | unchanged |
| Shared-load bank conflicts | 1,051,103 | 1,048,848 | -0.21% |
| Shared-load wavefronts | 3,384,809 | 3,396,281 | +0.34% |
| Eligible warps/cycle | 0.594 | 0.763 | **+28.5%** |
| Long-scoreboard cycles/instruction | 1.364 | 1.183 | **-13.2%** |
| Wait-stall ratio | 1.004 | 0.875 | **-12.9%** |
| Registers/thread | 59 | 67 | +8 |
| Shared memory | 46,848 B | 46,848 B | unchanged |

The SASS contains the eight `LDS.U16` level loads before
`WARPGROUP.DEPBAR.LE gsb0, 0x3`; packing into the reusable WGMMA fragment
follows the barrier. This is measured with Nsight Compute 2026.2.1, not an
inference from source code.

## Launch-specific promotion

A broad implementation improved the large unsplit gate/up launch but
regressed the narrow ordered-split QKV launch through register pressure and
did not reliably help split-16 down. The final dispatch is therefore:

| Site | W2 | W2.5 | W3 | W3.5 |
|:--|:--:|:--:|:--:|:--:|
| Grouped gate/up, K=2048, N=(8192,8192), split 1 | Phase 47 | **Prefetch** | **Prefetch** | Phase 47 |
| Grouped QKV, ordered split `(8,8,8)` | Phase 47 | Phase 47 | Phase 47 | Phase 47 |
| Non-grouped down, split 16 | Phase 47 | Phase 47 | Phase 47 | Phase 47 |

The final no-prefetch QKV and down audits are repeatable, CUDA Graph stable,
and inside the dense-P32 error bound. No speculative shape-wide policy was
promoted.

## Isolated grouped gate/up result

| M/K/N per child | W2.5 Phase 47 | W2.5 Phase 48 | Better | W3 Phase 47 | W3 Phase 48 | Better |
|:--|--:|--:|:--:|--:|--:|:--:|
| 1/2048/8192 x2 | 28.967 us | 28.052 us | Yes | 28.780 us | 27.214 us | Yes |
| 2/2048/8192 x2 | 28.431 us | 27.583 us | Yes | 28.726 us | 26.957 us | Yes |
| 4/2048/8192 x2 | 28.849 us | 27.985 us | Yes | 28.909 us | 27.057 us | Yes |
| 8/2048/8192 x2 | 29.277 us | 28.409 us | Yes | 29.353 us | 27.626 us | Yes |
| 16/2048/8192 x2 | 29.991 us | 29.180 us | Yes | 30.124 us | 28.262 us | Yes |

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples on
the physical 132-SM H100. It uses 30 warmups, 200 CUDA-event samples, and 50
warmed CUDA Graph replays per sample. Effective throughput counts logical
dense-equivalent FLOPs. Marlin and Machete are figurative W4 baselines;
ratios above one mean QVQ is faster. `Better` compares with Phase 47.

| W | MKN: gate/up x2; down | QVQ | vs Marlin W4 | vs Machete W4 | Better |
|--:|:--|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 47.965 us | 0.608x | 1.030x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 48.187 us | 0.648x | 1.027x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 48.699 us | 0.645x | 1.023x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 49.426 us | 0.595x | 1.011x | No |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 50.616 us | 0.644x | 0.987x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 48.553 us | 0.601x | 1.018x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 48.644 us | 0.642x | 1.018x | Yes |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 49.354 us | 0.636x | 1.009x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 49.915 us | 0.589x | 1.001x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 51.031 us | 0.639x | 0.979x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 47.637 us | 0.613x | 1.037x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 47.991 us | 0.650x | 1.031x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 48.552 us | 0.647x | 1.026x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 49.187 us | 0.598x | 1.016x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 50.294 us | 0.648x | 0.993x | Yes |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 49.995 us | 0.584x | 0.989x | Yes |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 50.930 us | 0.613x | 0.972x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 51.358 us | 0.612x | 0.970x | No |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 51.560 us | 0.571x | 0.969x | Yes |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 52.799 us | 0.618x | 0.946x | No |

Across all 20 rows, Phase 48 improves **1.0156x** geometrically and wins
16/20 cells. Its geometric ratios are **0.6195x Marlin W4** and
**1.0024x Machete W4**. All rows are repeatable, CUDA Graph stable, and
inside the dense-P32 accuracy bound.

## Correctness and resource scope

- 126 Hopper P32/grouped-P32 tests pass.
- Canonical checkpoint bytes, transient packed bytes, and persistent VRAM
  are unchanged.
- The extra eight-value decoded fragment is register-local; shared memory and
  workspace are unchanged.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase48_h100/production_mlp_gateup_register_prefetch_vs_phase47.json`
- `artifacts/a41_phase48_h100/register_prefetch_dispatch_audit.json`
- `artifacts/a41_phase48_h100/register_prefetch_profile.json`
- Nsight Compute report outside Git at
  `/root/qvq-profiler-artifacts/phase48-register-prefetch/`

## Next phase

Phase 49 should pre-pack the eight decoded FP16 values into the four 32-bit
registers consumed by WGMMA before the dependency wait. If compiler register
renaming can eliminate the post-wait unpack/copy chain, it preserves Phase
48's latency overlap while reducing the work exposed immediately after the
barrier. The experiment remains limited to grouped gate/up W2.5/W3 until the
same shape audit proves otherwise.
