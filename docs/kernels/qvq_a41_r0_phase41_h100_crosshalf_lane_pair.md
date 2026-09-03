# Phase 41: rejected cross-half level-lane pairing

Phase 41 tested whether the remaining lane-pair table conflicts could be
reduced without adding shared memory. The candidate paired lanes 16 apart on
one replicated 32-bit level-table word instead of pairing adjacent lanes. It
is exact and storage-neutral, but it regresses every rate's geometric mean and
is fully removed.

## Candidate mapping

Production assigns table word

\[
p=\left\lfloor lane/2\right\rfloor.
\]

The candidate instead assigns

\[
p=lane\bmod16,
\]

so lanes `l` and `l+16` share a replicated word. The low and high PGC views
remain the even and odd halfwords of that word. No level value, state, PGC
operation, WGMMA, shared allocation, checkpoint byte, or output boundary
changes.

## Matched H100 result

The physical H100 passed the strict 0% utilization / 0 MiB gate. Timing uses
30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample for grouped gate/up inner P32 plus paired recovery.

The rejected isolated experiment does not run a complete MLP, so Marlin and
Machete full-MLP columns are intentionally not inferred. `Better` is the
strict candidate comparison with Phase 40.

| Rate | M/K/N per child | Phase 40 | Cross-half | vs Phase 40 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|:--|--:|--:|--:|:--:|:--:|:--:|
| W2 | 1/2048/8192 x2 | 29.195 us | 29.502 us | 0.9896x | - | - | No |
| W2 | 2/2048/8192 x2 | 28.851 us | 28.958 us | 0.9963x | - | - | No |
| W2 | 4/2048/8192 x2 | 29.255 us | 29.328 us | 0.9975x | - | - | No |
| W2 | 8/2048/8192 x2 | 29.539 us | 29.628 us | 0.9970x | - | - | No |
| W2 | 16/2048/8192 x2 | 30.377 us | 30.445 us | 0.9978x | - | - | No |
| W2.5 | 1/2048/8192 x2 | 30.811 us | 30.808 us | 1.0001x | - | - | Yes |
| W2.5 | 2/2048/8192 x2 | 30.678 us | 30.762 us | 0.9973x | - | - | No |
| W2.5 | 4/2048/8192 x2 | 30.914 us | 30.984 us | 0.9977x | - | - | No |
| W2.5 | 8/2048/8192 x2 | 31.468 us | 31.529 us | 0.9981x | - | - | No |
| W2.5 | 16/2048/8192 x2 | 32.105 us | 32.172 us | 0.9979x | - | - | No |
| W3 | 1/2048/8192 x2 | 29.651 us | 29.395 us | 1.0087x | - | - | Yes |
| W3 | 2/2048/8192 x2 | 29.416 us | 29.491 us | 0.9974x | - | - | No |
| W3 | 4/2048/8192 x2 | 29.724 us | 29.832 us | 0.9964x | - | - | No |
| W3 | 8/2048/8192 x2 | 30.112 us | 30.224 us | 0.9963x | - | - | No |
| W3 | 16/2048/8192 x2 | 30.715 us | 30.854 us | 0.9955x | - | - | No |
| W3.5 | 1/2048/8192 x2 | 30.245 us | 30.475 us | 0.9924x | - | - | No |
| W3.5 | 2/2048/8192 x2 | 29.899 us | 29.980 us | 0.9973x | - | - | No |
| W3.5 | 4/2048/8192 x2 | 30.275 us | 30.410 us | 0.9956x | - | - | No |
| W3.5 | 8/2048/8192 x2 | 30.875 us | 30.978 us | 0.9967x | - | - | No |
| W3.5 | 16/2048/8192 x2 | 31.480 us | 31.598 us | 0.9963x | - | - | No |

Geometric means are 0.9956x, 0.9982x, 0.9988x, and 0.9956x for W2,
W2.5, W3, and W3.5 respectively.

## Why it fails

Each replicated row contains 32 halfwords but only 16 physical 32-bit words.
Any fixed assignment therefore maps two consumer lanes to one base bank. The
requested PGC indices vary by lane, so changing which two lanes share the word
usually changes the participants without removing the second wavefront. The
new lane-bit permutation also adds address dependencies. The result is a
small consistent loss rather than a reduction in serialized lookup service.

The candidate source is fully removed. Production remains Phase 40, and no
full-MLP run is warranted after the isolated gate fails.

## Next phase

Phase 42 should retain the accepted adjacent-lane table and test a third
in-flight decode/WGMMA fragment for W2, W2.5, and W3.5. W3 already uses depth
three. The intent is to overlap the unavoidable second shared-load wavefront
with independent register-sourced matrix work, without changing storage or
arithmetic.
