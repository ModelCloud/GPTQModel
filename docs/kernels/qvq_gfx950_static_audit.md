# gfx950 exported-artifact static audit, 2026-09-07

Scope: the 18 artifacts in the local `qwen-full-m1-m2048` export set used by
native ZML evaluation. Each HSACO SHA256 was checked against its manifest.
Exports predate ABI publication at `52506624`; the manifests record original
source/exporter hashes. This is not a post-commit executed-instruction profile.

Counts below are static assembly opcode lines matching AMD scalar/vector,
DS, global/flat/buffer/scratch families. They are not dynamic instructions,
loop-trip-weighted counts, timing, or a before/after speedup comparison.
VGPR is `.amdhsa_next_free_vgpr`; LDS is the exported launch shared-byte count.

```text
M     K      N      TB BANK STATIC_OPS MFMA VGPR LDS_BYTES
1     17408  5120    6  1         210    0   30       256
1      5120  1024    5  1         212    0   32       256
1      5120  1024    7  1         212    0   32       256
1      5120  10240   6  1         210    0   30       256
1      5120  12288   4  1         234    0   26       256
1      5120  17408   6  1         210    0   30       256
1      5120  17408   7  1         210    0   30       256
1      5120  6144    7  1         210    0   30       256
1      6144  5120    8  0         190    0   26       256
2048  17408  5120    6  1        2136   64  256    139264
2048   5120  1024    5  1         823   16  118     40960
2048   5120  1024    7  1         823   16  118     40960
2048   5120  10240   6  1        2136   64  256    139264
2048   5120  12288   4  1        2151   64  256    139264
2048   5120  17408   6  1        2136   64  256    139264
2048   5120  17408   7  1        2136   64  256    139264
2048   5120  6144    7  1        2136   64  256    139264
2048   6144  5120    8  0        2037   64  256    139264
```

Seven prefill specializations declare 256 VGPRs and 139264 LDS bytes. This
makes resource pressure a candidate for hardware investigation, not a measured
bottleneck. Decode uses a different kernel family and must be profiled separately.
No algebra rewrite or launch-geometry change is justified by these counts alone.

The M=2048 K=5120 N=17408 TB=6 assembly maps scalar signed division/remainder
correction around program-ID swizzling to `qvq_amd.py:675`. Investigate whether
nonnegative program-ID range information can remove this correction, while
preserving the complete grid permutation. This is only a source-correlated
hypothesis: no rewrite, correctness test, opcode delta, or speedup is claimed.

Still required: matched executed counters/trace, spills, occupancy and scheduler
evidence, full-operator correctness, and isolated warmed timing. The ongoing
GSM8K evaluation is not an isolated kernel performance benchmark.
