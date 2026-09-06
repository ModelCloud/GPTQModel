# Window row-reuse wave 10

The existing row-group reuse path was completed against the immutable F6 seed-7
window representation. Reuse limits 2 and 4 each covered 12 real projections
and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

The matched production-window baseline for the three M values missing from the
earlier decomposition wave was collected separately. All 288 row-reuse cases
and all 36 baseline cases passed the local correctness gate. Median
row-reuse/window full-layer ratios were:

| Reuse limit | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.971x | 0.969x | 0.935x | 0.947x | 0.883x | 0.809x | 0.738x | 0.589x | 0.379x | 0.252x | 0.161x | 0.136x |
| 4 | 0.996x | 0.997x | 0.959x | 0.987x | 0.920x | 0.879x | 0.840x | 0.754x | 0.560x | 0.414x | 0.303x | 0.263x |

The current implementation launches separate window calls and concatenates
their outputs for rows beyond the reuse limit. It therefore gets slower as M
grows; neither limit advances. Direct fragment reuse inside one CTA remains
the relevant follow-up.

Raw reports are in [the row-reuse directory](results/window-rowreuse-wave10/),
including the four matched baseline reports and all eight reuse reports.
