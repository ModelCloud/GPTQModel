# Window row-reuse wave 12

Reuse limits 8 and 16 were tested against the immutable F6 seed-7 window
representation. Each limit covered 12 real projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 288 cases passed the local correctness gate. Median row-reuse/window
full-layer ratios, computed against the matched production-window reports,
were:

| Reuse limit | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.963x | 0.975x | 0.962x | 0.983x | 0.909x | 0.914x | 0.888x | 0.835x | 0.705x | 0.593x | 0.490x | 0.449x |
| 16 | 0.975x | 0.972x | 0.963x | 0.992x | 0.968x | 0.947x | 0.930x | 0.914x | 0.835x | 0.795x | 0.716x | 0.683x |

The current row-group implementation remains slower than the production
window as M grows because it launches and concatenates multiple calls. It does
not advance. The raw reports are in [the wave-12 result directory](results/window-rowreuse-wave12/).
