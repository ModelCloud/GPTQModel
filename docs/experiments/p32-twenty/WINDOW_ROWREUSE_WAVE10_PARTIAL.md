# Window row-reuse wave 10 partial

Seven of eight queued row-reuse reports completed against the immutable F6
seed-7 window representation. The completed jobs cover 12 projections and 12
M values for reuse limits 2 and 4, except that the limit-2 GPU1 assignment was
still running when these results were committed. All 252 collected cases
passed the local correctness gate.

Compared with the archived no-reuse window timings on the common M values,
the median row-reuse/window ratios were:

| Reuse limit | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=128 | M=512 | M=2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.965x | 0.971x | 0.931x | 0.949x | 0.882x | 0.811x | 0.590x | 0.261x | 0.111x |
| 4 | 0.996x | 0.997x | 0.959x | 0.987x | 0.920x | 0.879x | 0.754x | 0.414x | 0.263x |

The existing implementation splits the activation rows into separate calls;
its extra launch and concatenation overhead makes it slower as M grows. These
results do not advance row reuse. The remaining reuse limit-2 GPU1 assignment
is retained in the queue and will be committed separately when complete.

Collected raw reports: [r2 GPU0](results/window-rowreuse-wave10/r2-gpu0.json),
[r2 GPU2](results/window-rowreuse-wave10/r2-gpu2.json), [r2
GPU3](results/window-rowreuse-wave10/r2-gpu3.json), [r4
GPU4](results/window-rowreuse-wave10/r4-gpu4.json), [r4
GPU5](results/window-rowreuse-wave10/r4-gpu5.json), [r4
GPU6](results/window-rowreuse-wave10/r4-gpu6.json), and [r4
GPU7](results/window-rowreuse-wave10/r4-gpu7.json).
