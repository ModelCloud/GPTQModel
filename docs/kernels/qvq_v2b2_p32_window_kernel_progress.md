# QVQ V2B2-P32 continuous-window Hopper kernel progression

This ledger tracks the accuracy-safe standard-P32 kernel. LR32 was a failed
experiment: it provided no speed advantage over this P32 path and introduced
quality regressions relative to P32, so its experimental kernel, dispatch,
quantization changes, tests, and benchmark records were removed from this PR.
The continuous-window representation is a lossless physical permutation of
canonical planar P32: it has the same word count, adds zero bits, and
reconstructs the identical K16 x N16 matrix.

Anchor-4 is a second lossless physical representation, accepted at `27573a3c`
as an exact format contract but not as a production kernel. It groups four
consecutive states into exactly four transition codes of storage:

| Rate | Transition bits | Exact Anchor-4 record | Bits / record | Original bits |
|---:|---:|---|---:|---:|
| W2 | 4 | state16 | 16 | 16 |
| W2.5 | 5 | state16 + lost4 | 20 | 20 |
| W3 | 6 | state16 + lost2 + next transition6 | 24 | 24 |
| W3.5 | 7 | state16 + lost5 + next transition7 | 28 | 28 |

The H200/CPU suite passes 17/17 Anchor-4 tests at every target rate: byte size,
canonical planar round-trip, all 128 states, decoded 256-value tiles, and exact
K16 x N16 matrix reconstruction. Combined with the existing window suite,
42/42 tests pass. Kernel experiments below show that the current RS-WGMMA
fragment ownership cannot yet consume those anchors economically.

## Measurement contract

- Device: physical GPU 0, NVIDIA H200, UUID
  `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, CC 9.0, 132 SMs.
- Model shapes: Qwen3.8-27B; the first optimization gate is M16 FP16 W3.
- Reference: each candidate is checked against its own dense standard-P32
  reconstruction with `max_abs <= 2e-3`.
- Timing: CUDA Graph external-event medians, 10 warmups and 40 measured launches.
- Comparator: Machete symmetric W4 group 128 is a performance reference, not a
  quality-equivalent format.
- CUTLASS/CuTe: 4.7.1, SM90a RS-WGMMA.

## Accepted progression

| Commit | Device | Rate | M | K | N | Kernel | Split | Median ms | Speedup vs planar P32 | xMachete | Max abs |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `63daa348` | H200 | W3 | 16 | 5120 | 17408 | planar P32 production | auto | 2.49670 | 1.000x | 0.017x | 2.10e-5 |
| `942c5ae6` | H200 | W3 | 16 | 5120 | 17408 | window RS-WGMMA | 4 | 0.12288 | 20.318x | 0.337x | 6.20e-5 |
| `00872584` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 4 | 0.07610 | 32.810x | 0.544x | 6.20e-5 |
| `8dea54a5` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 10 | 0.07378 | 33.840x | 0.561x | 3.05e-5 |
| `035044a3` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 10 | 0.07325 | 33.933x | 0.569x | 3.34e-5 |
| `8a0ab449` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 10 | 0.07282 | 34.125x | 0.575x | 3.05e-5 |
| `8fbf39d2` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 10 | 0.07181 | 34.627x | 0.575x | 3.34e-5 |
| `63daa348` | H200 | W3 | 16 | 17408 | 5120 | planar P32 production | auto | 2.50002 | 1.000x | 0.017x | 3.24e-5 |
| `942c5ae6` | H200 | W3 | 16 | 17408 | 5120 | window RS-WGMMA | 4 | 0.15272 | 16.370x | 0.278x | 3.09e-4 |
| `00872584` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 4 | 0.09253 | 27.019x | 0.459x | 3.09e-4 |
| `8dea54a5` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 34 | 0.07389 | 33.835x | 0.574x | 6.48e-5 |
| `035044a3` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 34 | 0.07246 | 34.684x | 0.591x | 4.96e-5 |
| `8a0ab449` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 34 | 0.07227 | 34.790x | 0.590x | 4.58e-5 |
| `8fbf39d2` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 34 | 0.07109 | 35.385x | 0.595x | 4.96e-5 |

The direct-window mapping was also checked at M16/K256/N64 and the TMA path at
M16/K256/N256.  Their maximum errors were 2.38e-6 and 3.34e-6.  The full window
format suite at `63daa348` passes 21/21 tests across W1-W3.5, including CPU/CUDA
word and state identity.

## Qwen3.8-27B all-rate M16 result

Benchmark head: `8e3c9f89`; all-rate kernel source: `874d9632`; rate-specific
split policy: `d9464071`.  Artifact:
`artifacts/h200_p32_window/qwen38_m16_p32_vs_machete_8e3c9f89.json`.
Every row uses the lossless standard-P32 continuous-window payload and passed
its own dense P32 reference with `max_abs <= 6.49e-5`.  Machete is symmetric W4
group 128 and is a speed reference rather than a quality-equivalent format.

| Rate | Shape | M | K | N | Window ms | Speedup vs planar P32 | Machete W4 ms | xMachete | Max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | Full Q+gate | 16 | 5120 | 12288 | 0.05654 | 31.893x | 0.03478 | 0.615x | 2.86e-5 |
| W2.5 | Full Q+gate | 16 | 5120 | 12288 | 0.05613 | 32.151x | 0.03478 | 0.620x | 3.05e-5 |
| W3 | Full Q+gate | 16 | 5120 | 12288 | 0.05464 | 32.991x | 0.03478 | 0.637x | 3.24e-5 |
| W3.5 | Full Q+gate | 16 | 5120 | 12288 | 0.05462 | 32.975x | 0.03478 | 0.637x | 5.72e-5 |
| W2 | Full K/V | 16 | 5120 | 1024 | 0.01410 | 11.506x | 0.02091 | 1.484x | 1.53e-5 |
| W2.5 | Full K/V | 16 | 5120 | 1024 | 0.01373 | 11.851x | 0.02091 | 1.523x | 1.72e-5 |
| W3 | Full K/V | 16 | 5120 | 1024 | 0.01379 | 11.818x | 0.02091 | 1.516x | 1.53e-5 |
| W3.5 | Full K/V | 16 | 5120 | 1024 | 0.01382 | 11.765x | 0.02091 | 1.513x | 1.53e-5 |
| W2 | Attention out | 16 | 6144 | 5120 | 0.03422 | 26.336x | 0.02426 | 0.709x | 3.43e-5 |
| W2.5 | Attention out | 16 | 6144 | 5120 | 0.03360 | 26.944x | 0.02426 | 0.722x | 4.58e-5 |
| W3 | Attention out | 16 | 6144 | 5120 | 0.03259 | 27.656x | 0.02426 | 0.744x | 3.81e-5 |
| W3.5 | Attention out | 16 | 6144 | 5120 | 0.03248 | 27.786x | 0.02426 | 0.747x | 3.81e-5 |
| W2 | Linear QKV | 16 | 5120 | 10240 | 0.05013 | 31.172x | 0.03478 | 0.694x | 3.24e-5 |
| W2.5 | Linear QKV | 16 | 5120 | 10240 | 0.04995 | 31.231x | 0.03478 | 0.696x | 3.43e-5 |
| W3 | Linear QKV | 16 | 5120 | 10240 | 0.04741 | 34.229x | 0.03478 | 0.734x | 6.48e-5 |
| W3.5 | Linear QKV | 16 | 5120 | 10240 | 0.04736 | 32.283x | 0.03478 | 0.734x | 5.91e-5 |
| W2 | Linear Z | 16 | 5120 | 6144 | 0.03414 | 27.978x | 0.02390 | 0.700x | 1.53e-5 |
| W2.5 | Linear Z | 16 | 5120 | 6144 | 0.03392 | 27.757x | 0.02390 | 0.705x | 1.53e-5 |
| W3 | Linear Z | 16 | 5120 | 6144 | 0.03355 | 28.495x | 0.02390 | 0.712x | 1.53e-5 |
| W3.5 | Linear Z | 16 | 5120 | 6144 | 0.03376 | 27.865x | 0.02390 | 0.708x | 5.15e-5 |
| W2 | MLP gate/up | 16 | 5120 | 17408 | 0.07616 | 33.637x | 0.04194 | 0.551x | 3.05e-5 |
| W2.5 | MLP gate/up | 16 | 5120 | 17408 | 0.07566 | 33.574x | 0.04194 | 0.554x | 3.62e-5 |
| W3 | MLP gate/up | 16 | 5120 | 17408 | 0.07382 | 34.622x | 0.04194 | 0.568x | 3.24e-5 |
| W3.5 | MLP gate/up | 16 | 5120 | 17408 | 0.07422 | 34.196x | 0.04194 | 0.565x | 5.72e-5 |
| W2 | MLP down | 16 | 17408 | 5120 | 0.07482 | 33.645x | 0.04262 | 0.570x | 4.58e-5 |
| W2.5 | MLP down | 16 | 17408 | 5120 | 0.07446 | 33.854x | 0.04262 | 0.572x | 4.58e-5 |
| W3 | MLP down | 16 | 17408 | 5120 | 0.07290 | 34.443x | 0.04262 | 0.585x | 4.96e-5 |
| W3.5 | MLP down | 16 | 17408 | 5120 | 0.07378 | 34.112x | 0.04262 | 0.578x | 5.72e-5 |

| Rate | Seven-shape xMachete geomean | Speedup vs planar-P32 geomean |
|---:|---:|---:|
| W2 | 0.718x | 26.647x |
| W2.5 | 0.726x | 26.871x |
| W3 | 0.743x | 27.709x |
| W3.5 | 0.741x | 27.304x |
| Combined | 0.732x | 27.129x |

The exact paired-window head `5dee9ad4` was rerun across the same complete
Qwen3.8-27B M16 matrix with 10 warmups and 40 measured launches. Artifact:
`artifacts/h200_p32_window/qwen38_m16_p32_vs_machete_5dee9ad4.json`. All 28
window rows passed their dense standard-P32 references; the worst maximum
absolute error was `6.49e-5`.

| Rate | Shape | M | K | N | Window ms | Planar-P32 speedup | Machete W4 ms | xMachete | Max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 | Full Q+gate | 16 | 5120 | 12288 | 0.05338 | 33.635x | 0.03442 | 0.645x | 2.86e-5 |
| W2.5 | Full Q+gate | 16 | 5120 | 12288 | 0.05296 | 33.997x | 0.03442 | 0.650x | 3.05e-5 |
| W3 | Full Q+gate | 16 | 5120 | 12288 | 0.05299 | 33.950x | 0.03442 | 0.649x | 3.24e-5 |
| W3.5 | Full Q+gate | 16 | 5120 | 12288 | 0.05315 | 33.802x | 0.03442 | 0.648x | 5.72e-5 |
| W2 | Full K/V | 16 | 5120 | 1024 | 0.01338 | 12.094x | 0.02086 | 1.560x | 1.53e-5 |
| W2.5 | Full K/V | 16 | 5120 | 1024 | 0.01304 | 12.429x | 0.02086 | 1.600x | 1.72e-5 |
| W3 | Full K/V | 16 | 5120 | 1024 | 0.01326 | 12.265x | 0.02086 | 1.573x | 1.53e-5 |
| W3.5 | Full K/V | 16 | 5120 | 1024 | 0.01325 | 12.234x | 0.02086 | 1.575x | 1.53e-5 |
| W2 | Attention out | 16 | 6144 | 5120 | 0.03190 | 28.337x | 0.02384 | 0.747x | 3.43e-5 |
| W2.5 | Attention out | 16 | 6144 | 5120 | 0.03195 | 28.299x | 0.02384 | 0.746x | 4.58e-5 |
| W3 | Attention out | 16 | 6144 | 5120 | 0.03134 | 28.704x | 0.02384 | 0.761x | 3.81e-5 |
| W3.5 | Attention out | 16 | 6144 | 5120 | 0.03152 | 28.662x | 0.02384 | 0.756x | 3.81e-5 |
| W2 | Linear QKV | 16 | 5120 | 10240 | 0.04728 | 33.021x | 0.03434 | 0.726x | 3.24e-5 |
| W2.5 | Linear QKV | 16 | 5120 | 10240 | 0.04698 | 33.194x | 0.03434 | 0.731x | 3.43e-5 |
| W3 | Linear QKV | 16 | 5120 | 10240 | 0.04576 | 35.206x | 0.03434 | 0.750x | 6.48e-5 |
| W3.5 | Linear QKV | 16 | 5120 | 10240 | 0.04622 | 32.954x | 0.03434 | 0.743x | 5.91e-5 |
| W2 | Linear Z | 16 | 5120 | 6144 | 0.03232 | 29.519x | 0.02349 | 0.727x | 1.53e-5 |
| W2.5 | Linear Z | 16 | 5120 | 6144 | 0.03237 | 29.119x | 0.02349 | 0.726x | 1.53e-5 |
| W3 | Linear Z | 16 | 5120 | 6144 | 0.03246 | 29.432x | 0.02349 | 0.724x | 1.53e-5 |
| W3.5 | Linear Z | 16 | 5120 | 6144 | 0.03275 | 28.744x | 0.02349 | 0.717x | 5.15e-5 |
| W2 | MLP gate/up | 16 | 5120 | 17408 | 0.07187 | 35.654x | 0.04184 | 0.582x | 3.05e-5 |
| W2.5 | MLP gate/up | 16 | 5120 | 17408 | 0.07131 | 35.630x | 0.04184 | 0.587x | 3.62e-5 |
| W3 | MLP gate/up | 16 | 5120 | 17408 | 0.07173 | 35.632x | 0.04184 | 0.583x | 3.24e-5 |
| W3.5 | MLP gate/up | 16 | 5120 | 17408 | 0.07259 | 34.847x | 0.04184 | 0.576x | 5.72e-5 |
| W2 | MLP down | 16 | 17408 | 5120 | 0.07136 | 35.267x | 0.04253 | 0.596x | 4.58e-5 |
| W2.5 | MLP down | 16 | 17408 | 5120 | 0.07122 | 35.409x | 0.04253 | 0.597x | 4.58e-5 |
| W3 | MLP down | 16 | 17408 | 5120 | 0.07157 | 35.166x | 0.04253 | 0.594x | 4.96e-5 |
| W3.5 | MLP down | 16 | 17408 | 5120 | 0.07222 | 34.915x | 0.04253 | 0.589x | 5.72e-5 |

| Rate | Seven-shape xMachete geomean | Speedup vs planar-P32 geomean |
|---:|---:|---:|
| W2 | 0.753x | 28.185x |
| W2.5 | 0.758x | 28.312x |
| W3 | 0.760x | 28.565x |
| W3.5 | 0.755x | 28.049x |
| Combined | 0.757x | 28.277x |

## Qwen3.8-27B full M1/M2/M4/M8/M16 comparison

Benchmark harness/head: `65379257`; kernel source: `8fbf39d2`. Artifact:
`artifacts/h200_p32_window/qwen38_m1_m2_m4_m8_m16_p32_vs_machete_65379257.json`.
All 140 exact-P32 rows passed their dense references; the worst maximum
absolute error was `7.44e-5`. The production prototype is currently fixed at
M16, so M1/M2/M4/M8 measurements include copying live rows into a persistent
zero-padded M16 input inside the timed CUDA graph. Machete and planar P32 run
at their native logical M.

Each full-matrix cell below is `P32 milliseconds / xMachete`.

| M | Shape | K | N | W2 | W2.5 | W3 | W3.5 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Full Q+gate | 5120 | 12288 | 0.05562/0.628x | 0.05504/0.635x | 0.05510/0.634x | 0.05536/0.631x |
| 1 | Full K/V | 5120 | 1024 | 0.01555/1.314x | 0.01550/1.318x | 0.01549/1.319x | 0.01549/1.319x |
| 1 | Attention out | 6144 | 5120 | 0.03419/0.702x | 0.03427/0.700x | 0.03370/0.712x | 0.03390/0.708x |
| 1 | Linear QKV | 5120 | 10240 | 0.04955/0.699x | 0.04928/0.703x | 0.04850/0.714x | 0.04859/0.713x |
| 1 | Linear Z | 5120 | 6144 | 0.03461/0.692x | 0.03453/0.694x | 0.03437/0.697x | 0.03419/0.701x |
| 1 | MLP gate/up | 5120 | 17408 | 0.07309/0.563x | 0.07325/0.562x | 0.07366/0.559x | 0.07472/0.551x |
| 1 | MLP down | 17408 | 5120 | 0.07294/0.584x | 0.07256/0.587x | 0.07346/0.580x | 0.07416/0.574x |
| 2 | Full Q+gate | 5120 | 12288 | 0.05482/0.630x | 0.05446/0.634x | 0.05469/0.632x | 0.05443/0.635x |
| 2 | Full K/V | 5120 | 1024 | 0.01501/1.362x | 0.01485/1.377x | 0.01488/1.374x | 0.01501/1.362x |
| 2 | Attention out | 6144 | 5120 | 0.03382/0.697x | 0.03366/0.701x | 0.03304/0.714x | 0.03339/0.706x |
| 2 | Linear QKV | 5120 | 10240 | 0.04912/0.702x | 0.04883/0.706x | 0.04781/0.721x | 0.04792/0.719x |
| 2 | Linear Z | 5120 | 6144 | 0.03416/0.695x | 0.03414/0.695x | 0.03395/0.699x | 0.03445/0.689x |
| 2 | MLP gate/up | 5120 | 17408 | 0.07338/0.563x | 0.07347/0.562x | 0.07374/0.560x | 0.07488/0.551x |
| 2 | MLP down | 17408 | 5120 | 0.07293/0.585x | 0.07277/0.586x | 0.07344/0.581x | 0.07424/0.574x |
| 4 | Full Q+gate | 5120 | 12288 | 0.05491/0.628x | 0.05461/0.631x | 0.05446/0.633x | 0.05440/0.634x |
| 4 | Full K/V | 5120 | 1024 | 0.01507/1.358x | 0.01486/1.377x | 0.01483/1.380x | 0.01493/1.371x |
| 4 | Attention out | 6144 | 5120 | 0.03360/0.701x | 0.03350/0.703x | 0.03309/0.712x | 0.03320/0.710x |
| 4 | Linear QKV | 5120 | 10240 | 0.04896/0.703x | 0.04861/0.708x | 0.04773/0.721x | 0.04819/0.714x |
| 4 | Linear Z | 5120 | 6144 | 0.03411/0.694x | 0.03392/0.698x | 0.03386/0.699x | 0.03430/0.690x |
| 4 | MLP gate/up | 5120 | 17408 | 0.07334/0.564x | 0.07341/0.563x | 0.07379/0.560x | 0.07502/0.551x |
| 4 | MLP down | 17408 | 5120 | 0.07294/0.581x | 0.07299/0.581x | 0.07363/0.576x | 0.07453/0.569x |
| 8 | Full Q+gate | 5120 | 12288 | 0.05498/0.629x | 0.05453/0.634x | 0.05482/0.631x | 0.05459/0.634x |
| 8 | Full K/V | 5120 | 1024 | 0.01504/1.377x | 0.01482/1.397x | 0.01485/1.394x | 0.01514/1.368x |
| 8 | Attention out | 6144 | 5120 | 0.03373/0.698x | 0.03366/0.699x | 0.03296/0.714x | 0.03342/0.704x |
| 8 | Linear QKV | 5120 | 10240 | 0.04912/0.700x | 0.04885/0.704x | 0.04781/0.719x | 0.04805/0.716x |
| 8 | Linear Z | 5120 | 6144 | 0.03392/0.688x | 0.03395/0.687x | 0.03389/0.688x | 0.03442/0.678x |
| 8 | MLP gate/up | 5120 | 17408 | 0.07315/0.562x | 0.07326/0.561x | 0.07368/0.558x | 0.07480/0.550x |
| 8 | MLP down | 17408 | 5120 | 0.07314/0.581x | 0.07301/0.583x | 0.07350/0.579x | 0.07434/0.572x |
| 16 | Full Q+gate | 5120 | 12288 | 0.05342/0.646x | 0.05299/0.652x | 0.05290/0.653x | 0.05277/0.654x |
| 16 | Full K/V | 5120 | 1024 | 0.01336/1.514x | 0.01325/1.527x | 0.01320/1.532x | 0.01349/1.499x |
| 16 | Attention out | 6144 | 5120 | 0.03208/0.740x | 0.03186/0.745x | 0.03141/0.755x | 0.03158/0.751x |
| 16 | Linear QKV | 5120 | 10240 | 0.04747/0.725x | 0.04698/0.732x | 0.04573/0.752x | 0.04606/0.747x |
| 16 | Linear Z | 5120 | 6144 | 0.03238/0.726x | 0.03235/0.727x | 0.03211/0.732x | 0.03269/0.720x |
| 16 | MLP gate/up | 5120 | 17408 | 0.07131/0.574x | 0.07133/0.574x | 0.07162/0.571x | 0.07261/0.564x |
| 16 | MLP down | 17408 | 5120 | 0.07077/0.598x | 0.07098/0.596x | 0.07110/0.595x | 0.07226/0.585x |

| M | W2 geomean | W2.5 geomean | W3 geomean | W3.5 geomean | All-rate geomean |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.711x | 0.714x | 0.716x | 0.712x | 0.713x |
| 2 | 0.716x | 0.718x | 0.721x | 0.715x | 0.718x |
| 4 | 0.715x | 0.718x | 0.721x | 0.715x | 0.717x |
| 8 | 0.715x | 0.718x | 0.720x | 0.712x | 0.716x |
| 16 | 0.748x | 0.751x | 0.756x | 0.748x | 0.751x |
| All M | 0.721x | 0.724x | 0.727x | 0.720x | 0.723x |

The isolated CuTe translation unit compiles all four specializations in 44
seconds on this host.  The complete window suite now passes 25/25 tests on the
H200, including exact TMA RS-WGMMA reconstruction at W2-W3.5.

## H200 NCU profile

Profiled head: `f36e1837`; kernel source: `00872584`.  Shape:
M16/K5120/N17408 W3 split 4.  Report:
`artifacts/h200_p32_window/profiles/qwen38_gate_w3_p32_tma_rs_wgmma_split4_f36e1837.ncu-rep`.

| Metric | Result |
|---|---:|
| NCU duration | 71.648 us |
| Grid / block | 1088 CTAs / 160 threads |
| Waves per SM | 1.18 |
| Warp instructions | 36.842M |
| Registers / thread | 56 |
| Static shared memory | 29.31 KiB |
| Local/shared spills | 0 / 0 |
| Theoretical / achieved occupancy | 54.69% / 41.59% |
| DRAM / L1 throughput | 10.36% / 78.44% |
| Shared wavefronts actual / ideal | 3.482M / 3.482M |
| Shared bank conflicts | 160 |
| Global PGC level loads | 2.785M |
| Shared window loads | 2.785M |
| TMA pipe utilization | 0.39% |
| No eligible warp | 40.28% |
| Long-scoreboard stall samples | 2,087 |
| Wait / math-throttle samples | 1,022 / 853 |

The dominant opcode mix is LOP3 8.863M (24.1%), IMAD 8.076M (21.9%), SHF
6.406M (17.4%), LDS 3.133M (8.5%), and LDG 2.785M (7.6%).  Every P32 state is
decoded exactly once, so recurrence deduplication is no longer available.  The
next experiments should target the random read-only PGC level loads and launch
tail without damaging the already conflict-free shared/TMA layout.

The accepted all-rate head `c07fe9de` was also profiled at W3.5,
M16/K5120/N17408, split 5.  Report:
`artifacts/h200_p32_window/profiles/qwen38_gate_w35_p32_tma_rs_wgmma_split5_c07fe9de.ncu-rep`.

| Metric | W3.5 result |
|---|---:|
| NCU duration | 69.152 us |
| Grid / waves per SM | 1,360 CTAs / 1.47 |
| Executed instructions | 36.786M |
| Registers / static shared | 56 / 31.36 KiB |
| L1 / L2 hit rate | 99.65% / 52.72% |
| L1/TEX / ALU utilization | 80.28% / 60.57% |
| Shared / tensor / TMA pipe | 4.28% / 4.28% / 0.40% |
| Shared conflicts / excessive wavefronts | 160 / 0 |
| Excessive global sectors | 33.520M (85% of all sectors) |
| No eligible warp | 38.66% |
| Long-scoreboard samples | 1,868 (744 not issued) |
| Long-scoreboard cycles per issue | 3.545 (30.8%) |

Commit `035044a3` replaces the two independent bank-bit shift/negate/AND
chains with one common shift and two 0/1 mask multiplies.  All 25 exact window
tests pass.  The matched W3 split-4 NCU capture drops executed instructions
from 36.570M to 35.863M (-1.9%) and duration from 71.65 to 70.14 us while
retaining 56 registers/thread, 29.31 KiB static shared memory, and a 99.72% L1
hit rate.  The event-timed Qwen3.8 MLP comparison against the prior all-rate
checkpoint is:

| Rate | Gate/up before ms | Gate/up `035044a3` ms | Speedup | Down before ms | Down `035044a3` ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| W2 | 0.07616 | 0.07315 | 1.041x | 0.07482 | 0.07243 | 1.033x |
| W2.5 | 0.07566 | 0.07376 | 1.026x | 0.07446 | 0.07371 | 1.010x |
| W3 | 0.07382 | 0.07325 | 1.008x | 0.07290 | 0.07246 | 1.006x |
| W3.5 | 0.07422 | 0.07374 | 1.007x | 0.07378 | 0.07312 | 1.009x |

Commit `8a0ab449` specializes the four W3 low-byte PGC lookups with explicit
wide global addresses.  The focused split-4 NCU capture measures 34.429M
executed instructions, another 1.434M reduction from `035044a3` and 2.141M
(-5.9%) from the original 36.570M profile.  Registers remain 56 and static
shared memory remains 29.31 KiB.  Applying the helper to every rate slowed W2
and W2.5 by roughly 2%, so the accepted code is deliberately W3-only; the
100-sample all-rate gate and all 25 exact P32 window tests pass.

Commit `8fbf39d2` decodes the `k` and `k+4` P32 states together.  Those states
share one bit shift and have an exact compile-time `2E`-word separation, so the
pair helper removes redundant bit-position, wrap, and address work without
changing the eight shared loads or four exact funnel shifts.  W3 split-4 NCU
drops from 34.429M to 33.903M instructions, registers fall from 56 to 55, and
duration falls from 70.18 to 69.89 us.  The 100-sample all-rate MLP gate is
monotonic: gate/up is 0.07227/0.07173/0.07235/0.07315 ms and down is
0.07070/0.07070/0.07117/0.07214 ms for W2/W2.5/W3/W3.5 respectively.

W3.5 executes essentially the same instruction count as the earlier W3
profile.  The remaining front-end cost is therefore common P32 state-address,
window-extraction, PGC-mix, and level-fetch work rather than a W3.5-only planar
expansion problem.

## Rejected experiments

| Head | Experiment | Gate M16/K5120/N17408 | Down M16/K17408/N5120 | Decision |
|---|---|---:|---:|---|
| `12b3a321` + working tree | Warp-distributed register PGC table | 0.10992 ms (0.687x baseline) | 0.14904 ms (0.620x baseline) | Rejected; exact, but an arbitrary lookup needs four requester-dependent shuffles and is 31-39% slower than the 99.7%-L1-hit read-only table. |
| `c07fe9de` + working tree | Eight-way lane-interleaved shared PGC table | W2 0.07453 ms (1.022x), W3 0.07594 ms (0.972x), W3.5 0.07707 ms (0.963x) | W2 0.07421 ms (1.008x), W3 0.07546 ms (0.966x), W3.5 0.07571 ms (0.974x) | Rejected; the extra shared footprint/occupancy loss outweighs reduced global lookup pressure at W3/W3.5. |
| `e143f6a1` + working tree | Current-window producer-contiguous decode with 16 gathers | W3 0.09174 ms (0.805x baseline) | W3 0.09141 ms (0.798x baseline) | Rejected; exact, but 44.249M instructions and the gather network exceed the consumer-owned window path. |
| `27573a3c` + working tree | Anchor-4 decoded independently in consumer ownership | W3 0.13920 ms (0.530x baseline) | W3 0.13818 ms (0.528x baseline) | Rejected; exact, but divergent step reconstruction raises instructions to 74.656M and registers to 72. |
| `27573a3c` + working tree | Anchor-4 producer decode, four shuffles, two mixed-index `movmatrix` | W3 0.08358 ms (0.883x baseline) | W3 0.08333 ms (0.875x baseline) | Rejected; exact and 55 registers with 96 shared conflicts, but 42.588M instructions remain above the 36.8M window baseline. |
| `27573a3c` + working tree | Proposed zero-shuffle two-`movmatrix` lower bound | NCU 0.07325 ms | not timed | Rejected; not exact because canonical anchors span two K-row/N-half quadrants. Even before exact routing it executes 40.413M instructions, missing the <32M gate. |
| `2186324a` + working tree | Stage the level-table pointer through shared memory to force a GPR address base | W3 0.07442 ms | W3 0.07333 ms | Rejected; NCU rises to 34.504M instructions, 29.44 KiB shared, 71.52 us, and 43.11% no-eligible cycles instead of deleting the second address instruction. |
| `5dee9ad4` + working tree | Pack two W3 PGC mixers into 16-bit lanes | W3 0.07510 ms | W3 0.07450 ms | Rejected; exact, but `PRMT` rises from 1.414M to 2.807M, total instructions rise from 33.903M to 35.307M, and NCU duration rises from 69.89 to 72.70 us with registers/shared unchanged. |

The zero-shuffle mismatch is not an epilogue-only N permutation. PTX assigns
destination lane `q` source rows `(2q, 2q+1)`; after the proposed row
permutation those are the desired bank pairs, but a canonical four-state anchor
also crosses K-row/N-half ownership. This makes the K permutation depend on N,
after WGMMA has already consumed the activation, so no output-column remap can
restore the exact matrix. The durable branch therefore retains only the exact
Anchor-4 conversion/reference contract, not any rejected CUDA path.

The Qwen3.8 MLP split sweep at `8dea54a5` accepted split 10 for gate/up and
split 34 for down.  These policies preserve K256 stage alignment, reduce the
partial-wave penalty seen in NCU, and are selected automatically for those two
exact shapes.  The default output is bitwise identical to explicitly requesting
the selected split.

## Coverage queue

| Priority | Coverage | State |
|---:|---|---|
| 1 | Warp-register PGC table versus read-only L1 | rejected; keep read-only L1 |
| 2 | Rate-specific split/grid policy for all seven Qwen shapes | accepted at `d9464071` |
| 3 | Generalize direct-window TMA RS-WGMMA to W2, W2.5, and W3.5 | accepted at `874d9632` |
| 4 | Qwen3.8 M1/M2/M4/M8 specializations | pending |
| 5 | Full seven-shape W2-W3.5 P32 versus Machete sweep | M16 complete; M1/M2/M4/M8 pending |
| 6 | Producer-contiguous four-state decode and fixed WGMMA register transpose | rejected for the current RS fragment ownership |
| 7 | Storage-neutral P32 Anchor-4 load-time repack | exact format accepted at `27573a3c`; CUDA mappings rejected |
