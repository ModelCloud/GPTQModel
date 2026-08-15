# QVQ YAQA 512-row full-16-layer sweep (2026-08-15)

This sweep compares canonical V2, V2+YAQA, and V2B2-P32+YAQA on all Q/K/V/O projections in all 16
Llama 3.2 1B decoder layers. It is the full-layer continuation of the earlier four-layer gate.

## Matched configuration

- Model: `unsloth/Llama-3.2-1B-Instruct`, local snapshot `5a8abab4a5d6f164389b1079fb721cfab8d7126c`.
- Dataset: NeuralMagic calibration `llm.parquet`.
- Quantized modules: 64 total, Q/K/V/O for decoder layers 0 through 15. MLP, embeddings, and LM head remained dense.
- Calibration: rows 0--511, 512 independent full rows, batch 1, no concatenation, no length limit.
- Evaluation: rows 512--1023, 512 independent full rows, batch 1, no concatenation, no length limit.
- YAQA Sketch-B: rows 1024--1535, 512 disjoint full rows, batch 8, 64 batches, 163,324 valid tokens.
- Shared validated Sketch-B cache: `artifacts/qvq_yaqa_512_full16/sketch_b_factors.pt`.
- Code revision used by the workers: `51dd27a1`.
- CUDA workers: A100-class `sm_80` GPUs with PyTorch 2.13.0+cu130 and Python 3.14 free-threaded mode.

## Complete results

Lower is better for Relative L2 and every KL column. Higher is better for Top-1/5/10 agreement.

```text
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| Rate | Arm             | Rel L2   | Local KL | Live KL  | Layer KL | Final KL | Top-1  | Top-5  | Top-10 |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| W1.5 | V2              | 0.449703 | 0.012284 | 0.096288 | 0.176816 | 0.229478 | 82.77% | 74.73% | 74.16% |
| W1.5 | V2 + YAQA       | 0.585690 | 0.073607 | 0.122874 | 0.092961 | 0.136739 | 86.73% | 79.79% | 79.28% |
| W1.5 | B2-P32 + YAQA   | 0.574036 | 0.063507 | 0.109521 | 0.086335 | 0.124472 | 87.40% | 80.48% | 80.06% |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| W2   | V2              | 0.322543 | 0.004764 | 0.046202 | 0.078559 | 0.103896 | 88.47% | 82.05% | 81.70% |
| W2   | V2 + YAQA       | 0.434324 | 0.021305 | 0.042147 | 0.036580 | 0.051819 | 91.68% | 86.35% | 86.10% |
| W2   | B2-P32 + YAQA   | 0.424314 | 0.020045 | 0.039853 | 0.033952 | 0.047674 | 92.11% | 86.71% | 86.50% |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| W2.5 | V2              | 0.230755 | 0.002087 | 0.021406 | 0.033271 | 0.049531 | 91.99% | 87.27% | 87.16% |
| W2.5 | V2 + YAQA       | 0.317290 | 0.009083 | 0.018857 | 0.016492 | 0.022321 | 94.16% | 90.40% | 90.21% |
| W2.5 | B2-P32 + YAQA   | 0.309422 | 0.008323 | 0.017591 | 0.015278 | 0.021398 | 94.58% | 90.56% | 90.47% |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| W3   | V2              | 0.165664 | 0.001019 | 0.010779 | 0.015849 | 0.023973 | 94.02% | 90.54% | 90.44% |
| W3   | V2 + YAQA       | 0.230037 | 0.004314 | 0.009210 | 0.007893 | 0.010748 | 96.20% | 93.05% | 92.92% |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
| W3.5 | V2              | 0.119218 | 0.000499 | 0.005549 | 0.007965 | 0.011924 | 96.04% | 93.02% | 93.05% |
| W3.5 | V2 + YAQA       | 0.166311 | 0.002254 | 0.004758 | 0.004011 | 0.005488 | 97.29% | 94.91% | 94.78% |
+------+-----------------+----------+----------+----------+----------+----------+--------+--------+--------+
```

V2B2-P32 supports W1 through W2.5, so no B2 arms were scheduled at W3 or W3.5.

## B2-P32 incremental effect

```text
+------+--------------------+------------------+----------------------+----------------------+----------------------+
| Rate | Final KL vs V2+YAQA| Final KL vs V2   | Top-1 vs V2+YAQA    | Top-5 vs V2+YAQA    | Top-10 vs V2+YAQA   |
+------+--------------------+------------------+----------------------+----------------------+----------------------+
| W1.5 |             -8.97% |          -45.76% |             +0.66 pp |             +0.68 pp |             +0.77 pp |
| W2   |             -8.00% |          -54.11% |             +0.43 pp |             +0.36 pp |             +0.40 pp |
| W2.5 |             -4.14% |          -56.80% |             +0.43 pp |             +0.17 pp |             +0.26 pp |
+------+--------------------+------------------+----------------------+----------------------+----------------------+
```

B2-P32+YAQA improved every reported aggregate over V2+YAQA at all three supported rates. Its Relative L2,
Local KL, Live KL, Layer KL, Final KL, and Top-N metrics all moved in the favorable direction.

## Selector and family diagnostics

```text
+------+-------------------+--------------+----------------------+-------------------------+----------------+
| Rate | Nonzero selectors | Entropy bits | Selector histogram   | Alternative-family uses | Runtime        |
+------+-------------------+--------------+----------------------+-------------------------+----------------+
| W1.5 |            50.04% |     0.999999 | 2,619,147 / 2,623,733| 0 / 24 / 15 / 25        | 3,428.63 sec   |
| W2   |            49.93% |     0.999999 | 2,624,863 / 2,618,017| 0 / 19 / 24 / 21        | 3,258.02 sec   |
| W2.5 |            50.05% |     0.999999 | 2,619,038 / 2,623,842| 0 / 21 / 21 / 22        | 3,210.75 sec   |
+------+-------------------+--------------+----------------------+-------------------------+----------------+
```

The near-one-bit entropy and balanced selector occupancy show that YAQA used both per-segment choices rather than
collapsing to canonical V2. The three alternative families were also all selected across modules at every rate.

## Raw artifacts

The source JSON and logs are under `artifacts/qvq_yaqa_512_full16/`. They are intentionally not committed because
the shared Sketch-B cache is approximately 1.6 GiB and the raw artifacts are reproducible from the configuration
above.
