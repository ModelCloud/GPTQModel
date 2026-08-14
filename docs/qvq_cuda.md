# QVQ CUDA kernel progress

The CUDA runtime now decodes `pgc16-v1` planar QVQ directly for every integer width from W1 through W8. The canonical
256-level FP16 table is process-cached per device and is not a module buffer, checkpoint tensor, or dequantized-weight
cache. The runtime does not retain a dequantized weight. The numerical oracle remains the independent PGC16 dense
reconstruction in `gptqmodel.quantization.qvq`.

## 2026-08-12 range-safe FP16 factorization

The CUDA GEMV always accumulated in FP32, but previously narrowed its inner result to FP16 before the output
Hadamard, `SV`, and bias epilogue. QVQ's factored intermediate can exceed FP16 even when the equivalent dense linear
and final output are finite. The first graph-safe workaround captured both FP16 and BF16 model forwards and selected
one on-device. It was correct but doubled substantial work.

The production path now returns the FP16-input GEMV accumulator in FP32 and uses a single fused output transform.
The transform reproduces every historical FP16 rounding point when the value is finite and retains FP32 only at an
operation that would overflow. This applies to the initial narrowing, every butterfly, `SV`, and bias. The input
transform similarly fuses `SU` with early normalization so `x * SU` cannot overflow before a mathematically finite
normalized transform. Ordinary finite input and output paths are bitwise identical to the previous FP16 operation
order. Composite Hadamard widths use the same rule in the Python fallback; BF16 is reserved for the uncommon
composite/narrow input-transform rescue. No persistent decoded-weight or activation cache is introduced.

The end-to-end decode benchmark used the full W4 Llama-3.2-1B QVQ checkpoint, FP16, M=1, Python 3.14.6 free-threaded
with `PYTHON_GIL=0`, PyTorch 2.13.0+cu130, and one PG506-230 `sm_80` GPU. Times are synchronized host medians. CUDA
graphs produce logits bitwise equal to eager execution.

```text
+---------------------------+----------+----------+-------------------+----------------------+
| Runtime                   | Eager ms | Graph ms | Graph/eager gain  | Accuracy             |
+---------------------------+----------+----------+-------------------+----------------------+
| Dual FP16+BF16 workaround | 88.9     | 54.3     | 1.64x             | finite, graph exact  |
| Range-safe single decode  | 55.8     | 15.7     | 3.55x             | finite, graph exact  |
+---------------------------+----------+----------+-------------------+----------------------+
```

The single-decode graph is 3.46x faster than the dual-workaround graph and remains about 13.8x faster than the
historical approximately 216 ms transient-decode baseline. A full-checkpoint comparison on 64 held-out calibration
rows `[512, 576)`, maximum length 128, produced exactly the same reported metrics before and after the change:

```text
+---------------------------+-------------+-------------+----------+----------+----------+
| Runtime                   | Fwd KLD     | JSD         | Top-1    | Top-5    | Rel-L2   |
+---------------------------+-------------+-------------+----------+----------+----------+
| Dual FP16+BF16 workaround | 0.023862908 | 0.005758327 | 0.933459 | 0.915211 | 0.114825 |
| Range-safe single decode  | 0.023862908 | 0.005758327 | 0.933459 | 0.915211 | 0.114825 |
+---------------------------+-------------+-------------+----------+----------+----------+
```

Validation covers all W1--W8 half-step rates, FP16/BF16, M=1/2/4/8/16/32, structured trellises, dense-reference
MSE/KLD/top-k gates, deterministic and non-default-stream launches, CUDA graphs, power-of-two and composite transform
widths, and overflow at the pre-scale, GEMV, butterfly, `SV`, and bias stages. The complete one-GPU suite passed 502
tests with two expected multi-GPU skips; the two-device free-threaded run passed both skipped concurrency tests.

## 2026-08-11 W1 extension

W1 keeps the `L=16`, `V=2` codec and emits one bit per scalar, so each 16x16 tile occupies eight `int32` trellis
words. It adds no checkpoint tensor and does not enable W1 for the unrelated `gptq_p` format. The W1 Viterbi rate has
four predecessor prefixes and 16,384 suffixes. Its CUDA kernel therefore opts into 64 KiB of dynamic shared memory;
the conservative 16-tile quantization batch bounds retained backpointers to about 132 MiB. The launch fails closed if
the device cannot provide that shared-memory limit.

The following synchronized CUDA-event run used Python 3.14.6 free-threaded, FP16, K=N=4096, 20 warmups, 100 kernel
samples, and three transient dense-reference samples on one PG506-230 `sm_80` GPU. The reference reconstructs the
dense weight inside every call; neither path retains a dequantized cache.

```text
+----+--------+---------+---------+-------------+--------------+--------+--------+
| M  | Ref ms | CUDA ms | Speedup | MSE         | Fwd KLD      | Top-1  | Top-5  |
+----+--------+---------+---------+-------------+--------------+--------+--------+
| 1  | 7.9780 | 0.2273  | 35.09x  | 4.62788e-6  | 2.80260e-45  | 1.0000 | 1.0000 |
| 2  | 7.9155 | 0.2232  | 35.46x  | 1.29918e-6  | 5.60519e-45  | 1.0000 | 1.0000 |
| 4  | 7.9206 | 0.2284  | 34.69x  | 1.75040e-6  | 5.65342e-9   | 1.0000 | 1.0000 |
| 8  | 8.0015 | 0.2365  | 33.83x  | 1.32420e-6  | 1.32951e-11  | 1.0000 | 1.0000 |
| 16 | 7.9565 | 0.2662  | 29.88x  | 1.97820e-6  | -5.44776e-9  | 1.0000 | 1.0000 |
| 32 | 7.9340 | 0.2826  | 28.07x  | 6.45861e-6  | 8.92031e-10  | 1.0000 | 1.0000 |
+----+--------+---------+---------+-------------+--------------+--------+--------+
```

All 48 DeepSeek V4 Flash attention/MoE shape rows at M=1/2/4/8/16/32 also passed against independent dense PGC16
reconstruction: top-1 and ordered top-5 were 1.0 throughout, maximum MSE was `2.67478e-5`, maximum forward KLD was
`2.25268e-6`, and transient-decode speedup ranged from 15.83x to 61.58x. The native W1 Viterbi path is bitwise exact
against eager for states and loss. Its measured batch progression was:

```text
+-------+-----------+---------+----------------+------------------+----------------+------------+
| Batch | Median ms | Tiles/s | Peak alloc MiB | Peak reserve MiB | Loss abs delta | Path exact |
+-------+-----------+---------+----------------+------------------+----------------+------------+
| 1     | 9.444     | 105.9   | 8.2            | 20.0             | 0.000e+00      | yes        |
| 4     | 9.415     | 424.8   | 33.0           | 32.0             | 0.000e+00      | yes        |
| 8     | 9.465     | 845.2   | 66.0           | 84.0             | 0.000e+00      | yes        |
| 16    | 9.874     | 1620.4  | 132.1          | 148.0            | 0.000e+00      | yes        |
+-------+-----------+---------+----------------+------------------+----------------+------------+
```

The complete CUDA suite passes 400 tests plus two expected one-visible-device skips, including W1--W8 FP16/BF16,
M=1/2/4/8/16/32, structured and extreme trellises, dense-reference error/KLD/top-k gates, deterministic replay,
non-default streams, and concurrent W1 Viterbi callers on two physical GPUs with `PYTHON_GIL=0`.

## 2026-08-11 DeepSeek V4 Flash split-K progression

The production checkpoint at `/monster/data/model/DeepSeek-V4-Flash-0731-BF16-Defused` was inspected through its
safetensors headers. Its attention and MoE projection shapes, rather than synthetic square matrices, now form a named
benchmark set in `scripts/benchmark_qvq_cuda.py`. Narrow projections use deterministic split-K: each block writes an
FP32 partial to a transient PyTorch tensor and a second kernel reduces the splits in a fixed order. There are no
floating-point atomics and no persistent dequantized-weight or workspace cache. Projections with at least 384 base
blocks remain on the pre-existing path; `q_b_proj` and `o_a_proj` are therefore controls, not claimed split-K wins.

The benchmark now reports and gates MSE, forward KLD, and top-1 agreement in addition to the existing dense-reference
checks. Defaults reject `MSE > 1e-3`, `KLD > 2e-4`, or `top-1 < 0.96875`. This run used Python 3.14.6 free-threaded,
PyTorch 2.13.0+cu130, CUDA 13.0, W4 FP16, and one physical PG506-230 `sm_80` GPU. The full W2--W8 FP16/BF16 CUDA test
matrix passed 157 tests on GPU 6, and the independent two-device free-threaded test passed on GPUs 6 and 7. Across all
48 real-shape W4 cases (including controls), maximum MSE was `2.39644e-05`, maximum KLD was `1.92373e-07`, and minimum
top-1 was `1.0000`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PYTHON_GIL=0 TORCH_CUDA_ARCH_LIST=8.0 \
  python scripts/benchmark_qvq_cuda.py --physical-gpu 7 \
  --shape-set deepseek-v4-flash-0731-all --bits 4 --m 1 2 4 8 16 32 \
  --iterations 100 --reference-iterations 3
```

```text
+----------------+------+-------+----+---------+------------+-------+-----------+------------+--------+
| Shape          | K    | N     | M  | Base ms | Split-K ms | Gain  | MSE       | Fwd KLD    | Top-1  |
+----------------+------+-------+----+---------+------------+-------+-----------+------------+--------+
| q_a_proj       | 4096 | 1024  | 1  | 0.2253  | 0.0737     | 3.06x | 2.385e-07 | 2.381e-08  | 1.0000 |
| q_a_proj       | 4096 | 1024  | 2  | 0.2181  | 0.0707     | 3.08x | 1.602e-07 | 1.401e-45  | 1.0000 |
| q_a_proj       | 4096 | 1024  | 4  | 0.2181  | 0.0686     | 3.18x | 2.519e-06 | 1.401e-45  | 1.0000 |
| q_a_proj       | 4096 | 1024  | 8  | 0.2191  | 0.0686     | 3.19x | 3.765e-07 | -5.656e-09 | 1.0000 |
| q_a_proj       | 4096 | 1024  | 16 | 0.2314  | 0.0727     | 3.18x | 1.569e-06 | 3.458e-09  | 1.0000 |
| q_a_proj       | 4096 | 1024  | 32 | 0.2806  | 0.0717     | 3.91x | 1.222e-06 | -9.846e-10 | 1.0000 |
| kv_proj        | 4096 | 512   | 1  | 0.4900  | 0.0594     | 8.25x | 0.000e+00 | 0.000e+00  | 1.0000 |
| kv_proj        | 4096 | 512   | 2  | 0.4685  | 0.0584     | 8.02x | 2.046e-12 | 1.490e-08  | 1.0000 |
| kv_proj        | 4096 | 512   | 4  | 0.4639  | 0.0584     | 7.94x | 1.863e-07 | -6.685e-10 | 1.0000 |
| kv_proj        | 4096 | 512   | 8  | 0.4900  | 0.0594     | 8.25x | 1.649e-07 | 4.617e-09  | 1.0000 |
| kv_proj        | 4096 | 512   | 16 | 0.4838  | 0.0635     | 7.62x | 3.481e-07 | -3.262e-09 | 1.0000 |
| kv_proj        | 4096 | 512   | 32 | 0.5253  | 0.0645     | 8.14x | 4.624e-07 | 2.716e-09  | 1.0000 |
| o_b_proj       | 8192 | 4096  | 1  | 0.4598  | 0.3144     | 1.46x | 6.042e-06 | -1.692e-10 | 1.0000 |
| o_b_proj       | 8192 | 4096  | 2  | 0.4454  | 0.3123     | 1.43x | 1.329e-06 | -1.066e-14 | 1.0000 |
| o_b_proj       | 8192 | 4096  | 4  | 0.4495  | 0.3062     | 1.47x | 1.381e-06 | 6.919e-09  | 1.0000 |
| o_b_proj       | 8192 | 4096  | 8  | 0.4608  | 0.3277     | 1.41x | 3.849e-06 | 1.244e-09  | 1.0000 |
| o_b_proj       | 8192 | 4096  | 16 | 0.4905  | 0.3809     | 1.29x | 3.740e-06 | -2.209e-09 | 1.0000 |
| o_b_proj       | 8192 | 4096  | 32 | 0.5734  | 0.4106     | 1.40x | 2.396e-05 | -1.367e-09 | 1.0000 |
| expert_gate_up | 4096 | 2048  | 1  | 0.2253  | 0.0891     | 2.53x | 2.414e-06 | -1.401e-45 | 1.0000 |
| expert_gate_up | 4096 | 2048  | 2  | 0.2181  | 0.0860     | 2.54x | 3.212e-07 | -1.234e-08 | 1.0000 |
| expert_gate_up | 4096 | 2048  | 4  | 0.2191  | 0.0891     | 2.46x | 1.362e-06 | 4.657e-09  | 1.0000 |
| expert_gate_up | 4096 | 2048  | 8  | 0.2243  | 0.0932     | 2.41x | 2.378e-07 | 3.900e-09  | 1.0000 |
| expert_gate_up | 4096 | 2048  | 16 | 0.2386  | 0.1055     | 2.26x | 1.216e-06 | 3.688e-09  | 1.0000 |
| expert_gate_up | 4096 | 2048  | 32 | 0.2816  | 0.1075     | 2.62x | 3.401e-06 | 1.555e-09  | 1.0000 |
| expert_down    | 2048 | 4096  | 1  | 0.1198  | 0.0891     | 1.34x | 2.536e-07 | 1.820e-08  | 1.0000 |
| expert_down    | 2048 | 4096  | 2  | 0.1167  | 0.0860     | 1.36x | 6.979e-07 | -1.205e-08 | 1.0000 |
| expert_down    | 2048 | 4096  | 4  | 0.4398  | 0.0881     | 4.99x | 3.364e-07 | -1.101e-10 | 1.0000 |
| expert_down    | 2048 | 4096  | 8  | 0.4444  | 0.0932     | 4.77x | 4.291e-07 | -7.714e-09 | 1.0000 |
| expert_down    | 2048 | 4096  | 16 | 0.4475  | 0.1044     | 4.29x | 4.971e-07 | 1.080e-09  | 1.0000 |
| expert_down    | 2048 | 4096  | 32 | 0.4521  | 0.1055     | 4.29x | 1.644e-06 | 3.559e-09  | 1.0000 |
| moe_router     | 4096 | 256   | 1  | 0.4710  | 0.0604     | 7.80x | 0.000e+00 | 5.605e-45  | 1.0000 |
| moe_router     | 4096 | 256   | 2  | 0.4618  | 0.0594     | 7.77x | 0.000e+00 | 2.803e-45  | 1.0000 |
| moe_router     | 4096 | 256   | 4  | 0.4582  | 0.0584     | 7.85x | 5.961e-08 | -1.393e-11 | 1.0000 |
| moe_router     | 4096 | 256   | 8  | 0.4598  | 0.0778     | 5.91x | 1.193e-07 | -3.984e-09 | 1.0000 |
| moe_router     | 4096 | 256   | 16 | 0.5125  | 0.0722     | 7.10x | 1.230e-07 | -1.155e-09 | 1.0000 |
| moe_router     | 4096 | 256   | 32 | 0.5356  | 0.0707     | 7.58x | 9.015e-07 | -1.572e-10 | 1.0000 |
+----------------+------+-------+----+---------+------------+-------+-----------+------------+--------+
```

## 2026-08-11 PGC16 CUDA validation

Environment: Python 3.14.6 free-threaded (`PYTHON_GIL=0`), PyTorch 2.13.0+cu130, CUDA 13.0, FP16, K=N=4096,
`TORCH_CUDA_ARCH_LIST=8.0`. Physical GPUs 6 and 7 were independently idle-gated and each benchmark process saw exactly
one PG506-230 (`sm_80`, 124 SM, 96 GiB). Timings are CUDA-event medians after 20 warmups and 100 candidate samples; the
transient baseline reconstructs the dense PGC16 weight inside every call. Tests compare against an independently
decoded FP32 dense matmul and report the output after conversion to FP16.

```text
+---+----+--------+---------+---------+-------------+-------------+-----------+-------------+--------------+--------+--------+
| W | M  | Ref ms | CUDA ms | Speedup | MAE         | RMSE        | Max abs   | Rel-L2      | Fwd KLD      | Top-1  | Top-5  |
+---+----+--------+---------+---------+-------------+-------------+-----------+-------------+--------------+--------+--------+
| 2 | 1  | 7.7261 | 0.2693  | 28.69x  | 1.99154e-05 | 0.000991994 | 0.0625    | 1.5626e-05  | 4.2039e-45   | 1.0000 | 1.0000 |
| 2 | 2  | 7.7384 | 0.2632  | 29.40x  | 6.03637e-05 | 0.00149614  | 0.0625    | 2.36062e-05 | -5.39876e-09 | 1.0000 | 1.0000 |
| 2 | 4  | 7.7435 | 0.2652  | 29.20x  | 3.99319e-05 | 0.00132006  | 0.0625    | 2.04147e-05 | 3.6257e-09   | 1.0000 | 1.0000 |
| 2 | 8  | 7.7394 | 0.2693  | 28.74x  | 4.16827e-05 | 0.00123597  | 0.0625    | 1.9168e-05  | 1.85945e-09  | 1.0000 | 1.0000 |
| 2 | 16 | 7.7384 | 0.2918  | 26.52x  | 5.33697e-05 | 0.00168187  | 0.125     | 2.62389e-05 | 1.34893e-09  | 1.0000 | 1.0000 |
| 2 | 32 | 7.7343 | 0.3195  | 24.21x  | 0.000310977 | 0.00394636  | 0.125     | 6.19313e-05 | 2.07095e-09  | 1.0000 | 1.0000 |
| 3 | 1  | 8.2719 | 0.2806  | 29.48x  | 7.39656e-05 | 0.00241992  | 0.125     | 3.76899e-05 | 1.44355e-08  | 1.0000 | 1.0000 |
| 3 | 2  | 8.2637 | 0.2734  | 30.22x  | 2.75979e-05 | 0.000945487 | 0.0625    | 1.49183e-05 | 1.11759e-08  | 1.0000 | 1.0000 |
| 3 | 4  | 8.2678 | 0.2714  | 30.47x  | 5.1153e-05  | 0.00166679  | 0.125     | 2.59903e-05 | 4.65661e-09  | 1.0000 | 1.0000 |
| 3 | 8  | 8.2698 | 0.2775  | 29.80x  | 4.7018e-05  | 0.00166946  | 0.125     | 2.61867e-05 | 6.07197e-09  | 1.0000 | 1.0000 |
| 3 | 16 | 8.2790 | 0.3000  | 27.59x  | 5.17489e-05 | 0.00157216  | 0.125     | 2.46349e-05 | -1.36336e-09 | 1.0000 | 1.0000 |
| 3 | 32 | 8.2637 | 0.3195  | 25.87x  | 0.000332152 | 0.00421706  | 0.125     | 6.63591e-05 | 3.08879e-08  | 1.0000 | 1.0000 |
| 4 | 1  | 7.7568 | 0.2345  | 33.08x  | 6.54334e-05 | 0.00157864  | 0.0625    | 2.49125e-05 | 2.8026e-45   | 1.0000 | 1.0000 |
| 4 | 2  | 7.7476 | 0.2263  | 34.24x  | 5.06146e-05 | 0.00172451  | 0.125     | 2.70277e-05 | 7.32144e-08  | 1.0000 | 1.0000 |
| 4 | 4  | 7.7496 | 0.2284  | 33.94x  | 5.65118e-05 | 0.00148873  | 0.0625    | 2.3092e-05  | 7.21775e-09  | 1.0000 | 1.0000 |
| 4 | 8  | 7.7558 | 0.2345  | 33.07x  | 7.44921e-05 | 0.00202744  | 0.125     | 3.19168e-05 | 1.62208e-09  | 1.0000 | 1.0000 |
| 4 | 16 | 7.7476 | 0.2488  | 31.14x  | 4.93295e-05 | 0.00141833  | 0.125     | 2.22231e-05 | 1.58905e-09  | 1.0000 | 1.0000 |
| 4 | 32 | 7.7517 | 0.2908  | 26.65x  | 0.000354595 | 0.0043969   | 0.125     | 6.9162e-05  | 3.34634e-09  | 1.0000 | 1.0000 |
| 5 | 1  | 8.2852 | 0.2488  | 33.30x  | 4.28641e-05 | 0.00198872  | 0.125     | 3.11808e-05 | -4.31183e-14 | 1.0000 | 1.0000 |
| 5 | 2  | 8.2852 | 0.2437  | 34.00x  | 4.09205e-05 | 0.00165506  | 0.125     | 2.54567e-05 | -1.06866e-11 | 1.0000 | 1.0000 |
| 5 | 4  | 8.2852 | 0.2499  | 33.16x  | 3.76345e-05 | 0.00112921  | 0.0625    | 1.77216e-05 | -1.75205e-08 | 1.0000 | 1.0000 |
| 5 | 8  | 8.2893 | 0.2560  | 32.38x  | 6.11666e-05 | 0.00172062  | 0.125     | 2.70588e-05 | 5.58761e-09  | 1.0000 | 1.0000 |
| 5 | 16 | 8.2893 | 0.2785  | 29.76x  | 5.29797e-05 | 0.0015412   | 0.125     | 2.43073e-05 | 2.7937e-09   | 1.0000 | 1.0000 |
| 5 | 32 | 8.3077 | 0.3205  | 25.92x  | 0.000331453 | 0.00412898  | 0.125     | 6.47399e-05 | 2.84322e-09  | 1.0000 | 1.0000 |
| 6 | 1  | 8.3292 | 0.2284  | 36.48x  | 5.71087e-06 | 0.000178379 | 0.0078125 | 2.78579e-06 | 7.00649e-45  | 1.0000 | 1.0000 |
| 6 | 2  | 8.3282 | 0.2222  | 37.48x  | 5.65862e-05 | 0.00177086  | 0.125     | 2.73449e-05 | 5.58794e-09  | 1.0000 | 1.0000 |
| 6 | 4  | 8.3077 | 0.2314  | 35.90x  | 4.76592e-05 | 0.00157019  | 0.125     | 2.48397e-05 | -7.39255e-09 | 1.0000 | 1.0000 |
| 6 | 8  | 8.3139 | 0.2345  | 35.45x  | 5.22381e-05 | 0.00140086  | 0.0625    | 2.19636e-05 | -2.81943e-11 | 1.0000 | 1.0000 |
| 6 | 16 | 8.3077 | 0.2509  | 33.11x  | 5.05856e-05 | 0.00147796  | 0.125     | 2.32471e-05 | 7.592e-10    | 1.0000 | 1.0000 |
| 6 | 32 | 8.3067 | 0.3082  | 26.95x  | 0.00032556  | 0.0040259   | 0.125     | 6.3049e-05  | -1.11611e-09 | 1.0000 | 1.0000 |
| 7 | 1  | 8.8556 | 0.2570  | 34.45x  | 5.61327e-05 | 0.00153475  | 0.0625    | 2.4633e-05  | 7.00649e-45  | 1.0000 | 1.0000 |
| 7 | 2  | 8.8525 | 0.2519  | 35.14x  | 3.53046e-05 | 0.00107636  | 0.0625    | 1.69156e-05 | -1.32604e-09 | 1.0000 | 1.0000 |
| 7 | 4  | 8.8556 | 0.2591  | 34.18x  | 5.71515e-05 | 0.00147874  | 0.0625    | 2.32318e-05 | 2.8026e-45   | 1.0000 | 1.0000 |
| 7 | 8  | 8.8556 | 0.2642  | 33.52x  | 6.23835e-05 | 0.00175181  | 0.125     | 2.73595e-05 | -6.63931e-10 | 1.0000 | 1.0000 |
| 7 | 16 | 8.8576 | 0.2908  | 30.46x  | 4.21021e-05 | 0.00131887  | 0.125     | 2.06992e-05 | -5.50053e-09 | 1.0000 | 1.0000 |
| 7 | 32 | 8.8535 | 0.3011  | 29.41x  | 0.000320773 | 0.00401947  | 0.125     | 6.27476e-05 | 1.46895e-09  | 1.0000 | 1.0000 |
| 8 | 1  | 7.7957 | 0.2202  | 35.41x  | 8.24183e-05 | 0.00249726  | 0.125     | 3.83549e-05 | -2.13163e-14 | 1.0000 | 1.0000 |
| 8 | 2  | 7.8060 | 0.2130  | 36.65x  | 6.46617e-05 | 0.00204226  | 0.125     | 3.18573e-05 | 5.60519e-45  | 1.0000 | 1.0000 |
| 8 | 4  | 7.8029 | 0.2171  | 35.94x  | 6.52042e-05 | 0.00181621  | 0.125     | 2.84399e-05 | -4.44561e-09 | 1.0000 | 1.0000 |
| 8 | 8  | 7.8008 | 0.2232  | 34.94x  | 5.06441e-05 | 0.00153648  | 0.125     | 2.38774e-05 | 6.71434e-10  | 1.0000 | 1.0000 |
| 8 | 16 | 7.7957 | 0.2396  | 32.53x  | 4.58069e-05 | 0.00140963  | 0.125     | 2.2213e-05  | 1.91389e-10  | 1.0000 | 1.0000 |
| 8 | 32 | 7.7957 | 0.2816  | 27.68x  | 0.000325428 | 0.0041307   | 0.125     | 6.48877e-05 | 6.27428e-09  | 1.0000 | 1.0000 |
+---+----+--------+---------+---------+-------------+-------------+-----------+-------------+--------------+--------+--------+
```

The PGC16 port initially rounded the fixed FP16 levels to BF16 for BF16 input and failed 23 of 42 dense-reference
cases. The corrected ABI always passes the canonical table as FP16. BF16 uses FP32 scalar FMA accumulation instead of
BF16 WMMA because BF16 tensor-core operands would change those format-defined levels. After the correction, all 84
W2--W8/M1--32 FP16 and BF16 cases pass, along with structured all-zero, all-one, and alternating-bit trellises,
accumulation/tail, deterministic-launch, non-default-stream, wrapper, and contract tests.

The historical tables below predate the PGC16 production-format decision: they measured the retained HYB-Q9 reference
decoder and must not be attributed to PGC16. They remain a useful historical regression and performance baseline. The
fresh NVIDIA PGC16 measurements and validation results above supersede them for the production format.

## 2026-08-11 historical HYB-Q9 correctness-first kernel

Environment: Python 3.14.6 free-threaded (`PYTHON_GIL=0`), PyTorch 2.13.0+cu130, CUDA 13.0, FP16, K=N=4096,
`TORCH_CUDA_ARCH_LIST=8.0`. Physical GPUs 6 and 7 were independently idle-gated and each process saw exactly one
PG506-230 (`sm_80`, 124 SM, 96 GiB). Timings are CUDA-event medians after 20 warmups; the candidate and cached dense
rows use 50 measurements and the slow transient reference uses three. The speedup baseline reconstructs the dense
QVQ inner weight inside every inference call. The cached dense ceiling is measured by the script but is intentionally
not a valid runtime design because it permanently defeats weight compression.

Command (split the bit list across physical GPUs 6 and 7 for the recorded parallel run):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<one-gpu-uuid> PYTHON_GIL=0 TORCH_CUDA_ARCH_LIST=8.0 \
  python scripts/benchmark_qvq_cuda.py --physical-gpu <6-or-7> --bits <partition> \
  --m 1 2 4 8 16 32 --k 4096 --n 4096 --iterations 50 --reference-iterations 3
```

```text
+------+----+--------------+--------------+---------+---------+-------------+--------+
| Bits | M  | Reference ms | QVQ CUDA ms | Speedup | Max abs | Fwd KLD     | Top-1  |
+------+----+--------------+--------------+---------+---------+-------------+--------+
| 2    | 1  | 7.1690       | 0.2591       | 27.67x  | 0.03125 | -2.63e-08   | 1.0000 |
| 2    | 2  | 7.2059       | 0.2529       | 28.49x  | 0.06250 | -4.61e-09   | 1.0000 |
| 2    | 4  | 7.1793       | 0.2529       | 28.38x  | 0.06250 |  4.99e-09   | 1.0000 |
| 2    | 8  | 7.1803       | 0.2580       | 27.83x  | 0.06250 | -4.22e-10   | 1.0000 |
| 2    | 16 | 7.1772       | 0.2785       | 25.77x  | 0.12500 |  5.76e-09   | 1.0000 |
| 2    | 32 | 7.1813       | 0.3901       | 18.41x  | 0.12500 | -2.63e-09   | 1.0000 |
| 3    | 1  | 7.7179       | 0.2693       | 28.66x  | 0.12500 |  5.61e-45   | 1.0000 |
| 3    | 2  | 7.7332       | 0.2632       | 29.39x  | 0.12500 |  0.00e+00   | 1.0000 |
| 3    | 4  | 7.7322       | 0.2662       | 29.04x  | 0.06250 |  1.92e-09   | 1.0000 |
| 3    | 8  | 7.7199       | 0.2744       | 28.13x  | 0.06250 | -9.70e-09   | 1.0000 |
| 3    | 16 | 7.7312       | 0.2939       | 26.31x  | 0.12500 | -8.55e-10   | 1.0000 |
| 3    | 32 | 7.7240       | 0.3973       | 19.44x  | 0.12500 | -2.65e-09   | 1.0000 |
| 4    | 1  | 7.1875       | 0.2314       | 31.06x  | 0.06250 |  2.24e-08   | 1.0000 |
| 4    | 2  | 7.1885       | 0.2253       | 31.91x  | 0.12500 | -4.64e-09   | 1.0000 |
| 4    | 4  | 7.1844       | 0.2273       | 31.60x  | 0.06250 | -1.05e-08   | 1.0000 |
| 4    | 8  | 7.1946       | 0.2335       | 30.82x  | 0.12500 |  1.28e-09   | 1.0000 |
| 4    | 16 | 7.2018       | 0.2478       | 29.06x  | 0.12500 |  3.85e-09   | 1.0000 |
| 4    | 32 | 7.2110       | 0.3523       | 20.47x  | 0.12500 | -1.95e-09   | 1.0000 |
| 5    | 1  | 10.1960      | 0.2499       | 40.81x  | 0.06250 |  0.00e+00   | 1.0000 |
| 5    | 2  | 7.7435       | 0.2447       | 31.64x  | 0.06250 | -2.27e-08   | 1.0000 |
| 5    | 4  | 7.7599       | 0.2509       | 30.93x  | 0.06250 |  1.12e-08   | 1.0000 |
| 5    | 8  | 7.7558       | 0.2580       | 30.06x  | 0.12500 |  1.77e-09   | 1.0000 |
| 5    | 16 | 7.7691       | 0.2796       | 27.79x  | 0.12500 |  1.84e-09   | 1.0000 |
| 5    | 32 | 7.7783       | 0.3922       | 19.83x  | 0.12500 |  2.94e-07   | 1.0000 |
| 6    | 1  | 7.7650       | 0.2273       | 34.16x  | 0.12500 |  4.07e-06   | 1.0000 |
| 6    | 2  | 7.7609       | 0.2212       | 35.09x  | 0.06250 | -5.57e-12   | 1.0000 |
| 6    | 4  | 7.7527       | 0.2273       | 34.10x  | 0.12500 | -1.13e-11   | 1.0000 |
| 6    | 8  | 7.7691       | 0.2324       | 33.42x  | 0.06250 |  3.40e-09   | 1.0000 |
| 6    | 16 | 7.7558       | 0.2468       | 31.43x  | 0.12500 |  2.18e-09   | 1.0000 |
| 6    | 32 | 7.7599       | 0.3574       | 21.71x  | 0.12500 | -2.26e-09   | 1.0000 |
| 7    | 1  | 8.3200       | 0.2540       | 32.76x  | 0.06250 | -5.62e-09   | 1.0000 |
| 7    | 2  | 8.3333       | 0.2478       | 33.63x  | 0.12500 | -1.82e-12   | 1.0000 |
| 7    | 4  | 8.3036       | 0.2560       | 32.44x  | 0.12500 | -2.38e-09   | 1.0000 |
| 7    | 8  | 8.3169       | 0.2611       | 31.85x  | 0.06250 |  1.52e-08   | 1.0000 |
| 7    | 16 | 8.2995       | 0.2980       | 27.85x  | 0.12500 | -2.04e-10   | 1.0000 |
| 7    | 32 | 8.2975       | 0.4393       | 18.89x  | 0.12500 | -9.31e-10   | 1.0000 |
| 8    | 1  | 7.2397       | 0.2304       | 31.42x  | 0.03125 | -1.43e-09   | 1.0000 |
| 8    | 2  | 7.2448       | 0.2232       | 32.45x  | 0.06250 | -5.20e-38   | 1.0000 |
| 8    | 4  | 7.2530       | 0.2284       | 31.76x  | 0.06250 |  9.29e-10   | 1.0000 |
| 8    | 8  | 7.2448       | 0.2355       | 30.76x  | 0.12500 | -5.59e-09   | 1.0000 |
| 8    | 16 | 7.2428       | 0.2499       | 28.99x  | 0.12500 |  7.11e-07   | 1.0000 |
| 8    | 32 | 7.2458       | 0.3753       | 19.31x  | 0.12500 |  1.30e-09   | 1.0000 |
+------+----+--------------+--------------+---------+---------+-------------+--------+
```

The tiny negative KLD values are floating-point roundoff around zero. Unit tests also cover FP16 and BF16 across the
same M matrix, a K=256 accumulation/tail shape at M=33 and N=80, five repeated deterministic launches, a non-default
CUDA stream, invalid layouts/dtypes/devices, and an explicit pre-launch architecture rejection.

## 2026-08-11 historical HYB-Q9 M32 Ampere WMMA progression

At M32, the decoded 16x16 tile is now reused by two Ampere WMMA warps with FP32 accumulators. M16 remains on the
scalar-FP32 kernel because W2 and W6 regressed under WMMA there. This is a measured `M >= 32` gate, not a device-name
heuristic. All W2--W8 FP16/BF16 dense-reference, KLD, top-1, deterministic, non-default-stream, M33 tail, and contract
tests pass after the change.

```text
+------+----+------------------+---------+---------+-------------+--------+
| Bits | M  | Scalar CUDA ms   | WMMA ms | Speedup | WMMA KLD    | Top-1  |
+------+----+------------------+---------+---------+-------------+--------+
| 2    | 32 | 0.3901           | 0.3226  | 1.21x   |  3.45e-09   | 1.0000 |
| 3    | 32 | 0.3973           | 0.3164  | 1.26x   | -3.17e-09   | 1.0000 |
| 4    | 32 | 0.3523           | 0.2857  | 1.23x   |  3.90e-09   | 1.0000 |
| 5    | 32 | 0.3922           | 0.3195  | 1.23x   |  2.97e-07   | 1.0000 |
| 6    | 32 | 0.3574           | 0.3062  | 1.17x   | -4.82e-11   | 1.0000 |
| 7    | 32 | 0.4393           | 0.2970  | 1.48x   |  1.15e-09   | 1.0000 |
| 8    | 32 | 0.3753           | 0.2775  | 1.35x   |  3.93e-09   | 1.0000 |
+------+----+------------------+---------+---------+-------------+--------+
```
