# Prism Ternary-Bonsai Q2_0 kernel log

## 2026-07-21: canonical Q2_0/PQ2_0 enablement and sm80 decode optimization

### Target

- Checkpoint: `/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf`
- Model: Prism ML/AI Ternary-Bonsai 1.7B, Qwen3 architecture.
- Quantization: ternary `{-1, 0, +1}` weights, approximately 1.58 logical bits per weight.
- Canonical storage: 128 weights in a 34-byte Q2_0 block: one FP16 scale and 32 packed-code bytes,
  or 2.125 stored bits per weight.
- `PQ2_0` uses the same payload and execution path.
- The local `Q2_0_g64` artifact reuses type 42 with incompatible 64-value/18-byte blocks and is rejected
  before tensor materialization.

### Environment

The profiling target was selected at runtime. Its physical PCI bus ID was `00000000:25:00.0`; implementation
decisions use the queried compute capability and never infer hardware from a fixed CUDA index.

```text
+-------------------------+-------------------------------------------------------+
| Item                    | Value                                                 |
+-------------------------+-------------------------------------------------------+
| GPU                     | NVIDIA PG506-230, sm80, 124 SMs, 98,304 MiB           |
| Host inventory          | 8 sm80 PG506-230/232 GPUs, 98,304 MiB each            |
| Driver                  | 610.43.02                                             |
| Python                  | 3.14.5, GIL enabled for measured runs                 |
| PyTorch / CUDA runtime  | 2.12.0+cu130 / 13.0                                  |
| nvcc                    | 13.0.88, cuda_13.0.r13.0/compiler.36424714_0           |
| Triton                  | 3.7.0                                                 |
| Nsight Systems          | 2024.6.2                                              |
| Nsight Compute          | 2025.3.1                                              |
| Kernel build            | Triton JIT, two warps, two stages; no AOT build flags |
+-------------------------+-------------------------------------------------------+
```

### Implementation

- Added exact NumPy and Torch Q2_0/PQ2_0 packing and dequantization.
- Added Q2_0/PQ2_0 backend selection and explicit incompatible-storage validation.
- Added a portable transposed-u32 Triton tensor-core path for multi-row/prefill workloads.
- Added an sm80-only output-major GEMV for batch-one decode on the four matrix shapes used by Bonsai 1.7B.
- The native decode kernel reads packed GGUF codes directly and retains only a transposed 21 MiB scale cache.
- Added `release_q2_prefill_cache()` to discard the 357 MiB prefill code cache after prefill. A later prefill
  rebuilds that cache.
- Preserved the Torch path and generic Triton fallback for unsupported shapes and compute capabilities.

### Correctness

- Sampled first, middle, and final rows from all 197 Q2 tensors against the F16 checkpoint: 1,554,432 values,
  zero MAE, zero maximum error, and 100% exact matches.
- Compared the optimized kernel against dense FP16-dequantized matmul using checkpoint tensors across the four
  representative shapes. Observed MAE was at most `0.000617`; maximum absolute error was at most `0.0078125`.
- Q2_0 and PQ2_0 smoke tests passed on sm80.
- End-to-end loading and generation completed with 196 quantized linear modules.

### CUDA-event results

Batch size was one. Prefill used 64 tokens; decode used one token with a growing KV cache. Direct runs used three
warmups and 20 measured iterations.

```text
+------------------------------------------+---------+--------+----------+----------+------------------+
| Path                                     | Stage   | Tokens | p50 ms   | p95 ms   | Throughput p50   |
+------------------------------------------+---------+--------+----------+----------+------------------+
| Baseline transposed tensor-core          | Prefill |     64 | 47.6800  | 52.8332  | 1342.28 tok/s    |
| Optimized hybrid                         | Prefill |     64 | 49.0358  | 52.0288  | 1305.17 tok/s    |
| Baseline transposed tensor-core          | Decode  |      1 | 45.9602  | 53.0489  |   21.76 tok/s    |
| Optimized native two-warp GEMV           | Decode  |      1 | 40.9078  | 44.4379  |   24.45 tok/s    |
| Native GEMV plus released prefill cache  | Decode  |      1 | 43.3196  | 44.1717  |   23.08 tok/s    |
+------------------------------------------+---------+--------+----------+----------+------------------+
```

The matched direct comparison shows 10.99% lower decode p50 and 12.35% higher p50 throughput. Prefill remains on
the tensor-core path; its difference is within observed run-to-run variance.

### Nsight Systems findings

The baseline trace contains one prefill and four decode steps. The optimized trace contains four low-VRAM decode
steps. Values below came from `nsys stats` and SQLite queries over the captured `prism_q2_0_decode` NVTX ranges.

```text
+-----------------------------------+------------+------------+------------------+
| Four-token decode metric          | Baseline   | Optimized  | Change           |
+-----------------------------------+------------+------------+------------------+
| NVTX range wall time              | 277.796 ms | 232.348 ms | -16.36%          |
| Q2 kernel GPU time                |  52.314 ms |  11.381 ms | -78.24%          |
| Q2 launches                       |        784 |        784 | unchanged        |
| Q2 median kernel duration         |  52.416 us |  10.112 us | 5.18x faster     |
| Steady-state H2D/D2H copies       |          0 |          0 | unchanged        |
+-----------------------------------+------------+------------+------------------+
```

After the change, each decode still launches approximately 1,591 GPU operations: 196 Q2 kernels and roughly 1,395
attention, normalization, elementwise, cache, and vocabulary-head operations. GPU-in-use was 13.6% in the optimized
decode range. The next investigation therefore targets host launch spacing, small-kernel fusion opportunities, and
static-cache CUDA graph feasibility rather than further arithmetic work inside the Q2 kernel alone.

### Nsight Compute findings

Representative matrix: `blk.0.attn_q.weight`, shape `2048 x 2048`, FP16 activation, one row.

```text
+----------------------+--------------------------+-----------------------+
| Metric               | Baseline tensor-core     | Native two-warp GEMV  |
+----------------------+--------------------------+-----------------------+
| Duration             | 55.46 us                 | 10.82 us              |
| Grid / block         | 64 / 128 threads         | 2048 / 64 threads     |
| Waves per SM         | 0.05                     | 0.52                  |
| Registers per thread | 48                       | 32                    |
| Achieved occupancy   | 6.27%                    | 40.84%                |
| SM throughput        | 4.44%                    | 36.51%                |
| DRAM throughput      | 0.83%                    | 4.54%                 |
+----------------------+--------------------------+-----------------------+
```

The selected output-major kernel launches one program per output and directly unpacks each GGUF row. A deeper
capture measured about 109 GB/s, approximately 69% L2 hit rate, no local-memory spills, and long-scoreboard stalls
as the dominant residual kernel stall. One-warp, four-warp, explicit-FMA, and transposed-u32 decode variants were
measured and rejected.

### VRAM

```text
+------------------------------------------+---------------+-------------+------------+---------------+
| State                                    | Allocated MiB | Qweight MiB | Cache MiB  | Cold peak MiB |
+------------------------------------------+---------------+-------------+------------+---------------+
| Loaded                                   |        963.79 |      357.00 |       0.00 |       3121.10 |
| Warmed with prefill cache                |       1357.51 |      357.00 |     357.00 |       3121.10 |
| Warmed after release_q2_prefill_cache()  |       1020.73 |      357.00 |      21.00 |       3121.10 |
+------------------------------------------+---------------+-------------+------------+---------------+
```

Cache release saves 336.78 MiB, or 24.81% of warmed allocated memory, and removes 94.12% of the Q2 launch cache.
It does not reduce the cold load peak.

### Reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
python scripts/profile_prism_q2_0.py \
  --mode model --device 0 --prompt-tokens 64 --warmup 3 --iterations 20

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/optimized_decode_low_vram \
  python scripts/profile_prism_q2_0.py \
    --mode model --device 0 --prompt-tokens 64 --warmup 3 --iterations 10 \
    --release-prefill-cache --capture --profile-prefill 0 --profile-decode 4

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --kernel-name regex:_gguf_q2_0_native_gemv_kernel_impl --launch-count 3 \
  --export artifacts/prism_q2_0_20260721/candidate_native_w2_attn_q_decode \
  --force-overwrite \
  python scripts/profile_prism_q2_0.py \
    --mode kernel --device 0 --tensor-name blk.0.attn_q.weight \
    --rows 1 --warmup 20 --iterations 50 --capture --profile-kernel 3
```

Raw local artifacts:

- `artifacts/prism_q2_0_20260721/baseline_model.nsys-rep`
- `artifacts/prism_q2_0_20260721/optimized_decode_low_vram.nsys-rep`
- `artifacts/prism_q2_0_20260721/baseline_attn_q_decode.ncu-rep`
- `artifacts/prism_q2_0_20260721/candidate_native_w2_attn_q_decode.ncu-rep`
- `artifacts/prism_q2_0_20260721/candidate_native_w2_deep.ncu-rep`

Binary profiler reports are intentionally kept out of Git history.

## 2026-07-21 follow-up: fused RMSNorm and CUDA Graph feasibility

The first implementation was committed as `a19d696c` (`feat: optimize Prism Q2 GGUF decode`). This follow-up used
that revision as its baseline and kept the same checkpoint, selected physical device, dtype, software stack, prompt,
and decode regime.

### Residual launch attribution

`nsys stats` over the first optimized low-VRAM decode showed 6,360 kernel launch APIs across four tokens: 5,576
`cudaLaunchKernel` calls and 784 Triton `cuLaunchKernelEx` calls. The trace contained 113 decomposed RMSNorms per
token: 56 layer norms, 56 Q/K head norms, and the final model norm. Each norm launched separate convert, square,
mean, epsilon/add, reciprocal-square-root, multiply, and convert work.

The next largest removable group was dynamic KV-cache concatenation, with 112 copy kernels per token. A static
cache replaced concatenation but was slower in eager execution because its position updates and indexed stores were
still launched individually. Static cache is useful only when those launches are captured as a graph.

### Fused RMSNorm

A single Triton kernel now performs FP32 variance accumulation, normalization, weight multiplication, and output
conversion. Installation is restricted to native Prism Q2_0/PQ2_0 models using `GGUFTritonKernel`, Qwen3 RMSNorm
modules resident on a live sm80 CUDA device, FP16/BF16 inputs, and the model's 128- or 2048-element norm widths.
The original Torch expression remains the fallback for every other device, dtype, width, and backend.

The full model installed 113 fused norms. Against the unfused model, final-logit MAE was `0.00130442`, maximum
absolute error was `0.00976562`, and greedy argmax was identical. The focused sm80 test observed at most `0.00390625`
absolute RMSNorm output error.

```text
+--------------------------------------+-----------+-----------+-----------+----------------+
| CUDA-event path                      | p50 ms    | p95 ms    | tok/s     | Change at p50  |
+--------------------------------------+-----------+-----------+-----------+----------------+
| Original Q2 tensor-core decode       | 45.9602   | 53.0489   |  21.76    | baseline       |
| Native Q2 GEMV decode                | 40.9078   | 44.4379   |  24.45    | -10.99%        |
| Native Q2 GEMV plus fused RMSNorm    | 34.6775   | 35.4625   |  28.84    | -24.55% total  |
| Static cache plus fused RMSNorm      | 38.5148   | 39.0997   |  25.96    | rejected       |
+--------------------------------------+-----------+-----------+-----------+----------------+
```

The production RMSNorm path improves the already optimized decode by another 15.23%. The matched 64-token prefill
improved from `49.0358` to `40.2994` ms p50, or 17.82%.

The matched Systems capture reduced the four-token decode NVTX range from `232.348` to `178.509` ms under profiler
overhead, a further 23.17% reduction. Launches fell from approximately 1,591 to 800 GPU operations per token, a
49.72% reduction. The fused RMSNorm itself accounted for 452 launches over four tokens, `1.337373` ms total GPU
time, and `2.880` us median duration. It replaced several launches per norm.

```text
+----------------------------------+--------------------+--------------------+
| Four-token Systems metric        | Native Q2 GEMV     | + fused RMSNorm    |
+----------------------------------+--------------------+--------------------+
| Decode NVTX wall time            | 232.348 ms         | 178.509 ms         |
| GPU operations per token         | about 1,591        | 800                |
| cudaLaunchKernel calls           | 5,576              | 1,960              |
| cuLaunchKernelEx calls           | 784                | 1,236              |
| GPU in-use                       | 13.6%               | 12.4%              |
+----------------------------------+--------------------+--------------------+
```

GPU-in-use fell because the fusion removed work faster than it removed the remaining host gaps. This confirms that
the eager path remains host-launch-bound rather than indicating a GPU regression.

### Whole-decode CUDA Graph experiment

A fixed batch-one `StaticCache` experiment captures one complete decode step, including the 196 Q2 projections,
fused norms, FlashAttention, vocabulary head, greedy argmax, token feedback, cache updates, and position increment.
The graph retains all static input, output, cache, and position tensors for its lifetime. Dropping those Python
references invalidates the captured addresses and must not be allowed by a production wrapper.

After two eager warmups and 20 measured graph replays, the final token matched 22 eager static-cache decode steps
exactly. A further 20 profiled replays advanced the cache from length 86 to 106 without error.

```text
+--------------------------------------+-----------+-----------+-----------+--------------------+
| Path                                 | p50 ms    | p95 ms    | tok/s     | vs fused eager     |
+--------------------------------------+-----------+-----------+-----------+--------------------+
| Dynamic cache plus fused RMSNorm     | 34.6775   | 35.4625   |  28.84    | baseline           |
| Static-cache CUDA Graph replay       |  5.8540   |  5.8859   | 170.82    | 5.92x throughput   |
+--------------------------------------+-----------+-----------+-----------+--------------------+
```

Nsight recorded 20 `cudaGraphLaunch` calls instead of thousands of per-operation host launches. The profiler's
single synchronization covering those replays took `110.847420` ms; CUDA events outside the active profiler range
provide the lower-overhead latency numbers above.

The graph candidate allocated `1403.44` MiB and peaked at `1418.00` MiB. That is 382.71 MiB more live allocation
than the low-VRAM eager state (`1020.73` MiB). It also has fixed batch, cache capacity, attention implementation,
and tensor-address requirements. EOS checks or per-token streaming would require bounded replay chunks and host
synchronization. For those reasons graph replay is validated as a high-value follow-up direction but is not enabled
by default in the model loader.

### Remaining ranked opportunities

```text
+----------+-----------------------------------------+----------------------+--------------------------------------------+
| Priority | Candidate                               | Launches/token       | Evidence / constraint                      |
+----------+-----------------------------------------+----------------------+--------------------------------------------+
| 1        | Production static-cache graph wrapper   | Captures about 800   | 5.92x measured; fixed-shape and VRAM trade |
| 2        | Fuse gate/up Q2 GEMV plus SwiGLU        | Could remove 84      | Same input and output width in 28 MLPs     |
| 3        | Fuse Q/K/V projection launch            | Could remove 56      | Same input; three distinct output widths   |
| 4        | Dynamic-cache copy reduction            | 112 observed         | Static eager was slower without graphs     |
+----------+-----------------------------------------+----------------------+--------------------------------------------+
```

Gate/up/SwiGLU fusion is the best remaining eager-kernel candidate: each layer currently launches two Q2 GEMVs,
SiLU, and a multiply for tensors with the same input and intermediate width. It needs a matched real-checkpoint
kernel implementation and register/occupancy validation before retention. That candidate was subsequently measured
and promoted in the Torch 2.13 follow-up below.

### Follow-up reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_fused_graph \
  python scripts/profile_prism_q2_graph.py \
    --device 0 --prompt-tokens 64 --warmup 3 --iterations 20 \
    --max-cache-len 256 --graph-warmup 2 --capture \
    --profile-dynamic 4 --profile-graph 20

nsys stats --force-export=true \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_api_sum,cuda_gpu_sum,cuda_kern_exec_sum \
  artifacts/prism_q2_0_20260721/followup_fused_graph.nsys-rep
```

Additional local artifact:

- `artifacts/prism_q2_0_20260721/followup_fused_graph.nsys-rep`

## 2026-07-21 Torch 2.13 revalidation: fused Q2 SwiGLU

The authoritative target environment was updated while this investigation was running. All results in this section
were therefore rerun with `/root/vm314t`; the Torch 2.12 results above remain useful historical measurements but are
not used as the controlled baseline for the new fusion.

```text
+----------------------+------------------------------------------------------+
| Component            | Torch 2.13 target                                    |
+----------------------+------------------------------------------------------+
| Python               | 3.14.5                                               |
| PyTorch / CUDA       | 2.13.0+cu130 / CUDA runtime 13.0                     |
| Transformers         | 5.7.0.dev0                                           |
| Triton               | 3.7.1                                                |
| Driver               | 610.43.02                                            |
| Nsight               | Systems 2024.6.2; Compute 2025.3.1                  |
| GPU                  | NVIDIA PG506-230, PCI 0000:25:00.0                   |
| Capability / SMs     | sm80 / 124                                           |
| Memory               | 98,304 MiB                                           |
| Workload             | FP16, batch-one decode, 64-token prompt              |
| Fused projection     | gate/up: 1 x 2048 -> 6144, Q2_0                     |
| Launch               | grid 6144, block 64, 2 warps, 1 stage, Triton JIT   |
+----------------------+------------------------------------------------------+
```

### Fused gate/up Q2 GEMV and SwiGLU

The new kernel loads each 128-element activation block once, decodes the gate and up Q2 rows into separate FP32
accumulators, and applies FP32 SiLU plus the FP16 elementwise product before storing one intermediate tensor. It
replaces two Q2 GEMV launches, SiLU, and multiply with one launch in each of 28 MLP layers.

Installation is deliberately narrow: native Q2_0/PQ2_0 through `GGUFTritonKernel`, Qwen3 with SiLU, exact
`2048 -> 6144` gate/up shapes, bias-free and adapter-free projections on the same live sm80 device, FP16 input,
eval mode, and exactly one flattened row. Prefill, larger batches, BF16, training, other architectures, other GPU
capabilities, biases, and adapters execute the original MLP forward unchanged.

The target-stack checkpoint comparison installed all 28 fusions and produced zero sampled-logit MAE, zero maximum
absolute sampled-logit error, and identical greedy argmax. The focused sm80 test also compares fused output against
the two production Q2 GEMVs and validates that multi-row prefill takes the original path. CUDA Graph replay ended at
the same token (`93`) as the equivalent eager static-cache sequence.

The real-checkpoint launch sweep used five warmups and 50 timed decode steps per candidate in one process:

```text
+----------------------+-----------+-----------+-----------+
| Launch               | p50 ms    | p95 ms    | tok/s     |
+----------------------+-----------+-----------+-----------+
| Unfused gate/up      | 33.1090   | 35.8255   | 30.20     |
| 2 warps, 1 stage     | 29.5188   | 32.3266   | 33.88     |
| 2 warps, 2 stages    | 30.6662   | 31.9934   | 32.61     |
| 4 warps, 1 stage     | 30.9197   | 32.3145   | 32.34     |
| 4 warps, 2 stages    | 31.2724   | 32.8027   | 31.98     |
| 8 warps, 1 stage     | 31.3841   | 33.8463   | 31.86     |
| 8 warps, 2 stages    | 31.0697   | 32.0345   | 32.19     |
+----------------------+-----------+-----------+-----------+
```

The selected two-warp, one-stage launch lowers the controlled same-process median by 10.84%. A separate production
low-VRAM run, including loader installation and cache release, measured:

```text
+--------------------------------------+-----------+-----------+-----------+
| Torch 2.13 production path           | p50 ms    | p95 ms    | tok/s     |
+--------------------------------------+-----------+-----------+-----------+
| 64-token prefill                     | 40.4209   | 45.6800   | 1583.34   |
| Dynamic-cache eager decode           | 30.9765   | 34.1600   |   32.28   |
| Static-cache eager decode            | 37.5890   | 44.9284   |   26.60   |
| Static-cache CUDA Graph replay       |  5.5900   |  5.5951   |  178.89   |
+--------------------------------------+-----------+-----------+-----------+
```

Static eager remains rejected. Graph replay is 5.54x the low-VRAM dynamic eager throughput, but remains opt-in
research because of its fixed-address, fixed-capacity, and host-synchronization constraints. Across the complete
project, the initial 45.9602 ms decode measurement and the final 30.9765 ms production measurement differ by
32.60%; that end-to-end comparison spans the environment update, while 10.84% above is the controlled Torch 2.13
SwiGLU contribution.

### Matched Nsight Systems attribution

The four-token dynamic range contains 2,864 GPU operations, or 716 per token. Relative to the retained fused-RMS
trace, this removes exactly 84 operations per token. Native Q2 GEMV launches fall from 196 to 140 per token, and 28
fused SwiGLU kernels replace the gate/up pair plus two elementwise launches.

```text
+----------------------------------+--------------------+--------------------+
| Four-token Systems metric        | Fused RMSNorm      | + fused SwiGLU     |
+----------------------------------+--------------------+--------------------+
| GPU operations per token         | 800                | 716                |
| cudaLaunchKernel calls           | 1,960              | 1,732              |
| cuLaunchKernelEx calls           | 1,236              | 1,128              |
| Native Q2 GEMV launches          | 784                | 560                |
| Fused SwiGLU launches            | 0                  | 112                |
+----------------------------------+--------------------+--------------------+
```

The Torch 2.13 Systems range was `185.753` ms under profiler overhead. It is not compared as a latency delta with
the earlier Torch 2.12 trace; CUDA events provide the controlled latency result. The fused kernel accounted for
`4.189847` ms over 112 calls and had a `37.408` us median duration.

### Nsight Compute findings

```text
+--------------------------------+------------------+
| Metric                         | Fused SwiGLU     |
+--------------------------------+------------------+
| Duration                       | 37.95 us         |
| Grid / block                   | 6144 / 64        |
| Registers per thread           | 30               |
| Local-memory spills            | 0                |
| Theoretical / achieved occ.    | 100% / 71.53%    |
| SM throughput                  | 66.47%           |
| Memory / DRAM throughput       | 24.51% / 7.64%   |
| Memory throughput              | 186.75 GB/s      |
| L1 / L2 hit rate               | 77.43% / 58.84%  |
| Waves per SM                   | 1.55             |
+--------------------------------+------------------+
```

The kernel is compute- and dependency-latency-limited, not DRAM-bandwidth-limited. Long-scoreboard waits consume
about 5.5 of 15.52 cycles between issued instructions (35.44%). Nsight flags the 1.55-wave partial tail, but every
measured four- and eight-warp launch was slower. Output tiling or a one-warp variant remains an investigation, not a
retained change.

### Torch 2.13 VRAM

```text
+------------------------------+---------------+--------------+----------+-----------+
| State                        | Allocated MiB | Reserved MiB | Peak MiB | Cache MiB |
+------------------------------+---------------+--------------+----------+-----------+
| Loaded                       |        963.79 |      1586.00 |  3121.10 |      0.00 |
| Q2 prefill cache released    |       1019.31 |      1680.00 |  3121.10 |     21.00 |
| Dynamic decode warmed        |       1025.33 |      1680.00 |  3121.10 |     21.00 |
| Static-cache graph           |       1406.94 |      1786.00 |  1421.51 |       n/a |
+------------------------------+---------------+--------------+----------+-----------+
```

The SwiGLU fusion adds no persistent weight cache. The graph uses 381.61 MiB more live allocation than warmed
low-VRAM eager decode; Q2 cache release and the 3,121.10 MiB cold-load peak are unchanged.

### Torch 2.13 reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_0.py \
  --mode model --device 0 --prompt-tokens 64 --warmup 5 --iterations 50 \
  --release-prefill-cache

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_torch213_fused_swiglu_graph \
  /root/vm314t/bin/python scripts/profile_prism_q2_graph.py \
    --device 0 --prompt-tokens 64 --warmup 3 --iterations 20 \
    --max-cache-len 256 --graph-warmup 2 --capture \
    --profile-dynamic 4 --profile-graph 20

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section WarpStateStats \
  --kernel-name regex:_prism_q2_swiglu_gemv_kernel --launch-count 1 \
  --export artifacts/prism_q2_0_20260721/followup_torch213_swiglu_w2s1 \
  --force-overwrite \
  /root/vm314t/bin/python scripts/profile_prism_q2_graph.py \
    --device 0 --prompt-tokens 64 --warmup 2 --iterations 5 \
    --max-cache-len 128 --graph-warmup 2 --capture \
    --profile-dynamic 1 --profile-graph 0
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_torch213_fused_swiglu_graph.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_swiglu_graph.sqlite`
- `artifacts/prism_q2_0_20260721/followup_torch213_swiglu_w2s1.ncu-rep`

The next eager candidate at this stage was Q/K/V Q2 launch fusion (up to 56 launches per token). It was subsequently
implemented and retained in the investigation below. A production graph wrapper still has the largest measured
upside, while dynamic KV-cache copies remain a secondary target only if they can be removed without reproducing the
static-eager regression.

## 2026-07-21 continued investigation: fused Q/K/V Q2 projection

### Design and fallback boundary

The retained Q/K/V kernel combines the three output-major projections into one 4096-program grid: 2048 query rows,
1024 key rows, and 1024 value rows. It uses the same two-warp, one-stage direct Q2 decode and returns three views over
one output allocation. This removes two launches in each of 28 attention layers and avoids the separate partial-wave
tails of the 2048-, 1024-, and 1024-program grids.

The Qwen3 attention implementation invokes `q_proj`, `k_proj`, then `v_proj` with the same input tensor. The query
wrapper attaches key/value results to that input tensor only until the value wrapper consumes them. This avoids
module-global cross-request state. A missing or mismatched handoff simply calls the original projection, so a changed
call order remains correct. Installation has the same native Q2_0/PQ2_0, exact shape, bias/adapter-free, live sm80,
and backend gates as the other Prism specializations. Batch sizes above one, prefill, BF16, training, moved devices,
other architectures, and unsupported projections use their original forwards.

The real checkpoint installed 28 attention fusions. A one-token decode comparison produced zero sampled-logit MAE,
zero maximum absolute error, and identical greedy argmax. Focused tests compare each fused Q/K/V tensor against the
three production kernels, verify that the tensor-local handoff is removed after value projection, and exercise the
multi-row fallback. Static-cache graph replay continued to end at token `93`, matching eager exactly.

### Launch selection and controlled latency

The initial real-checkpoint launch screen selected two warps and one stage:

```text
+----------------------+-----------+-----------+-----------+
| Launch               | p50 ms    | p95 ms    | tok/s     |
+----------------------+-----------+-----------+-----------+
| Unfused Q/K/V        | 33.5933   | 38.2018   | 29.77     |
| 1 warp, 1 stage      | 32.2749   | 37.4947   | 30.98     |
| 1 warp, 2 stages     | 27.9301   | 30.8405   | 35.80     |
| 2 warps, 1 stage     | 27.3137   | 28.5973   | 36.61     |
| 2 warps, 2 stages    | 28.5138   | 31.2576   | 35.07     |
| 4 warps, 1 stage     | 27.4017   | 28.3898   | 36.49     |
| 4 warps, 2 stages    | 27.5092   | 30.4644   | 36.35     |
+----------------------+-----------+-----------+-----------+
```

Because independent runs showed substantial host and clock variation, the final claim comes from an AB/BA toggle of
the true original projection forwards and the production fusion in one loaded model. Each cell used ten warmups and
50 timed tokens with a fresh 64-token cache:

```text
+---------+----------------------+-----------+-----------+-----------+
| Round   | Path                 | p50 ms    | p95 ms    | tok/s     |
+---------+----------------------+-----------+-----------+-----------+
| A       | Unfused Q/K/V        | 31.3201   | 37.2460   | 31.93     |
| A       | Production fused     | 28.0771   | 28.9867   | 35.62     |
| B       | Production fused     | 28.5102   | 29.4715   | 35.08     |
| B       | Unfused Q/K/V        | 31.9928   | 33.4224   | 31.26     |
+---------+----------------------+-----------+-----------+-----------+
```

The two orderings show 10.35% and 10.89% lower median latency. The averaged medians are 31.6565 ms unfused and
28.2937 ms fused, a 10.62% reduction. The initial project measurement (45.9602 ms) and this final controlled eager
result differ by 38.44%, although that end-to-end comparison spans the Torch environment update.

### Matched Nsight Systems attribution

```text
+----------------------------------+--------------------+--------------------+
| Four-token Systems metric        | + fused SwiGLU     | + fused Q/K/V      |
+----------------------------------+--------------------+--------------------+
| Decode NVTX wall time            | 185.753 ms         | 142.442 ms         |
| GPU operations per token         | 716                | 660                |
| cudaLaunchKernel calls           | 1,732              | 1,732              |
| cuLaunchKernelEx calls           | 1,128              | 904                |
| Native Q2 GEMV launches          | 560                | 224                |
| Fused Q/K/V launches             | 0                  | 112                |
+----------------------------------+--------------------+--------------------+
```

The matched Torch 2.13 profiled range falls by 23.32%. Q/K/V fusion removes 56 GPU operations per token, leaving
660 total, 58.52% fewer than the approximately 1,591 operations/token after the first native Q2 optimization. Only
the attention output and MLP down projections remain as standalone native Q2 GEMVs (56/token); fused SwiGLU and
fused Q/K/V contribute 28 launches/token each. The Q/K/V kernel accounted for `1.316030` ms over 112 calls and had
an `11.744` us Systems median.

### Nsight Compute findings

```text
+--------------------------------+------------------+
| Metric                         | Fused Q/K/V      |
+--------------------------------+------------------+
| Duration                       | 12.80 us         |
| Grid / block                   | 4096 / 64        |
| Registers per thread           | 32               |
| Local-memory spills            | 0                |
| Theoretical / achieved occ.    | 100% / 55.53%    |
| SM throughput                  | 41.43%           |
| Memory / DRAM throughput       | 26.77% / 7.68%   |
| Memory throughput              | 186.67 GB/s      |
| L1 / L2 hit rate               | 82.49% / 62.98%  |
| Waves per SM                   | 1.03             |
+--------------------------------+------------------+
```

The combined grid nearly fills one maximum-residency wave; only 128 programs enter the partial wave. It remains
latency-limited, with long-scoreboard waits consuming about 5.0 of 15.12 cycles between issued instructions
(32.82%). Two- and four-output program tiles produced exact results but no stable kernel-time advantage, so the
one-output layout remains selected.

### Graph and VRAM after all retained fusions

The complete static-cache graph now measures `5.1979` ms p50, `5.2029` ms p95, or `192.39` tokens/s. This is 5.44x
the averaged controlled eager throughput. Allocated, reserved, and peak memory remain `1406.94`, `1786.00`, and
`1421.51` MiB; low-VRAM eager remains `1025.33` MiB warmed. Q/K/V fusion adds no persistent weight cache and uses
one transient combined output allocation in place of three projection outputs.

### Additional reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python \
  artifacts/prism_q2_0_20260721/followup_qkv_experiment.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_torch213_fused_qkv_graph \
  /root/vm314t/bin/python scripts/profile_prism_q2_graph.py \
    --device 0 --prompt-tokens 64 --warmup 5 --iterations 50 \
    --max-cache-len 256 --graph-warmup 2 --capture \
    --profile-dynamic 4 --profile-graph 20

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
ncu --profile-from-start off \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section WarpStateStats \
  --kernel-name regex:_prism_q2_qkv_gemv_kernel --launch-count 1 \
  --export artifacts/prism_q2_0_20260721/followup_torch213_qkv_w2s1 \
  --force-overwrite \
  /root/vm314t/bin/python scripts/profile_prism_q2_graph.py \
    --device 0 --prompt-tokens 64 --warmup 2 --iterations 5 \
    --max-cache-len 128 --graph-warmup 2 --capture \
    --profile-dynamic 1 --profile-graph 0
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_torch213_fused_qkv_graph.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_fused_qkv_graph.sqlite`
- `artifacts/prism_q2_0_20260721/followup_torch213_qkv_w2s1.ncu-rep`

The remaining eager trace is dominated by dynamic KV-cache concatenation/copies, attention elementwise work, and
host gaps around many small kernels. Static eager already showed that replacing concatenation alone is a regression;
the next high-confidence product gain is therefore a bounded production CUDA Graph wrapper, not another broad eager
cache rewrite.

## 2026-07-21 continued investigation: reusable generation graphs

### Torch 2.13 built-in compile path (rejected)

Transformers 5.7 has an automatic `torch.compile` decode path when `generate()` uses `cache_implementation="static"`.
The first attempt exposed a compatibility issue in the fused Q/K/V handoff: Dynamo fake tensors could not execute
`delattr()` on the tensor-local cache. Clearing the attribute by assigning `None` preserves eager lifetime behavior,
passes the fused-QKV regression, and lets compilation proceed.

Compilation is not a performance win for this backend. In an eight-token generation smoke test, Dynamo specialized
`_get_q2_native_scale()` on individual module weight pointers, reached its 128-recompile limit, and fell back through
many graph breaks:

```text
+----------------------+--------------+-----------+-----------+---------------+
| HF generation path   | Cold host ms | p50 ms/8 | tok/s     | Peak MiB      |
+----------------------+--------------+-----------+-----------+---------------+
| Dynamic, no compile  |      1888.40 |  241.1664 |     33.17 |       1351.14 |
| Static, no compile   |       360.12 |  308.3111 |     25.95 |       1360.24 |
| Static + compile     |     36106.51 |  704.4339 |     11.36 |       2571.07 |
+----------------------+--------------+-----------+-----------+---------------+
```

All completed paths produced the same `(1, 72)` token tensor and final token `63`. The compile-compatible Q/K/V
cleanup is retained, but automatic compilation is rejected for Prism Q2 inference until the quantized operations are
represented to Dynamo without per-module pointer guards.

### Retained graph runner and correctness boundary

`StaticCUDAGraphGreedyRunner` is an explicit, reusable low-level path. It captures one whole static-cache decode step,
resets only device-side cache lengths between requests, prefills the new prompt, and replays the graph. It records each
generated token on the same stream and returns raw token IDs. The graph can serve different unpadded prompt lengths up
to its cache capacity without recapture.

The supported boundary is deliberately narrow: eval-mode FP16 Qwen3, batch one, full attention, raw `torch.long`
CUDA input IDs, greedy argmax, one capture stream, and no concurrent use. Sampling, logits processors, padding masks,
beam search, streaming, and arbitrary stopping criteria remain on `model.generate()`. EOS IDs are supported by
truncating the visible result at the first EOS after the bounded graph finishes; this preserves output semantics but
does not save compute after EOS.

An optional `capture_prefill=True` specialization captures fixed-length prefill into a second graph and shares the
private graph pool with decode. It requires `release_prefill_cache=False`, owns strong references to all Q2 code-cache
tensors used by the captured graph, and accepts only the prompt length used during capture. The default remains the
flexible-prompt mode.

Dense FP16 regression coverage compares both graph modes with ordinary Qwen3 greedy generation, checks exact token
equality, output shape and dtype, two prompt lengths on the reusable decode graph, repeated cache reuse, one-token
generation, EOS truncation, and fixed-prefill prompt-length rejection. The real 2.125-bpw checkpoint produced the full
same 84-token output and final token `63` across dynamic, flexible-graph, and fixed-prefill-graph paths.

Reusable quantized graphs also require stable kernel argument addresses. Rebuilding a Q2 prefill cache originally
allocated a new scale tensor and retired the scale address embedded in the decode graph. The retained cache change
reuses the native scale tensor when reconstructing the expanded prefill code cache, then releases back to that same
address. A focused sm80 regression checks the scale `data_ptr()` across release, rebuild, and second release.

### Flexible graph latency and VRAM policies

The following are synchronized per-request CUDA-event measurements for a synthetic 64-token prompt and 20 generated
tokens. Each graph used the minimum 84-token capacity and produced exact output:

```text
+----------------------------+-----------+-----------+-----------+---------------+---------------+----------+
| Matched run / path         | p50 ms    | p95 ms    | tok/s     | Allocated MiB | Reserved MiB  | Peak MiB |
+----------------------------+-----------+-----------+-----------+---------------+---------------+----------+
| Low-memory dynamic         |  610.9047 |  633.3039 |     32.74 |       1329.48 |       1700.00 |  1351.14 |
| Flexible graph, release Q2 |  167.0922 |  169.3872 |    119.69 |       1011.42 |       1726.00 |  1369.89 |
| Retained-cache dynamic     |  600.0057 |  643.7761 |     33.33 |       1329.48 |       1700.00 |  1351.14 |
| Flexible graph, retain Q2  |  134.2351 |  136.6094 |    148.99 |       1347.10 |       1726.00 |  1369.89 |
+----------------------------+-----------+-----------+-----------+---------------+---------------+----------+
```

Low-memory graph mode reduces matched latency by 72.65% (3.66x throughput) and lowers persistent live allocation by
318.06 MiB versus dynamic generation. Retaining the expanded prefill cache reduces matched latency by 77.63% (4.47x
throughput). It is 19.66% faster than low-memory graph mode across the separate controlled runs, at a cost of
335.68 MiB more persistent allocation. Peak memory is similar because low-memory mode still constructs the expanded
cache transiently during prefill.

The corrected synchronized AB/BA capacity comparison selected the minimum cache size. Each cell used five warmups and
20 individually synchronized requests:

```text
+---------+----------------+-----------+-----------+-----------+
| Round   | Cache capacity | p50 ms    | p95 ms    | tok/s     |
+---------+----------------+-----------+-----------+-----------+
| A       |             84 |  128.0732 |  136.5668 |    156.16 |
| A       |            256 |  134.6145 |  144.2920 |    148.57 |
| B       |            256 |  129.1540 |  141.5527 |    154.85 |
| B       |             84 |  128.6298 |  134.5848 |    155.49 |
+---------+----------------+-----------+-----------+-----------+
```

The averaged medians are `128.3515` ms at capacity 84 and `131.8843` ms at capacity 256, so exact capacity is 2.68%
faster and uses 18.82 MiB less live allocation. An earlier unsynchronized queued-throughput screen was discarded and
is not used for the latency claim.

### Fixed-prefill graph result

Capturing the fixed 64-token prefill removes the remaining eager launch train. The retained-cache, exact-capacity
comparison is:

```text
+------------------------------+-----------+-----------+-----------+---------------+---------------+----------+
| Path                         | p50 ms    | p95 ms    | tok/s     | Allocated MiB | Reserved MiB  | Peak MiB |
+------------------------------+-----------+-----------+-----------+---------------+---------------+----------+
| Dynamic generation           |  640.7518 |  656.4819 |     31.21 |       1329.48 |       1700.00 |  1351.14 |
| Fixed-prefill + decode graph |   94.0001 |   94.0274 |    212.77 |       1347.39 |       1726.00 |  1357.27 |
+------------------------------+-----------+-----------+-----------+---------------+---------------+----------+
```

The fixed specialization lowers full-request median latency by 85.33% and raises throughput 6.82x. Capture costs
`346.03` ms once; the first post-capture request was `107.66` ms. Live allocation is 17.91 MiB above dynamic and only
0.29 MiB above flexible retained-cache graph mode.

### Matched Nsight Systems attribution

Nsight Systems 2024.6.2 used CUDA graph-level tracing, so each replay appears as one graph record. Profiler wall times
are used only for matched attribution; the CUDA-event table above is the latency result.

The flexible graph trace contained an eager prefill followed by 19 decode replays:

```text
+--------------------------------+------------------+------------------+
| 20-token profiled range        | Dynamic         | Flexible graph   |
+--------------------------------+------------------+------------------+
| NVTX wall time                 | 913.676 ms       | 152.038 ms       |
| Exposed kernel launches        | 15,206           | 1,088            |
| CUDA graph launches            | 0                | 19               |
| GPU records including memops   | 15,308           | 1,130            |
| GPU-in-use union               | 12.71%           | 71.09%           |
+--------------------------------+------------------+------------------+
```

The flexible path exposes 92.72% fewer execution launches. Its 19 graph executions total `94.347968` ms under tracing,
or `4.9657` ms each on average. The 22 output/cache copies consume only `0.052832` ms of GPU time and `0.207655` ms of
host API time, so moving token recording outside the graph for pointer-safe correctness is immaterial.

The final fixed-prefill trace reduces the request to one prefill graph, 19 decode graphs, two standalone kernels, and
22 copies:

```text
+--------------------------------+------------------+------------------+
| 20-token profiled range        | Dynamic         | Fixed graph      |
+--------------------------------+------------------+------------------+
| NVTX wall time                 | 825.768 ms       | 108.500 ms       |
| Exposed execution launches     | 13,638           | 22               |
| CUDA graph launches            | 0                | 20               |
| GPU records including memops   | 13,740           | 44               |
| GPU-in-use union               | 12.76%           | 99.34%           |
+--------------------------------+------------------+------------------+
```

This is a 99.84% reduction in exposed execution launches and an 86.86% reduction in matched profiler wall time. The
prefill graph took `12.812545` ms under tracing; the 19 decode graphs had a `4.995540` ms median. At 99.34% GPU-in-use,
the fixed path has moved the residual bottleneck from host launch gaps to graph-contained GPU work. Further gains now
require reducing kernel execution time or changing the quantized compute/attention algorithm, not more host launch
fusion.

### Reproduction and new artifacts

```bash
# Flexible low-memory graph.
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 2 --iterations 5 \
  --paths dynamic graph

# Flexible fast graph with retained prefill cache.
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 2 --iterations 5 \
  --paths dynamic graph --retain-prefill-cache

# Fixed-prompt prefill and decode graphs.
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 2 --iterations 5 \
  --paths dynamic graph --retain-prefill-cache --capture-prefill

# Final fixed-graph Systems capture.
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_torch213_fixed_prefill_graph \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 2 --iterations 3 \
    --paths dynamic graph --retain-prefill-cache --capture-prefill --capture

# Corrected synchronized capacity AB/BA.
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python \
  artifacts/prism_q2_0_20260721/followup_graph_cache_abba.py
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_torch213_production_graph.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_production_graph.sqlite`
- `artifacts/prism_q2_0_20260721/followup_torch213_fixed_prefill_graph.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_fixed_prefill_graph.sqlite`
- `artifacts/prism_q2_0_20260721/followup_graph_cache_abba.py`

## 2026-07-21 post-graph investigation: RMSNorm decode launch geometry

Once the fixed graph reached 99.34% GPU-in-use, the 113 RMSNorm kernels per token became a small compute-side target.
A direct CUDA-event microbenchmark was host dominated and discarded. The retained decision comes from 100-launch
NVTX-separated Nsight Systems ranges, four Nsight Compute reports, and a synchronized full-request AB/BA comparison.

### Systems launch screen

All variants produced bit-identical FP16 outputs in the screen:

```text
+----------------------+----------------+----------------+------------------+
| Decode norm shape    | Four-warp p50 | Selected p50  | Selected launch  |
+----------------------+----------------+----------------+------------------+
| 1 x 2048             |       2.880 us |       2.624 us | 8 warps (-8.89%) |
| 8 x 128              |       2.528 us |       2.240 us | 1 warp (-11.39%) |
| 16 x 128             |       2.560 us |       2.304 us | 1 warp (-10.00%) |
+----------------------+----------------+----------------+------------------+
```

The selector applies these launches only to FP16 and the exact profiled decode row counts. FP16 prefill shapes,
other row counts, BF16, CPU, non-sm80 devices, unsupported widths, and the original Torch fallback retain the existing
four-warp behavior.

### Nsight Compute validation

```text
+----------------+-------+----------+------+-----------+----------------+---------------------+
| Shape          | Warps | Duration | Regs | Ach. occ. | Cycles / issue | Dominant stalls     |
+----------------+-------+----------+------+-----------+----------------+---------------------+
| 1 x 2048       |     4 |  4.03 us |   32 |     6.17% |          18.54 | scoreboard 8.8      |
| 1 x 2048       |     8 |  3.94 us |   24 |    10.64% |          22.88 | scoreboard 8.6      |
| 16 x 128       |     4 |  3.68 us |   16 |     5.95% |          37.46 | scoreboard 14.7     |
| 16 x 128       |     1 |  3.20 us |   20 |     1.54% |          29.92 | immediate const 12.2 |
+----------------+-------+----------+------+-----------+----------------+---------------------+
```

Nsight Compute replay overhead makes its absolute durations higher than the Systems trace, but both tools select the
same launch geometries. The width-2048 launch trades more threads for lower register use and higher achieved occupancy.
The width-128 kernel is a tiny 16-block grid; one warp finishes sooner despite lower device-wide occupancy.

### Full fixed-graph AB/BA

Each cell used ten warmups and 30 individually synchronized requests. Both graphs used captured 64-token prefill,
19 decode replays, retained Q2 prefill cache, and exact 84-token capacity:

```text
+---------+-----------+-----------+-----------+-----------+
| Round   | Path      | p50 ms    | p95 ms    | tok/s     |
+---------+-----------+-----------+-----------+-----------+
| A       | 4-warps   |   94.3104 |   94.3883 |    212.07 |
| A       | Selected  |   93.8179 |   94.0315 |    213.18 |
| B       | Selected  |   93.7370 |   93.8165 |    213.36 |
| B       | 4-warps   |   94.3094 |   94.3931 |    212.07 |
+---------+-----------+-----------+-----------+-----------+
```

The averaged median falls from `94.3099` to `93.7774` ms, a further 0.565% reduction. Full generated token tensors
remain identical with final token `63`. This is a deliberately small final gain; larger launch-only fusions are no
longer attractive now that they execute inside a 99.34%-busy graph.

### Reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --sample=none --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_torch213_rms_warp_screen \
  /root/vm314t/bin/python \
    artifacts/prism_q2_0_20260721/followup_rms_warp_screen.py --capture

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python \
  artifacts/prism_q2_0_20260721/followup_rms_graph_abba.py
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_torch213_rms_warp_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_rms_warp_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_torch213_rms_2048_w4.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_rms_2048_w8.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_rms_128x16_w4.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_torch213_rms_128x16_w1.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_rms_warp_screen.py`
- `artifacts/prism_q2_0_20260721/followup_rms_ncu.py`
- `artifacts/prism_q2_0_20260721/followup_rms_graph_abba.py`

## 2026-07-21 Transformers 5.14.1 forward-compatibility retest

### Updated stack and scope

The branch was retested without changing or downgrading Transformers. The active `/root/vm314t` environment was:

- Python 3.14.5 free-threading build.
- PyTorch `2.13.0+cu130`, CUDA runtime 13.0, and Triton 3.7.1.
- Transformers 5.14.1.
- NVIDIA `PG506-230`, compute capability 8.0, 124 SMs, and 102,191,202,304 bytes of memory. The process was
  restricted to one runtime-probed GPU with `CUDA_VISIBLE_DEVICES=0`; no fixed physical CUDA index is assumed by the
  implementation.
- Model `/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf`, native 2.125-bpw Q2_0,
  FP16 activations, batch one.

### Transformers API drift and retained fixes

The first 161-test focused run produced 159 passes, one optional skip, and one failure in the legacy cache-length
bridge. Transformers 5.14.1 made `CacheLayerMixin.get_max_length()` abstract and changed the deprecated
cache-level `get_max_cache_shape()` into an alias that calls `get_max_length()`. The previous compatibility
implementation could therefore recurse if the legacy method had to be restored on that API boundary.

The bridge now reads the selected cache layer directly, supports either the current layer `get_max_length()` method
or the older layer `get_max_cache_shape()` method, and treats an empty/out-of-range dynamic cache as unbounded. The
regression fixture implements the current abstract method and explicitly exercises an empty cache.

A real load also exposed a deprecation message from reading `config.torch_dtype`. Transformers 5.14.1 exposes that
name as a warning property even when modern `config.dtype` exists but is unset. `get_hf_config_dtype()` now avoids
the deprecated property on modern configs while retaining a genuinely stored legacy `torch_dtype` value. A
config-only GGUF smoke test then reported `model_type=qwen3`, `dtype_before=None`, and `dtype_after=torch.float16`
without the Transformers deprecation message.

No dependency version, lock file, or environment package was changed. These are forward-compatibility changes in
GPT-QModel and its tests.

After both fixes and their new regressions were included, the final consolidated focused run completed with 162
passes, one optional skip, and no failures in 257.85 seconds. CUDA correctness cases executed on sm_80 rather than
skipping; the only skip was the existing optional PEFT/AWQ compatibility probe.

### Real-model generation validation

A short dynamic/static/flexible-graph smoke test used an eight-token prompt, four generated tokens, one warmup, and
two measured requests. Every path returned the identical `(1, 12)` token tensor with final token 16:

```text
+---------+-------------+-------------+-----------+---------------+----------+
| Path    | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Peak MiB |
+---------+-------------+-------------+-----------+---------------+----------+
| Dynamic |    131.7114 |    132.3130 |     30.37 |       1329.15 |  1343.13 |
| Static  |    147.5360 |    148.8717 |     27.11 |       1330.37 |  1351.60 |
| Graph   |     90.6321 |     90.9786 |     44.13 |       1004.43 |  1361.34 |
+---------+-------------+-------------+-----------+---------------+----------+
```

The final warmed comparison used the production 64-token prompt, 20 generated tokens, two warmups, five measured
requests, retained Q2 cache, fixed captured prefill, and exact 84-token graph capacity. Dynamic and graph generation
returned identical `(1, 84)` output with final token 63:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Dynamic generation    |    555.9844 |    557.3684 |     35.97 |       1329.48 |       1700.00 |  1351.14 |
| Fixed prefill + graph |     93.3028 |     93.3902 |    214.36 |       1347.39 |       1724.00 |  1357.27 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

On this retest the fixed graph lowered median request latency by 83.22% and raised throughput 5.96x. It used 17.91
MiB more live allocation and 6.13 MiB more peak allocation than dynamic generation. Capture cost 2.167 seconds and
the first post-capture request took 107.15 ms; those one-time costs are excluded from the warmed table.

### Reproduction and checks

```bash
CUDA_VISIBLE_DEVICES=0 /root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_cuda_graph_generate.py

CUDA_VISIBLE_DEVICES=0 /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 2 --iterations 5 \
  --paths dynamic graph --retain-prefill-cache --capture-prefill

/root/vm314t/bin/ruff check gptqmodel/utils/hf.py tests/test_hf_config_compat.py
git diff --check
```

## 2026-07-21 continued investigation: fixed-graph child-kernel attribution

### Node-level capture on Transformers 5.14.1

The earlier graph-level Systems trace established 99.34% GPU-in-use, but intentionally represented each CUDA graph
as one opaque operation. A new `--cuda-graph-trace=node` capture exposed every child node for one warmed fixed
64-token prefill plus 20-token greedy request. The unprofiled CUDA-event measurement immediately before collection
was `93.2671` ms p50, or `214.44` generated tokens/s. It retained the Q2 prefill cache, captured prefill, used an
exact 84-token capacity, and returned the expected `(1, 84)` tensor ending in token `63`.

The SQLite export identifies a graph node by `graphNodeId`. Nodes executed once belong to the prefill graph; nodes
executed 19 times belong to the reused decode graph. This separates stages without inferring from kernel names:

```text
+----------+---------------------+------------------+----------------------+-------------------------+
| Stage    | Unique kernel nodes | Kernel instances | Kernel GPU time      | Memory-node GPU time    |
+----------+---------------------+------------------+----------------------+-------------------------+
| Prefill  |               1,085 |            1,085 | 13.003904 ms         | 0.004352 ms             |
| Decode   |                 917 |           17,423 | 96.279931 ms total   | 0.913532 ms total       |
| / token  |                 917 |              917 |  5.067365 ms         | 0.048081 ms             |
+----------+---------------------+------------------+----------------------+-------------------------+
```

Decode memory nodes comprise 551 device-to-device copies taking `0.876380` ms and 19 four-byte memsets taking
`0.037152` ms. Prefill has one eight-byte copy and one four-byte memset. Another 22 standalone device-to-device
copies take only `0.047840` ms. Node tracing adds measurement overhead, so these absolute child sums are attribution
data, not a replacement for the unprofiled 93.2671 ms request result.

### Dominant kernels

The following complete filtered table retains a residual row. Shares are within kernel time for the named stage and
come from the SQLite query over `CUPTI_ACTIVITY_KIND_KERNEL`; the family names are the Nsight short kernel names.

```text
+---------+-------------------------------------------+------------------+----------+------------+----------+
| Stage   | Kernel family                             | Source / owner   | Calls    | Total ms   | Share    |
+---------+-------------------------------------------+------------------+----------+------------+----------+
| Prefill | _gguf_q2_0_u32_fused_matmul_kernel_impl  | gguf_triton.py   |      196 |   9.495338 |   73.02% |
| Prefill | elementwise_kernel                        | PyTorch          |      366 |   1.362077 |   10.47% |
| Prefill | vectorized_elementwise_kernel             | PyTorch          |      232 |   0.500509 |    3.85% |
| Prefill | _prism_q2_rms_norm_kernel                 | rms_norm.py      |      113 |   0.371135 |    2.85% |
| Prefill | gemv2T_kernel_val                         | cuBLAS lm_head   |        1 |   0.333791 |    2.57% |
| Prefill | fmha_cutlassF_f16_aligned_64x128_rf_sm80  | PyTorch SDPA     |       28 |   0.322399 |    2.48% |
| Prefill | CatArrayBatchedCopy                       | PyTorch cache    |       57 |   0.310816 |    2.39% |
| Prefill | index_elementwise_kernel                  | PyTorch cache    |       56 |   0.235775 |    1.81% |
| Prefill | residual                                  | mixed            |       36 |   0.072064 |    0.55% |
+---------+-------------------------------------------+------------------+----------+------------+----------+
| Decode  | _prism_q2_swiglu_gemv_kernel             | q2_swiglu.py     |      532 |  19.812432 |   20.58% |
| Decode  | _gguf_q2_0_native_gemv_kernel_impl       | gguf_triton.py   |    1,064 |  17.375894 |   18.05% |
| Decode  | elementwise_kernel                        | PyTorch          |    4,807 |  16.519946 |   17.16% |
| Decode  | vectorized_elementwise_kernel             | PyTorch          |    4,959 |   9.060234 |    9.41% |
| Decode  | gemv2T_kernel_val                         | cuBLAS lm_head   |       19 |   6.304944 |    6.55% |
| Decode  | _prism_q2_qkv_gemv_kernel                | q2_qkv.py        |      532 |   6.244975 |    6.49% |
| Decode  | fmha_cutlassF_f16_aligned_64x128_rf_sm80  | PyTorch SDPA     |      532 |   5.696564 |    5.92% |
| Decode  | _prism_q2_rms_norm_kernel                 | rms_norm.py      |    2,147 |   5.107348 |    5.30% |
| Decode  | CatArrayBatchedCopy                       | PyTorch cache    |    1,064 |   4.646133 |    4.83% |
| Decode  | index_elementwise_kernel                  | PyTorch cache    |    1,064 |   4.108885 |    4.27% |
| Decode  | residual                                  | mixed            |      703 |   1.402576 |    1.46% |
+---------+-------------------------------------------+------------------+----------+------------+----------+
```

The 56 native Q2 GEMV nodes per decode form two exact 28-node duration clusters. The short cluster accounts for
`5.321842` ms over 532 launches at `10.003` us average; the long cluster accounts for `12.054052` ms at `22.658` us.
From the model's one attention-output and one MLP-down projection per layer, these are respectively the
`2048 -> 2048` attention output and `6144 -> 2048` MLP down projections. Both currently use a 2,048-program,
two-warp, two-stage launch. The longer down projection is therefore the next bounded launch-geometry target.

### Overlap opportunities

```text
+--------------------------+-------------------------+--------------------------------------+------------+
| Region                   | Measured time / share   | Dependency evidence                  | Decision   |
+--------------------------+-------------------------+--------------------------------------+------------+
| Decode graph kernels     | 5.067365 ms / token     | One stream; layer and token dataflow | No overlap |
| Decode graph memory ops  | 0.048081 ms / token     | Cache writes feed same-token SDPA    | Low ceiling|
| Inter-token decode       | 19 serial replays       | Token N+1 consumes token N argmax    | No overlap |
| Standalone output copies | 0.047840 ms / request   | Final generated-token assembly       | Negligible |
+--------------------------+-------------------------+--------------------------------------+------------+
```

The previously measured 99.34% GPU-in-use and this child-node breakdown agree: batch-one greedy generation no
longer has useful host/GPU overlap to recover. Multi-request batching could create independent work, but changes the
latency and concurrency objective and is outside this single-request kernel comparison.

### Ranked fusion and tuning candidates

```text
+--------------------------+--------------------------+----------------------+-------------------------------+
| Producer / target        | Consumer / source        | Intermediate         | Evidence and next action      |
+--------------------------+--------------------------+----------------------+-------------------------------+
| MLP down Q2 GEMV         | residual add             | FP16 [1, 2048]       | 12.054052 ms; sweep K=6144    |
| PyTorch elementwise set  | attention/cache chain    | mixed FP16/index     | 35.66% combined; map eagerly  |
| fused Q2 SwiGLU          | MLP down Q2 GEMV         | FP16 [1, 6144]       | 20.58% + 12.52%; hard fusion  |
| QKV + RoPE/cache writes  | PyTorch SDPA             | FP16 Q/K/V           | dependencies block direct fuse|
| FP16 vocabulary GEMV     | argmax reduction         | FP16 [1, vocab]      | 6.79%; profile before custom  |
+--------------------------+--------------------------+----------------------+-------------------------------+
```

SwiGLU-to-down fusion is not immediately retained: down outputs each depend on all 6,144 independently generated
SwiGLU values, and sm80 has no grid-wide synchronization suitable for this layout. A launch sweep of the existing
long down-projection GEMV is lower risk. The mixed elementwise/cache group needs a matched eager mapping trace before
any source-level fusion proposal; graph names alone are insufficient attribution.

### Reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_fixed_graph_nodes \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --retain-prefill-cache --capture-prefill --capture

nsys stats --force-export=true --format=csv --report=cuda_gpu_kern_sum \
  artifacts/prism_q2_0_20260721/followup_tf514_fixed_graph_nodes.nsys-rep
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_tf514_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_fixed_graph_nodes.sqlite`

## 2026-07-21 continued optimization: exact fused Prism Q2 rotary embedding

### Rejected MLP-down launch retune

The first post-attribution experiment screened the real `blk.0.ffn_down.weight` tensor, shape `2048 x 6144`, with
one, two, four, and eight warps and one or two stages. Every variant retained the same dense-reference error:
`0.00057839` MAE and `0.00390625` maximum absolute error. Two-warp variants were bit-identical to production; other
warp counts changed only reduction ordering.

A 100-launch-per-range Systems screen initially appeared to favor one warp and one stage:

```text
+------------+------------+------------+------------+
| Launch     | Median us  | Min us     | Max us     |
+------------+------------+------------+------------+
| 1w / 1s    |     19.072 |     18.656 |     35.264 |
| 1w / 2s    |     19.200 |     18.560 |  3,680.054 |
| 2w / 1s    |     19.584 |     19.328 |  3,910.165 |
| 2w / 2s    |     19.616 |     19.328 |  4,064.150 |
| 4w / 1s    |     28.000 |     27.648 |  3,337.464 |
| 4w / 2s    |     27.968 |     27.712 |  2,852.665 |
| 8w / 1s    |     43.872 |     43.648 |  1,302.877 |
| 8w / 2s    |     43.904 |     43.552 |  2,920.217 |
+------------+------------+------------+------------+
```

The repeated screen keeps one layer's Q2 weights cache-hot. Nsight Compute and the full model showed that this is
not representative of 28 layers with different weights:

```text
+--------------------------------+-------------+-------------+
| Nsight Compute metric          | 1w / 1s     | 2w / 2s     |
+--------------------------------+-------------+-------------+
| Replay duration                | 36.928 us   | 23.136 us   |
| Block / grid                   | 32 / 2,048  | 64 / 2,048  |
| Registers per thread           | 31          | 32          |
| Achieved occupancy             | 23.70%      | 44.16%      |
| SM throughput                  | 21.36%      | 46.13%      |
| Compute-memory throughput      |  7.76%      | 18.99%      |
| DRAM throughput                |  6.44%      |  6.29%      |
| Long-scoreboard cycles / issue | 11.54       |  8.15       |
+--------------------------------+-------------+-------------+
```

Two separately captured full graphs provided the deciding AB/BA result. Each cell used ten warmups and 50
individually synchronized requests:

```text
+---------+------------+-----------+-----------+-----------+
| Round   | Down GEMV  | p50 ms    | p95 ms    | tok/s     |
+---------+------------+-----------+-----------+-----------+
| A       | 2w / 2s    |   93.3837 |   93.6091 |    214.17 |
| A       | 1w / 1s    |   99.0290 |   99.6915 |    201.96 |
| B       | 1w / 1s    |   98.9691 |   99.3741 |    202.08 |
| B       | 2w / 2s    |   93.4938 |   93.9608 |    213.92 |
+---------+------------+-----------+-----------+-----------+
```

The averaged median regressed from `93.4387` to `98.9990` ms, or 5.95%, while generation remained exact. The
one-warp result is rejected and the existing two-warp/two-stage down projection remains unchanged. This is also why
the final decision is not based on an isolated repeated-weight microbenchmark.

### Matched eager source mapping

A single static-cache decode at position 64 was captured with PyTorch operator NVTX ranges. Kernel/runtime
correlation mapped the dominant generic elementwise work back to the Transformers 5.14.1 Qwen3 implementation:

```text
+-------------------+-------+------------+-------------------------------------------+
| PyTorch operation | Calls | Total ms   | Qwen3 source                              |
+-------------------+-------+------------+-------------------------------------------+
| aten::mul         |   114 |   0.478176 | four RoPE multiplies/layer plus two setup  |
| aten::neg         |    56 |   0.201792 | two rotate-half negations/layer           |
| aten::cat         |    57 |   0.289280 | two rotate-half concatenations/layer + one |
| aten::add         |   142 |   0.322237 | two RoPE adds and residual adds per layer |
| aten::copy_       |    59 |   0.330368 | cache and layout copies                   |
| aten::index_copy_ |    56 |   0.251679 | static K/V cache updates                  |
+-------------------+-------+------------+-------------------------------------------+
```

For each of 28 layers, generic Q/K rotary embedding launches four multiplies, two negations, two concatenations,
and two additions: 280 dependent kernels per token. This was the largest coherent remaining fusion boundary.

### Retained exact Q/K rotary kernel

The new Triton kernel handles all 16 query heads and eight key heads in one 24-program launch. It reads each
128-element head and the shared cosine/sine row, directly applies the half rotation, and writes separate query and
key tensors. To match the three generic PyTorch elementwise steps bit-for-bit, inline PTX explicitly performs FP32
multiplication, FP16 round-to-nearest, FP32 addition, and the final FP16 round. Random supported-shape comparisons
then had zero MAE, zero maximum error, and exact tensor equality; the real model had zero sampled-logit error.

Installation remains Prism-only and deliberately narrow:

- The loader must already have installed the native Prism Q2 Q/K/V specialization on an exact Qwen3
  2,048-hidden, 16-query-head, eight-key-head, 128-head-dimension model resident on a runtime-probed sm80 device.
- The fused path requires eval mode, FP16, batch one, one decode row, contiguous head values, matching cosine/sine
  tensors, and `unsqueeze_dim=1`.
- Prefill, larger batches, training, BF16/FP32, moved devices, non-sm80 GPUs, different layouts or dimensions, and
  unavailable Triton execute the installed Transformers function unchanged.
- Each attention receives a clone of its live Transformers forward function with a private globals dictionary in
  which only `apply_rotary_pos_emb` is replaced. Transformers is not patched globally. If a future upstream forward
  no longer exposes that function boundary, installation safely returns zero instead of copying or assuming an old
  implementation.

A 200-launch Systems screen kept every candidate bit-exact and selected four warps by a small margin:

```text
+-------+-----------+-----------+-----------+
| Warps | Median us | Min us    | Max us    |
+-------+-----------+-----------+-----------+
|     1 |     2.400 |     2.368 |     2.560 |
|     2 |     2.400 |     2.368 |     2.464 |
|     4 |     2.368 |     2.336 |     2.432 |
+-------+-----------+-----------+-----------+
```

At less than 2.5 us this kernel is launch-latency-bound; the four-warp delta has only about a 0.02% full-request
ceiling. Cross-layer fusion is blocked by attention, residual, and MLP dependencies, so deeper tuning of this kernel
is not a useful next target.

### Controlled end-to-end latency and VRAM

The fixed-graph AB/BA used two separately captured graphs in one loaded model. Each cell had ten warmups and 40
individually synchronized 64-prompt/20-generated-token requests:

```text
+---------+----------------+-----------+-----------+-----------+
| Round   | Rotary path    | p50 ms    | p95 ms    | tok/s     |
+---------+----------------+-----------+-----------+-----------+
| A       | Transformers   |   93.3227 |   93.4407 |    214.31 |
| A       | Fused exact    |   81.0301 |   81.1874 |    246.82 |
| B       | Fused exact    |   81.0429 |   81.4203 |    246.78 |
| B       | Transformers   |   93.3258 |   93.5116 |    214.30 |
+---------+----------------+-----------+-----------+-----------+
```

The averaged median falls from `93.3243` to `81.0365` ms, a 13.17% latency reduction and 1.152x throughput gain.
Sampled logits and the full `(1, 84)` greedy token tensor were bit-identical; both ended in token `63`.

A matched dynamic-cache AB/BA toggled only the 28 model-local attention forwards. Twelve synchronized requests per
cell, after three warmups, gave averaged medians of `635.8336` ms generic and `573.6196` ms fused: 9.78% lower
latency and 1.108x higher throughput, again with exact tokens. The causal AB/BA is used instead of comparing separate
dynamic runs, which showed substantial host and clock variation.

The final five-iteration production run, with the retained four-warp kernel installed automatically, measured:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Fixed prefill + graph |     80.6697 |     80.8385 |    247.92 |       1347.39 |       1726.00 |  1357.27 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

Relative to the pre-rotary Transformers 5.14.1 production row (`93.3028` ms and `214.36` tokens/s), this is 13.54%
lower latency and 15.66% higher throughput. Allocated and peak memory are unchanged; the two-MiB reserved difference
is allocator capacity, not live storage. The fusion adds no persistent weight or activation cache.

### Post-fusion Systems attribution

The matched node trace confirms that each layer's ten rotary kernels became one:

```text
+------------------------------+----------------+----------------+-------------+
| Decode metric, 19 tokens     | Before rotary | Fused rotary   | Change      |
+------------------------------+----------------+----------------+-------------+
| Unique kernel nodes / token  |            917 |            665 | -27.48%     |
| Kernel instances             |         17,423 |         12,635 | -4,788      |
| Kernel GPU time              |   96.279931 ms |   80.551146 ms | -16.34%     |
| D2D copies                   |            551 |            551 | unchanged   |
| Memsets                      |             19 |             19 | unchanged   |
+------------------------------+----------------+----------------+-------------+
```

The new kernel takes `1.057565` ms over 532 launches, or `1.988` us average. Prefill remains unchanged at 1,085
kernel nodes and `13.009567` ms because multi-row rotary uses the original path.

```text
+-------------------------------------------+--------+------------+----------+
| Post-fusion decode kernel family          | Calls  | Total ms   | Share    |
+-------------------------------------------+--------+------------+----------+
| _prism_q2_swiglu_gemv_kernel             |    532 |  19.808914 |   24.59% |
| _gguf_q2_0_native_gemv_kernel_impl       |  1,064 |  17.384372 |   21.58% |
| vectorized_elementwise_kernel             |  3,895 |   7.043623 |    8.74% |
| elementwise_kernel                        |  1,615 |   6.333484 |    7.86% |
| gemv2T_kernel_val                         |     19 |   6.309906 |    7.83% |
| _prism_q2_qkv_gemv_kernel                |    532 |   6.222738 |    7.73% |
| fmha_cutlassF_f16_aligned_64x128_rf_sm80  |    532 |   5.712786 |    7.09% |
| _prism_q2_rms_norm_kernel                 |  2,147 |   5.107092 |    6.34% |
| index_elementwise_kernel                  |  1,064 |   4.162546 |    5.17% |
| _prism_q2_fused_rotary_kernel             |    532 |   1.057565 |    1.31% |
| residual                                  |    684 |   1.408120 |    1.75% |
+-------------------------------------------+--------+------------+----------+
```

SwiGLU and the remaining attention-output/MLP-down Q2 projections now account for 46.17% of decode kernel time.
Further material gains require a different quantized compute algorithm or a more invasive residual/cache boundary.
The two K/V `index_copy_` operations per layer are a bounded future graph target, but their measured ceiling is much
smaller than rotary and they are retained unchanged until a custom cache can preserve every Transformers fallback.

### Validation and reproduction

The focused CUDA regression run completed with 14 passes on sm80; no CUDA test skipped. It covered exact rotary
output, model-local installation, CPU/non-Q2 fallback, training and prefill fallback, QKV, SwiGLU, RMSNorm, and CUDA
graph generation. The expanded compatibility run completed with 164 passes and one unrelated skip across the HF
compatibility, internal GGUF, weight-only, Prism Q2, and CUDA graph suites. Ruff and `git diff --check` also passed.
Transformers remained at 5.14.1 throughout.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_prism_q2_rotary.py tests/test_prism_q2_qkv.py \
  tests/test_prism_q2_swiglu.py tests/test_prism_rms_norm.py \
  tests/test_cuda_graph_generate.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_prism_q2_rotary.py \
  tests/test_cuda_graph_generate.py

/root/vm314t/bin/ruff check \
  gptqmodel/nn_modules/triton_utils/q2_rotary.py \
  gptqmodel/models/loader.py tests/test_prism_q2_rotary.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_rotary_fixed_graph_nodes \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --retain-prefill-cache --capture-prefill --capture
```

New raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_down_gemv_sweep.py`
- `artifacts/prism_q2_0_20260721/followup_down_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_down_gemv_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_down_gemv_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_tf514_down_w1s1.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_down_w2s2.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_static_decode_mapping.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_static_decode_mapping.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_static_decode_mapping.sqlite`
- `artifacts/prism_q2_0_20260721/followup_rotary_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_rotary_dynamic_abba.py`
- `artifacts/prism_q2_0_20260721/followup_rotary_warp_screen.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_rotary_warp_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_rotary_warp_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_tf514_rotary_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_rotary_fixed_graph_nodes.sqlite`

## Transformers 5.14.1 follow-up: fixed-cache GQA fusion

The post-rotary profile was continued on the same runtime and hardware: Python 3.14.5t, Torch 2.13.0+cu130,
CUDA 13.0, Triton 3.7.1, Transformers 5.14.1, and the runtime-probed NVIDIA PG506-230 (`sm80`, 124 SMs,
102,191,202,304 bytes). No dependency was changed or downgraded.

### Remaining attention launch boundary

A new eager decode trace after the retained rotary fusion mapped all 663 GPU kernels back through CUDA runtime
correlation IDs to the innermost PyTorch NVTX operation. The fixed-cache attention boundary was now the largest
coherent generic launch group:

```text
+-------------------------------+-------+------------+----------------------------------------------+
| Operation                     | Calls | Total ms   | Role                                         |
+-------------------------------+-------+------------+----------------------------------------------+
| aten::copy_                   |    56 |   0.324224 | materialize repeated K and V, two per layer  |
| aten::fill_                   |    84 |   0.138400 | boolean-mask preparation, three per layer    |
| aten::where                   |    28 |   0.070848 | convert boolean mask, one per layer           |
| efficient attention / FMHA    |    28 |   0.324192 | one attention kernel per layer                |
| aten::index_copy_             |    56 |   0.263455 | static K/V cache writes, retained separately  |
| residual aten::add            |    56 |   0.127488 | two residual additions per layer              |
+-------------------------------+-------+------------+----------------------------------------------+
```

Transformers 5.14.1 does not select native grouped-query attention when a static boolean mask is present. It
expands each eight-head K/V cache to 16 heads, prepares an additive mask, runs FMHA, and then materializes the
transposed output. For every layer this is seven kernel launches plus a D2D layout copy before the following output
projection.

Torch 2.13's `enable_gqa=True` was screened before writing a custom kernel. With the required masked shape it fell
back to a 17-kernel math path. It was numerically close (`0.00002519` MAE, `0.00048828` maximum absolute error) but
materially slower:

```text
+---------------------+------------------+---------------+
| Path                | Eager median us  | Graph p50 us  |
+---------------------+------------------+---------------+
| repeat K/V + SDPA   |          136.640 |        30.592 |
| native GQA SDPA     |          286.960 |        57.600 |
+---------------------+------------------+---------------+
```

The native GQA path is rejected. GPT-QModel does not patch Torch or Transformers to force it.

### Retained Triton grouped-query attention

The retained kernel uses eight programs, one per K/V head. Each program evaluates the two associated query heads,
the 84-position boolean mask, softmax, and the value reduction. A padded 16-row by 128-column tile lets both matrix
products use Tensor Cores; only the two live rows are stored. The output is written directly as `(1, 1, 16, 128)`,
so the later transpose/contiguous copy also disappears. Its masked softmax explicitly returns zeros for an all-false
mask, matching SDPA instead of producing a `-inf - -inf` NaN.

The production boundary is deliberately exact and Prism-only:

- Installation requires the already-installed native Prism Q2 QKV path on an exact Qwen3 model with hidden size
  2,048, 16 query heads, eight K/V heads, head dimension 128, and a runtime-probed `sm80` device.
- Execution requires eval mode, FP16 contiguous Q/K/V tensors with shapes `(1,16,1,128)` and
  `(1,8,84,128)`, a contiguous boolean `(1,1,1,84)` mask, zero dropout, the exact Qwen3 scale, and no position bias.
- Prefill, dynamic cache lengths, training, additive masks, BF16/FP32, moved devices, sliding attention, alternate
  dimensions, non-sm80 GPUs, output-attention requests, and unavailable Triton call the captured Transformers
  attention interface unchanged.
- Each attention forward is cloned with a private globals dictionary whose `ALL_ATTENTION_FUNCTIONS` entry is a
  model-local proxy. Neither the Transformers module global nor its registry is mutated. If upstream changes remove
  that lookup boundary, installation returns zero and retains the upstream implementation.

The isolated boolean-mask screen used 100 warmups and 500 CUDA-event samples. The reference was the exact
Transformers operation sequence for `Q=(1,16,1,128)`, `K/V=(1,8,84,128)`, FP16, with 65 visible positions:

```text
+----------------------+-----------------+----------------+
| Path                 | Eager median us | Graph p50 us   |
+----------------------+-----------------+----------------+
| repeat K/V + SDPA    |         161.792 |         30.720 |
| Triton GQA, 4 warps  |          46.080 |         15.360 |
| Triton GQA, 8 warps  |          45.056 |         15.360 |
+----------------------+-----------------+----------------+
```

The attention output had `0.00003946` MAE and `0.00048828` maximum absolute error against SDPA. A matched Systems
capture of ten calls per path exposed the GPU-only difference:

```text
+-------------------------------+------------------+-------------------+
| Boundary component            | Kernels / call   | GPU us / call     |
+-------------------------------+------------------+-------------------+
| repeated-K/V FMHA             |                1 |           11.3888 |
| K/V repeat materialization    |                2 |            8.9408 |
| boolean-mask fills            |                3 |            5.1199 |
| boolean-mask conversion       |                1 |            2.5055 |
| reference total               |                7 |           27.9550 |
| Triton GQA, 4 warps           |                1 |            5.8623 |
| Triton GQA, 8 warps           |                1 |            5.5968 |
+-------------------------------+------------------+-------------------+
```

Eight warps are retained. The 0.266 us Systems advantage is small, but both choices are numerically identical and
there is no full-request reason to prefer the slower launch.

### Nsight Compute characterization

The production eight-warp kernel was captured for one full 51-pass Nsight Compute metric set:

```text
+--------------------------------+------------------+
| Metric                         | Value            |
+--------------------------------+------------------+
| Instrumented duration          | 7.74 us          |
| Grid / block                   | 8 / 256 threads  |
| Registers per thread           | 64               |
| Dynamic shared memory / block  | 36.86 KiB        |
| Local-memory spills            | 0                |
| Theoretical occupancy          | 50.00%           |
| Achieved occupancy             | 12.57%           |
| Waves per SM                   | 0.02             |
| L1/TEX throughput              | 42.11%           |
| DRAM throughput                | 1.90%            |
| SM throughput                  | 0.73%            |
+--------------------------------+------------------+
```

The low device-wide utilization is caused by the intentionally tiny eight-block grid, not spilling or DRAM
saturation. Doubling work merely to create more blocks has little room below the roughly 5.6 us uninstrumented
launch. The full-request result, rather than Nsight's local small-grid advisory, is the selection criterion.

### Full-model correctness and AB/BA

Two separately captured CUDA graphs toggled only the model-local GQA attention forwards; both retained the exact
rotary kernel and every earlier Prism optimization. Each AB/BA cell used ten warmups and 40 individually synchronized
64-prompt/20-generated-token requests:

```text
+---------+--------------------+-----------+-----------+-----------+
| Round   | Attention path     | p50 ms    | p95 ms    | tok/s     |
+---------+--------------------+-----------+-----------+-----------+
| A       | repeat K/V + SDPA  |   80.7076 |   80.9463 |    247.81 |
| A       | Triton GQA         |   70.4343 |   70.6370 |    283.95 |
| B       | Triton GQA         |   70.4256 |   70.5068 |    283.99 |
| B       | repeat K/V + SDPA  |   80.7209 |   80.8437 |    247.77 |
+---------+--------------------+-----------+-----------+-----------+
```

The averaged median falls from `80.7142` to `70.4300` ms: 12.74% lower latency and a 1.146x speedup. The sampled
decode logits had `0.00186154` MAE and `0.015625` maximum absolute error, while the full `(1,84)` generated tensor
remained exact and ended in token `63`.

A broader correctness sweep independently ran both paths over four prompts and all 76 fused decode steps:

```text
+--------+-------------+----------------+--------------+-------------+
| Prompt | Logits MAE  | Max abs error  | Tokens exact | Final token |
+--------+-------------+----------------+--------------+-------------+
|      0 |  0.00132839 |     0.01562500 | yes          |          63 |
|      1 |  0.00124979 |     0.00976562 | yes          |         195 |
|      2 |  0.00162221 |     0.01562500 | yes          |        1119 |
|      3 |  0.00143974 |     0.01562500 | yes          |         195 |
+--------+-------------+----------------+--------------+-------------+
| Total  |  0.00141003 |     0.01562500 | yes          |           - |
+--------+-------------+----------------+--------------+-------------+
```

The model-local interface wrapper caches its resolved upstream interface and tests shape before dtype or device
metadata, keeping unsupported paths cheap. A dynamic-cache AB/BA, where every GQA call must fall back, remained
token-exact and averaged `558.6427` ms without the wrapper versus `549.6578` ms with it (`-1.61%`). The individual
dynamic cells varied substantially, so this is treated as evidence of no reproducible fallback regression rather
than a dynamic-path speedup.

### Production latency, VRAM, and launch count

The final 20-iteration automatic-loader run measured:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Fixed prefill + graph |     70.5029 |     70.7602 |    283.68 |       1347.39 |       1726.00 |  1356.88 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

Relative to the retained rotary production row, this is 12.60% lower latency and 14.42% more throughput. Relative
to the first Transformers 5.14.1 production baseline (`93.3028` ms, `214.36` tokens/s), the combined retained work
is 24.44% lower latency and 32.34% more throughput. Allocated and reserved VRAM are unchanged; peak differs by less
than 0.4 MiB. The GQA path adds no persistent cache or weight storage.

The final graph-node trace confirms the launch removal over 19 decode tokens:

```text
+------------------------------+----------------+----------------+-------------+
| Decode metric, 19 tokens     | Rotary only    | Rotary + GQA   | Change      |
+------------------------------+----------------+----------------+-------------+
| Kernel nodes / token         |            665 |            497 | -25.26%     |
| Kernel instances             |         12,635 |          9,443 | -3,192      |
| Kernel GPU time              |   80.551146 ms |   69.108049 ms | -14.21%     |
| D2D copies                   |            551 |             19 | -532        |
| Memsets                      |             19 |             19 | unchanged   |
+------------------------------+----------------+----------------+-------------+
```

Each layer replaces seven old kernel nodes with one GQA kernel, removing six kernels per layer or 168 kernel nodes
per token. Writing the final output layout directly removes one more D2D node per layer, for 196 fewer GPU graph
nodes per token in total. Prefill remains on Transformers and is stable at 1,085 kernel nodes and `13.000336` ms.

```text
+-------------------------------------------+--------+------------+----------+
| Post-GQA decode kernel family             | Calls  | Total ms   | Share    |
+-------------------------------------------+--------+------------+----------+
| _prism_q2_swiglu_gemv_kernel             |    532 |  19.793142 |   28.64% |
| _gguf_q2_0_native_gemv_kernel_impl       |  1,064 |  17.512560 |   25.34% |
| gemv2T_kernel_val                         |     19 |   6.291147 |    9.10% |
| _prism_q2_qkv_gemv_kernel                |    532 |   6.204284 |    8.98% |
| _prism_q2_rms_norm_kernel                 |  2,147 |   5.107087 |    7.39% |
| vectorized_elementwise_kernel             |  2,299 |   4.541015 |    6.57% |
| index_elementwise_kernel                  |  1,064 |   4.123828 |    5.97% |
| _prism_q2_gqa_attention_kernel            |    532 |   3.092598 |    4.48% |
| _prism_q2_fused_rotary_kernel             |    532 |   0.988925 |    1.43% |
| elementwise_kernel_with_index             |    608 |   0.937463 |    1.36% |
+-------------------------------------------+--------+------------+----------+
```

SwiGLU plus the remaining Q2 projections now consume 53.98% of decode kernel time. The next bounded graph target is
the static-cache update boundary: two `index_copy_` launches per layer plus position bookkeeping account for about
5.06 ms. Fusing those writes with attention would require intercepting the cache before `Cache.update`, so it is
not folded into this change without a dedicated cache-correctness study. The quantized compute kernels remain the
larger long-term target.

### Validation and reproduction

The expanded Torch 2.13 / Transformers 5.14.1 suite completed with 166 passes and one unrelated skip. It includes
the real sm80 GQA kernel, numerical comparison against SDPA, all-false-mask parity, model-local registry isolation,
additive-mask, dynamic-cache, training and CPU/non-Q2 fallbacks, all earlier Prism kernels, HF compatibility,
GGUF/weight-only, and CUDA graph tests. Ruff and `git diff --check` passed.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_prism_q2_rotary.py \
  tests/test_prism_q2_attention.py tests/test_cuda_graph_generate.py

/root/vm314t/bin/ruff check \
  gptqmodel/nn_modules/triton_utils/q2_attention.py \
  gptqmodel/models/loader.py tests/test_prism_q2_attention.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
ncu --set full --profile-from-start off --target-processes all \
  --kernel-name regex:_prism_q2_gqa_attention_kernel --launch-count 1 \
  --force-overwrite \
  --export artifacts/prism_q2_0_20260721/followup_tf514_gqa_attention \
  /root/vm314t/bin/python \
    artifacts/prism_q2_0_20260721/followup_gqa_ncu.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_gqa_fixed_graph_nodes \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill --capture
```

Additional raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_tf514_post_rotary_static_mapping.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_post_rotary_static_mapping.sqlite`
- `artifacts/prism_q2_0_20260721/followup_gqa_sdpa_screen.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_gqa_sdpa_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_gqa_sdpa_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_triton_gqa_attention_screen.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_triton_gqa_bool_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_triton_gqa_bool_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_gqa_ncu.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_gqa_attention.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_gqa_support_probe.py`
- `artifacts/prism_q2_0_20260721/followup_gqa_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_gqa_dynamic_abba.py`
- `artifacts/prism_q2_0_20260721/followup_gqa_correctness_sweep.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_gqa_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_gqa_fixed_graph_nodes.sqlite`

## Transformers 5.14.1 follow-up: fused static-cache decode update

The next investigation retained the same runtime and runtime-probed device: Python 3.14.5t, Torch
2.13.0+cu130, CUDA 13.0, Triton 3.7.1, Transformers 5.14.1, and the NVIDIA PG506-230 (`sm80`, 124 SMs,
102,191,202,304 bytes). No dependency was changed or downgraded.

### Cache boundary attribution and retained design

For each decode layer, Transformers' `StaticLayer.update` launched two `index_copy_` kernels plus three position
bookkeeping kernels. A ten-call Nsight Systems capture measured the isolated boundary as follows:

```text
+--------------------------------+----------------+-------------------+
| Static-cache component         | Kernels / call | GPU us / call     |
+--------------------------------+----------------+-------------------+
| key/value index_copy_          |              2 |            8.9567 |
| cache-position add             |              1 |            1.9648 |
| cumulative-length add_         |              1 |            1.8848 |
| one-element arange             |              1 |            1.6192 |
| Transformers total             |              5 |           14.4255 |
| Triton, 4 warps                |              1 |            2.0960 |
| Triton, 8 warps                |              1 |            2.0992 |
+--------------------------------+----------------+-------------------+
```

The retained kernel reads the device-side cumulative length, writes one `(1,8,1,128)` FP16 K/V pair directly
into the two `(1,8,84,128)` static-cache tensors, synchronizes the block, and advances the length. It needs no
host readback and remains capturable by the existing CUDA graph. Eight warps are retained because the Systems
measurements were effectively tied while the CUDA-event graph screen was consistently lower:

```text
+----------------------+-----------------+----------------+
| Path                 | Eager median us | Graph p50 us   |
+----------------------+-----------------+----------------+
| Transformers update  |          79.872 |         19.456 |
| Triton, 4 warps      |          50.176 |         18.432 |
| Triton, 8 warps      |          49.152 |         17.408 |
+----------------------+-----------------+----------------+
```

The specialization is deliberately local to Prism's static graph runner:

- Installation requires a Qwen3 config with hidden size 2,048, 16 query heads, eight K/V heads, head dimension
  128, an 84-token full-attention `StaticCache`, all 28 already-specialized native Prism Q2 attention layers, and
  a runtime-probed `sm80` CUDA device.
- Execution requires eval mode plus exact contiguous FP16 state/cache shapes on the installed device. Prefill,
  training, alternate shapes or dtypes, sliding/dynamic caches, non-Prism models, CPU, and non-sm80 devices call
  the captured Transformers 5.14.1 `StaticLayer.update` implementation unchanged.
- The installer validates every layer before changing any instance and patches only the two cache objects owned
  by this runner: the production cache and its graph-warmup scratch cache. It does not alter a Transformers class,
  registry, or module global.

An initial implementation retained bound `update` methods on each layer. That formed a self-cycle for the scratch
cache and increased persistent allocation by about 9.2 MiB. It was rejected. The final implementation stores the
unbound upstream method and uses weak references for the layer and model. A regression test destroys the scratch
cache and proves its layer is collected; final allocated, reserved, and peak VRAM all return to the pre-change
values.

### Nsight Compute characterization

One production eight-warp launch was captured with the full 51-pass Nsight Compute metric set:

```text
+--------------------------------+------------------+
| Metric                         | Value            |
+--------------------------------+------------------+
| Instrumented duration          | 3.30 us          |
| Grid / block                   | 1 / 256 threads  |
| Registers per thread           | 32               |
| Dynamic shared memory / block  | 4.10 KiB         |
| Local-memory spills            | 0                |
| Theoretical occupancy          | 100.00%          |
| Achieved occupancy             | 12.19%           |
| Waves per SM                   | 0.00             |
| L1/TEX throughput              | 38.87%           |
| L2 throughput                  | 0.47%            |
| DRAM throughput                | 0.11%            |
| SM throughput                  | 0.05%            |
+--------------------------------+------------------+
```

The low device-wide utilization is the expected consequence of one tiny block for one layer. There are no spills,
register or shared-memory limits, or DRAM pressure to solve. Splitting the 2,048 elements into more launches or
blocks would work against the launch-removal objective; full-request timing remains the selection criterion.

### Full-model latency, VRAM, and graph structure

The robust production run used 20 graph warmups followed by 50 individually synchronized 64-prompt/20-generated-
token requests. Generated tensors and the final token were exact against the pre-cache-fusion path:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| GQA, upstream cache   |     70.5029 |     70.7602 |    283.68 |       1347.39 |       1726.00 |  1356.88 |
| GQA, fused cache      |     65.8662 |     65.9629 |    303.65 |       1347.39 |       1726.00 |  1356.88 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

The retained cache path is 6.58% lower latency and 7.04% higher throughput than the prior GQA production row. The
complete retained Prism work is now 29.41% lower latency and 41.65% higher throughput than the first Transformers
5.14.1 production baseline (`93.3028` ms, `214.36` tokens/s), a 1.417x end-to-end speedup, with no persistent or
peak VRAM increase. A separate cache-only AB/BA averaged `77.2342` versus `67.6078` ms, but its four cells were
too noisy to use as the headline result; the longer production sample and graph-node trace are the decision data.

The post-cache Nsight Systems trace over 19 decode tokens confirms exactly four removed launches per layer:

```text
+------------------------------+----------------+----------------+-------------+
| Decode metric, 19 tokens     | GQA only       | GQA + cache    | Change      |
+------------------------------+----------------+----------------+-------------+
| Kernel nodes / token         |            497 |            385 | -22.54%     |
| Kernel instances             |          9,443 |          7,315 | -2,128      |
| Kernel GPU time              |   69.108049 ms |   63.269126 ms | -8.45%      |
| D2D copies                   |             19 |             19 | unchanged   |
| Memsets                      |             19 |             19 | unchanged   |
+------------------------------+----------------+----------------+-------------+
```

The fused cache kernel itself accounts for 532 calls and `1.212441` ms, or 2.279 us per layer. Prefill remains on
the Transformers path. The final decode attribution is now:

```text
+-------------------------------------------+--------+------------+----------+
| Post-cache decode kernel family           | Calls  | Total ms   | Share    |
+-------------------------------------------+--------+------------+----------+
| _prism_q2_swiglu_gemv_kernel             |    532 |  19.799545 |   31.29% |
| _gguf_q2_0_native_gemv_kernel_impl       |  1,064 |  17.513701 |   27.68% |
| gemv2T_kernel_val                         |     19 |   6.292360 |    9.95% |
| _prism_q2_qkv_gemv_kernel                |    532 |   6.191499 |    9.79% |
| _prism_q2_rms_norm_kernel                 |  2,147 |   5.091347 |    8.05% |
| _prism_q2_gqa_attention_kernel            |    532 |   3.097908 |    4.90% |
| vectorized elementwise kernels            |  1,235 |   2.439869 |    3.86% |
| _prism_q2_static_cache_update_kernel      |    532 |   1.212441 |    1.92% |
| _prism_q2_fused_rotary_kernel             |    532 |   0.987035 |    1.56% |
+-------------------------------------------+--------+------------+----------+
```

SwiGLU plus native Q2 projections now consume 58.98% of decode kernel time. They are the dominant next
investigation; the 9.95% vocabulary GEMV, 9.79% fused QKV projection, and 8.05% RMSNorm are the next bounded
families. The remaining cache work is only 1.92%, so further cache micro-tuning cannot materially move the request.

### Correctness, lifecycle, and reproduction

The cache tests compare prefill fallback and three decode updates bit-for-bit with Transformers 5.14.1, validate
the returned cache aliases and device-side cumulative length, replay the fused update through a CUDA graph, check
the training fallback, reject non-Q2/CPU installation, prevent double installation, and prove scratch-cache
collection. The expanded suite completed with 168 passes and one unrelated skip. Ruff and `git diff --check`
passed.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_prism_q2_rotary.py \
  tests/test_prism_q2_attention.py tests/test_prism_q2_cache.py \
  tests/test_cuda_graph_generate.py

/root/vm314t/bin/ruff check \
  gptqmodel/nn_modules/triton_utils/q2_cache.py \
  gptqmodel/utils/cuda_graph.py tests/test_prism_q2_cache.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
ncu --set full --profile-from-start off \
  --kernel-name regex:_prism_q2_static_cache_update_kernel --launch-count 1 \
  --force-overwrite \
  --export artifacts/prism_q2_0_20260721/followup_tf514_cache_update \
  /root/vm314t/bin/python \
    artifacts/prism_q2_0_20260721/followup_cache_ncu.py

CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_cache_fixed_graph_nodes \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill --capture
```

Additional raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_cache_update_screen.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_cache_update_screen.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_cache_update_screen.sqlite`
- `artifacts/prism_q2_0_20260721/followup_cache_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_cache_ncu.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_cache_update.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_cache_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_cache_fixed_graph_nodes.sqlite`

## 2026-07-22 continued optimization: residual-fused Q2 output projections

The post-cache trace left two native Q2 projections per decoder layer: the `2048 -> 2048` attention output and
the `6144 -> 2048` MLP down projection. Each projection was immediately followed by a generic FP16 residual-add
kernel. This is a dependency-safe fusion boundary: every output program already owns one final hidden element, so
it can load the corresponding residual and produce the same rounded result without cross-program synchronization.

The work retained Transformers 5.14.1, Torch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1 throughout. Performance
profiling used runtime-probed PG506-230 devices with `sm80`, 124 SMs, and 102,191,202,304 bytes. GPUs 6 and 7 were
reserved and were not used for this follow-up.

### Retained model-local implementation

The new kernel preserves the original numerical sequence: reduce the Q2 projection in FP32, round the projection
to FP16, add the FP16 residual in FP32, and round the final result to FP16. Both real projection shapes are exact
against the old native Q2 GEMV followed by PyTorch `add`.

The installation boundary remains deliberately narrow:

- The loader must already have installed native Prism Q2 Q/K/V and SwiGLU paths on an eval-mode Qwen3 model with
  hidden size 2,048, intermediate size 6,144, SiLU, 16 query heads, eight K/V heads, head dimension 128, and all
  layers on one runtime-probed `sm80` device.
- Both output projections must be bias-free, adapter-free `GGUFTritonKernel` Q2_0 modules with exact
  `2048 -> 2048` and `6144 -> 2048` shapes. Batch-one, one-token, contiguous FP16 execution uses the fused path;
  prefill, training, other shapes/dtypes/devices, CPU, non-sm80, non-Q2, and moved modules use the captured original
  methods.
- The installer fingerprints the live decoder forward signature, its exact four accessed submodules, and its two
  residual additions before replacing anything. If a later Transformers Qwen3 forward changes that structure,
  installation returns zero and leaves the upstream implementation untouched.
- Only the 28 model-owned decoder layers and their 56 projection instances are changed. Original methods are stored
  unbound and layer/model references are weak, avoiding the reference-cycle issue found during cache work. No Torch
  or Transformers class, function, registry, or module global is patched.
- A per-context handoff marks whether the projection consumed the pending residual without placing temporary tensors
  on shared modules. Concurrent Python inference contexts cannot see each other's residual. If an unexpected live
  attention/MLP path does not call that projection, the wrapper performs the original outer add; if a lower
  projection gate fails, it runs the upstream projection plus residual. Both cases preserve correctness rather than
  silently dropping the residual.

### Exact real-model correctness

The unit path compares both projection shapes, a complete decoder layer, direct projection fallback, disabled-path
fallback, two-token prefill, eval/training gates, and CUDA graph replay bit-for-bit. A separate real-model dynamic-
cache sweep disabled and enabled only the residual specialization across four prompts and all 80 returned score
tensors:

```text
+--------+-------------+---------------+--------------+-------------+
| Prompt | Logits MAE  | Max abs error | Tokens exact | Final token |
+--------+-------------+---------------+--------------+-------------+
|      0 |  0.00000000 |    0.00000000 | yes          |          63 |
|      1 |  0.00000000 |    0.00000000 | yes          |       10161 |
|      2 |  0.00000000 |    0.00000000 | yes          |         211 |
|      3 |  0.00000000 |    0.00000000 | yes          |       17133 |
+--------+-------------+---------------+--------------+-------------+
| Total  |  0.00000000 |    0.00000000 | yes          |           - |
+--------+-------------+---------------+--------------+-------------+
```

The dynamic path is host-dominated and noisy: an AB/BA averaged `638.2857` ms disabled and `636.8082` ms enabled
(`-0.23%`). This is treated as evidence of no fallback/interface regression, not a dynamic-generation speedup.

### Uncontended graph AB/BA and production result

The first graph AB/BA completed before another long-running CUDA workload began holding contexts and about 68 GiB
on the original profiling device. It captured separate baseline and candidate graphs, changing only the residual
fusion. Each cell used ten warmups and 40 individually synchronized 64-prompt/20-generated-token requests:

```text
+---------+--------------------+-----------+-----------+-----------+
| Round   | Residual path      | p50 ms    | p95 ms    | tok/s     |
+---------+--------------------+-----------+-----------+-----------+
| A       | separate add       |   65.8621 |   66.1314 |    303.66 |
| A       | fused Q2 GEMV      |   62.2940 |   62.4030 |    321.06 |
| B       | fused Q2 GEMV      |   62.3063 |   62.4364 |    320.99 |
| B       | separate add       |   66.5324 |   67.7854 |    300.61 |
+---------+--------------------+-----------+-----------+-----------+
```

The averaged median falls from `66.1972` to `62.3002` ms: 5.89% lower latency and a 1.063x speedup. Generated
tensors are exact and end in token `63`.

A later production run on idle, runtime-probed GPU 2 used 20 warmups and 50 synchronized requests. Its median is
stable with the uncontended candidate, while intermittent system activity still created unrelated upper-tail
outliers; the contaminated p95 is deliberately not reported:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Cache fusion only     |     65.8662 |     65.9629 |    303.65 |       1347.39 |       1726.00 |  1356.88 |
| + residual fusion     |     62.1307 | contended   |    321.90 |       1347.39 |       1726.00 |  1356.88 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

The final median is 5.67% lower and throughput is 6.01% higher than the retained cache row. Relative to the first
Transformers 5.14.1 production baseline (`93.3028` ms, `214.36` tokens/s), all retained work is now 33.41% lower
latency and 50.17% higher throughput, a 1.502x end-to-end speedup. Live, reserved, and peak VRAM are unchanged.

### Graph-node attribution

The post-fusion Systems node trace confirms exactly two removed residual kernels per layer. Absolute cross-run
kernel time is shown only as attribution because this capture overlapped the external CUDA workload; node counts
and graph structure are unaffected:

```text
+------------------------------+----------------+----------------+-------------+
| Decode metric, 19 tokens     | Cache only     | + residual     | Change      |
+------------------------------+----------------+----------------+-------------+
| Kernel nodes / token         |            385 |            329 | -14.55%     |
| Kernel instances             |          7,315 |          6,251 | -1,064      |
| Observed kernel GPU time     |   63.269126 ms |   51.766284 ms | attribution |
| D2D copies                   |             19 |             19 | unchanged   |
| Memsets                      |             19 |             19 | unchanged   |
+------------------------------+----------------+----------------+-------------+
```

Prefill remains on the original decoder path and stays at 1,085 kernel nodes. The fused projection family has 1,064
calls over 19 decode tokens. Its two exact 532-call clusters are `3.644884` ms total / `6.851` us average for the
attention output and `9.706403` ms / `18.245` us for the MLP down projection.

```text
+-------------------------------------------+--------+------------+----------+
| Post-residual decode kernel family        | Calls  | Total ms   | Share    |
+-------------------------------------------+--------+------------+----------+
| _prism_q2_swiglu_gemv_kernel             |    532 |  17.144369 |   33.12% |
| _prism_q2_residual_gemv_kernel           |  1,064 |  13.351287 |   25.79% |
| gemv2T_kernel_val                         |     19 |   6.026512 |   11.64% |
| _prism_q2_qkv_gemv_kernel                |    532 |   5.308912 |   10.26% |
| _prism_q2_rms_norm_kernel                 |  2,147 |   4.404166 |    8.51% |
| _prism_q2_gqa_attention_kernel            |    532 |   2.707994 |    5.23% |
| _prism_q2_static_cache_update_kernel      |    532 |   1.051539 |    2.03% |
| _prism_q2_fused_rotary_kernel             |    532 |   0.864795 |    1.67% |
| vectorized elementwise kernels            |    171 |   0.334935 |    0.65% |
+-------------------------------------------+--------+------------+----------+
```

Quantized SwiGLU, fused residual projections, and QKV now comprise 69.17% of observed decode kernel time. The next
bounded launch-removal candidate is the 11.64% dense vocabulary GEMV followed immediately by argmax in the explicit
greedy graph runner. Any future experiment should remain runner-local because general model callers require logits.

### Nsight Compute characterization

Both production shapes were captured with the full 51-pass metric set. The instrumented durations are not used as
request timing, but the resource data rules out spills, register pressure, shared-memory pressure, and DRAM
saturation:

```text
+--------------------------------+------------------+------------------+
| Metric                         | 2048 -> 2048     | 6144 -> 2048     |
+--------------------------------+------------------+------------------+
| Instrumented duration          | 8.93 us          | 22.82 us         |
| Grid / block                   | 2048 / 64        | 2048 / 64        |
| Registers per thread           | 32               | 32               |
| Dynamic shared memory / block  | 8 bytes          | 8 bytes          |
| Local-memory spills            | 0                | 0                |
| Theoretical occupancy          | 100.00%          | 100.00%          |
| Achieved occupancy             | 38.14%           | 43.51%           |
| Waves per SM                   | 0.52             | 0.52             |
| L1/TEX throughput              | 28.87%           | 23.67%           |
| DRAM throughput                | 5.62%            | 8.38%            |
| SM throughput                  | 31.04%           | 34.05%           |
| Long-scoreboard stall share    | 49.33%           | 71.50%           |
+--------------------------------+------------------+------------------+
```

The longer projection remains dependency-latency-bound. The earlier one/two/four/eight-warp full-model study
already found the current two-warp geometry fastest when 28 distinct weight tensors are used; the local metric
advisories do not justify reopening the rejected one-warp configuration.

### Validation and reproduction

The final expanded suite completed on runtime-probed physical GPU 5 (`PG506-230`, `sm80`, 124 SMs,
102,191,202,304 bytes) with
170 passes and one unrelated skip. It includes exact residual projections and decoder output, live-forward
fingerprint rejection, direct/disabled/prefill/training fallbacks, CUDA graph replay, all earlier Prism kernels,
Transformers 5.14 compatibility, GGUF/weight-only, and dense CUDA graph generation. Ruff and `git diff --check`
passed.

```bash
CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_prism_q2_rotary.py \
  tests/test_prism_q2_attention.py tests/test_prism_q2_cache.py \
  tests/test_prism_q2_residual.py tests/test_cuda_graph_generate.py

/root/vm314t/bin/ruff check \
  gptqmodel/nn_modules/triton_utils/q2_residual.py \
  gptqmodel/models/loader.py tests/test_prism_q2_residual.py

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 20 --iterations 50 \
  --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
ncu --set full --profile-from-start off \
  --kernel-name regex:_prism_q2_residual_gemv_kernel --launch-count 2 \
  --force-overwrite \
  --export artifacts/prism_q2_0_20260721/followup_tf514_residual_gemv \
  /root/vm314t/bin/python \
    artifacts/prism_q2_0_20260721/followup_residual_ncu.py
```

Additional raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_residual_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_residual_production_abba.py`
- `artifacts/prism_q2_0_20260721/followup_residual_correctness_sweep.py`
- `artifacts/prism_q2_0_20260721/followup_residual_ncu.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_residual_gemv.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_residual_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_residual_fixed_graph_nodes.sqlite`

## 2026-07-22 continued optimization: one-warp Q2 SwiGLU

The post-residual trace made the fused gate/up/SwiGLU kernel the largest decode family. Its 6,144-program grid had
been retained at two warps and one stage after the original Torch 2.13 dynamic-decode screen. The now fully fused
static graph changes that decision: one warp performs the same per-output reduction with substantially fewer
executed instructions, and a full-model AB/BA confirms that the lower instruction count outweighs its lower
occupancy.

All new tests and profiles in this section ran only on physical GPU 5, a runtime-probed NVIDIA PG506-230 (`sm80`,
124 SMs, 102,191,202,304 bytes). Physical GPUs 6 and 7 remained reserved and untouched. The environment remained
Python 3.14.5t, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, and Transformers 5.14.1. The retained source change is
only `num_warps=2 -> 1` for the already exact-shape, Prism-only `2048 -> 6144` Q2_0 SwiGLU decode launch; every
existing model, dtype, shape, device, training, prefill, and backend fallback is unchanged.

### Correctness sweep

The one- and two-warp kernels were bit-identical for the random unit and Nsight inputs. Different warp reduction
orders can produce small downstream model differences, so correctness was also checked at the actual generation
boundary. The graph AB/BA retained the exact `(1, 84)` greedy token tensor and final token `63`; its last-step logits
had `0.00118783` MAE and `0.00781250` maximum absolute difference.

A separate dynamic-cache sweep compared all 80 score tensors over four prompts:

```text
+--------+-------------+---------------+--------------+-------------+
| Prompt | Logits MAE  | Max abs error | Tokens exact | Final token |
+--------+-------------+---------------+--------------+-------------+
|      0 |  0.00119068 |    0.01562500 | yes          |          63 |
|      1 |  0.00146696 |    0.01562500 | yes          |       10161 |
|      2 |  0.00125864 |    0.01562500 | yes          |         211 |
|      3 |  0.00133796 |    0.01562500 | yes          |       17133 |
+--------+-------------+---------------+--------------+-------------+
| Total  |  0.00131356 |    0.01562500 | yes          |           - |
+--------+-------------+---------------+--------------+-------------+
```

The focused regression now also captures and replays the fused SwiGLU under a CUDA Graph, changes its input in
place, and compares the replay with the unfused Q2 projection/SwiGLU reference. Prefill continues through the
original Transformers MLP path.

### Full-graph AB/BA and production result

Two independent graphs were captured in one loaded model, differing only in the SwiGLU warp count. Every cell used
ten warmups and 40 individually synchronized 64-prompt/20-generated-token requests:

```text
+---------+----------------+-----------+-----------+-----------+
| Round   | SwiGLU launch  | p50 ms    | p95 ms    | tok/s     |
+---------+----------------+-----------+-----------+-----------+
| A       | two warps      |   62.0570 |   62.1655 |    322.28 |
| A       | one warp       |   59.0664 |   59.2450 |    338.60 |
| B       | one warp       |   59.0863 |   59.2123 |    338.49 |
| B       | two warps      |   62.0964 |   62.2476 |    322.08 |
+---------+----------------+-----------+-----------+-----------+
```

The averaged median falls from `62.0767` to `59.0764` ms: 4.83% lower request latency and a 1.051x speedup. The
clean automatic-loader production run used 20 warmups and 50 synchronized requests on idle GPU 5:

```text
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                  | CUDA p50 ms | CUDA p95 ms | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Two-warp SwiGLU       |     62.1307 | contended   |    321.90 |       1347.39 |       1726.00 |  1356.88 |
| One-warp SwiGLU       |     59.0674 |     59.2384 |    338.60 |       1347.39 |       1726.00 |  1356.88 |
+-----------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

This production median is 4.93% lower and throughput is 5.19% higher than the retained residual-fusion row. Relative
to the first Transformers 5.14.1 production baseline (`93.3028` ms, `214.36` tokens/s), the cumulative retained work
is now 36.69% lower latency and 57.96% higher throughput, a 1.580x end-to-end speedup. Live, reserved, and peak VRAM
are unchanged. Graph capture took `1611.2639` ms once; the first post-capture request took `59.5258` ms.

### Nsight Compute explanation

The full 51-pass metric set captured the two-warps/one-stage launch first and the one-warp/one-stage launch second on
the same synthetic production-shape tensors. Outputs were bit-identical. The result explains why selecting by
occupancy alone would be wrong:

```text
+-----------------------------------+------------------+------------------+
| Metric                            | Two warps        | One warp         |
+-----------------------------------+------------------+------------------+
| Instrumented duration             | 37.824 us        | 30.336 us        |
| Grid / block                      | 6,144 / 64       | 6,144 / 32       |
| Executed instructions             | 14,954,496       | 8,429,568        |
| Registers per thread              | 30               | 42               |
| Local-memory spills               | 0                | 0                |
| Theoretical occupancy             | 100.00%          | 50.00%           |
| Achieved occupancy                | 71.20%           | 36.42%           |
| Waves per SM                      | 1.55             | 1.55             |
| Compute throughput                | 66.15%           | 46.54%           |
| DRAM throughput                   | 7.68%            | 9.58%            |
| Measured memory throughput        | 187.64 GB/s      | 233.91 GB/s      |
| Warp cycles / issued instruction  | 15.44            | 10.90            |
| Long-scoreboard cycles / issue    | 5.49             | 6.49             |
+-----------------------------------+------------------+------------------+
```

The one-warp launch cuts executed instructions by 43.63% and instrumented duration by 19.80%. It has fewer active
warps and a larger long-scoreboard share, but also eliminates the cross-warp reduction barrier, greatly reduces
math-pipe/not-selected pressure, and completes the fixed 1.55 waves sooner. There are no spills or persistent
buffers. The final Systems trace retains exactly 329 decode kernel nodes per token: launch structure is unchanged.
Across the 19 decode replays, the one-warp family takes `15.507713` ms over 532 calls (`29.150` us average), 9.55%
less than the separate post-residual trace despite slower timings for every other major family in the new capture.
The same-run AB/BA and Compute replay, rather than that cross-trace delta, support the retained latency claim.

### Dense vocabulary head investigation

The next named family was also bounded before changing code. The tied dense FP16 vocabulary head is
`151669 x 2048`, or 592.46 MiB. In the real post-residual node trace, its cuBLAS GEMV averaged `317.326` us over 20
calls and streamed the 621,236,224-byte matrix at about 1.958 TB/s effective bandwidth. The following argmax was
only `10.659` us per call. An isolated GPU-5 graph probe likewise put the complete head-plus-argmax near the memory-
streaming limit.

A Triton direct-greedy implementation would still have to read the entire dense matrix and perform a second-stage
global winner reduction. It would therefore keep the current two-launch structure while replacing a near-bandwidth-
ceiling cuBLAS GEMV. Avoiding the FP16 logits buffer has only a 0.29 MiB VRAM ceiling, and removing all measured
argmax time has less than a 0.4% request-latency ceiling. Runtime quantization of the tied embedding/head could
change that cost, but it changes model weights and accuracy rather than only the inference kernel. No vocabulary
head change is retained.

### Validation and reproduction

The final expanded suite completed on physical GPU 5 with 170 passes and one unrelated skip. It covers the new
one-warp CUDA Graph replay, all prior Prism kernels, Transformers 5.14 compatibility, GGUF/weight-only behavior, and
dense graph generation. Ruff and `git diff --check` passed.

```bash
CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python -m pytest -q \
  tests/test_hf_config_compat.py tests/test_internal_gguf.py \
  tests/test_weight_only_config.py tests/test_weight_only.py \
  tests/test_prism_q2_qkv.py tests/test_prism_q2_swiglu.py \
  tests/test_prism_rms_norm.py tests/test_prism_q2_rotary.py \
  tests/test_prism_q2_attention.py tests/test_prism_q2_cache.py \
  tests/test_prism_q2_residual.py tests/test_cuda_graph_generate.py

/root/vm314t/bin/ruff check \
  gptqmodel/nn_modules/triton_utils/q2_swiglu.py tests/test_prism_q2_swiglu.py

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 20 --iterations 50 \
  --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
ncu --set full --profile-from-start off \
  --kernel-name regex:_prism_q2_swiglu_gemv_kernel --launch-count 2 \
  --force-overwrite \
  --export artifacts/prism_q2_0_20260721/followup_tf514_swiglu_w2_w1 \
  /root/vm314t/bin/python \
    artifacts/prism_q2_0_20260721/followup_swiglu_warp_ncu.py

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_swiglu_w1_fixed_graph_nodes \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill --capture
```

Additional raw local artifacts, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_vocab_head_probe.py`
- `artifacts/prism_q2_0_20260721/followup_swiglu_w1_graph_abba.py`
- `artifacts/prism_q2_0_20260721/followup_swiglu_w1_correctness_sweep.py`
- `artifacts/prism_q2_0_20260721/followup_swiglu_warp_ncu.py`
- `artifacts/prism_q2_0_20260721/followup_tf514_swiglu_w2_w1.ncu-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_swiglu_w1_fixed_graph_nodes.nsys-rep`
- `artifacts/prism_q2_0_20260721/followup_tf514_swiglu_w1_fixed_graph_nodes.sqlite`

### Post-retune output-tiling screen

One final bounded screen tested whether a program should compute two or four adjacent SwiGLU outputs and reuse each
128-element activation load. Every configuration was bit-identical to the retained one-output/one-warp kernel. The
screen used 100 warmups followed by 20 synchronized batches of 100 launches on GPU 5; it is a cache-hot kernel
screen, not an end-to-end latency claim:

```text
+------------------+-------+-----------+--------------------+
| Outputs / program | Warps | p50 us    | Change vs retained |
+------------------+-------+-----------+--------------------+
|                1 |     1 |   29.0816 | baseline           |
|                2 |     1 |   42.5677 | +46.37%            |
|                2 |     2 |   46.0186 | +58.24%            |
|                2 |     4 |   33.5821 | +15.48%            |
|                4 |     2 |   33.9405 | +16.71%            |
|                4 |     4 |   38.4461 | +32.20%            |
|                4 |     8 |   39.5469 | +35.99%            |
+------------------+-------+-----------+--------------------+
```

Even the best tiled launch is 15.48% slower. Sharing the already cache-resident activation does not compensate for
the larger accumulator state and lower CTA-level parallelism while the two Q2 weight rows still have to be decoded
in full. Output tiling is rejected without spending another full-model capture. The remaining residual projection
family also already has a full-model one-warp rejection: the `6144 -> 2048` path regressed 5.95%, while the earlier
representative `2048 -> 2048` study rejected one warp as well. Those launch geometries remain at two warps.

The next plausible material boundary is therefore cross-projection persistence: compute SwiGLU, synchronize, then
consume it in the down projection without returning to the host launch stream. That requires a cooperative/persistent
CUDA design with a true grid-wide synchronization point; the current 6,144-CTA Triton layout cannot safely provide
one. It is left as a future CUDA-kernel investigation rather than introducing an unsafe in-place dependency.

Additional raw local artifact, intentionally excluded from Git:

- `artifacts/prism_q2_0_20260721/followup_swiglu_output_tile.py`

## 2026-07-22 single-GPU context capacity and prefill/decode scaling

The current Prism Q2 eager path was measured separately for parallel prefill and single-token autoregressive decode.
The run used commit `b3e49249`, the native Q2_0 GGUF checkpoint
`/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf`, batch 1, FP16 activations and KV cache,
`logits_to_keep=1`, and `GGUFTritonKernel` at 2.125 estimated bits per weight. The loader installed the current
Prism-only RMSNorm, QKV, rotary, GQA attention, residual-decoder, and one-warp SwiGLU kernels.

Only physical GPU 5 was used: runtime probing reported an NVIDIA PG506-230 at PCI `00000000:A5:00.0`, `sm80`, 124
SMs, and 102,191,202,304 bytes of CUDA-visible memory. Physical GPUs 6 and 7 remained reserved and untouched. The
software environment was Python 3.14.5t, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, and Transformers 5.14.1; no
dependency was downgraded. The loaded configuration reports `max_position_embeddings=32768`. The source GGUF
metadata identifies a 32,768-token context with YaRN factor 4 from an original 8,192-token context.

Each shape received one full-prefill warmup followed by two CUDA-event-timed prefills. The last timed dynamic cache
was retained for two decode warmups and 20 CUDA-event-timed decode tokens. The table therefore compares parallel
prompt throughput with sequential token throughput; their tokens/s values describe different execution regimes and
should not be interpreted as equal-latency work.

```text
+---------+----------------+---------------+-------------+---------------+--------------+---------------+-----------+--------------+----------+-------+
| context | prefill p50 ms | prefill tok/s | host p50 ms | decode p50 ms | decode tok/s | decode p95 ms | alloc MiB | reserved MiB | peak MiB | valid |
+---------+----------------+---------------+-------------+---------------+--------------+---------------+-----------+--------------+----------+-------+
| 128     |         48.386 |       2645.39 |      48.875 |        28.222 |        35.43 |        31.249 |   1344.15 |      1704.00 |  1350.98 | yes   |
| 512     |         53.415 |       9585.25 |      53.537 |        26.730 |        37.41 |        31.813 |   1393.40 |      1760.00 |  1431.38 | yes   |
| 2,048   |        112.186 |      18255.42 |     112.312 |        25.656 |        38.98 |        29.548 |   1585.41 |      1960.00 |  1658.14 | yes   |
| 8,192   |        520.081 |      15751.40 |     520.239 |        26.068 |        38.36 |        27.517 |   2353.46 |      2790.00 |  2647.23 | yes   |
| 32,768  |       2840.582 |      11535.67 |    2840.838 |        26.439 |        37.82 |        30.475 |   5425.65 |      7526.00 |  6593.61 | yes   |
+---------+----------------+---------------+-------------+---------------+--------------+---------------+-----------+--------------+----------+-------+
```

All configured lengths completed with finite logits and the expected cache length. Thus the largest supported and
fully prefetched context tested is the model's complete 32,768-token configured window; it does not OOM on this GPU.
At that boundary, prefill takes 2.841 seconds at 11,535.67 prompt tokens/s, while steady dynamic-cache decode takes
26.439 ms/token at 37.82 tokens/s. Prefill throughput peaks at 18,255.42 tokens/s for the 2,048-token case. Decode
remains between 35.43 and 38.98 tokens/s across the configured sweep, while long-prefill throughput declines after
2K as attention work grows. The 32K live allocation is 5,425.65 MiB and the measured peak is 6,593.61 MiB.

### Memory-only cache/decode OOM boundary

A second probe answers the narrower hardware-capacity question beyond the model's supported context. It creates a
full Transformers 5.14 `StaticCache` for all 28 layers, marks it occupied, and executes one real Prism model decode
at the last position. This exercises the loaded model, current Q2 kernels, FP16 KV allocation, and a full-length
attention read. It does **not** prefill real tokens, validate positional quality, or enlarge the model's trained or
configured context; lengths beyond 32,768 are memory-capacity diagnostics only. One FP16 K+V token consumes 114,688
bytes (112 KiB) for 28 layers, eight KV heads, and head dimension 128.

```text
+----------------+--------+-----------+-----------------+----------+----------+
| cache capacity | status | decode ms | cache alloc MiB | peak MiB | free MiB |
+----------------+--------+-----------+-----------------+----------+----------+
| 32,768         | pass   |    97.892 |         3584.30 |  5681.24 | 91134.31 |
| 65,536         | pass   |   128.097 |         7168.30 |  9521.33 | 87102.31 |
| 131,072        | pass   |   252.998 |        14336.30 | 17201.52 | 79294.31 |
| 262,144        | pass   |   500.577 |        28672.30 | 32561.89 | 63678.31 |
| 524,288        | pass   |   996.170 |        57344.30 | 63282.64 | 32958.31 |
| 1,048,576      | OOM    |         - |               - |        - |        - |
| 786,432        | pass   |  1599.972 |        86016.30 | 94003.52 |  2238.31 |
| 917,504        | OOM    |         - |               - |        - |        - |
| 851,968        | OOM    |         - |               - |        - |        - |
| 819,200        | OOM    |         - |               - |        - |        - |
| 802,816        | pass   |  1647.739 |        87808.30 | 95923.53 |   318.31 |
| 811,008        | OOM    |         - |               - |        - |        - |
| 806,912        | OOM    |         - |               - |        - |        - |
+----------------+--------+-----------+-----------------+----------+----------+
```

The maximum passing memory-capacity trial is 802,816 tokens and the first failing trial is 806,912 tokens, bounding
the one-GPU full-cache/decode ceiling to `[802816, 806912)` at 4,096-token resolution. The passing cache alone uses
87,808.30 MiB; total peak allocation reaches 95,923.53 MiB and only 318.31 MiB remains free. These single, unwarmed
static-cache decode samples characterize the capacity probe and should not be compared directly with the warmed
dynamic-cache decode p50 values above. After cleanup, GPU 5 returned to 0 MiB used and 0% utilization.

### Reproduction and validation

The reusable benchmark is `scripts/profile_prism_q2_context.py`. Ruff, Python byte-compilation, `git diff --check`,
the complete real-model speed sweep, and the OOM search passed. The script catches and clears expected allocator OOMs
between binary-search trials.

```bash
CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_context.py

/root/vm314t/bin/ruff check scripts/profile_prism_q2_context.py
/root/vm314t/bin/python -m py_compile scripts/profile_prism_q2_context.py
git diff --check
```

## 2026-07-22 eager decode bottleneck: graph dispatch, not a mega-kernel

The 35--39 token/s context-sweep result is the eager dynamic-cache path, not the optimized production graph path.
A fresh decode-only Systems capture at commit `b8e61b84` isolated four warmed tokens after a 128-token prefill. It
used the same Q2_0 checkpoint, batch one, FP16 activations, dynamic KV cache, and current Prism kernel suite as the
context sweep. The profiler range excludes checkpoint load, JIT, allocator growth, and prefill.

Only physical GPU 5 was used. Runtime probing reported an NVIDIA PG506-230 at PCI `00000000:A5:00.0`, `sm80`, 124
SMs, and 102,191,202,304 bytes. GPU 5 was otherwise idle; physical GPUs 6 and 7 remained untouched. The environment
remained Python 3.14.5t, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, Transformers 5.14.1, Nsight Systems 2024.6.2,
and Nsight Compute 2025.3.1. No dependency was changed or downgraded.

### Eager decode attribution

`nsys stats` reports 1,520 GPU kernels across the four-token range, or 380 per token. Their summed GPU execution is
only 13.266926 ms, while the first-to-last GPU span is 123.439599 ms. Actual kernel work is therefore 3.316732
ms/token and the uncovered dispatch spacing is 27.543168 ms/token. The Systems utilization rule independently
reports 10.8% GPU in-use. CUDA API attribution contains 1,016 `cuLaunchKernelEx` plus 500 `cudaLaunchKernel` calls;
the GPU is waiting for Python/PyTorch dispatch rather than a slow individual kernel.

```text
+------------------------------+--------------------------------------+-------+----------+--------+-----------+
| Kernel family                | Source                               | Calls | Total ms | Share  | Calls/tok |
+------------------------------+--------------------------------------+-------+----------+--------+-----------+
| Prism Q2 SwiGLU              | triton_utils/q2_swiglu.py            |   112 | 3.288899 | 24.79% |        28 |
| Prism Q2 residual GEMV       | triton_utils/q2_residual.py          |   224 | 3.172707 | 23.91% |        56 |
| Dense vocabulary GEMV        | Qwen3 LM head / cuBLAS               |     4 | 1.327030 | 10.00% |         1 |
| Prism Q2 QKV                 | triton_utils/q2_qkv.py               |   112 | 1.301079 |  9.81% |        28 |
| Prism RMSNorm                | triton_utils/rms_norm.py             |   452 | 1.174425 |  8.85% |       113 |
| FlashAttention split/combine | PyTorch dynamic attention            |   224 | 1.628973 | 12.28% |        56 |
| Dynamic-cache concatenation  | PyTorch CatArrayBatchedCopy          |   224 | 0.928568 |  7.00% |        56 |
| Prism fused rotary           | triton_utils/q2_rotary.py            |   112 | 0.267871 |  2.02% |        28 |
| Other position/logit helpers | PyTorch / Transformers               |    56 | 0.177374 |  1.34% |        14 |
+------------------------------+--------------------------------------+-------+----------+--------+-----------+
| Total                        |                                      | 1,520 |13.266926 |100.00% |       380 |
+------------------------------+--------------------------------------+-------+----------+--------+-----------+
```

```text
+-------------------------------+---------------+--------------+----------------+
| Four-token eager metric       | Total         | Per token    | Share of span  |
+-------------------------------+---------------+--------------+----------------+
| First-to-last GPU span        | 123.439599 ms | 30.859900 ms |        100.00% |
| Executing GPU kernels         |  13.266926 ms |  3.316732 ms |         10.75% |
| Uncovered host/launch spacing | 110.172673 ms | 27.543168 ms |         89.25% |
+-------------------------------+---------------+--------------+----------------+
```

### Matched production fix

The synchronized, non-profiler benchmark compared 20 warmups and 50 requests for normal dynamic generation and the
already retained `StaticCUDAGraphGreedyRunner`. Both paths used a 64-token prompt, generated 20 greedy tokens, and
returned the exact same `(1, 84)` tensor ending in token 63. The graph captures the fixed-shape prefill and one whole
decode step, including all 28 layers, cache updates, vocabulary head, argmax, token feedback, and position update.

```text
+---------------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Path                      | p50 ms / 20 | p95 ms / 20 | tokens/s  | Allocated MiB | Reserved MiB  | Peak MiB |
+---------------------------+-------------+-------------+-----------+---------------+---------------+----------+
| Dynamic-cache eager       |    513.6926 |    577.9255 |     38.93 |       1329.48 |       1700.00 |  1351.14 |
| Fixed-prefill CUDA Graph  |     58.9629 |     59.0751 |    339.20 |       1347.39 |       1724.00 |  1356.89 |
+---------------------------+-------------+-------------+-----------+---------------+---------------+----------+
```

Graph dispatch lowers matched request latency by 88.52% and improves throughput 8.712x. Its live-allocation cost is
17.91 MiB, with 24 MiB more reserved and 5.75 MiB more measured peak allocation. This is the applicable launch-
bottleneck fix. It remains explicit because its valid boundary is batch-one greedy Qwen3, FP16, full attention,
fixed cache capacity and tensor addresses, one stream, and no concurrent use. Sampling, arbitrary masks, beam search,
streaming, and unsupported devices remain on `model.generate`.

### Follow-up graph trace

A matched child-node trace captured one warmed fixed-prefill request plus 19 decode replays. Nsight instrumentation
expanded its wall time, so synchronized CUDA events above remain the latency authority. Within the trace, 7,338
child kernels consume 67.273618 ms inside a 69.921768 ms first-to-last GPU span. Copies add 0.083519 ms and memsets
0.041088 ms, for 67.398225 ms of GPU work and 96.39% GPU in-use. Only 2.523543 ms remains uncovered across the
entire instrumented request, versus 110.172673 ms across just four eager tokens.

```text
+-----------------------------------+-------+-----------+--------+------------------------------------+
| Whole-request graph family        | Calls | Total ms  | Share  | Scope                              |
+-----------------------------------+-------+-----------+--------+------------------------------------+
| Prism Q2 SwiGLU                   |   532 | 15.507848 | 23.1%  | 28 x 19 decode steps               |
| Prism Q2 residual GEMV            | 1,064 | 14.967757 | 22.2%  | 56 x 19 decode steps               |
| Q2 prefill fused matmul           |   196 |  9.463708 | 14.1%  | captured prefill only              |
| Dense vocabulary GEMV             |    20 |  6.622703 |  9.8%  | prefill plus 19 decode steps       |
| Prism Q2 QKV                      |   532 |  6.148081 |  9.1%  | 28 x 19 decode steps               |
| Prism RMSNorm                     | 2,260 |  5.428756 |  8.1%  | prefill plus decode                |
| Prism GQA attention               |   532 |  3.093321 |  4.6%  | 28 x 19 decode steps               |
| Prism static-cache update         |   532 |  1.211380 |  1.8%  | 28 x 19 decode steps               |
| Prism fused rotary                |   532 |  0.987994 |  1.5%  | 28 x 19 decode steps               |
| Other prefill/decode helpers      |     - |  3.842070 |  5.7%  | position, mask and output helpers  |
+-----------------------------------+-------+-----------+--------+------------------------------------+
| Total kernels                     | 7,338 | 67.273618 |100.0%  | one complete request               |
+-----------------------------------+-------+-----------+--------+------------------------------------+
```

### Mega-kernel decision

A bounded mega-graph experiment captured all 19 dependent decode steps into one graph launch instead of replaying a
one-token graph. It was token-exact and passed repeated dense-model replay, but the real checkpoint measured 58.9896
ms p50 and 339.04 tokens/s, versus 58.9629 ms and 339.20 tokens/s for the retained graph. VRAM was unchanged. The
standalone sequence capture took 7.294 seconds; capture timing is initialization-order dependent, but it is clearly
more expensive and provides no steady-state gain. The experiment was removed.

The two dominant retained kernels were also checked against their existing Compute reports:

```text
+-------------------------------+----------+-------+-------+-----------+-----------+----------------+
| Kernel                        | NCU us   | Grid  | Block | Registers | SM SOL    | Achieved occ.  |
+-------------------------------+----------+-------+-------+-----------+-----------+----------------+
| SwiGLU, retained one warp     |   30.336 | 6,144 |    32 |        42 |    46.54% |         36.42% |
| Residual 2048 -> 2048         |    8.928 | 2,048 |    64 |        32 |    31.04% |         38.14% |
| Residual 6144 -> 2048         |   22.816 | 2,048 |    64 |        32 |    34.05% |         43.51% |
+-------------------------------+----------+-------+-------+-----------+-----------+----------------+
```

These are actual dequantization/GEMV costs once graph dispatch keeps the GPU fed. A true MLP mega-kernel would need
to produce 6,144 independent SwiGLU values, perform a grid-wide barrier, and then consume all of them in each of
2,048 down-projection reductions. Triton has no safe Ampere grid-wide synchronization for that producer/consumer
layout. A cooperative CUDA kernel would have to restrict its grid to simultaneously resident blocks, reducing the
parallelism of both already-tuned grids, while still writing the intermediate through global memory. Extending this
across attention or decoder layers adds further global barriers and cannot remove Q2 weight traffic. With the graph
already at 96.39% GPU in-use, that risk is not supported by a measurable dispatch ceiling.

```text
+------------------------------+-----------------------------+--------------------------------+------------------+
| Boundary                     | Dependency                  | Candidate                      | Decision         |
+------------------------------+-----------------------------+--------------------------------+------------------+
| Python -> 380 ops/token      | Host dispatch only          | Whole-token CUDA Graph         | Retain; 8.712x   |
| Decode token N -> N+1        | Argmax token feedback       | Multi-token mega graph         | Reject; no gain  |
| SwiGLU -> down projection    | Full 6,144-value reduction  | Cooperative persistent CUDA    | Reject for sm80  |
| Residual -> following norm   | Full-vector RMS reduction   | Cross-grid layer mega-kernel   | Reject for sm80  |
| Dense head -> argmax         | Full vocabulary winner      | Direct greedy head             | Prior <0.4% cap  |
+------------------------------+-----------------------------+--------------------------------+------------------+
```

### Retained instrumentation and reproduction

`scripts/profile_prism_q2_context.py` now labels its decode columns explicitly as **eager** and has `--capture` to
place only warmed dynamic-decode iterations inside a profiler range. No speculative mega-kernel or sequence graph is
retained. Raw `.nsys-rep`, SQLite, and rejected-screen data remain local under `artifacts/` and outside Git.

```bash
CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,cublas --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_dynamic_decode_current \
  /root/vm314t/bin/python scripts/profile_prism_q2_context.py \
    --lengths 128 --prefill-iterations 1 --decode-warmup 10 \
    --decode-iterations 4 --skip-cache-search --capture

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
/root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
  --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 20 --iterations 50 \
  --paths dynamic graph --graph-warmup 2 --retain-prefill-cache --capture-prefill

CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=1 \
nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,cublas --cuda-graph-trace=node --sample=none \
  --force-overwrite=true \
  --output=artifacts/prism_q2_0_20260721/followup_tf514_graph_decode_current \
  /root/vm314t/bin/python scripts/profile_prism_q2_generate.py \
    --device 0 --prompt-tokens 64 --new-tokens 20 --warmup 1 --iterations 1 \
    --paths graph --graph-warmup 2 --retain-prefill-cache --capture-prefill --capture

nsys stats --force-export=true \
  --report=nvtx_gpu_proj_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum \
  artifacts/prism_q2_0_20260721/followup_tf514_graph_decode_current.nsys-rep

/root/vm314t/bin/ruff check scripts/profile_prism_q2_context.py
/root/vm314t/bin/python -m py_compile scripts/profile_prism_q2_context.py
git diff --check
```
