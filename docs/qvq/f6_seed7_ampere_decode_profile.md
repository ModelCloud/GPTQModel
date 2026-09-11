# F6 Seed-7 QVQ P32 decode profiling

## Verdict

The first decode optimization target is the compiler-materialized per-layer K/V
cache slicing path, not a QVQ arithmetic kernel. At maximum context, the
captured decode step performs 32 explicit 256 MiB device-to-device copies:
one K and one V layer slice for each of 16 transformer layers. They consume
7.932 ms per decode step, or 21.96% of listed GPU activity. XLA HLO, thunk
metadata, buffer assignment, and the Nsight Systems timeline agree on the
operation count and byte geometry.

The next targets are unified attention and the dominant QVQ P32 M1 kernel.
Attention is the largest kernel contributor and is underfilled/dependency
limited in the focused replay. The dominant P32 kernel launches only eight
blocks on a 124-SM GPU and spends substantial executed instruction work on
bit manipulation, address arithmetic, and shared-memory loads.

No production source change is proposed from profiling evidence alone.

## Reproducibility

- Model: `modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`
- QvQ revision: `03c326d1314f707016f5159c9ccfa81dabe78f9a`
- ZML-Ultra revision: `ffac15de3b2bf3d39a038edacf39570dd2913221`
- Native Fast API ABI: version 1
- GPU: NVIDIA PG506-230, Ampere SM80, 124 SMs, 98304 MiB
- GPU UUID: `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`
- Context: 131072 tokens
- Batch: 1
- KV capacity: 8192 pages
- Decode warmups: 3
- Nsight Systems measured samples: 2
- Focused Nsight Compute launch count: 1 per target
- Native library SHA-256:
  `92db84934908746dbad6b9e3b503261ca91bc0b2f473d81df11dfe11d09bc2d8`
- SM80 QVQ cubin SHA-256:
  `fccaab45b58067ecf8c3979a4b1e5dcc1a4849e36121cd82a3350ab5e27917be`
- Full commands and environment:
  `artifacts/ncu/f6_seed7_ampere_decode_20260911/run_targeted_ncu.sh`,
  `artifacts/ncu/f6_seed7_ampere_decode_20260911/zml_native_fast_abi_decode_profile.py`,
  and `artifacts/ncu/f6_seed7_ampere_decode_20260911/idle_gate_exec.py`
- Core artifact hashes:
  `artifacts/ncu/f6_seed7_ampere_decode_20260911/SHA256SUMS.txt`
- Complete 146-file run-directory manifest:
  `artifacts/ncu/f6_seed7_ampere_decode_20260911/SHA256SUMS.full.txt`

The harness overwrites the final KV slot on repeated decode samples so that
the sequence remains at the model's 131072-token limit. GPU execution was
leased by UUID and gated on three consecutive idle samples. The final GPU
state was 0 MiB used, 0% GPU utilization, and 0% memory utilization.

## Nsight Systems attribution

The two-sample trace contains 72.267 ms of listed GPU activity, or 36.133 ms
per sample. The ordinary application benchmark measured approximately
38.8-39.0 ms per maximum-context batch-1 decode step.

```text
+------+--------------------------------------+--------+-----------+----------+
| Rank | GPU activity                         | ms/step| GPU share | launches |
+------+--------------------------------------+--------+-----------+----------+
| 1    | kernel_unified_attention_3d_ptr      | 14.776 | 40.89%    | 16       |
| 2    | QVQ P32 M1 <6,1,128,16,2,0,...,1>   |  8.495 | 23.51%    | 16       |
| 3    | 256 MiB D2D layer-cache copies       |  7.932 | 21.95%    | 32       |
| 4    | p32_rank8_project_kernel<8>          |  0.718 |  1.99%    | 80       |
| 5    | gemm_fusion_dot_10386                |  0.379 |  1.05%    | 16       |
| 6    | minor P32 M1 <6,1,...8192,32,2048,0> |  0.340 |  0.94%    | 16       |
+------+--------------------------------------+--------+-----------+----------+
```

Counts above are per decode sample; the CSV stores totals over two samples.

## K/V copy correlation

Nsight Systems measured 64 exact copies of 268435456 bytes over two decode
samples. That is 32 copies and 8 GiB of copied payload per step. Copy duration
was 240.577-255.968 us, with a 247.878 us mean. The measured copies immediately
follow the cache update kernels on the same CUDA stream.

The XLA decode module independently contains exactly 32 `kCopy` thunks,
annotated `wrapped_slice` through `wrapped_slice.31`. Each wrapped slice:

1. consumes a full updated cache value shaped
   `f16[16,16384,16,8,64]`;
2. slices one layer to `f16[1,16384,16,8,64]`;
3. materializes 268435456 bytes;
4. bitcasts to `f16[16384,16,8,64]`; and
5. becomes the K or V operand of `kernel_unified_attention_3d_ptr`.

The 32 copies therefore correspond to 16 layers times two cache operands.
The HLO even/odd slice pairs feed the attention custom call's
`key_cache_ptr` and `value_cache_ptr` operands.

This proves compiler materialization of the per-layer cache slices. It does
not prove that a copy can be deleted while preserving XLA buffer ownership,
donation, stream ordering, command-buffer capture, and attention ABI
requirements.

## Focused Nsight Compute results

```text
+----------------+----------+---------+---------+-------+------+-------+-------+
| Target         | Duration | Grid    | DRAM %  | SM %  | Regs | Waves | Warps |
+----------------+----------+---------+---------+-------+------+-------+-------+
| Attention      | 1.051 ms | 1x8x16  | 10.73%  |10.93% | 124  | 0.13  | 3.23% |
| P32 M1 main    | 559.8 us | 8x1x1   | 0.00001%| 2.19% |  64  | 0.01  | 6.25% |
| P32 M1 minor   |  10.0 us | 2x1x32  | 0.00053%| 9.23% |  64  | 0.06  | 6.25% |
| Rank-8 project | 150.4 us |1x8192x1 | 22.26%  |39.15% |  31  | 8.26  |93.88% |
+----------------+----------+---------+---------+-------+------+-------+-------+
```

### Attention

- Block: 64 threads.
- Dynamic shared memory: 6.144 KiB/block.
- L2 throughput: 21.19%.
- DRAM rate in replay: 262.422 GB/s.
- Eligible warps: 0.233 per active cycle.
- Issue active: 0.23 per active cycle.
- No measured local or shared spilling.
- Highest raw warp-stall metrics: wait 1.321, long scoreboard 1.283,
  short scoreboard 0.470, barrier 0.173 instructions.

The small grid, 0.13 waves/SM, 124 registers/thread, and low eligible-warp
rate indicate underfill and dependency latency. The measured 10.73% DRAM
throughput does not indicate HBM saturation.

### Dominant P32 M1

- Block: 128 threads.
- Static shared memory: 6.848 KiB/block.
- Eligible warps: 0.342 per active cycle.
- Issue active: 0.34 per active cycle.
- No measured local or shared spilling.
- Highest raw warp-stall metrics: selected 1.000, wait 0.907,
  barrier 0.462, short scoreboard 0.313 instructions.

Eight blocks on 124 SMs are the primary structural limitation. The replay's
near-zero DRAM counter is not used as evidence that the production kernel has
no memory traffic.

### Rank-8 projection

- Block: 256 threads.
- L2 throughput: 19.23%.
- Eligible warps: 1.473 per active cycle.
- Issue active: 0.27 per active cycle.
- Highest raw warp-stall metrics: LG throttle 26.859, long scoreboard
  13.572, MIO throttle 5.248, not selected 4.432 instructions.

This kernel is highly occupied but load/store and dependency-pipeline
constrained. Its smaller end-to-end contribution makes it a lower priority.

## Source-correlated SASS

The dominant P32 launch uses 64 registers/thread, 6848 bytes static shared
memory, no stack, and no local memory. Source correlation mapped 1728
instruction addresses and left 72 unmapped.

Top executed instruction families in the captured source-correlated P32
table:

```text
+--------+----------------------------+
| Opcode | Executed warp instructions |
+--------+----------------------------+
| LOP3   |                  1,786,520 |
| SHF    |                  1,225,536 |
| IMAD   |                  1,214,304 |
| LDS    |                  1,130,496 |
| HADD2  |                    786,432 |
| FFMA   |                    524,288 |
+--------+----------------------------+
```

The most active source regions are:

- line 201: IMAD, LOP3, and LDS, approximately 788k combined;
- line 198: SHF, LOP3, and LDS, approximately 786k combined;
- line 688: HADD2, 786k;
- lines 36-38: SHF/LOP3/IMAD mixing;
- lines 191-192: address and bit-mixing work;
- lines 1130-1136: FFMA accumulation.

These map to window-state extraction, PGC16 mixing/masking, level lookup,
shared-memory traffic, and accumulation. Candidate work is exact
address/bit-state reuse and less repeated decode bookkeeping, not a precision
change.

The rank-8 launch uses 31 registers/thread and no reported stack, local, or
shared memory. All 240 instruction addresses were mapped. Its leading
executed families were 8.389M LDG, 4.194M FFMA, 4.194M HADD2, 1.245M IADD3,
and 1.180M IMAD, consistent with its load-pipeline stall profile.

## Overlap opportunities

```text
+----------------------------------+----------+-------------------------------+
| Region                           | Evidence | Assessment                    |
+----------------------------------+----------+-------------------------------+
| Cache slice D2D vs attention     | Same     | No overlap in captured graph; |
|                                  | stream   | remove materialization first  |
| Per-layer attention vs next MLP  | Serial   | Dependency blocks direct      |
|                                  | graph    | overlap                        |
| Rank-8 project/epilogue          | 80+80    | Launch fusion may reduce small|
|                                  | launches | overhead; low total priority  |
| Tiny XLA concatenate/convert ops | Many     | Aggregate before pursuing;    |
|                                  | launches | each is individually small    |
+----------------------------------+----------+-------------------------------+
```

## Fusion and ABI candidates

```text
+------+----------------------------------+------------------------------------+
| Rank | Candidate                        | Required proof                     |
+------+----------------------------------+------------------------------------+
| 1    | Pass full K/V cache plus layer   | No 256 MiB copies; same outputs,   |
|      | index/offset into attention ABI  | donation, graph and stream safety  |
| 2    | Represent atLayer as a true      | XLA/PJRT alias or sub-buffer proof |
|      | sub-buffer rather than a tensor  | without materialized kCopy         |
| 3    | Increase P32 M1 grid/split       | End-to-end gain after reduction;   |
|      | parallelism                      | localized drift within campaign    |
| 4    | Reuse P32 state/address work     | Matched source SASS and numerical  |
|      | across decoded pairs             | equality/accuracy gates            |
| 5    | Fuse rank-8 project + epilogue   | Launch reduction with measured     |
|      | where layouts permit             | whole-decode gain                  |
+------+----------------------------------+------------------------------------+
```

## Ranked optimization targets

1. **Eliminate or avoid per-layer K/V slice materialization.**
   Measured contribution: 7.932 ms/step. Complete elimination is a
   non-overlap-adjusted upper bound of about 20.4% of the 38.9 ms application
   latency, not a forecast. The likely design direction is an attention ABI
   that accepts the full cache and a layer index or byte offset.
2. **Improve attention launch parallelism and dependency hiding.**
   Measured contribution: 14.776 ms/step. Investigate segmentation, CTA
   geometry, register lifetime, and independent work per CTA while preserving
   paged-attention semantics.
3. **Increase dominant P32 M1 useful parallelism.**
   Measured contribution: 8.495 ms/step. Test more grid/split work only with
   the reduction cost included; then reduce repeated SHF/LOP3/IMAD/LDS work.
4. **Address rank-8 load-pipeline pressure.**
   Measured contribution: 0.718 ms/step. Consider layout/vector-load and
   project/epilogue fusion only after the top three targets.
5. **Aggregate tiny launch overhead.**
   Pursue only after separately quantifying launch and parameter-update gaps;
   the current trace does not justify a persistent or mega-kernel rewrite.

## Measurement boundaries

### Measured

- Kernel and copy durations/counts from a warmed Nsight Systems capture.
- Focused per-kernel Nsight Compute replay counters.
- Executed source-correlated instruction counts from Nsight Compute.
- Static resource usage from Nsight Compute and `cuobjdump`.
- Exact XLA HLO shapes, buffer assignments, thunk count, and attention
  operands.

### Inferred

- The copies arise from ZML `PagedKvCache.atLayer` slices after `updateAt`,
  because the compiler graph has exactly the same 16-layer K/V structure and
  the materialized values feed the attention K/V operands.
- Attention and P32 are structurally underfilled.
- A full-cache-plus-layer-offset attention ABI is the most direct design
  candidate.

### Unresolved

- Whether XLA/PJRT can express a safe alias/sub-buffer for these dynamic cache
  state values without changing the attention ABI.
- Whether donation or command-buffer ownership currently forces
  materialization.
- Whether additional P32 split parallelism wins after reduction overhead.
- Whether attention grid changes preserve all paged and ragged cases.

### Profiler limitations

- Nsight Compute replay changes application timing and profiles one launch;
  its durations are not substituted for the ordinary decode benchmark.
- Cache-control was `none`; replay cache state is not guaranteed to match
  every production launch.
- Kernel-specific DRAM percentages cannot establish whole-workload HBM
  saturation.
- Static SASS size is not treated as executed instruction count.
- The 7.932 ms copy-removal figure is an upper bound before implementation,
  not a validated speedup.

## Artifact index

The derived tables, capture harness, workload records, and manifests are
committed under `artifacts/ncu/f6_seed7_ampere_decode_20260911/`. The raw
reports and disassembly remain in the durable model benchmark directory
because the focused NCU reports are approximately 498 MiB each and the
compressed full SASS dump is 125 MiB. Committed CSV copies use LF line endings;
the manifests authenticate the original run-directory files.

Committed:

- `nsys_decode_b1_gpu_activity_breakdown.csv`
- `nsys_decode_b1_d2d_256m_adjacency.csv`
- `ncu_target_kernel_summary.csv`
- `ncu_p32_m1_main_source_hotspots.csv`
- `ncu_p32_rank8_source_hotspots.csv`
- `xla_decode_cache_copy_summary.csv`
- `ncu_attention_full_workload.json`
- `ncu_p32_m1_main_full_workload.json`
- `ncu_p32_rank8_workload.json`
- `run_targeted_ncu.sh`
- `zml_native_fast_abi_decode_profile.py`
- `idle_gate_exec.py`
- `SHA256SUMS.txt`
- `SHA256SUMS.full.txt`

Retained raw:

- `nsys_decode_b1_maxctx.nsys-rep`
- `nsys_decode_b1_maxctx.sqlite`
- `ncu_attention_full.ncu-rep`
- `ncu_p32_m1_main_full.ncu-rep`
- `ncu_p32_rank8.ncu-rep`
- `qvq_nvdisasm_p32_m1_main.sass`
- `qvq_nvdisasm_p32_rank8.sass`
- `qvq_cuobjdump_sass.txt.gz`
- `xla_dump_decode/`
