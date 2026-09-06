# QvQ CPU/GPU Fisher experiment

This explicit collector combines grouped GPU statistics with CPU factor finalization.
The GPU performs the model forward/backward, projections, Grams, and diagonal
reductions. Two CPU workers normalize and construct ready factors while later
GPU work continues. The workers consume required host copies, adding no tensor
transfers. All queued work and transfers finish before the collector returns.

The measured scope is the local **Qwen3.5-27B geometry proxy**, all 400 targets,
BF16 model, IEEE FP32 statistics, B2/T64/R256, sixteen calibration rows starting
at row 64, seed 20260908, without activation checkpointing. This is a warmed
result; graph construction and first allocation are setup costs. Larger-batch
CPU Gram offload was slower and is not enabled.

The normal library collector is unchanged. This experiment checks the retained
collector's source hash, Torch build, H200/SM90 hardware, batch geometry,
free-threaded Python, and CPU topology. Unsupported calls use the normal collector.
Only tested activation/gradient geometries enter the grouped path. A single
cached graph plan is protected against concurrent replay; changed module geometry
or CPU affinity invalidates it. `collector.clear_cache()` releases the graph plan
and owned CPU workers. Returned factors use the canonical `YaqaGramSketch` class.

From the repository root on the tested machine:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
PYTHON_GIL=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=300000 \
numactl --cpunodebind=0,1,2,3 --membind=0 \
/root/venv-py3.14t-gil0/bin/python scripts/benchmark_qvq_yaqa_qwen38.py \
  --model /monster/data/model/Qwen3.5-27B --model-label Qwen3.5-27B \
  --dataset /monster/data/model/dataset/nm-calibration/llm.parquet \
  --all-targets --rows 16 --batch-size 2 --sequence-length 64 \
  --row-start 64 --seed 20260908 --arms streaming_256 \
  --accumulator-device cuda --no-activation-checkpointing \
  --cpu-threads 16 --cpu-interop-threads 1 --warmup 6 --repeats 9 \
  --idle-max-driver-memory-mib 80 \
  --collector-source scripts/experiments/qvq_cpu_sidecar/paired.py \
  --verify-source gptqmodel/quantization/qvq_yaqa.py \
  --output /tmp/qvq-cpu-gpu-paired.json

/root/venv-py3.14t-gil0/bin/python \
  scripts/experiments/qvq_cpu_sidecar/summarize.py /tmp/qvq-cpu-gpu-paired.json
```

`paired.py` alternates the accepted GPU collector, grouped GPU collector, and
CPU/GPU collector with one loaded model. Read its `comparison_arm` tags or use
`summarize.py`; the benchmark's aggregate table combines the arms and is not a
speedup comparison. The initial reference check compares all 800 CPU/GPU factors
bitwise, including source, diagonal, source diagonal, normalizer, and seed.

For CPU/GPU alone, select `collector.py` instead, with `--warmup 2 --repeats 3`.
Setting `QVQ_CPU_FINALIZE=0` keeps grouped GPU statistics and disables CPU
finalization. Omitting `--collector-source` uses the accepted GPU baseline.

The 80 MiB accounting allowance is specific to this pinned-memory workload.
An isolated probe measured 11 MiB of driver baseline plus 2 MiB per GiB of pinned
host allocation. With 32 GiB pinned, the difference between total GPU memory and
per-process accounting was 75 MiB; returning the pinned cache reduced it to
11 MiB. The paired report records pinned-host allocation. Foreign GPU processes
remain rejected and startup still requires three idle samples.

oneDNN 3.12.0 was installed with OpenMP at `/opt/qvq/onednn-3.12.0`; AMX BF16/INT8
and AVX-512 FP32 dispatch were verified separately. The retained collector uses
exact FP32 math and Torch CPU finalization. It does not claim an AMX speedup.
See [experiment evidence](../../../../docs/qvq_fisher_cpu_sidecar.md) for rejected
CPU Gram/diagonal alternatives, instruction audits, and numerical checks.

Final post-profile paired result (three warmed samples per arm):

| Collector | Median seconds | Speedup over current GPU |
|---|---:|---:|
| Current GPU | 4.012427 | 1.000x |
| Grouped GPU | 2.359056 | 1.701x |
| Grouped GPU + CPU finalization | 2.307548 | 1.739x |

All 800 factors matched bitwise. Most gain comes from grouping GPU work; the
incremental CPU gain is small and noisy. A separate six-pair comparison measured
2.313712 seconds for grouped GPU versus 2.295275 seconds with CPU finalization.
Eight focused lifecycle/fallback/serialization tests passed. Full quantized-model
save/load/inference was not run for this opt-in experiment.

The [final machine-readable report](../../../../docs/experiments/qvq_fisher_cpu_gpu_result_20260906.json)
contains samples, hardware, source hashes, numerical checks, and instruction audit.
Raw evidence is preserved at `/root/qvq-sidecar-artifacts/20260906-final/`, with a
SHA-256 manifest. The packaged GPU instruction summary matches the audited grouped
GPU predecessor; post-profile real-operator checks and full-model timing passed.
