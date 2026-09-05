#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/root/venv-py3.14t/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_JOBS=8 NINJAFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2
export GPTQMODEL_TORCH_EXTENSIONS_DIR=/root/work/p32-twenty-jit
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python - <<'PY'
import os, subprocess, time
uuid = os.environ['CUDA_VISIBLE_DEVICES']
for sample in range(3):
    row = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-gpu=index,pci.bus_id,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in row.split(',')]
    assert fields[2] == uuid and int(fields[3]) <= 8 and int(fields[4]) == 0, row
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    assert uuid not in processes, processes
    print('Idle gate', sample+1, row, 'memory tolerance=8MiB', flush=True)
    time.sleep(1)
PY
exec python -u scripts/qvq_quantize.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /root/work/p32-twenty-data/f6-seed7-c4-checkpoint \
  --quant-config scripts/p32_twenty/f6_seed7.json \
  --calibration-dataset /root/work/p32-twenty-data/lifecycle.parquet \
  --calibration-dataset-split train --calibration-row-start 0 --calibration-rows 128 \
  --yaqa-dataset /root/work/p32-twenty-data/teacher_calibration.parquet \
  --yaqa-dataset-split train --yaqa-row-start 0 --yaqa-rows 10178 \
  --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 --qvq-telemetry
