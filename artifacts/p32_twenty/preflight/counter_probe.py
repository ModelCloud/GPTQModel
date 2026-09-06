import os
import subprocess
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
uuid = 'GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2'
os.environ['CUDA_VISIBLE_DEVICES'] = uuid
for i in range(3):
    row = subprocess.check_output(['nvidia-smi', '--id='+uuid, '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in row.split(',')]
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    assert fields[0] == uuid and int(fields[1]) <= 8 and int(fields[2]) == 0, row
    assert uuid not in procs, procs
    print('idle sample', i+1, row, flush=True)
    time.sleep(1)
import torch
x = torch.ones(4096, device='cuda')
for _ in range(5):
    y = x + 1
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
y = x + 1
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
assert torch.equal(y, torch.full_like(x, 2))
