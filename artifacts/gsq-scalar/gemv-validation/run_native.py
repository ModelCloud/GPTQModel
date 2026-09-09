import os, subprocess, time
uuid=os.environ['CUDA_VISIBLE_DEVICES']
assert uuid.startswith('GPU-') and ',' not in uuid and os.environ.get('GPU_ALLOCATOR_LEASE_ID')
for i in range(3):
 info=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.total,utilization.gpu,driver_version','--format=csv,noheader,nounits'],text=True).strip()
 values=[x.strip() for x in info.split(',')]
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert values[2]==uuid and int(values[4])<=8 and int(values[6])==0 and uuid not in processes,info
 print('IDLE',i+1,info,flush=True)
 time.sleep(1)
import pytest
raise SystemExit(pytest.main(['-q','-s','tests/test_gsq_awq_gemv.py','-k','native_reloaded','--junitxml=/tmp/gsq-gemv-native.xml']))
