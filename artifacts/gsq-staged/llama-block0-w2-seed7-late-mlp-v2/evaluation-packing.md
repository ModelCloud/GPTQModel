# W2 late-MLP packing audit

All seven real projections preserved integer assignments (zero mismatches); no learned scale rounded to zero in FP16 storage. See `packing-audit.json`. This CPU decoder audit complements the actual serialized portable TorchLinear propagation in `../propagation-w2-late-mlp-v2`.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MAX_JOBS=4 PYTHONPATH=. /root/venv-py3.14t/bin/python scripts/audit_gsq_staged_packing.py artifacts/gsq-staged/llama-block0-w2-seed7-late-mlp-v2
```

Log: `../logs/packing-w2-late-mlp-v2.log`. Source: `scripts/audit_gsq_staged_packing.py` SHA256 `664e1d7be81643c5ee683b473d60a6e781dc8c6cf087a50dadfa34c355e641a7`.
