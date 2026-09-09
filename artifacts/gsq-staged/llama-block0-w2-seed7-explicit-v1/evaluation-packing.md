# W2 portable packing audit

Completed. All seven saved staged projections packed and in-memory reloaded with TorchLinear W2/group128/FP16 scales. No new calibration data or full-model inference.

Artifact: /root/polly-work/qvq-gsq/artifacts/gsq-staged/llama-block0-w2-seed7-explicit-v1

CLI from /root/polly-work/qvq-gsq: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m scripts.audit_gsq_staged_packing artifacts/gsq-staged/llama-block0-w2-seed7-explicit-v1`

Log: /root/polly-work/qvq-gsq/artifacts/gsq-staged/logs/llama-block0-w2-seed7-explicit-v1-packing.log SHA256 18bf7b1cfcf6c9f6a7ef82a240f907893aaa40ba55c5a4c16855751618595df5

```json
{
  "self_attn.q_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 7.700290405310284e-11
  },
  "self_attn.k_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.3676811072560469e-10
  },
  "self_attn.v_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 2.948282677023295e-12
  },
  "self_attn.o_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 5.450283453722893e-12
  },
  "mlp.gate_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.6585009407488194e-11
  },
  "mlp.up_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.4517124481694399e-11
  },
  "mlp.down_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.3594367818947628e-11
  }
}
```
