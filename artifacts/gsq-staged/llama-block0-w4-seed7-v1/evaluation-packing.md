# Portable GPTQ packing audit

Run: llama-block0-w4-seed7-v1; arm: staged. Status: completed.

Source artifact: /root/polly-work/qvq-gsq/artifacts/gsq-staged/llama-block0-w4-seed7-v1

Full QVQ base commit: 548d883e7aac1fb1858ad0daff061a1c774f365e; uncommitted audit script SHA256: 4acfa758fed3e50755d24fe081bde030c9f6a115ac31408d7b5231c8e9bdd696

CLI from /root/polly-work/qvq-gsq: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m scripts.audit_gsq_staged_packing artifacts/gsq-staged/llama-block0-w4-seed7-v1`

W4/group128 portable TorchLinear, FP32 weight input, FP16 stored scales. CPU, no GPU/graphs/attention execution. No new dataset; this audits saved learned weights only. In-memory state-dict reload, not disk-payload reload or full model inference. ZML: N/A.

Log: /root/polly-work/qvq-gsq/artifacts/gsq-staged/logs/llama-block0-w4-seed7-v1-packing.log; SHA256 70cd0711c9e820910c384d9f8be034495a68e26beb4245ea285337f48b6f5eb7

```json
{
  "self_attn.q_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 5.74332630065566e-11
  },
  "self_attn.k_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 9.510213450081295e-11
  },
  "self_attn.v_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 3.3472541145079804e-12
  },
  "self_attn.o_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 1,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 5.32153401205937e-12
  },
  "mlp.gate_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.3957812336484743e-11
  },
  "mlp.up_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.1449856687772986e-11
  },
  "mlp.down_proj": {
    "assignment_mismatches": 0,
    "negative_scales": 0,
    "zero_stored_scales": 0,
    "scale_storage_dtype": "torch.float16",
    "decoded_weight_mse": 1.1856749956851154e-11
  }
}
```
