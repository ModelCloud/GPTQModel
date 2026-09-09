# FP8 QKV propagation diagnostic

run_id: propagation-seed7-v1; arms: f6, baseline, gsq_weight, gsq_calibrated.
Status: completed. Stored output: /root/polly-work/qvq-gsq/artifacts/gsq-fp8/propagation-seed7-v1.
QVQ commit: 72d2bf7ef21713e34e7c2ef7ae078ee364086865 plus hash-bound archived local sources.
ZML: N/A, not used. This is selected QKV substitution into canonical F6,
not a complete native FP8 model export or a production inference benchmark.

Exact command, from /root/polly-work/qvq-gsq:

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python -m scripts.validate_gsq_fp8_propagation --layers artifacts/gsq-fp8/real-qkv-seed7-v2 --output artifacts/gsq-fp8/propagation-seed7-v1
```

All four arms used the same 32 held-out token sequences (6367 tokens), dense
FP32 teacher, eager attention, batch 1, cache disabled and no CUDA graphs.
No tokenizer or prompt rendering was rerun; exact token IDs are hash-bound
through inputs.json in provenance.json. GPU 0: PG506-230 SM80,
GPU-737e2423-874a-23a4-1126-dfbe3e77c294, PCI DE:00.0. Three idle samples passed.
Torch 2.15.0.dev20260817+cu130, CUDA 13.0, Transformers 5.15.0,
Triton 3.8.0+gitdf3f91dd. TF32 disabled. No custom kernel compilation.

Raw logits for every document and arm (including teacher) are retained as
.pt files. Per-document metrics, token counts and logits hashes are in
report.json. Full stdout/stderr is in execution.log. Preparation binds source
models, dense tokenizer files, datasets, local payloads and executable sources
in provenance.json. Manifest hashes cover all output files.

Calibrated GSQ improves KL versus ordinary FP8 QKV: delta -0.00079134155,
paired-document 95% bootstrap CI [-0.00095252600,-0.00067029045].
Top-10 agreement declines by 0.06125334 percentage points, with CI excluding
zero. Logit MSE, Top-1 and Top-5 deltas are noise-consistent in this 32-document
sample. Weight-only GSQ matches baseline logits exactly. These are mixed
propagated results; expand disjoint data/layers/seeds before promotion.

Bootstrap uses 10000 document resamples, seed 7, token-weighted means;
paired-bootstrap.json retains full intervals. No task accuracy measured.
