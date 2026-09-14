# Scale-gradient reduction repeatability

Identical real saved Q-projection weights, parameters and uniform samples were replayed 20 times. Default CUDA reduction differed by up to 1.6689300537109375e-6 across 25,674 scale-gradient elements. With PyTorch deterministic algorithms enabled, all repeats were exact. This isolates a source of training variability; it does not prove this is the only nondeterministic operation or establish model-quality gains.

Both implementations accumulate group-scale gradients through `scatter_add_`: local `gsq_training.py` and pinned author `src/quantization/gumbel_quantizer_2bit.py` lines 89–91.

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/check_gsq_scale_reduction.py --source artifacts/gsq-staged/llama-block0-w2-seed7-config-v3/stages.pt --output artifacts/gsq-staged/scale-reduction-repeatability.json
```

Log: `logs/scale-reduction-repeatability.log.gz`. Payload/source hashes, runtime and GPU inventory are in `scale-reduction-repeatability.json`. Reference author commit: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
