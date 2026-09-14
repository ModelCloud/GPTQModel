# Author quantizer, Lion, and hard-export parity

Nine CUDA cases: W2/W3/W4 × FP32/FP32, BF16/BF16, BF16/FP32 weight/logit dtype. Exact equality for forward, both gradients, 20-update Lion trajectory logits/scales, and final hard weights. Small matched-noise fixtures establish arithmetic parity, not complete training-procedure or real-model quality equivalence.

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/check_gsq_author_parity.py --author /tmp/gsq-author-reproduction --lion-wheel /tmp/gsq-lion-reference/lion_pytorch-0.2.5-py3-none-any.whl --output artifacts/gsq-staged/author-parity-hard-export.json
```

Log: `logs/author-parity-hard-export.log.gz`; NVIDIA PG506-230 on exclusive UUID lease with three idle checks.

```json
{
  "commit": "bebfd81fcf384363e4b15b72e35656c9f35af083",
  "author_commit": "03fc16484c369e3127225615d5e03e8d3a6043e3",
  "files": {
    "scripts/check_gsq_author_parity.py": "e65fe5cb8184c9df158261777acde5b951ae447003f04e02922dcfbed785e75f",
    "gptqmodel/quantization/gsq_training.py": "a8d5b3cd045c25384598e5d9f3109c97a75f5d01c8ced5bdb16bb01836cc1028",
    "/tmp/gsq-lion-reference/lion_pytorch-0.2.5-py3-none-any.whl": "77089f28bda82f1cee829cabab8a104fe8aabf66af1941b0fc6ce400aabb28cc",
    "/tmp/gsq-author-reproduction/src/quantization/gumbel_quantizer_2bit.py": "26f6ed44ef4652ed7046702239f9bdc25d798c797d0ecf4c7f1149edec2137a8",
    "/tmp/gsq-author-reproduction/src/quantization/gumbel_quantizer_int.py": "6573d7d5ba572bbad8b2914710b277f3cdab95cf993a16414c72024edfde276d"
  }
}
```
