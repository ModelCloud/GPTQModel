# Configured staged W2 block execution

State: complete. Real Llama 3.2 1B Instruct, seed7, 16 train / 32 disjoint held-out documents. Experimental block API, FP32, group128, ordinary GPTQ initialization. Full settings, source/model hashes and GPU details follow.

```sh
PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/validate_gsq_staged_llama.py --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output artifacts/gsq-staged/llama-block0-w2-seed7-config-repeat-v4 --bits 2 --epochs 10 --qk-steps 2000 --damp-percent .01
```

```json
{
  "metrics": {
    "baseline": 0.000618326096142135,
    "staged": 0.0003325881021339364
  },
  "relative_change_percent": -46.21153721167154,
  "log": "artifacts/gsq-staged/logs/llama-block0-w2-seed7-config-repeat-v4.log",
  "log_sha256": "36fb920a5cde389285aa2f0fae0daec80e1702eb83be737979647f758b173fe4"
}
```

Weights and training losses are not bitwise identical across unchanged CUDA runs; see `comparison.json`. This run does not inherit earlier payloads' final-logit evidence. The initial strict parity check failed and is preserved in the v3 `entry-parity.json` and `../logs/config-v3-parity.log`.

```json
{
  "state": "complete",
  "commit": "5d9f1eb7df200780e8733341e7af1b37409bdd43",
  "source_hashes": {
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_staged_llama.py": "050bd9f542f96da1841c8a737938ae8d77e94ff64dd9e9c8f8d9dcc3948eb7e0",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training.py": "e4829e9b8b6490ebb0fb9d712e1a6947896cb09b21ea2aa2775b4adbe21996d4",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training_config.py": "bf458506f08a198d834593096a6a63d2272ce7c842a947e39ce9032cb77646f6",
    "/root/polly-work/qvq-gsq/artifacts/gsq-scalar/gptq-w4-seed7-v2/inputs.json": "1fad35585b53bcf26beb66acb0c1fa5b5062404d889cbe02cd87c43cddedaff1",
    "/root/polly-work/qvq-gsq/artifacts/gsq-scalar/gptq-w4-seed7-v2/provenance.json": "747b19ae35e95e21f06c02f34a54c55d28cecea86aaecdbe7e30c06463d458b8",
    "/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
    "/monster/data/model/Llama-3.2-1B-Instruct/config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
  },
  "inventory": "0, 00000000:DE:00.0, GPU-737e2423-874a-23a4-1126-dfbe3e77c294, NVIDIA PG506-230, 0, 0",
  "torch": "2.15.0.dev20260817+cu130",
  "cuda": "13.0",
  "seed": 7,
  "bits": 2,
  "epochs": 10,
  "qk_steps": 2000,
  "group_size": 128,
  "attention": "eager",
  "cache": false,
  "graphs": false,
  "precision": "float32",
  "source_model": "/monster/data/model/Llama-3.2-1B-Instruct",
  "scope": "block0, local reconstruction; no paper-reproduction claim",
  "gsq_training": {
    "enabled": true,
    "seed": 7,
    "epochs": 10,
    "qk_steps": 2000,
    "damp_percent": 0.01,
    "assignment_lr": 0.0001,
    "scale_lr": 5e-05,
    "weight_decay": 1.0,
    "betas": [
      0.9,
      0.95
    ],
    "temperature": [
      2.0,
      0.05
    ],
    "multiplier": [
      100.0,
      500.0
    ],
    "warmup_steps": 0,
    "min_lr": 0.1,
    "decay": "cosine"
  },
  "qk_learning_rate_decay": "constant",
  "mlp_initializer_timing": "after_attention",
  "qk_damp_percent": 0.01,
  "initializer_damp_percent": 0.01,
  "started_utc": "2026-09-09T09:56:49.676502+00:00",
  "argv": [
    "scripts/validate_gsq_staged_llama.py",
    "--inputs",
    "artifacts/gsq-scalar/gptq-w4-seed7-v2",
    "--output",
    "artifacts/gsq-staged/llama-block0-w2-seed7-config-repeat-v4",
    "--bits",
    "2",
    "--epochs",
    "10",
    "--qk-steps",
    "2000",
    "--damp-percent",
    ".01"
  ],
  "run_id": "llama-block0-w2-seed7-config-repeat-v4",
  "weighting": "unweighted documents; not YAQA 1.25/N-mode reproduction",
  "gpu_properties": "_CudaDeviceProperties(name='NVIDIA PG506-230', major=8, minor=0, total_memory=97457MB, multi_processor_count=124, uuid=737e2423-874a-23a4-1126-dfbe3e77c294, pci_bus_id=222, pci_device_id=0, pci_domain_id=0, L2_cache_size=48MB)",
  "initializer": {
    "self_attn.q_proj": {
      "loss": 0.00401970509163796,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "self_attn.k_proj": {
      "loss": 0.0018991010082093987,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "self_attn.v_proj": {
      "loss": 3.86399491130119e-05,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "self_attn.o_proj": {
      "loss": 3.6901146379174117e-06,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "mlp.gate_proj": {
      "loss": 0.0027891715515314945,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "mlp.up_proj": {
      "loss": 0.002140536307022747,
      "damp": 0.009999999776482582,
      "samples": 3767
    },
    "mlp.down_proj": {
      "loss": 1.383828355742602e-05,
      "damp": 0.009999999776482582,
      "samples": 3767
    }
  },
  "heldout": [
    {
      "elements": 374784,
      "baseline_sse": 201.6371654609233,
      "staged_sse": 93.57535905290113
    },
    {
      "elements": 524288,
      "baseline_sse": 268.9656104950283,
      "staged_sse": 162.5615847214873
    },
    {
      "elements": 413696,
      "baseline_sse": 227.72612558742986,
      "staged_sse": 122.08679409106509
    },
    {
      "elements": 245760,
      "baseline_sse": 131.6985245766103,
      "staged_sse": 55.62211766683376
    },
    {
      "elements": 524288,
      "baseline_sse": 242.06756688642616,
      "staged_sse": 138.50431769723093
    },
    {
      "elements": 514048,
      "baseline_sse": 220.2638772081244,
      "staged_sse": 131.9446302033679
    },
    {
      "elements": 524288,
      "baseline_sse": 455.99980527269423,
      "staged_sse": 182.31816265025708
    },
    {
      "elements": 393216,
      "baseline_sse": 288.4139035893474,
      "staged_sse": 136.37466139082093
    },
    {
      "elements": 350208,
      "baseline_sse": 196.8114816080141,
      "staged_sse": 102.00221967662449
    },
    {
      "elements": 524288,
      "baseline_sse": 260.40308395409835,
      "staged_sse": 148.21972983149803
    },
    {
      "elements": 524288,
      "baseline_sse": 297.45335008164886,
      "staged_sse": 176.80286519162664
    },
    {
      "elements": 524288,
      "baseline_sse": 316.35818175730174,
      "staged_sse": 192.09108542621283
    },
    {
      "elements": 303104,
      "baseline_sse": 203.7402491956413,
      "staged_sse": 99.91958968258605
    },
    {
      "elements": 466944,
      "baseline_sse": 260.9742349871046,
      "staged_sse": 155.56896341684146
    },
    {
      "elements": 524288,
      "baseline_sse": 371.5390567478093,
      "staged_sse": 191.8730514545028
    },
    {
      "elements": 264192,
      "baseline_sse": 260.63275234655543,
      "staged_sse": 101.74326238823599
    },
    {
      "elements": 524288,
      "baseline_sse": 293.7800773574379,
      "staged_sse": 194.5648794249039
    },
    {
      "elements": 524288,
      "baseline_sse": 355.4588717678584,
      "staged_sse": 206.18454611620768
    },
    {
      "elements": 329728,
      "baseline_sse": 156.16950701264187,
      "staged_sse": 69.92913398816813
    },
    {
      "elements": 196608,
      "baseline_sse": 117.20278782792509,
      "staged_sse": 32.95002588753184
    },
    {
      "elements": 237568,
      "baseline_sse": 226.29763189677567,
      "staged_sse": 118.75531554081942
    },
    {
      "elements": 176128,
      "baseline_sse": 129.53716168830977,
      "staged_sse": 50.38988921139544
    },
    {
      "elements": 165888,
      "baseline_sse": 157.7736371044227,
      "staged_sse": 69.70436302145326
    },
    {
      "elements": 116736,
      "baseline_sse": 105.91642189892536,
      "staged_sse": 31.57140516606239
    },
    {
      "elements": 102400,
      "baseline_sse": 121.64191519832195,
      "staged_sse": 28.274846172048843
    },
    {
      "elements": 524288,
      "baseline_sse": 302.3763424370926,
      "staged_sse": 177.783017425575
    },
    {
      "elements": 524288,
      "baseline_sse": 329.1768007961418,
      "staged_sse": 180.63156415723378
    },
    {
      "elements": 524288,
      "baseline_sse": 283.2336425345176,
      "staged_sse": 162.45510054679622
    },
    {
      "elements": 524288,
      "baseline_sse": 308.57872302501283,
      "staged_sse": 186.39406286797916
    },
    {
      "elements": 524288,
      "baseline_sse": 401.41850279470316,
      "staged_sse": 274.42557150481906
    },
    {
      "elements": 524288,
      "baseline_sse": 255.73066680351832,
      "staged_sse": 152.8117158766788
    },
    {
      "elements": 524288,
      "baseline_sse": 313.7571965741589,
      "staged_sse": 208.78730654554533
    }
  ],
  "payload_sha256": "c36f47270c49b204a4a7ebd7b76c7b44a14c7667907e2c60679237321020c9df",
  "finished_utc": "2026-09-09T09:57:29.098544+00:00"
}
```
