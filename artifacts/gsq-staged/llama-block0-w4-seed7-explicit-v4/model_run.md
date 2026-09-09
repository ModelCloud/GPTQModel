# Explicit-backward staged GSQ run

Completed execution; experimental block artifact, not a portable full-model export. ZML N/A.

Artifact: /root/polly-work/qvq-gsq/artifacts/gsq-staged/llama-block0-w4-seed7-explicit-v4

CLI from /root/polly-work/qvq-gsq: `PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python -m scripts.validate_gsq_staged_llama --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output artifacts/gsq-staged/llama-block0-w4-seed7-explicit-v4 --bits 4 --epochs 10 --qk-steps 2000`

Scoped FP32 local reconstruction; unweighted documents, not F6/N production evaluation. Teacher/initializer/hyperparameters match v3, updated relaxation order and explicit backward.

```json
{
  "baseline_mse": 1.9465117556922764e-05,
  "staged_mse": 2.3208134140236772e-05,
  "relative_change_percent": 19.229355139356997,
  "heldout_documents": 32,
  "elements": 13039616,
  "log_path": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/logs/llama-block0-w4-seed7-explicit-v4.log",
  "log_sha256": "836eadabd5298ddd5498fda155595f993a77424ddd3aaf9e27a1db4b02438c6d"
}
```

```json
{
  "state": "complete",
  "commit": "bebfd81fcf384363e4b15b72e35656c9f35af083",
  "source_hashes": {
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_staged_llama.py": "81817edae9f932d0589616d123149ecc899dbabfa158987ed219af71d9302167",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training.py": "b7500317bbcbcae187577baad3a9fc654e13935c3937a493001691f3939706b3",
    "/root/polly-work/qvq-gsq/artifacts/gsq-scalar/gptq-w4-seed7-v2/inputs.json": "1fad35585b53bcf26beb66acb0c1fa5b5062404d889cbe02cd87c43cddedaff1",
    "/root/polly-work/qvq-gsq/artifacts/gsq-scalar/gptq-w4-seed7-v2/provenance.json": "747b19ae35e95e21f06c02f34a54c55d28cecea86aaecdbe7e30c06463d458b8",
    "/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
    "/monster/data/model/Llama-3.2-1B-Instruct/config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
  },
  "inventory": "0, 00000000:DE:00.0, GPU-737e2423-874a-23a4-1126-dfbe3e77c294, NVIDIA PG506-230, 0, 0",
  "torch": "2.15.0.dev20260817+cu130",
  "cuda": "13.0",
  "seed": 7,
  "bits": 4,
  "epochs": 10,
  "qk_steps": 2000,
  "group_size": 128,
  "attention": "eager",
  "cache": false,
  "graphs": false,
  "precision": "float32",
  "source_model": "/monster/data/model/Llama-3.2-1B-Instruct",
  "scope": "block0, local reconstruction; no paper-reproduction claim",
  "training": {
    "assignment_lr": 0.0001,
    "scale_lr": 5e-05,
    "betas": [
      0.9,
      0.95
    ],
    "weight_decay": 1.0,
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
  "qk_damp_percent": 0.01,
  "initializer_damp_percent": 0.1,
  "started_utc": "2026-09-09T09:36:19.953054+00:00",
  "argv": [
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_staged_llama.py",
    "--inputs",
    "artifacts/gsq-scalar/gptq-w4-seed7-v2",
    "--output",
    "artifacts/gsq-staged/llama-block0-w4-seed7-explicit-v4",
    "--bits",
    "4",
    "--epochs",
    "10",
    "--qk-steps",
    "2000"
  ],
  "run_id": "llama-block0-w4-seed7-explicit-v4",
  "weighting": "unweighted documents; not YAQA 1.25/N-mode reproduction",
  "gpu_properties": "_CudaDeviceProperties(name='NVIDIA PG506-230', major=8, minor=0, total_memory=97457MB, multi_processor_count=124, uuid=737e2423-874a-23a4-1126-dfbe3e77c294, pci_bus_id=222, pci_device_id=0, pci_domain_id=0, L2_cache_size=48MB)",
  "initializer": {
    "self_attn.q_proj": {
      "loss": 0.00031633430239943507,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "self_attn.k_proj": {
      "loss": 0.00015190336284632387,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "self_attn.v_proj": {
      "loss": 4.724508192452506e-06,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "self_attn.o_proj": {
      "loss": 4.052339474746489e-07,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "mlp.gate_proj": {
      "loss": 0.00024042670341331698,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "mlp.up_proj": {
      "loss": 0.00018654877227467252,
      "damp": 0.10000000149011612,
      "samples": 3767
    },
    "mlp.down_proj": {
      "loss": 2.6922102870218373e-06,
      "damp": 0.10000000149011612,
      "samples": 3767
    }
  },
  "heldout": [
    {
      "elements": 374784,
      "baseline_sse": 5.598115849717133,
      "staged_sse": 6.387457221665813
    },
    {
      "elements": 524288,
      "baseline_sse": 9.600127091852404,
      "staged_sse": 11.563046178423468
    },
    {
      "elements": 413696,
      "baseline_sse": 7.37979242168872,
      "staged_sse": 8.69045811605456
    },
    {
      "elements": 245760,
      "baseline_sse": 3.6212935517798623,
      "staged_sse": 3.8401291227267444
    },
    {
      "elements": 524288,
      "baseline_sse": 8.232950440870988,
      "staged_sse": 9.73590112987906
    },
    {
      "elements": 514048,
      "baseline_sse": 7.701203633140372,
      "staged_sse": 9.175934687163043
    },
    {
      "elements": 524288,
      "baseline_sse": 10.398193271964598,
      "staged_sse": 12.478449557885959
    },
    {
      "elements": 393216,
      "baseline_sse": 8.16028804935539,
      "staged_sse": 9.647490395461219
    },
    {
      "elements": 350208,
      "baseline_sse": 6.12718561914317,
      "staged_sse": 7.157845224287507
    },
    {
      "elements": 524288,
      "baseline_sse": 8.772552879017677,
      "staged_sse": 10.630231779822752
    },
    {
      "elements": 524288,
      "baseline_sse": 10.17343848279516,
      "staged_sse": 12.428778186757643
    },
    {
      "elements": 524288,
      "baseline_sse": 10.650552595158956,
      "staged_sse": 12.876217364324896
    },
    {
      "elements": 303104,
      "baseline_sse": 6.104766889815329,
      "staged_sse": 7.081456207507851
    },
    {
      "elements": 466944,
      "baseline_sse": 8.974996898750467,
      "staged_sse": 11.051059643821993
    },
    {
      "elements": 524288,
      "baseline_sse": 10.548831543425196,
      "staged_sse": 12.984423490937662
    },
    {
      "elements": 264192,
      "baseline_sse": 6.028954835801644,
      "staged_sse": 7.05634257796382
    },
    {
      "elements": 524288,
      "baseline_sse": 10.969181887137742,
      "staged_sse": 13.35666976447789
    },
    {
      "elements": 524288,
      "baseline_sse": 11.35965466493133,
      "staged_sse": 13.978012704211631
    },
    {
      "elements": 329728,
      "baseline_sse": 4.399363822826501,
      "staged_sse": 4.8227327762500805
    },
    {
      "elements": 196608,
      "baseline_sse": 2.466687485482919,
      "staged_sse": 2.388811318773288
    },
    {
      "elements": 237568,
      "baseline_sse": 7.239983865950856,
      "staged_sse": 8.438432620285202
    },
    {
      "elements": 176128,
      "baseline_sse": 3.4237508128164755,
      "staged_sse": 3.5997306413899355
    },
    {
      "elements": 165888,
      "baseline_sse": 4.300958247854185,
      "staged_sse": 4.8990758518832855
    },
    {
      "elements": 116736,
      "baseline_sse": 2.2796981087839674,
      "staged_sse": 2.2316409197955815
    },
    {
      "elements": 102400,
      "baseline_sse": 2.03528933672812,
      "staged_sse": 1.9426775266971987
    },
    {
      "elements": 524288,
      "baseline_sse": 10.53074948676996,
      "staged_sse": 12.655333595723054
    },
    {
      "elements": 524288,
      "baseline_sse": 10.373623364419611,
      "staged_sse": 12.494199667651852
    },
    {
      "elements": 524288,
      "baseline_sse": 9.300681377435922,
      "staged_sse": 11.280153977906222
    },
    {
      "elements": 524288,
      "baseline_sse": 10.285297698229659,
      "staged_sse": 12.595506842025685
    },
    {
      "elements": 524288,
      "baseline_sse": 15.675897643930108,
      "staged_sse": 19.484529187552997
    },
    {
      "elements": 524288,
      "baseline_sse": 8.80118443765646,
      "staged_sse": 10.704129331501992
    },
    {
      "elements": 524288,
      "baseline_sse": 12.302412041900105,
      "staged_sse": 14.968299654367808
    }
  ],
  "payload_sha256": "dc6ef7f18f19280c2d0dd019d47dbae5c8b0d66498ce5d198a47f9228d97d41b",
  "finished_utc": "2026-09-09T09:37:03.444444+00:00"
}
```
