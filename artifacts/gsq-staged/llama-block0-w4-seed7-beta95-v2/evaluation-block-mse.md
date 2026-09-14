# Staged GSQ beta2 comparison

Completed execution; not paper parity. Selected-layer experimental artifact.

Lion betas [.9,.95]; all remaining settings match v1. Source model, dataset/token IDs, runtime identity, timestamps, and executed-source hashes are recorded below. ZML N/A.

CLI from /root/polly-work/qvq-gsq: `PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python -m scripts.validate_gsq_staged_llama --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output artifacts/gsq-staged/llama-block0-w4-seed7-beta95-v2 --bits 4 --epochs 2`

```json
{
  "baseline_mse": 1.9465117556922764e-05,
  "staged_mse": 9.150931394428358e-05,
  "relative_change_percent": 370.1195031403153,
  "baseline_per_document_exactly_matches_v1": true,
  "log_path": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/logs/llama-block0-w4-seed7-beta95-v2.log",
  "log_sha256": "b53f31c8a86393f6f811b9ddf89e40fac0489bcb7722b06138c5796458a92b8c"
}
```

```json
{
  "state": "complete",
  "commit": "548d883e7aac1fb1858ad0daff061a1c774f365e",
  "source_hashes": {
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_staged_llama.py": "e67db8bdf2545f6605a0f46b5a0ce1c1a43690462d6df9d33997d0c973294e6e",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training.py": "a69136dfec842ee57e77fa904efa19a464fb871dfef233c9e7cd797e0ed8a3e1",
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
  "epochs": 2,
  "group_size": 128,
  "attention": "eager",
  "cache": false,
  "graphs": false,
  "precision": "float32",
  "source_model": "/monster/data/model/Llama-3.2-1B-Instruct",
  "scope": "block0, local reconstruction; no paper-reproduction claim",
  "started_utc": "2026-09-09T09:23:58.458548+00:00",
  "argv": [
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_staged_llama.py",
    "--inputs",
    "artifacts/gsq-scalar/gptq-w4-seed7-v2",
    "--output",
    "artifacts/gsq-staged/llama-block0-w4-seed7-beta95-v2",
    "--bits",
    "4",
    "--epochs",
    "2"
  ],
  "run_id": "llama-block0-w4-seed7-beta95-v2",
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
      "staged_sse": 33.10269843858963
    },
    {
      "elements": 524288,
      "baseline_sse": 9.600127091852404,
      "staged_sse": 40.39880556256954
    },
    {
      "elements": 413696,
      "baseline_sse": 7.37979242168872,
      "staged_sse": 36.45732365177977
    },
    {
      "elements": 245760,
      "baseline_sse": 3.6212935517798623,
      "staged_sse": 29.22787369831563
    },
    {
      "elements": 524288,
      "baseline_sse": 8.232950440870988,
      "staged_sse": 38.10247883982301
    },
    {
      "elements": 514048,
      "baseline_sse": 7.701203633140372,
      "staged_sse": 37.36159864874869
    },
    {
      "elements": 524288,
      "baseline_sse": 10.398193271964598,
      "staged_sse": 41.526066127808406
    },
    {
      "elements": 393216,
      "baseline_sse": 8.16028804935539,
      "staged_sse": 37.528719291379694
    },
    {
      "elements": 350208,
      "baseline_sse": 6.12718561914317,
      "staged_sse": 33.87034063680343
    },
    {
      "elements": 524288,
      "baseline_sse": 8.772552879017677,
      "staged_sse": 40.01165966037911
    },
    {
      "elements": 524288,
      "baseline_sse": 10.17343848279516,
      "staged_sse": 41.871758848144154
    },
    {
      "elements": 524288,
      "baseline_sse": 10.650552595158956,
      "staged_sse": 42.552427350580004
    },
    {
      "elements": 303104,
      "baseline_sse": 6.104766889815329,
      "staged_sse": 33.614901777980236
    },
    {
      "elements": 466944,
      "baseline_sse": 8.974996898750467,
      "staged_sse": 39.84916040845671
    },
    {
      "elements": 524288,
      "baseline_sse": 10.548831543425196,
      "staged_sse": 42.54892248478379
    },
    {
      "elements": 264192,
      "baseline_sse": 6.028954835801644,
      "staged_sse": 33.544332380658275
    },
    {
      "elements": 524288,
      "baseline_sse": 10.969181887137742,
      "staged_sse": 42.64700470988505
    },
    {
      "elements": 524288,
      "baseline_sse": 11.35965466493133,
      "staged_sse": 43.92265882797317
    },
    {
      "elements": 329728,
      "baseline_sse": 4.399363822826501,
      "staged_sse": 31.2085203892173
    },
    {
      "elements": 196608,
      "baseline_sse": 2.466687485482919,
      "staged_sse": 27.52744643508179
    },
    {
      "elements": 237568,
      "baseline_sse": 7.239983865950856,
      "staged_sse": 35.44721973930604
    },
    {
      "elements": 176128,
      "baseline_sse": 3.4237508128164755,
      "staged_sse": 28.96608945579138
    },
    {
      "elements": 165888,
      "baseline_sse": 4.300958247854185,
      "staged_sse": 29.938271269129075
    },
    {
      "elements": 116736,
      "baseline_sse": 2.2796981087839674,
      "staged_sse": 26.364491044266757
    },
    {
      "elements": 102400,
      "baseline_sse": 2.03528933672812,
      "staged_sse": 25.94045094956615
    },
    {
      "elements": 524288,
      "baseline_sse": 10.53074948676996,
      "staged_sse": 41.569711237405656
    },
    {
      "elements": 524288,
      "baseline_sse": 10.373623364419611,
      "staged_sse": 41.58288778666463
    },
    {
      "elements": 524288,
      "baseline_sse": 9.300681377435922,
      "staged_sse": 39.84863674755999
    },
    {
      "elements": 524288,
      "baseline_sse": 10.285297698229659,
      "staged_sse": 41.458986369193475
    },
    {
      "elements": 524288,
      "baseline_sse": 15.675897643930108,
      "staged_sse": 50.802608896804344
    },
    {
      "elements": 524288,
      "baseline_sse": 8.80118443765646,
      "staged_sse": 39.441957146554536
    },
    {
      "elements": 524288,
      "baseline_sse": 12.302412041900105,
      "staged_sse": 45.010305445703864
    }
  ],
  "payload_sha256": "d2834d044c1570a4093cd2ab62e52fa006230f948d6835a24b13f153d8d46602",
  "finished_utc": "2026-09-09T09:24:15.886178+00:00"
}
```

Q/K update count and main sampling settings still differ from the author training procedure; no full-model KL or native packed reload is established here.
