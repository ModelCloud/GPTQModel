# Full W4 signed GPTQ baseline

State: complete. Run ID: full-model-w4-nm128-signed-baseline-v1. Arm: baseline.
Stored experiment path: /root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w4-nm128-signed-baseline-v1. This is not yet a sealed canonical snapshot.
All 16 layers quantized; all 32 held-out public reload logits exactly match the
pre-export model and are finite. GSQ is disabled. This matched staged-path signed
GPTQ initializer is not the package-default true-sequential GPTQ recipe.

The user requested W4 after the failed W2 test. This run uses 128 native NM
calibration documents (47,005 tokens), seed 7, group128, FP32 initialization and
FP16 GPTQ-v2 export. Five epochs are recorded but unused with GSQ disabled.
QVQ commit: 5cc76af5c715f6089663d00e5cc5cefb61445afd. ZML Ultra: not used. Exact executed dirty source copies
are in executed-source/ and their hashes are recorded below.

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/validate_gsq_full_llama.py --inputs artifacts/gsq-staged/nm128-seed7-v1 --output artifacts/gsq-staged/full-model-w4-nm128-signed-baseline-v1 --arm baseline --bits 4 --train-samples 128 --initializer gptq_signed --batch-size 64 --microbatch-size 16 --epochs 5
```

Quantization stdout/stderr: quantization.log; SHA256: 95867d8ad21c9d0ffcc3968119bd6be99ed6a8ad34a0a5c7a60f525bd2217466.
Held-out metrics: summary.json and per-document rows below. Full Platinum evaluation
is queued under ../gsm8k-platinum-w4-nm128-signed-v1/baseline-full.
Calibration selection and exact source revisions/hashes are preserved in
../nm128-seed7-v1/provenance.json and inputs.json, copied here as inputs.json.

Complete effective configuration, source hashes, timing, hardware/runtime, model
output hashes and held-out evidence:

```json
{
  "state": "complete",
  "arm": "baseline",
  "run_id": "full-model-w4-nm128-signed-baseline-v1",
  "commit": "5cc76af5c715f6089663d00e5cc5cefb61445afd",
  "argv": [
    "scripts/validate_gsq_full_llama.py",
    "--inputs",
    "artifacts/gsq-staged/nm128-seed7-v1",
    "--output",
    "artifacts/gsq-staged/full-model-w4-nm128-signed-baseline-v1",
    "--arm",
    "baseline",
    "--bits",
    "4",
    "--train-samples",
    "128",
    "--initializer",
    "gptq_signed",
    "--batch-size",
    "64",
    "--microbatch-size",
    "16",
    "--epochs",
    "5"
  ],
  "source_model": "/monster/data/model/Llama-3.2-1B-Instruct",
  "gsq_training": {
    "enabled": false,
    "initializer": "gptq_signed",
    "seed": 7,
    "epochs": 5,
    "batch_size": 64,
    "microbatch_size": 16,
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
  "bits": 4,
  "group_size": 128,
  "train_precision": "float32",
  "export_precision": "float16",
  "calibration_samples": 128,
  "calibration_tokens": 47005,
  "calibration_token_cap": 2048,
  "weighting": "unweighted documents; not full paper calibration or F6/N-mode reproduction",
  "started_utc": "2026-09-09T12:47:45.936313+00:00",
  "inventory": "0, 00000000:DE:00.0, GPU-737e2423-874a-23a4-1126-dfbe3e77c294, NVIDIA PG506-230, 0, 0",
  "torch": "2.15.0.dev20260817+cu130",
  "cuda": "13.0",
  "gpu": "_CudaDeviceProperties(name='NVIDIA PG506-230', major=8, minor=0, total_memory=97457MB, multi_processor_count=124, uuid=737e2423-874a-23a4-1126-dfbe3e77c294, pci_bus_id=222, pci_device_id=0, pci_domain_id=0, L2_cache_size=48MB)",
  "deterministic_algorithms": true,
  "cublas_workspace_config": ":4096:8",
  "source_hashes": {
    "/root/polly-work/qvq-gsq/scripts/validate_gsq_full_llama.py": "8e8d5dd336777eb124cdf1eaccf4c83f234ad4ddb6afad698f321eb4964ae548",
    "/root/polly-work/qvq-gsq/gptqmodel/looper/gsq_training_model.py": "83de5f493aec6e7d3a3304f68f3fe0f94f086ab7d8286ffb2c1a843c5dbf0836",
    "/root/polly-work/qvq-gsq/gptqmodel/looper/gsq_training_capture.py": "2827ca33ac06fffa6ecf5b1ed31a3eb41fbee2d7479fe7d7d3ed375ae0934a53",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training.py": "8398cb70bea7a3e0c72c705c92d10b58ca0ac6e5ae891ee226a9bc1c2c19be01",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_training_config.py": "a4aaff10d7644698b7e6cf254618fe268b7f818a96f95f3ae1d9ea5716a1adea",
    "/root/polly-work/qvq-gsq/scripts/p32_twenty/scorecard.py": "9b1ee07dab08216454375b2d2d683b99825743ec500c4468cb886b79edf3020f",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_initialization.py": "5704da92d82222eba9f96cf8f64800c50d78fbeac5550dbc862c0a225fd485c6",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gptq.py": "55cd501f15349781a33b6ef272f115ad8be4583e63f7eab6c7a0ecea6f6626a6",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/gsq_batching.py": "85c2b847a346459dc925be7051ea26dc75018caf81428ae912666123453643e2",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/quantizer.py": "d7428a392eb261f159655878e16b1fc57411396bd2976a5f4b8338ee8524dfcb",
    "/root/polly-work/qvq-gsq/gptqmodel/quantization/config.py": "a3bf792ebdd5e29c31663aea33809f7a5965581923cad5a854788a42e00cff9c",
    "/root/polly-work/qvq-gsq/artifacts/gsq-staged/nm128-seed7-v1/inputs.json": "893d981d8398efe64e111a22a6996f3e23f073b6460f124fc061097c8bb19dda",
    "/root/polly-work/qvq-gsq/artifacts/gsq-staged/nm128-seed7-v1/provenance.json": "9c01b66ac65c74bc4de96117bc2df1ed91d6e30ccd0333d46f0ce4333f8f107e",
    "/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
    "/monster/data/model/Llama-3.2-1B-Instruct/config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
  },
  "rows": [
    {
      "kl_teacher_candidate": 0.13930969924444722,
      "top1_agreement": 0.825136612021858,
      "top5_agreement": 0.8032786885245902,
      "top10_agreement": 0.8060109289617486,
      "tokens": 183,
      "mse": 0.5518479347229004,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "a276002d371d5a2f92a4c81afac40e339cee5fade60aeb955b696c2ca2926e48"
    },
    {
      "kl_teacher_candidate": 0.12196278686367082,
      "top1_agreement": 0.84375,
      "top5_agreement": 0.8171875000000001,
      "top10_agreement": 0.8152343750000001,
      "tokens": 256,
      "mse": 0.5090386271476746,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "54e5d9e70fc81394b021ac3730d3f0a2359332fa414e8356481df27ec1dc561c"
    },
    {
      "kl_teacher_candidate": 0.2473429041987638,
      "top1_agreement": 0.7524752475247525,
      "top5_agreement": 0.7821782178217822,
      "top10_agreement": 0.7727722772277228,
      "tokens": 202,
      "mse": 0.7090200781822205,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "77fc36e615e0770543e76c44667024ce698ea9e1aa7c104077578b3bb45e5d37"
    },
    {
      "kl_teacher_candidate": 0.23834081702487647,
      "top1_agreement": 0.775,
      "top5_agreement": 0.765,
      "top10_agreement": 0.7491666666666668,
      "tokens": 120,
      "mse": 0.8008235096931458,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "98fa511e791d7a535aed95823c3382f9a48ddcf5ff614e268e94b4c3d5cdefae"
    },
    {
      "kl_teacher_candidate": 0.18133314035161868,
      "top1_agreement": 0.82421875,
      "top5_agreement": 0.77890625,
      "top10_agreement": 0.779296875,
      "tokens": 256,
      "mse": 0.671233594417572,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "4759b956a23a4e70b3bd9b36144399f6a967bb77df751c933a9d48ada46b14b2"
    },
    {
      "kl_teacher_candidate": 0.155005840503943,
      "top1_agreement": 0.8087649402390438,
      "top5_agreement": 0.8111553784860559,
      "top10_agreement": 0.8099601593625498,
      "tokens": 251,
      "mse": 0.5076804757118225,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "e10d0725412a2dd71bbfa587d42bcf02f8370e0866fb6309c7bbc277b9b0bbb0"
    },
    {
      "kl_teacher_candidate": 0.13014975286835467,
      "top1_agreement": 0.80859375,
      "top5_agreement": 0.8,
      "top10_agreement": 0.8054687500000001,
      "tokens": 256,
      "mse": 0.5621685981750488,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "a3e0e5710e60cfa9c902d21ba88c852634c1fd405a069ddc664889180a12e216"
    },
    {
      "kl_teacher_candidate": 0.30084576352119397,
      "top1_agreement": 0.7552083333333334,
      "top5_agreement": 0.7468750000000001,
      "top10_agreement": 0.7375000000000002,
      "tokens": 192,
      "mse": 0.9571062922477722,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "9a556067cf7ef9317da4136842e310452216af027017171a20a029415b8cf15b"
    },
    {
      "kl_teacher_candidate": 0.22772635380137435,
      "top1_agreement": 0.7719298245614035,
      "top5_agreement": 0.7742690058479532,
      "top10_agreement": 0.7824561403508773,
      "tokens": 171,
      "mse": 0.7508854866027832,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2447604609de8849282c9837d3a506264ec673041120075777bf3be5796dce2a"
    },
    {
      "kl_teacher_candidate": 0.21744849465549893,
      "top1_agreement": 0.77734375,
      "top5_agreement": 0.7585937500000001,
      "top10_agreement": 0.7718750000000001,
      "tokens": 256,
      "mse": 0.6838564276695251,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "c3dd6247b4a94611ed4d2068eacaa78b8e47c2c816c3f201080c60e8a9496108"
    },
    {
      "kl_teacher_candidate": 0.38707686532581,
      "top1_agreement": 0.734375,
      "top5_agreement": 0.7015625000000001,
      "top10_agreement": 0.6898437500000001,
      "tokens": 256,
      "mse": 1.342284917831421,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2306de4fe1a934807ce55dac8e2258c8746808310b1a2ef16c055fd9cc885a1c"
    },
    {
      "kl_teacher_candidate": 0.22638710549027607,
      "top1_agreement": 0.81640625,
      "top5_agreement": 0.7562500000000001,
      "top10_agreement": 0.739453125,
      "tokens": 256,
      "mse": 0.891841471195221,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "786c11e0425302971af5291ef68fcac7778e512c37b4b2a4ba382e4298e3d16d"
    },
    {
      "kl_teacher_candidate": 0.4982198210691927,
      "top1_agreement": 0.6486486486486487,
      "top5_agreement": 0.6567567567567568,
      "top10_agreement": 0.6682432432432432,
      "tokens": 148,
      "mse": 1.3220843076705933,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "275cacf7924579eac7b6a7a79d53f8cb3772a2f76addca4bb90aa69b8ebd3b08"
    },
    {
      "kl_teacher_candidate": 0.270957465753811,
      "top1_agreement": 0.7807017543859649,
      "top5_agreement": 0.7508771929824563,
      "top10_agreement": 0.7416666666666668,
      "tokens": 228,
      "mse": 0.9679388403892517,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "5397e46f73beb08cffcd0d8096e20a7441a6fb3855fde84e1e3d14cb76bd2ad3"
    },
    {
      "kl_teacher_candidate": 0.20114803046908436,
      "top1_agreement": 0.796875,
      "top5_agreement": 0.7679687500000001,
      "top10_agreement": 0.78046875,
      "tokens": 256,
      "mse": 0.661406397819519,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "86601f0b0b6666abd311e83f50bb5b3128c6464da16c0605bfd5daf1797068b8"
    },
    {
      "kl_teacher_candidate": 0.21442365946462943,
      "top1_agreement": 0.7906976744186046,
      "top5_agreement": 0.7224806201550388,
      "top10_agreement": 0.7356589147286822,
      "tokens": 129,
      "mse": 0.9885391592979431,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "73c9fa02a1d5c9699b35ca97798c92364071ee9d2a098ef437c59410fb4706b5"
    },
    {
      "kl_teacher_candidate": 0.3050846815342969,
      "top1_agreement": 0.74609375,
      "top5_agreement": 0.71796875,
      "top10_agreement": 0.7390625000000001,
      "tokens": 256,
      "mse": 0.8190847635269165,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "ac23c801e12ae61f5a95428a1ce243c159cf80ea8a4073d5aeac03490aca9452"
    },
    {
      "kl_teacher_candidate": 0.1739257809108329,
      "top1_agreement": 0.8671875,
      "top5_agreement": 0.759375,
      "top10_agreement": 0.753515625,
      "tokens": 256,
      "mse": 0.8884286880493164,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "9260271464222c17d68055266f93a531cb27b67d0a9e9e24ead03e72a543ad84"
    },
    {
      "kl_teacher_candidate": 0.1912645306625178,
      "top1_agreement": 0.8322981366459627,
      "top5_agreement": 0.7875776397515528,
      "top10_agreement": 0.7950310559006211,
      "tokens": 161,
      "mse": 0.7045228481292725,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "233c1919521264f41ca601795000b353a1d3ccd6e18200d82790cd2d17be2bbb"
    },
    {
      "kl_teacher_candidate": 0.1741875315227739,
      "top1_agreement": 0.8020833333333334,
      "top5_agreement": 0.8125,
      "top10_agreement": 0.8114583333333334,
      "tokens": 96,
      "mse": 0.7373772859573364,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "e0d13b6c30a5b62b537d73d42b33694b781313fdc03b8b41364b2d178fe2ada4"
    },
    {
      "kl_teacher_candidate": 2.1094655058674032,
      "top1_agreement": 0.46551724137931033,
      "top5_agreement": 0.4810344827586207,
      "top10_agreement": 0.47241379310344833,
      "tokens": 116,
      "mse": 3.250689744949341,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "7ab2864d6a0def3e3f2f13f85c64997df70b9ff7a62af8e22ba0824f43fb8f41"
    },
    {
      "kl_teacher_candidate": 0.46548309056134496,
      "top1_agreement": 0.6976744186046512,
      "top5_agreement": 0.7116279069767443,
      "top10_agreement": 0.7104651162790698,
      "tokens": 86,
      "mse": 1.055975317955017,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "3b18a85fe624a495d8d703893b3bf2a5ecdca759d804c150366a97d12e9d3cf5"
    },
    {
      "kl_teacher_candidate": 0.5373946631496127,
      "top1_agreement": 0.6419753086419753,
      "top5_agreement": 0.6518518518518519,
      "top10_agreement": 0.6493827160493827,
      "tokens": 81,
      "mse": 1.0982342958450317,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "0c4d614c7b5751c83837c3dbf170a75381c1d217d3956404f8acfb0c8ebec142"
    },
    {
      "kl_teacher_candidate": 0.44881177101523906,
      "top1_agreement": 0.6666666666666666,
      "top5_agreement": 0.6877192982456141,
      "top10_agreement": 0.6842105263157895,
      "tokens": 57,
      "mse": 0.8744688630104065,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "6d4ac25ae9bb8bad19c9cf1f81792999afec3b546c4aba6290ee1c54ff688a2c"
    },
    {
      "kl_teacher_candidate": 0.3728814114561534,
      "top1_agreement": 0.6,
      "top5_agreement": 0.716,
      "top10_agreement": 0.6920000000000001,
      "tokens": 50,
      "mse": 0.9573333859443665,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "ff6ecd56d932edc982b64494d131f36db3f25fa1497470a2cca05e524637d7ba"
    },
    {
      "kl_teacher_candidate": 0.3200201645195109,
      "top1_agreement": 0.7265625,
      "top5_agreement": 0.73828125,
      "top10_agreement": 0.755859375,
      "tokens": 256,
      "mse": 1.0080742835998535,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "f1ec6863b73a2d44eeea7215fd6e2542518f369ebe3acc1c60cedd777db76abc"
    },
    {
      "kl_teacher_candidate": 0.16465418510354557,
      "top1_agreement": 0.796875,
      "top5_agreement": 0.7835937500000001,
      "top10_agreement": 0.784375,
      "tokens": 256,
      "mse": 0.6295894384384155,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2bc87d35f494968ee4e1cbb954d6b7536a6d9c955ae7b85ae760786387acf7df"
    },
    {
      "kl_teacher_candidate": 0.1885955055698366,
      "top1_agreement": 0.76953125,
      "top5_agreement": 0.8,
      "top10_agreement": 0.80859375,
      "tokens": 256,
      "mse": 0.6941256523132324,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "c4def389fbb5b79cc96fb4d23a54ff229e461e11ba823a8b1bde3b8cb06fc00c"
    },
    {
      "kl_teacher_candidate": 0.21483403450056582,
      "top1_agreement": 0.76171875,
      "top5_agreement": 0.7742187500000001,
      "top10_agreement": 0.784765625,
      "tokens": 256,
      "mse": 0.8119811415672302,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "f80b575d2b451b02e209d76fec902e7511e90d5b28c315a6e913209bf92b314f"
    },
    {
      "kl_teacher_candidate": 0.4950401152591557,
      "top1_agreement": 0.78125,
      "top5_agreement": 0.62890625,
      "top10_agreement": 0.6125,
      "tokens": 256,
      "mse": 1.7593669891357422,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "d1409448f8b8485b5b1b604a249e6b13ec378afa5b320a2a11c95fd259254234"
    },
    {
      "kl_teacher_candidate": 0.1888339398778198,
      "top1_agreement": 0.8671875,
      "top5_agreement": 0.7351562500000001,
      "top10_agreement": 0.7234375000000001,
      "tokens": 256,
      "mse": 1.0362707376480103,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "5f315df7e409e6a5fc86542e0ead0fb6a200def065ee5d80647d2e7a6e8fa115"
    },
    {
      "kl_teacher_candidate": 0.204322351119539,
      "top1_agreement": 0.73046875,
      "top5_agreement": 0.7351562500000001,
      "top10_agreement": 0.7609375,
      "tokens": 256,
      "mse": 0.7763140797615051,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "8571b5f526794fcf90e2133ac76430d0aa36a105f0d4844278bf9294f4b7507d"
    }
  ],
  "finished_utc": "2026-09-09T12:51:40.368897+00:00",
  "model_hashes": {
    "config.json": "bf7ab2aa43a87e74f2b36a0078a2aead87038e0850a7058b555d28bfbb07ae2e",
    "generation_config.json": "7cfdf52ffe0e0c1c95ab0dc42aafc4117ba4c0d9804905cebb08eee36c6b7144",
    "quantize_config.json": "7ce70e4f5cad0cfd78c3dbd95697b44b9dd56fe6ce96260d380aafa1d56d5fbe",
    "model-00001-of-00017.safetensors": "4b1eac4477fae279be6d4d6fdf0d35c3cdcd4a40f2df166c352cab403b88bf19",
    "model-00002-of-00017.safetensors": "00ecc272ae7f637e1204fba82b13f91acd5b9ea13b86879cad2b868e2ebd2ed6",
    "model-00003-of-00017.safetensors": "9990ed0f3d477590c485b82a09f3a87180913e694947bc42fbac1871747c3d72",
    "model-00004-of-00017.safetensors": "d53393f5d6b894874a4c711f453796c45cc20277054aca06c01ba718309ed8fd",
    "model-00005-of-00017.safetensors": "6e1833498f330595da73bc649dd3a52ef92b0a1aef1824721867f483523efa5c",
    "model-00006-of-00017.safetensors": "5bd9a6d4187c5a5c5cb0bc7a8680824d095acd3a391b1aa7aaa8e1848580b3d9",
    "model-00007-of-00017.safetensors": "fc92e4b19f5d05f8b61b4bd8288226c1637134f176a3081ffa50ae74edb97431",
    "model-00008-of-00017.safetensors": "0dc19b2b3e1abe35958c3146904cc6fb7986c364a0eda47216837c94e6591f99",
    "model-00009-of-00017.safetensors": "596be52986dd7c2cfb8f321b4e79a8afd8f545462e798151e482013237d78cc3",
    "model-00010-of-00017.safetensors": "49f36d3eec855fa34230784641fcc5dafb9d379eae6b72a6caf47a0a9cc57650",
    "model-00011-of-00017.safetensors": "c2453b3894bbbe949a333d4b1b7bff469214de8efa4b1317b7d2eb0c69dd9f88",
    "model-00012-of-00017.safetensors": "9a9bdcadba46ab3db9fc6476e038fd4c6d4ed474aa1402afcb995f7fbe5d8b4d",
    "model-00013-of-00017.safetensors": "9c9e62ec11d37f9c612433eb2502952297720a8a9d7af79814f0ff6968d56d24",
    "model-00014-of-00017.safetensors": "dac0cfd636f5d24cd645fbe77ab78bc1468b2ab832b3c6d6d3413553afdceef6",
    "model-00015-of-00017.safetensors": "e7f03422b1902c296082a8ec84d351c3ef9447d0e5318a4789e7613022c068d6",
    "model-00016-of-00017.safetensors": "51330525bc894e7ae89cc2aa30d29e812db9e6accfe895aae0417a1091d6b118",
    "model-00017-of-00017.safetensors": "b562c6945710b198b5cc84e006a0fc944e7beacb71918a3897ecbe50e291b5d5",
    "model.safetensors.index.json": "7b2b487d6577403b4cba4073b4d07fd5614a3aeb79324dad9295d4ea450a2655",
    "chat_template.jinja": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
    "tokenizer_config.json": "70cf4aae347755bdd07709be21edd11f9ab14b2697371b3016a52595149cb354",
    "tokenizer.json": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "gsq_training_run.json": "463db241b38246d7b99673833774ca19f755162805e07826531724a64121efb7"
  }
}
```
