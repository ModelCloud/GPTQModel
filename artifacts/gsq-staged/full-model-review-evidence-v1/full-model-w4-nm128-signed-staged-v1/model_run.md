# Full W4 staged GSQ experiment

State: complete. Run ID: full-model-w4-nm128-signed-staged-v1; arm: staged.
Stored experiment path: /root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w4-nm128-signed-staged-v1; not yet a sealed canonical snapshot.
All 16 blocks completed; all 32 held-out public reload logits exactly match and are finite.
W4/group128 signed GPTQ prior, 128 NM documents, seed 7, five attention/MLP epochs,
batch64/micro16, Q/K 2000 updates, FP32 training and FP16 GPTQ-v2 export.
QVQ commit: 5cc76af5c715f6089663d00e5cc5cefb61445afd; ZML Ultra not used. Exact dirty source copies are in executed-source/.
This is an experimental staged trainer, not full paper reproduction. All five
held-out final-logit metrics regress with paired intervals excluding zero.
Full Platinum evaluation is queued; no task recovery claim is made.

```sh
PYTHONPATH=.:/root/polly-work/Evalution CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- /root/venv-py3.14t/bin/python scripts/validate_gsq_full_llama.py --inputs artifacts/gsq-staged/nm128-seed7-v1 --output artifacts/gsq-staged/full-model-w4-nm128-signed-staged-v1 --arm staged --bits 4 --train-samples 128 --initializer gptq_signed --batch-size 64 --microbatch-size 16 --epochs 5
```

Quantization log: quantization.log, SHA256 b0ab36f42d98c1ce76115d6724b32db4f4ce4649dd774569eee02b68d0e04945.
Calibration provenance: ../nm128-seed7-v1/provenance.json; exact selected tokens: inputs.json.
Held-out results: summary.json; paired analysis: ../full-model-w4-nm128-signed-comparison-v1/.
Platinum records: ../gsm8k-platinum-w4-nm128-signed-v1/staged-full/.

Complete effective configuration, runtime, timestamps, source/output hashes and per-document results:

```json
{
  "state": "complete",
  "arm": "staged",
  "run_id": "full-model-w4-nm128-signed-staged-v1",
  "commit": "5cc76af5c715f6089663d00e5cc5cefb61445afd",
  "argv": [
    "scripts/validate_gsq_full_llama.py",
    "--inputs",
    "artifacts/gsq-staged/nm128-seed7-v1",
    "--output",
    "artifacts/gsq-staged/full-model-w4-nm128-signed-staged-v1",
    "--arm",
    "staged",
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
    "enabled": true,
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
  "started_utc": "2026-09-09T12:51:57.623769+00:00",
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
      "kl_teacher_candidate": 0.2420842322696261,
      "top1_agreement": 0.7978142076502732,
      "top5_agreement": 0.7551912568306012,
      "top10_agreement": 0.751912568306011,
      "tokens": 183,
      "mse": 0.7360679507255554,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "a276002d371d5a2f92a4c81afac40e339cee5fade60aeb955b696c2ca2926e48"
    },
    {
      "kl_teacher_candidate": 0.21442880445458376,
      "top1_agreement": 0.8125,
      "top5_agreement": 0.790625,
      "top10_agreement": 0.7781250000000001,
      "tokens": 256,
      "mse": 0.7067745327949524,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "54e5d9e70fc81394b021ac3730d3f0a2359332fa414e8356481df27ec1dc561c"
    },
    {
      "kl_teacher_candidate": 0.39445393018516983,
      "top1_agreement": 0.693069306930693,
      "top5_agreement": 0.7257425742574257,
      "top10_agreement": 0.7178217821782178,
      "tokens": 202,
      "mse": 1.0210047960281372,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "77fc36e615e0770543e76c44667024ce698ea9e1aa7c104077578b3bb45e5d37"
    },
    {
      "kl_teacher_candidate": 0.3696649393187579,
      "top1_agreement": 0.725,
      "top5_agreement": 0.7183333333333334,
      "top10_agreement": 0.7166666666666667,
      "tokens": 120,
      "mse": 1.000120997428894,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "98fa511e791d7a535aed95823c3382f9a48ddcf5ff614e268e94b4c3d5cdefae"
    },
    {
      "kl_teacher_candidate": 0.32985120887549796,
      "top1_agreement": 0.77734375,
      "top5_agreement": 0.7093750000000001,
      "top10_agreement": 0.7097656250000001,
      "tokens": 256,
      "mse": 1.201372742652893,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "4759b956a23a4e70b3bd9b36144399f6a967bb77df751c933a9d48ada46b14b2"
    },
    {
      "kl_teacher_candidate": 0.24842992152789783,
      "top1_agreement": 0.7888446215139442,
      "top5_agreement": 0.7665338645418327,
      "top10_agreement": 0.7609561752988048,
      "tokens": 251,
      "mse": 0.7725849151611328,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "e10d0725412a2dd71bbfa587d42bcf02f8370e0866fb6309c7bbc277b9b0bbb0"
    },
    {
      "kl_teacher_candidate": 0.20258593492144467,
      "top1_agreement": 0.80859375,
      "top5_agreement": 0.77734375,
      "top10_agreement": 0.78515625,
      "tokens": 256,
      "mse": 0.6704878211021423,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "a3e0e5710e60cfa9c902d21ba88c852634c1fd405a069ddc664889180a12e216"
    },
    {
      "kl_teacher_candidate": 0.4619362677204751,
      "top1_agreement": 0.6822916666666666,
      "top5_agreement": 0.6802083333333334,
      "top10_agreement": 0.6635416666666667,
      "tokens": 192,
      "mse": 1.30034339427948,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "9a556067cf7ef9317da4136842e310452216af027017171a20a029415b8cf15b"
    },
    {
      "kl_teacher_candidate": 0.3479565750304916,
      "top1_agreement": 0.7309941520467836,
      "top5_agreement": 0.7309941520467836,
      "top10_agreement": 0.7450292397660819,
      "tokens": 171,
      "mse": 0.8309988379478455,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2447604609de8849282c9837d3a506264ec673041120075777bf3be5796dce2a"
    },
    {
      "kl_teacher_candidate": 0.28488912807492145,
      "top1_agreement": 0.75,
      "top5_agreement": 0.7328125000000001,
      "top10_agreement": 0.74609375,
      "tokens": 256,
      "mse": 0.824263870716095,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "c3dd6247b4a94611ed4d2068eacaa78b8e47c2c816c3f201080c60e8a9496108"
    },
    {
      "kl_teacher_candidate": 0.5042526514210786,
      "top1_agreement": 0.65625,
      "top5_agreement": 0.665625,
      "top10_agreement": 0.6785156250000001,
      "tokens": 256,
      "mse": 1.678918480873108,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2306de4fe1a934807ce55dac8e2258c8746808310b1a2ef16c055fd9cc885a1c"
    },
    {
      "kl_teacher_candidate": 0.40155355257955694,
      "top1_agreement": 0.74609375,
      "top5_agreement": 0.7015625000000001,
      "top10_agreement": 0.6953125,
      "tokens": 256,
      "mse": 1.300674319267273,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "786c11e0425302971af5291ef68fcac7778e512c37b4b2a4ba382e4298e3d16d"
    },
    {
      "kl_teacher_candidate": 0.8567643900089558,
      "top1_agreement": 0.5135135135135135,
      "top5_agreement": 0.5797297297297298,
      "top10_agreement": 0.5972972972972973,
      "tokens": 148,
      "mse": 2.1541566848754883,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "275cacf7924579eac7b6a7a79d53f8cb3772a2f76addca4bb90aa69b8ebd3b08"
    },
    {
      "kl_teacher_candidate": 0.32607762517218075,
      "top1_agreement": 0.7236842105263158,
      "top5_agreement": 0.7228070175438597,
      "top10_agreement": 0.7271929824561404,
      "tokens": 228,
      "mse": 1.0682395696640015,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "5397e46f73beb08cffcd0d8096e20a7441a6fb3855fde84e1e3d14cb76bd2ad3"
    },
    {
      "kl_teacher_candidate": 0.326083295712013,
      "top1_agreement": 0.7265625,
      "top5_agreement": 0.70859375,
      "top10_agreement": 0.73828125,
      "tokens": 256,
      "mse": 0.8762065768241882,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "86601f0b0b6666abd311e83f50bb5b3128c6464da16c0605bfd5daf1797068b8"
    },
    {
      "kl_teacher_candidate": 0.3259042991877754,
      "top1_agreement": 0.7674418604651163,
      "top5_agreement": 0.6868217054263567,
      "top10_agreement": 0.6976744186046512,
      "tokens": 129,
      "mse": 1.2444074153900146,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "73c9fa02a1d5c9699b35ca97798c92364071ee9d2a098ef437c59410fb4706b5"
    },
    {
      "kl_teacher_candidate": 0.3345670625784355,
      "top1_agreement": 0.7265625,
      "top5_agreement": 0.6890625,
      "top10_agreement": 0.716015625,
      "tokens": 256,
      "mse": 1.0262056589126587,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "ac23c801e12ae61f5a95428a1ce243c159cf80ea8a4073d5aeac03490aca9452"
    },
    {
      "kl_teacher_candidate": 0.2277413805991644,
      "top1_agreement": 0.80859375,
      "top5_agreement": 0.7484375000000001,
      "top10_agreement": 0.7488281250000001,
      "tokens": 256,
      "mse": 0.8895602226257324,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "9260271464222c17d68055266f93a531cb27b67d0a9e9e24ead03e72a543ad84"
    },
    {
      "kl_teacher_candidate": 0.30133251775040015,
      "top1_agreement": 0.7142857142857143,
      "top5_agreement": 0.7590062111801242,
      "top10_agreement": 0.7521739130434784,
      "tokens": 161,
      "mse": 0.9654389023780823,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "233c1919521264f41ca601795000b353a1d3ccd6e18200d82790cd2d17be2bbb"
    },
    {
      "kl_teacher_candidate": 0.32015312600199636,
      "top1_agreement": 0.75,
      "top5_agreement": 0.75,
      "top10_agreement": 0.7562500000000001,
      "tokens": 96,
      "mse": 0.8461523056030273,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "e0d13b6c30a5b62b537d73d42b33694b781313fdc03b8b41364b2d178fe2ada4"
    },
    {
      "kl_teacher_candidate": 1.7027431356950349,
      "top1_agreement": 0.49137931034482757,
      "top5_agreement": 0.48620689655172417,
      "top10_agreement": 0.4698275862068966,
      "tokens": 116,
      "mse": 3.249368667602539,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "7ab2864d6a0def3e3f2f13f85c64997df70b9ff7a62af8e22ba0824f43fb8f41"
    },
    {
      "kl_teacher_candidate": 0.6396497761522408,
      "top1_agreement": 0.627906976744186,
      "top5_agreement": 0.6697674418604651,
      "top10_agreement": 0.6488372093023256,
      "tokens": 86,
      "mse": 1.2404204607009888,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "3b18a85fe624a495d8d703893b3bf2a5ecdca759d804c150366a97d12e9d3cf5"
    },
    {
      "kl_teacher_candidate": 0.6864454323766322,
      "top1_agreement": 0.6049382716049383,
      "top5_agreement": 0.6222222222222223,
      "top10_agreement": 0.6061728395061728,
      "tokens": 81,
      "mse": 1.3445099592208862,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "0c4d614c7b5751c83837c3dbf170a75381c1d217d3956404f8acfb0c8ebec142"
    },
    {
      "kl_teacher_candidate": 0.6090993712385382,
      "top1_agreement": 0.5789473684210527,
      "top5_agreement": 0.6280701754385964,
      "top10_agreement": 0.6157894736842107,
      "tokens": 57,
      "mse": 1.2458363771438599,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "6d4ac25ae9bb8bad19c9cf1f81792999afec3b546c4aba6290ee1c54ff688a2c"
    },
    {
      "kl_teacher_candidate": 0.7121643621462858,
      "top1_agreement": 0.6,
      "top5_agreement": 0.62,
      "top10_agreement": 0.62,
      "tokens": 50,
      "mse": 1.3353629112243652,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "ff6ecd56d932edc982b64494d131f36db3f25fa1497470a2cca05e524637d7ba"
    },
    {
      "kl_teacher_candidate": 0.43217578567526094,
      "top1_agreement": 0.69921875,
      "top5_agreement": 0.71796875,
      "top10_agreement": 0.720703125,
      "tokens": 256,
      "mse": 1.3172941207885742,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "f1ec6863b73a2d44eeea7215fd6e2542518f369ebe3acc1c60cedd777db76abc"
    },
    {
      "kl_teacher_candidate": 0.28886882061923586,
      "top1_agreement": 0.75,
      "top5_agreement": 0.7429687500000001,
      "top10_agreement": 0.744140625,
      "tokens": 256,
      "mse": 1.024982213973999,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "2bc87d35f494968ee4e1cbb954d6b7536a6d9c955ae7b85ae760786387acf7df"
    },
    {
      "kl_teacher_candidate": 0.29358980232978593,
      "top1_agreement": 0.765625,
      "top5_agreement": 0.7468750000000001,
      "top10_agreement": 0.7582031250000001,
      "tokens": 256,
      "mse": 1.1777452230453491,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "c4def389fbb5b79cc96fb4d23a54ff229e461e11ba823a8b1bde3b8cb06fc00c"
    },
    {
      "kl_teacher_candidate": 0.3371360876939967,
      "top1_agreement": 0.734375,
      "top5_agreement": 0.7265625,
      "top10_agreement": 0.733984375,
      "tokens": 256,
      "mse": 0.96284419298172,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "f80b575d2b451b02e209d76fec902e7511e90d5b28c315a6e913209bf92b314f"
    },
    {
      "kl_teacher_candidate": 0.5595545516931748,
      "top1_agreement": 0.7890625,
      "top5_agreement": 0.61328125,
      "top10_agreement": 0.5960937500000001,
      "tokens": 256,
      "mse": 1.9640947580337524,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "d1409448f8b8485b5b1b604a249e6b13ec378afa5b320a2a11c95fd259254234"
    },
    {
      "kl_teacher_candidate": 0.29949892462441663,
      "top1_agreement": 0.83203125,
      "top5_agreement": 0.70625,
      "top10_agreement": 0.7132812500000001,
      "tokens": 256,
      "mse": 1.0896176099777222,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "5f315df7e409e6a5fc86542e0ead0fb6a200def065ee5d80647d2e7a6e8fa115"
    },
    {
      "kl_teacher_candidate": 0.2979372404951562,
      "top1_agreement": 0.73828125,
      "top5_agreement": 0.6828125,
      "top10_agreement": 0.710546875,
      "tokens": 256,
      "mse": 1.0774260759353638,
      "reload_exact": true,
      "reload_max_abs": 0.0,
      "teacher_sha256": "8571b5f526794fcf90e2133ac76430d0aa36a105f0d4844278bf9294f4b7507d"
    }
  ],
  "finished_utc": "2026-09-09T13:07:09.566310+00:00",
  "model_hashes": {
    "config.json": "456051833559de06c1c934c529294e9617aabf393b5da06af3a0cc45f52885ea",
    "generation_config.json": "7cfdf52ffe0e0c1c95ab0dc42aafc4117ba4c0d9804905cebb08eee36c6b7144",
    "quantize_config.json": "919bb9e30d345849d41dd6ab19b1db71bbaab79e98298677aee3172c91f7d182",
    "model-00001-of-00017.safetensors": "1c949d2b9304fc4e5379549d43e4d434bba8f0d756cda91f195b3ae6ba2504df",
    "model-00002-of-00017.safetensors": "b12bb4d57511b6f3c2db2fcfc22cbecbb05b8f2aa7d9433d413808ac33999fd9",
    "model-00003-of-00017.safetensors": "33801136584d03ef2be66960d40ea15df46795ab6d770874916a9de5a2feb337",
    "model-00004-of-00017.safetensors": "c506f2028af6055f78fd12286256fb5e7c9f21024efaeff34044ba1223b0c208",
    "model-00005-of-00017.safetensors": "1cf67bf853de127aecdf1c66e7378fcb26d7501fd047b8b12f0c0924fd8d3309",
    "model-00006-of-00017.safetensors": "a0dd6331b70561dfad99890858dac091e36a68df8d18a626c817998cf6f2d65c",
    "model-00007-of-00017.safetensors": "3ccf8d7108c83dbcd7f24658cb019e179d7f50e40c5d9da332d4c2798ee121c4",
    "model-00008-of-00017.safetensors": "c99bdeed54c698ae72f5f6c44e34247cf4d9601388a09d54f81c105f85c3acdf",
    "model-00009-of-00017.safetensors": "5470830c08994e468ec2912910fa6d6ba256ef7f4729dd1d8b5ea58e751fa377",
    "model-00010-of-00017.safetensors": "a1720eff8f1497fc023a1338c33a172737a2610218929cbd58c6dac3aaa2f463",
    "model-00011-of-00017.safetensors": "702960317b17e98013e7d5e063e459fe741b2648a1232b60bdad43f1ae044880",
    "model-00012-of-00017.safetensors": "5f5940b4e034bea810f7ad8c983762e2363f6109463810503e8581831b91a893",
    "model-00013-of-00017.safetensors": "4d86f559cc8334877a006a64a0b1ab212caa8d5c92ca2c998048dc8c87ecbf93",
    "model-00014-of-00017.safetensors": "f062fd4922e76c1867e073eeb7abd9ab957750f3383c87eacc8ea5a8f86ec089",
    "model-00015-of-00017.safetensors": "2287dc9ef8b06a54f121d6e3f40cbb44c6e63c66ecd99d3a47cc7fb6c444884e",
    "model-00016-of-00017.safetensors": "59f894de69b4c6b3fdb8c827ab83de54eea5c2eaab9589aedcadef9bf120afe3",
    "model-00017-of-00017.safetensors": "b562c6945710b198b5cc84e006a0fc944e7beacb71918a3897ecbe50e291b5d5",
    "model.safetensors.index.json": "7b2b487d6577403b4cba4073b4d07fd5614a3aeb79324dad9295d4ea450a2655",
    "chat_template.jinja": "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
    "tokenizer_config.json": "70cf4aae347755bdd07709be21edd11f9ab14b2697371b3016a52595149cb354",
    "tokenizer.json": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "gsq_training_run.json": "7a5e9f4b71f7b03c91f0411b001cbc6f1d6d3b4a6f3c94c00431c679c7868387"
  }
}
```
