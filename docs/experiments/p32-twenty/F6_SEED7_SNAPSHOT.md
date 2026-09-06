# F6 seed 7: fixed experiment snapshot

Use this existing checkpoint directly for the twenty-experiment study. Do not re-quantize or substitute calibration data.

Snapshot directory:

```text
/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7
```

This is the historical **F6 YAQA 1.25×, seed 7** snapshot. Preserve its saved configuration, including
source weights `yaqa=1.25` and `nm=1.0`. It is distinct from the earlier unweighted F6 seed-0 recipe.
The previously attempted C4 quantization was cancelled and is not a study artifact.

## Provenance

| Field | Saved value |
| --- | --- |
| Model | /monster/data/model/Llama-3.2-1B-Instruct |
| Quantizer commit | `0f6cb786c071d00cbba76e7da9f29df80a8e5521` |
| Python | 3.14.7 |
| Torch | 2.13.0+cu130 |
| Python GIL enabled | True |
| YAQA seed | 7 |
| Checkpoint tensors in index | 558 |
| Weight shards | 17 |
| Total shard bytes | 915535582 |

All indexed shards were present at documentation time (2026-09-05). The SHA-256 values below were
computed from the current files. This documents artifact identity and file completeness; it does not
claim a new inference evaluation or a comparison against the stored per-tensor hash manifest.

## Complete saved quantization configuration

Exact contents of `quantize_config.json`:

```json
{
  "bits": 2,
  "dynamic": {
    "+:^model\\.layers\\.(6|8)\\.mlp\\.up_proj$": {
      "bits": 4,
      "format": "qvq"
    },
    "+:^model\\.layers\\.[0-9]+\\.self_attn\\.q_proj$": {
      "bits": 2
    },
    "+:^model\\.layers\\.[0-9]+\\.self_attn\\.k_proj$": {
      "bits": 2.5
    },
    "+:^model\\.layers\\.[0-9]+\\.self_attn\\.v_proj$": {
      "bits": 3.5
    },
    "+:^model\\.layers\\.[0-9]+\\.self_attn\\.o_proj$": {
      "bits": 4,
      "format": "qvq"
    },
    "+:^model\\.layers\\.[0-9]+\\.mlp\\.gate_proj$": {
      "bits": 3
    },
    "+:^model\\.layers\\.[0-9]+\\.mlp\\.down_proj$": {
      "bits": 3
    },
    "+:^model\\.layers\\.[0-9]+\\.mlp\\.up_proj$": {
      "bits": 3.5
    }
  },
  "group_size": -1,
  "lm_head": false,
  "method": "qvq",
  "quant_method": "qvq",
  "format": "qvq_v2b2_p32",
  "checkpoint_format": "qvq_v2b2_p32",
  "pack_dtype": "int32",
  "meta": {
    "calibration_paths": [
      "parquet-train.arrow"
    ],
    "quantizer": [
      "gptqmodel:7.3.3+ultra-local-git-0f6cb786"
    ],
    "timestamp": "2026-09-03T03:57",
    "uri": "https://github.com/modelcloud/gptqmodel",
    "damp_percent": null,
    "damp_auto_increment": null,
    "static_groups": null,
    "true_sequential": true,
    "mse": null,
    "scale_search": null,
    "gptaq": null,
    "foem": null,
    "act_group_aware": null,
    "fallback": {
      "strategy": "rtn",
      "threshold": "0.5%",
      "smooth": null
    },
    "offload_to_disk": false,
    "offload_to_disk_path": null,
    "pack_impl": "cpu",
    "gc_mode": "interval",
    "shard_strategy": "per_layer",
    "wait_for_submodule_finalizers": false,
    "auto_forward_data_parallel": true,
    "weight_only_quant_threads": null,
    "dense_vram_strategy": "exclusive",
    "dense_vram_strategy_devices": null,
    "moe_vram_strategy": "exclusive",
    "moe_vram_strategy_devices": null,
    "fused_forward": null,
    "native_kernel_replay": false
  },
  "sym": true,
  "codebook": "pgc16-v1",
  "trellis_window": 16,
  "vector_size": 2,
  "bank_count": 2,
  "tile_rows": 16,
  "tile_cols": 16,
  "rounding": "yaqa",
  "yaqa": {
    "seed": 7,
    "regularization": 0.15,
    "regularization_by_rate": [
      [
        2.0,
        0.15
      ],
      [
        2.5,
        0.02
      ],
      [
        3.0,
        0.02
      ],
      [
        3.5,
        0.02
      ],
      [
        4.0,
        0.02
      ]
    ],
    "minimum_sequences": 10178,
    "batch_size": 1,
    "chat_template": {
      "enabled": false,
      "content_weight": 0.97
    },
    "activation_checkpointing": true,
    "mps_cleanup_interval": 8,
    "sequence_sort": "desc",
    "source_weight_column": "source_name",
    "source_weights": [
      [
        "yaqa",
        1.25
      ],
      [
        "nm",
        1.0
      ]
    ],
    "max_factor_bytes_per_pass": null,
    "v2b2_family_mode": "reselect",
    "sample_strategy": "full",
    "spectral_refinement": false,
    "spectral_ranks": [
      8,
      16,
      32
    ],
    "spectral_lambdas": [
      0.1,
      0.25,
      0.5,
      1.0
    ],
    "spectral_push": false,
    "spectral_push_alphas": [
      0.25,
      0.5,
      1.0
    ],
    "spectral_localized": false,
    "spectral_localized_alphas": [
      0.25,
      0.5,
      1.0
    ],
    "spectral_localized_max_segments": 8,
    "spectral_localized_max_changes": 1,
    "spectral_localized_replay_candidates": 0,
    "spectral_localized_direct_replay_candidates": 0
  },
  "viterbi_pruning": {
    "mode": "auto",
    "strategy": "norm_band",
    "exact": true,
    "fallback": "baseline"
  },
  "incoherence": "rht",
  "module_scale_search": false,
  "output_channel_scale_optimization": false,
  "viterbi_objective": "euclidean",
  "tail_biting_candidates": 1,
  "viterbi_minimum_proxy_improvement": 0.0
}
```

## Recorded quantization datasets

These bindings are copied from the saved quantization run, including the verified historical Fisher manifest.
They describe the datasets used to produce this snapshot; no new dataset is selected here.

```json
{
  "calibration": {
    "config": null,
    "content_sha256": "26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef",
    "identity_manifest": null,
    "identity_manifest_sha256": null,
    "manifest_verified": false,
    "row_start": 0,
    "rows": 128,
    "source": "/monster/data/model/dataset/nm-calibration/llm.parquet",
    "split": "train"
  },
  "replay_confirmation": null,
  "replay_search": null,
  "validation": null,
  "yaqa": {
    "config": null,
    "content_sha256": "5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39",
    "identity_manifest": "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.manifest.json",
    "identity_manifest_sha256": "90f17c84200edc4cf25bd745729507c6e32b76f34622b3f44b34aabe8e10131e",
    "manifest_verified": true,
    "row_start": 0,
    "rows": 10178,
    "source": "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet",
    "split": "train"
  }
}
```

## Recorded disjointness evidence

```json
{
  "calibration_bindings": [
    {
      "path": "/monster/data/model/dataset/nm-calibration/llm.parquet",
      "selected_rows_sha256": "a74675bfed3834ac1c3140b2d7fb7d93d90a76608cc08515a5019a1cb2075ff4",
      "selection": {
        "row_start": 0,
        "rows": 10000,
        "strategy": "prefix"
      },
      "sha256": "26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef",
      "size_bytes": 12792319,
      "slices": [
        {
          "row_start": 0,
          "rows": 128
        }
      ]
    },
    {
      "path": "/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet",
      "sha256": "2140541facb66112428212b3a36d51a7735393b28c79db59c2429f6e51ed57ef",
      "size_bytes": 711386,
      "slices": [
        {
          "row_start": 0,
          "rows": 182
        }
      ]
    },
    {
      "path": "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet",
      "sha256": "5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39",
      "size_bytes": 6283798,
      "slices": [
        {
          "row_start": 0,
          "rows": 10178
        }
      ]
    }
  ],
  "evaluation_bindings": {
    "d300": {
      "path": "/root/qvq-data/divergence300-v1/divergence300-development.jsonl",
      "sha256": "701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2",
      "size_bytes": 53374772
    },
    "d300_locked": {
      "path": "/root/qvq-data/divergence300-v1/divergence300-locked.jsonl",
      "sha256": "17151e98b2e34587c9af58a6736c875c82854564f45c763f027090f35b8e8f58",
      "size_bytes": 59290198
    },
    "gsm8k_platinum": {
      "config": "main",
      "dataset": "madrylab/gsm8k-platinum",
      "split": "test"
    }
  },
  "path": "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json",
  "sha256": "f283eca649cbf1d2dcc160c202bc131c2462d4b3485b34c0d4bb5237d85a9d4e",
  "status": "pass",
  "strict_required": true
}
```

## File identity

| File relative to snapshot | Bytes | SHA-256 |
| --- | ---: | --- |
| `quantize_config.json` | 3826 | `e4f6db43d71c91d91570a5e8e19e5d97bd67e4142cf116f1e77549aa2debefce` |
| `qvq_quantize_run.json` | 1462860 | `b148618d3a0a3a961dafe6aab406509e73fed6f941e2c53ec40f147a37d4db57` |
| `model.safetensors.index.json` | 45132 | `6ad9b35d6dcc10bccfa9bf47ba5ed4958795beeb145050415be03e1e6e81b697` |
| `checkpoint_hashes.json` | 116391 | `ed335f88b24d7e998e8f4aa668f66ca9eb2caad53b8c1d250f24a53dcd77c4e7` |
| `config.json` | 5093 | `479cbae6f86d46d8422e61e86ecb5724264bfd14dade57f357db95de47da2e3c` |
| `tokenizer.json` | 17209920 | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
| `tokenizer_config.json` | 426 | `9bf3439e95c394672875c156fbb182c514ddb9645188feb318b49270af29cdcd` |
| `model-00001-of-00017.safetensors` | 24264302 | `d711d6cdd6eeb90bc4d2afdc90ef0a2a31c914216491681115ae77a3c1faf108` |
| `model-00002-of-00017.safetensors` | 24264302 | `a814ad8c0ae0dcd5f1bf1800c081e34ba056006eee078fd0539377918f27829b` |
| `model-00003-of-00017.safetensors` | 24264302 | `c0b0de9c4b6132d6eda35df390fec90f3205eb1a1a524a673109aea59170371e` |
| `model-00004-of-00017.safetensors` | 24264302 | `71ac42121682ae22f78a661face6f08dabfc6c149dcca01d3f4abc33cb337e17` |
| `model-00005-of-00017.safetensors` | 24264302 | `a097d2459c2723ac4b8020cddcb13d5d536708f816299f5b2b06394f1ea848d2` |
| `model-00006-of-00017.safetensors` | 24264302 | `36fa455afcd0a610bbc63f1b58c14c850077c34fbd98eb464de3fc1e981c01bf` |
| `model-00007-of-00017.safetensors` | 25247133 | `3d5fdb9c731ae3a740078e8410bc1e42200eecc6dab4c5c4320751234908986d` |
| `model-00008-of-00017.safetensors` | 24264302 | `dc9713306fd9c11c622b0aea9ade31f26f8c829654ab54d5616e583a47a5f898` |
| `model-00009-of-00017.safetensors` | 25247133 | `0dc2f1a419f6e74804ddfcb81f7a06eba9c63867c9a77db5f9e3b516f21b1fd0` |
| `model-00010-of-00017.safetensors` | 24264302 | `1a7278a992f77a2ae9e5df389a5d7c2bdbb588b341248730b2747b6531d76dfa` |
| `model-00011-of-00017.safetensors` | 24264334 | `32d65db4b2801103e373e0cf351fab2411130db4beef41ff11c66c04bb7aa585` |
| `model-00012-of-00017.safetensors` | 24264334 | `6272f1111ce49e4268593e7c6e940632a834491cb701710e16871f5c1a93bedb` |
| `model-00013-of-00017.safetensors` | 24264334 | `3f508ac92d95e7f76e3339b013f358459f0c1045f57ff850ad5a5ca9b979b959` |
| `model-00014-of-00017.safetensors` | 24264334 | `be2fc3580719c48cae52ddc8b1f204a51457498e6fd217bc9661c521e8216c83` |
| `model-00015-of-00017.safetensors` | 24264334 | `3a35d1ce091ad76370f0fcf6fe66ae394e806a8ebd22727853b4268b9c215667` |
| `model-00016-of-00017.safetensors` | 24264334 | `ce37069a179cc38ac97e4e5f862924db184df5f05ae81d4d06a2aac202b13401` |
| `model-00017-of-00017.safetensors` | 525340896 | `b562c6945710b198b5cc84e006a0fc944e7beacb71918a3897ecbe50e291b5d5` |

## Existing evaluation artifacts

These reports already exist in the snapshot directory; they were not rerun for this documentation commit.

- `post_quant_eval_divergence300.json`
- `post_quant_eval_gsm8k_platinum.json`
- `post_quant_eval_mmlu_humanities_fa2_cb_decode_graph_20260904.json`
- `post_quant_eval_mmlu_suite_fa2_cb_decode_graph_20260903.json`
- `post_quant_eval_result_divergence300.json`
- `post_quant_eval_result_divergence300_0c6d2c0e7630.json`
- `post_quant_eval_result_tasks.json`
- `post_quant_eval_result_tasks_04dbbf61a969.json`
- `post_quant_eval_result_tasks_a8cfcc725678.json`
- `post_quant_eval_result_tasks_decc555283e9.json`
- `post_quant_eval_result_tasks_f3aae27d375b.json`
