# TeleChat2 reproducible evidence

Base: `bf4003dabad138a9245e55c87151d354a632bb97`, the existing parent target
after the human main merge. No prerequisite-copy commits are part of this unit.
No shared code, fixtures, dependencies, CI workflows or PR lifecycle changed.

Environment: Python 3.14.6t, `PYTHON_GIL=0`, Transformers 5.23.0,
Torch 2.15.0+cpu, published Tokenicer 0.0.19 (`direct_url.json` absent).
SentencePiece 0.2.1 and 0.2.3 were installed and directly checked, not inferred
from a requirements file. Both run the exact real vocabulary fixtures.

## Regression sensitivity

| Probe | Observed intended failure | Repair/control |
| --- | --- | --- |
| Original config `auto_map` | Six selection/security reds | Six local config selections green |
| HF5 Python-backend BOS/EOS serialization | Public model/tokenizer reload reds | Owned serialization retains both construction flags |
| Native headless causal load | Did not raise; random `lm_head.weight` initialized | Explicit missing-weight failure, then complete loads still pass |
| Global K/V split instead of per-KV-head interleave | 8 numerical failures | 4 MQA controls pass; unmodified suite green |
| Remove half table rounding | 12 numerical failures | Unmodified suite green |
| Ignore post-layernorm residual policy | 6 numerical failures | 6 prenorm controls pass; unmodified suite green |

Each numerical mutation was applied individually to the actual runtime source,
tested with `-k independent_decoder`, and restored before final qualification.
The expected outputs use separately written source equations, not port helpers.

## Commands

Owned tests:

```bash
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest -q \
  tests/test_telechat2_local_port.py tests/module_tree/test_telechat2_local_port.py
```

Required AWQ onboarding baseline plus owned and shared loader/processor/fusion
consumers (one **actually observed** unrelated old-API test deselected):

```bash
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest -q \
  tests/test_telechat2_local_port.py \
  tests/module_tree/test_telechat2_local_port.py \
  tests/test_awq_scale_search_safety.py \
  tests/test_awq_moe_router_scaling.py \
  tests/test_awq_empty_groups.py \
  tests/test_awq_position_kwargs.py \
  tests/test_awq_scale_search_memory.py \
  tests/test_bloom_awq_replay.py \
  tests/module_tree/test_awq_relationships.py \
  tests/module_tree/test_bloom_awq_relationships.py \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_definition_subset_ordering.py \
  tests/test_local_model_loading.py tests/test_local_model_paths.py \
  tests/test_local_tokenizer_loading.py tests/test_local_tokenicer_options.py \
  tests/test_local_processor_loading.py tests/module_tree/test_fusable_models.py \
  -k 'not test_quantized_reload_retains_once_normalized_wrapper'
```

The shared test still calls
`loaded.quantize([sample, sample], batch_size=1, calibration_data_min_length=1)`
at the recorded base; the main-merged API deliberately rejects it. Before
exclusion the combined command produced **901 passed, 463 skipped, one failure**
in 29.96s. The parent was given the precise node for its shared migration.
No old-API shim or shared-file fix was added to this port.

The required AWQ baseline alone, together with owned tests, independently
passed 661 tests / 3 existing absent remote-checkpoint skips in 33.54s.
The 460 additional skips are NVIDIA Marlin hardware-only fusion cases, not
untested TeleChat cases. Two source metadata configs warn about legacy
`rope_scaling`; shared native fixtures add their own warnings.

SentencePiece 0.2.1 full owned run: **87 passed, zero skipped**, 4 warnings,
21.34s. After switching from dynamic lowercase SP aliases to canonical public
SWIG methods and `LoadFromSerializedProto`, the real-tokenizer subset was
rechecked: **14 passed**, 44 intentionally unselected, 2 SWIG warnings, 7.67s.
These are the same methods supported on 0.2.3. Final current-version results
are **87 owned passed, zero skipped**, 2 warnings, 37.65s; combined **901 passed,
463 skipped, one deselected**, 15 warnings, 48.26s. These final two processes
ran concurrently on the CPU VM, so their elapsed times are not benchmarks.

Additional manual check: repeated seeded native sampling compared every score
and sequence while preserving both caller and root generation configs. This
does not qualify a trained model's sampling quality.

Gates:

```bash
PATH=/home/ubuntu/.venvs/gptqmodel-tf5/bin:$PATH format/check.sh
uvx --from ruff==0.14.2 ruff check gptqmodel docs/eora tests setup.py
PYRIGHT_PYTHON_FORCE_VERSION=latest pyright \
  --pythonpath /home/ubuntu/.venvs/gptqmodel-tf5/bin/python \
  gptqmodel/models/ports/telechat2
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m compileall -q \
  gptqmodel/models/ports/telechat2 tests/test_telechat2_local_port.py \
  tests/module_tree/test_telechat2_local_port.py
git diff --check
```

The format gate reports the available Ruff 0.17.5 despite its installer pin;
the explicit isolated **Ruff 0.14.2 whole gate also passes**. Pyright's final
owned runtime check reports **0 errors, 0 warnings**. An initial run used the
wrong interpreter and another overlapped a dependency switch; neither is
represented as the final type-check result. No active pre-commit hook exists
in this checkout. Package discovery includes `gptqmodel.models.ports.telechat2`;
all 14 fixture SHA-256s match the manifest. The source PDF is byte-identical to
the pinned upstream copy. No trained tensor payloads were downloaded or saved
in the repository; all generated test weight payloads are safetensors.

Parent-only integration: exact proposed fusion reason in
`central_fixture_proposals.json`; registered module plans/AWQ family set are
unchanged. Central docs, main README and existing PR diagrams remain parent
owned. Package data must include LICENSE, NOTICE, PDF, JSON and gzip assets;
in particular preserve the model-license PDF instead of shipping its fixtures
under an Apache-only assumption. See README/NOTICE for license and hardware,
cache, quality and pretrained headless-model limitations.
