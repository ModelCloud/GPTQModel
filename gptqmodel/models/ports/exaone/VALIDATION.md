# GPT-QModel EXAONE 3 validation and integration

## Environment and scope

Linux CPU; CPython 3.14t with `PYTHON_GIL=0`, Transformers 5.23.0,
Torch 2.15.0+cpu. Commands below run from the repository root using
`/home/ubuntu/.venvs/gptqmodel-tf5/bin/python` as `python`.
No downloaded checkpoint Python was executed, no trained tensor payload was
downloaded, and no model/quantization algorithm outside the EXAONE unit changed.

| Qualification | Evidence |
| --- | --- |
| Architecture selection | No native `exaone` config/model mapping; package-owned config/model win all four trust/bypass combinations, including invalid-config failure followed by successful retry |
| Real released ownership | Exact full meta `state_dict()` matches four pinned safetensors-index schemas: EXAONE 3.0 7.8B and EXAONE 3.5 2.4B/7.8B/32B; tied-head alias handled explicitly |
| Live module plan | CPU/meta MHA/GQA/MQA; all seven linear and both excluded RMSNorm owners resolve; exact groups, Q/K/V roles, prefixes and existing shared fixture asserted |
| Numerical behavior | Independent source-equation oracle in FP32/FP16/BF16; six original RoPE types; rotated cached keys/values and continuation; MHA/GQA/MQA, embedding input, labels and tuple outputs |
| Generation/cache | Eager/SDPA, greedy/beam, causality, left padding, dynamic/static generation, cache reorder/crop/reset, long-to-short RoPE isolation and gradient-checkpointed backward |
| Failure/retry | Invalid config, malformed 2D/4D masks and legacy tuples; injected completed-write failures in decoder, final norm, head, loss and output encoding restore empty/populated dynamic/static caches, then match fresh full forward |
| Serialization | Dense tied/untied safetensors tensor/logit reload; real public AWQ calibration, packing of every projection, safetensors save and packed inference reload |
| AWQ policy | No fold/search/replay groups; actual plain quantization with `fallback=None`, tied/untied embeddings and Llama 3 RoPE; all 14 linears packed; nontrivial norm gains unchanged |

## Commands and measured results

```sh
PYTHON_GIL=0 python -m pytest -q \
  tests/test_exaone_local_port.py \
  tests/module_tree/test_exaone_local_port.py -x
```

**73 passed, 1 upstream deprecation warning in 12.17s.** The warning is the
Transformers 5 legacy method alias `gradient_checkpointing_enable`; the actual
backward and cache-disable assertions pass.

The same two EXAONE suites plus local config/path/checkpoint/processor suites,
InternLM/Baichuan/Brumby/Afmoe sibling suites, registered plan fixtures and
explicit plan contracts finish **709 passed, 2 skipped in 14.94s**. This uses
the broader command below with the two EXAONE suites added and the full leaf
inventory/AWQ-family suites omitted. It includes the final rollback changes.

The shared loader/registry/sibling regression command is:

```sh
PYTHON_GIL=0 python -m pytest -q \
  tests/test_local_model_loading.py tests/test_local_model_paths.py \
  tests/test_checkpoint_local_model.py tests/test_local_processor_loading.py \
  tests/test_internlm_local_port.py tests/test_baichuan_local_port.py \
  tests/test_brumby_local_port.py tests/test_afmoe_local_port.py \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_explicit_module_plan.py \
  tests/module_tree/test_registry_leaf_coverage.py \
  tests/test_awq_scaling_groups.py
```

**908 passed, 26 skipped, 3 xfailed, 2 failed in 149.56s** on parent base
`7e7832668` plus this port. Both failures reproduce unchanged at that base in
a separate worktree (**2 failed in 3.02s**), with these exact nodes:

```text
tests/module_tree/test_registry_leaf_coverage.py::test_every_decoder_leaf_is_declared_or_explicitly_excluded[lfm3]
tests/module_tree/test_registry_leaf_coverage.py::test_every_decoder_leaf_is_declared_or_explicitly_excluded[lfm3_vl]
```

Both report pre-existing undeclared `conv.conv` and `self_attn.g_proj` owners.
No LFM3 file was edited. Existing unsupported optional kernels/checkpoints and
known registry exclusions account for skips/xfails; they are not EXAONE proof.

```sh
python -m ruff check gptqmodel/models/definitions/exaone.py \
  gptqmodel/models/ports/exaone tests/test_exaone_local_port.py \
  tests/module_tree/test_exaone_local_port.py tests/models/test_exaone.py
pyright --pythonpath /home/ubuntu/.venvs/gptqmodel-tf5/bin/python \
  --pythonversion 3.14 --level error gptqmodel/models/ports/exaone
python -m compileall -q gptqmodel/models/ports/exaone \
  tests/test_exaone_local_port.py tests/module_tree/test_exaone_local_port.py
git diff --check
```

Ruff, compileall and whitespace checks pass; Pyright 1.1.407 reports
**0 errors, 0 warnings** for the port package.

## Red/green evidence

No missing symbols or collection failure is counted as red evidence.

| Regression node | Red behavior | Green behavior |
| --- | --- | --- |
| `test_public_dispatch_ignores_malicious_checkpoint_code[False-False]` | Original EXAONE definition at parent base rejects metadata with the HF error demanding execution of checkpoint custom code; port/test import and collection succeed | Definition-owned config/model load without checkpoint code, across all trust/bypass combinations; current test additionally rejects bad dimensions then retries successfully |
| `test_cache_reorder_crop_reset_and_failed_mask_retry` | Initial port did not reject an undersized key mask (`DID NOT RAISE`) | Shape validation happens before writes; warm cache unchanged, valid continuation/reorder/crop/reset matches full forward |
| `test_padded_generation_uses_current_transformers_cache_api[static-eager]` | Initial mask validation called removed `StaticCache.get_max_cache_shape`, raising AttributeError after actual generation started; reproduced separately | Current `get_mask_sizes` API supports static eager/SDPA generation parity |
| `test_failure_after_cache_writes_rolls_back_and_retries[decoder-False]` (before cold/warm parametrization) | Deliberate failure after a completed mutating decoder block left cache length 4 rather than 2 | Model/LM rollback restores previous layer tensors/counters; extended matrix covers cold/warm dynamic/static state and decoder/norm/head/loss/encoding sites |

## Parent-owned integration requests

No central JSON change is required: `MODEL_MAP['exaone']` already maps to
`ExaOneQModel`, and `registered_module_plans.json` already contains the exact
`ExaOneQModel/gptq` case. The port's live test asserts both existing entries.
The definition's module tree remains unchanged; GPTQ and AWQ linear groups
are both explicitly checked.

Add an EXAONE row to `docs/transformers5-model-ports.md`, linking
`../gptqmodel/models/ports/exaone/PROVENANCE.md` and this document. Copy the
selection/rollback diagram and before/after table from PROVENANCE.md into
that central reference (already README-linked) and the existing PR body.
The shared MANIFEST already includes port `.md`, `.json` and `NOTICE*` files.

Adding `exaone` only to `tests/test_awq_scaling_groups.py`'s
`PLAIN_QUANTIZATION_FAMILIES` does not add a family test: it is currently absent
from `FAMILY_MODEL_TYPES` and native `AutoConfig`. If central onboarding is
desired, also add it to that family list, return
`tests.test_exaone_local_port.tiny_config()` in `_tiny_config`, and construct
the model through `ExaOneQModel.loader.from_config(config)` rather than
`AutoModelForCausalLM` (follow the existing Agnes special case). Its policy is
plain, not neutral folding. The port-specific full public AWQ round trips
already run without these optional central additions.

## Unresolved limits (not qualification claims)

**GPU:** CUDA/ROCm/other accelerator kernels, FP16/BF16 device accuracy,
multi-GPU GIL=0 concurrency, compilation, FlashAttention and fusion are unrun.

**Uncached checkpoints:** Official EXAONE 3.0 needs gated access (HTTP 401).
Trained 3.0/3.5 forward, generation, tokenizer/chat-template integration and
evaluation were not run; no multi-GB payload is cached. Byte-identical source
verification and index/meta audits do not establish trained model quality.

**Explicit cache limits:** Legacy tuples, nonstandard sliding/quantized layer
types and offloaded caches are rejected, not silently approximated. No
thread-sharing guarantee is made for a caller sharing one mutable cache.
