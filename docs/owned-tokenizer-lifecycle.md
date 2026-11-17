<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Definition-owned tokenizer construction and normalization

GPT-QModel selects the definition's tokenizer class before normalizing its
instance. `LocalModelLoader.load_tokenizer` delegates legacy special-token
metadata preparation to `Tokenicer.prepare_load_kwargs`, then calls only the
package-owned class. Checkpoint `auto_map`, `trust_remote_code` and
`code_revision` cannot replace that class. Native tokenizer paths without an
owned class retain the selected model config and disable checkpoint Python
when the model has a local loader.

Preloading reads tokenizer JSON as data and preserves asset-selection kwargs
and explicit caller overrides. Reusable repairs belong in Tokenicer;
source vocabularies, BOS/EOS behavior and source-specific serialization belong
in their packaged tokenizer classes.

`LocalModelLoader.tokenicer_options` declares immutable normalization defaults
as option pairs. `load_hf_tokenizer` merges them with `setdefault`: an explicit
caller value, including `False` or `None`, wins. These options reach
`Tokenicer.load` for owned, native and already-constructed tokenizers; they do
not enter an owned tokenizer constructor. Options cannot replace the selected
config or checkpoint-code permission. Historical SentencePiece conversion is
an explicit provider profile, never inferred here from a model name or asset
hash. Tokenicer owns the conversion and its native-provider eligibility gate.

```mermaid
flowchart TD
    A["load_hf_tokenizer: selected model_config"] --> B{"Definition declares tokenizer_class?"}
    A --> O["Merge tokenicer_options defaults; explicit caller values win"]
    O --> G
    B -->|Yes| C["LocalModelLoader.load_tokenizer: strip checkpoint code selectors"]
    C --> D["Tokenicer.prepare_load_kwargs: selected JSON metadata only"]
    D --> E["Owned class from_pretrained"]
    D -->|Malformed selected metadata| F["Fail before construction; repair and retry"]
    E -->|Missing or corrupt assets| F
    E --> G["Tokenicer.load: normalize constructed instance once"]
    B -->|No| H["Tokenicer.load: native loading with selected config and effective trust"]
    H -->|Unavailable native tokenizer or denied code| F
    H --> I{"return_tokenicer?"}
    G --> I
    I -->|True: internal model load| J["BaseQModel.__init__: retain Tokenicer wrapper"]
    I -->|False: public helper default| K["Return native tokenizer"]
    J --> L["Processor receives wrapper.tokenizer; shared object preserved"]
    style F fill:#fdd,stroke:#c00
```

| Before | After | Code evidence |
| --- | --- | --- |
| Owned constructors received legacy dictionary-valued special tokens directly | Prepare metadata through the upstream API before construction | `models/local.py:72-80` |
| Internal loaders returned a normalized native tokenizer, and the constructor normalized it again | Internal calls request the wrapper; the constructor retains it | `utils/hf.py:2802-2845`, `models/loader.py` calls with `return_tokenicer=True`, `models/base.py:1054-1056` |
| A generic processor fixture inherited a release-specific tokenizer binding | Explicitly declare the fixture's generic tokenizer; retain separate real-release source tests | `tests/test_local_processor_loading.py`, `tests/test_internlm_tokenizers_local_port.py` |

The dependency floor is `tokenicer>=0.0.19`. At qualification time, the API is
in the **unreleased** candidate
[`c14e1c8c062c7b7cc9a981054366f98bde015fa1`](https://github.com/ModelCloud/Tokenicer/commit/c14e1c8c062c7b7cc9a981054366f98bde015fa1),
[Tokenicer PR #66](https://github.com/ModelCloud/Tokenicer/pull/66).
Published 0.0.18 lacks this API. The exact candidate was installed from source;
this dependency floor is a release prerequisite, not a claim of a PyPI release.

## Validation

An earlier combined suite passed **296 tests, no skips** in 24.34 seconds
(15 warnings). The current consumer/config/InternLM/norm-teacher and guard
batch passes **288 tests, one optional PEFT skip** in 31.90 seconds. The
profile-forwarding and tokenizer-consumer batch passes **48 tests, no skips**
in 3.62 seconds. These batches overlap and are not additive.
Restoring the prior `load_hf_tokenizer` implementation in memory produces
**six intended failures and six controls passing in 0.19s**: package defaults
are missing, or the profile option leaks into the owned constructor. The
profile cases also pin native/owned/instance identity, explicit False/None,
exact IDs/decode, once-only wrapping and rejection of security/config keys.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_GIL=0 \
  /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest -q --tb=short -ra -o addopts= \
  tests/test_local_tokenicer_options.py tests/test_local_tokenizer_loading.py
```
The public suite checks owned class selection, hostile checkpoint metadata,
legacy special-token flags and decode, input-asset immutability, subfolders,
missing/corrupt asset repair/retry, native return types and one normalization
per model load. The quantized consumer packs real AWQ linears and performs two
safetensors save/load cycles with identical token IDs, wrapper ownership and
logits. Removing preloading reproduces the legacy token-list construction
failure. Restoring the old native-tokenizer return reproduces duplicate
normalization in dense and quantized public loads (five failing cases in
0.45 seconds). Removing preloading fails at the intended native
`List[Union[str, AddedToken]]` check in 0.47 seconds.

The full real-release tokenizer matrix is not yet green on candidate
`c14e1c8`: three Baichuan2-7B public cases expose special-token-map precedence
in Tokenicer's preloader. They expect source IDs `[1, 46955, 2, 0]`; the
candidate splits the embedded EOS. The source assertions remain intact; this
requires an upstream repair before the dependency is releasable.

```bash
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest \
  tests/test_local_tokenizer_loading.py tests/test_local_model_loading.py \
  tests/test_local_processor_loading.py tests/test_trust_remote_code_isolation.py \
  tests/test_internlm_tokenizers_local_port.py tests/test_internlm_local_port.py \
  tests/test_internlm2_local_port.py tests/module_tree/test_internlm_local_port.py \
  tests/module_tree/test_internlm2_local_port.py tests/test_local_model_paths.py \
  -q --tb=short -ra -o addopts=
```

Qualification uses Python 3.14.6t with `PYTHON_GIL=0`, Torch 2.15 CPU,
Transformers 5.23.0, Tokenicer candidate 0.0.19 and SentencePiece 0.2.1.
It does not qualify GPU execution, trained-checkpoint quality or concurrent
model execution. Real-release source contracts are in each port's provenance
and tokenizer reference.
