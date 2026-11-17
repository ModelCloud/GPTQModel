# Model implementation selection and checkpoint code

GPT-QModel targets Transformers 5. Model definitions use native Transformers
implementations or modeling code shipped inside `gptqmodel`. Checkpoint
`auto_map` entries are untrusted data and cannot override a local port.

## Selection policy

`load_model_config` reads configuration data with
`PreTrainedConfig.get_config_dict`, preserving revision, authentication and
subfolder options. A definition with a `LocalModelLoader` owns both its config
and model class. It imports those classes lazily, calls the config class
directly and constructs the model with `_from_config` or `from_pretrained`.
It never registers checkpoint classes globally with Transformers.

```mermaid
flowchart TD
    A["GPTQModel.load / from_pretrained / from_quantized"] --> B["load_model_config: read config data"]
    B --> C{"Definition loader is LocalModelLoader?"}
    C -- yes --> D["LocalModelLoader.load_config: package config class"]
    D --> E["LocalModelLoader.from_config / from_pretrained: package model class"]
    C -- no --> F["resolve_trust_remote_code"]
    F --> G{"Declared remote-only layout?"}
    G -- no --> H{"Native Transformers loader mapping?"}
    H -- yes --> I["trust_remote_code=False: native config and model"]
    H -- no --> J{"Explicit trust AND safety bypass?"}
    G -- yes --> J
    J -- yes --> K["Checkpoint Python allowed"]
    J -- no --> L["RemoteCodeSecurityError or unsupported native config"]
    style L fill:#ffcdd2,stroke:#b71c1c
```

The existing security bypass is `GPTQ_BYPASS_SAFETY_CHECKS=1`. It is additional
consent: it does not turn trust on. For a remote-only implementation, both that
flag and the explicit API argument `trust_remote_code=True` are necessary.
Local ports ignore both for model/config selection. Native implementations
continue to override remote trust. A known remote-only checkpoint layout
(currently Bailing Hybrid MLA, selected by `q_lora_rank`) cannot silently fall
back to an incompatible native model.

For GGUF checkpoints, the explicitly selected `gguf_file` remains the config
source through the data read, trust preflight and final construction, including
when a subfolder is selected. An unrelated `config.json` cannot veto that
selection or introduce a remote-only layout marker. This also allows a
GGUF-only subfolder to load without a JSON config.

```mermaid
flowchart TD
    A["load_model_config: gguf_file and subfolder"] --> B["PreTrainedConfig.get_config_dict: selected GGUF metadata"]
    B --> C["local_model_loader: metadata model_type"]
    C --> D{"Definition owns local classes?"}
    D -- yes --> E["LocalModelLoader.load_config: same GGUF selection"]
    D -- no --> F["resolve_trust_remote_code: preserve gguf_file and subfolder"]
    F --> G["AutoConfig.from_pretrained: same GGUF selection"]
    F --> H["Unsupported selected metadata: permission gate or clear error"]
    style H fill:#ffcdd2,stroke:#b71c1c
```

Tokenizers receive remote trust only when both explicit trust and the safety
bypass are present. This gate does not change Tokenicer normalization.
Tokenizer construction also receives the policy-selected config as Transformers'
`config` argument, separately from Tokenicer's `model_config` normalization
argument. This prevents AutoTokenizer from importing checkpoint `AutoConfig`
while initializing a tokenizer for a local or native model, even when tokenizer
trust and the bypass are both enabled.

```mermaid
flowchart TD
    A["load_model_config: selected config"] --> B["load_hf_tokenizer: model_config"]
    B --> C["Tokenicer.load: model_config and config both receive selected config"]
    C --> D["AutoTokenizer: reuse config; skip AutoConfig discovery"]
    D --> E{"Tokenizer checkpoint Python requested?"}
    E -- no --> F["Native tokenizer"]
    E -- yes --> G{"Explicit trust AND safety bypass?"}
    G -- yes --> H["Checkpoint tokenizer allowed"]
    G -- no --> I["Native fallback or rejection; no checkpoint Python"]
    style I fill:#ffcdd2,stroke:#b71c1c
```

Processor loads use the resolved model trust. Definition pre-load hooks that
import checkpoint modules declare `before_model_load_requires_remote_code` and
run only when remote trust has been resolved and the bypass is enabled.

## Local model ports

Model ports use the shared loader contract. Put the config and model
implementation under a model-specific package
inside `gptqmodel`, preserving upstream license/attribution and recording the
source revision. Set the definition's loader explicitly:

```python
from ..local import LocalModelLoader

loader = LocalModelLoader(
    config_class="gptqmodel.models.model_name.configuration_model_name.ModelConfig",
    model_class="gptqmodel.models.model_name.modeling_model_name.ModelForCausalLM",
)
```

These strings are trusted source metadata; never construct them from checkpoint
fields. Both must name classes in `gptqmodel`. The model class must implement
the Transformers 5 `PreTrainedModel` construction contract. Construction keeps
the existing shared model-load lock; the frozen loader metadata does not cache
mutable model instances or modify Transformers' global registries.

Remove obsolete Transformers 4 pins, remote import hooks and remote-trust
requirements when wiring each model. Port composite config/model construction as
well: nested `auto_map` must not cause checkpoint code execution. Qualify
checkpoint key/conversion mappings, generation/cache behavior, save/reload,
real-model module-plan ownership, and the model's explicit AWQ policy. Native
support must be verified against the checkpoint layout, not inferred from
similar model names. Do not replace an architecture with a lookalike.

## Load, save, export and retry boundaries

The public dispatchers, ordinary and quantized loaders, quantized re-save shell,
and compressed-tensors dequantizer use `load_model_config`. The dequantizer
also selects the definition's local model loader for shell construction.
`prepare_remote_model_init_compat` resolves the policy before any dynamic import.
The Kimi K3/K3.1 reference loaders require the same security bypass in addition
to explicit trust. The bypass is checked on each attempt; a previous successful
remote load is not permission for the next load.
Direct SGLang and vLLM engine helpers also enforce the bypass before forwarding
remote trust to an external engine. Their runtime architecture support remains
the responsibility of those engines.

| Mechanism | Previous behavior (base revision `bb36e7fb2`) | Current behavior |
| --- | --- | --- |
| Remote-only implementation | Explicit trust alone allowed checkpoint Python (`gptqmodel/utils/hf.py:1011-1052`) | Explicit trust and the existing security bypass are both required |
| Native model with a remote-only layout marker | Kept remote trust (`gptqmodel/utils/hf.py:1024-1038`) | Local port, explicit bypass, or a clear failure; no incompatible native fallback |
| Config selection | `AutoConfig.from_pretrained` selected checkpoint config classes (`gptqmodel/models/auto.py:643`, `gptqmodel/models/loader.py:1887,2440`) | Definition-owned config class first, then gated AutoConfig |
| Re-save | Loaded config directly through AutoConfig (`gptqmodel/models/writer.py:2587`) | Uses the same local/native/security policy |
| Remote pre-load hooks | Could import checkpoint modules on a native load (`gptqmodel/models/loader.py:917-933`) | Declared remote hooks require resolved trust and the bypass |
| Tokenizer config discovery | AutoTokenizer could reload checkpoint `AutoConfig` despite a policy-selected local config | `load_hf_tokenizer` supplies the selected config through the existing Tokenicer API |

Tests in `tests/test_local_model_loading.py` exercise public load rejection,
actual checkpoint config and model imports under explicit bypass, permission
revocation, native/local precedence, all three public dispatchers, lazy local
meta construction, native multimodal mappings and subfolder selection.
They establish dispatch and security behavior, not numerical qualification of
the separate model ports.
