# Model adapter map

## Main integration points

| Concern | Source |
| --- | --- |
| Adapter contract and tree builder | `gptqmodel/models/base.py` |
| Dense reference definition | `gptqmodel/models/definitions/llama.py` |
| MoE reference definition | `gptqmodel/models/definitions/qwen3_moe.py` |
| Imports and public `MODEL_MAP` | `gptqmodel/models/auto.py` |
| MoE lifecycle implementations | `gptqmodel/models/moe_lifecycle.py` |
| Reusable model test harness | `tests/models/model_test.py` |
| Small model example | `tests/models/test_tinyllama.py` |

## Module-tree notation

The top-level path reaches the repeating layers and uses `#` for the layer index. A dictionary then describes modules in forward order.

| Syntax | Meaning |
| --- | --- |
| `name:!` | Participates in inference/capture ordering but is not quantized. |
| `name:?` | Capture-only node; not quantized. |
| `name:0`, `name:1`, ... | Relative grouping; entries with the same numeric group share a block. |
| `name:moe` | Marks the scope as mixture-of-experts. |
| `runtime|checkpoint` | Ordered aliases; the first is the runtime shell name and later names are checkpoint alternatives. |
| nested `#` | Expert-index placeholder expanded using `dynamic_expert_index`. |

Flags can be combined. Confirm the builder output rather than reasoning from punctuation alone.

## Typical dense grouping

For a Llama-like layer, Q/K/V projections share the attention input and usually form one group, the attention output projection follows in another, gate/up projections share the MLP input, and down projection follows them. Norms are present for capture/order but are not quantized. This pattern is only a template; read the target forward method.

## MoE checks

- Expert count resolves from the correct root, text, or nested thinker config.
- Router/gate quantization policy is deliberate.
- Expert placeholders expand in forward order for both GPTQ and AWQ lifecycles.
- Shared experts remain in their actual segment rather than being moved to the end.
- Lifecycle hooks match gate/up/down or fused projection structure.
- Tiny tests cover at least two experts so index expansion is observable.
