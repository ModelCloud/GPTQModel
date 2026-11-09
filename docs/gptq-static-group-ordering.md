<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPTQ static-group ordering

With `static_groups=True`, `GPTQ._quantize_impl` measures each group's scale and
zero point on the original weight columns. Group Aware Reordering then permutes
columns for the GPTQ update. The returned weight and `g_idx` use original column
order, so the static scale and zero arrays must also retain original group order.
This alignment was corrected in PR #3719.

## Previous flow

```mermaid
flowchart TD
    A["Measure static scales and zeros on original columns"] --> B["Reorder columns with final_perm"]
    B --> C["Quantize each column from its source static group"]
    C --> D["Return Q and g_idx to original column order"]
    D --> E["Apply inverse global group permutation to static scales and zeros"]
    E --> F["pack_block indexes the wrong group parameters"]
    classDef failure fill:#fdd,stroke:#c33
    class F failure
```

## Current flow

```mermaid
flowchart TD
    A["GPTQ._quantize_impl measures static groups on original W"] --> B["compose_final_perm reorders W and H"]
    B --> C["static_group_perm maps each reordered column to its source group"]
    C --> D["Eager loop or gptq_block_update computes Q"]
    D --> E["invert_perm restores original column order"]
    E --> F{"static_groups?"}
    F -- Yes --> G["Keep static scales and zeros in original group order"]
    F -- No --> H["Reorder dynamic scales and zeros with inverse global group permutation"]
    G --> I["g_idx selects the matching group for pack_block"]
    H --> I
    I --> J["TorchLinear.dequantize_weight reconstructs packed Q"]
    J --> K{"Independent oracle agrees?"}
    K -- No --> L["Regression test fails"]
    classDef failure fill:#fdd,stroke:#c33
    class L failure
```

| Mechanism | Previous | Current |
| --- | --- | --- |
| Static group lookup | `gptqmodel/quantization/gptq.py:2388` selected the source group after `final_perm` | Both eager and fused updates retain that mapping |
| Returned metadata | `gptqmodel/quantization/gptq.py:2500` inverted the static scales and zeros despite original-order `g_idx` | `gptqmodel/quantization/gptq.py:2500` preserves static order; dynamic groups still invert |
| Detection | The 2×4 test checked returned Q only | `tests/test_gptq_static_group_order.py` checks Q, scale, zero, `g_idx`, packed words, and dequantization against an independent affine oracle |

The packed roundtrip test uses 32 input and output channels, two different
16-column affine groups, a global group swap, and a local column permutation.
The CUDA fused/eager parity case uses a forced group swap when CUDA and the
fused extension are available.
