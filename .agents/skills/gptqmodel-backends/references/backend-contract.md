# Backend contract checklist

## Integration points

| Area | Source |
| --- | --- |
| Public backend enum and aliases | `gptqmodel/utils/backend.py` |
| Base classes and capability schema | `gptqmodel/nn_modules/qlinear/__init__.py` |
| Dynamic discovery and support maps | `gptqmodel/utils/importer.py` |
| Native extension registry | `gptqmodel/extension.py` |
| Existing implementations | `gptqmodel/nn_modules/qlinear/` |
| Selection tests | `tests/kernels/test_selection.py` |
| Hierarchy tests | `tests/kernels/test_qlinear_hierarchy.py` |
| Fallback tests | `tests/kernels/test_fallback.py` |

## Review questions

- Does the class inherit the packing and quantization semantics it actually uses?
- Are `SUPPORTS_METHODS`, `SUPPORTS_FORMATS`, bit widths, group sizes, `desc_act`, symmetry, sharding, training, padding, devices, platforms, pack dtypes, and activation dtypes explicit?
- Is auto-selection priority lower or higher than existing backends for a measured reason?
- Does explicit selection report the original availability or capability failure?
- Does auto selection fall back without changing quantization semantics?
- Are architecture checks based on runtime compute capability rather than product-name strings or fixed device indices?
- Does packing preserve the on-disk layout expected by save/reload and external consumers?
- Does `post_init` allocate derived state on the right device and release temporary buffers?

For a new kernel backend, require at least one shape that exercises each tail or padding path. Include multiple batch/token regimes when the launch strategy changes between matrix-vector and matrix-matrix workloads.
