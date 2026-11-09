# NF12 installed extension selection

`gptqmodel.utils.nf12_ad.load` can reuse an installed upstream `nf12` package
or build the vendored extension. PyTorch creates a `torch.ops.nf12` namespace
when it is accessed, even if no operators are registered. The installed path
therefore checks that the three runtime operators used by GPT-QModel—`encode`,
`decode`, and `linear`—are callable before accepting that package.

```mermaid
flowchart TD
    A["nf12_ad.load"] --> B{"CUDA device available?"}
    B -- No --> X["Return False with availability error"]
    B -- Yes --> C["_load_installed imports nf12"]
    C --> D{"Package outside vendored source and load succeeds?"}
    D -- No --> F["_build_and_load vendored extension"]
    D -- Yes --> E{"encode, decode, linear callable in torch.ops.nf12?"}
    E -- Yes --> G["Reuse installed operators"]
    E -- No --> F
    F --> H{"Vendored build and load succeeds?"}
    H -- Yes --> I["_register_fakes; return True"]
    H -- No --> Y["Return False with build error"]
    G --> I
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y failure;
```

| Mechanism | Before | Current |
| --- | --- | --- |
| Installed package acceptance | `hasattr(torch.ops, "nf12")` accepted an empty namespace (`e0991f5:gptqmodel/utils/nf12_ad.py:180`) | `_load_installed` checks the callable `encode`, `decode`, and `linear` operators (`gptqmodel/utils/nf12_ad.py:180-184`) |
| Partial or no-op installed package | `_LOADED` could become `True`; the first operator call then failed | The installed path returns `False` and `load` tries the vendored extension; if that build fails, `load` reports the error and callers can use the Torch codec |

The fake installed-package cases in `tests/test_nf12_ad_runtime_load.py`
exercise empty, partial, and complete operator sets. No quantization math,
checkpoint representation, or inference storage changes in this flow.
