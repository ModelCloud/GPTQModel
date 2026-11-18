# Third-Party Licenses

This file records the principal third-party components in the ZDTaichu5.0-9B
Hugging Face weight repository. File-level notices take precedence over this
summary.

| Component | Scope | Upstream | Governing terms |
|---|---|---|---|
| Qwen3.5-9B | Language backbone, tokenizer, and architecture | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache License 2.0 |
| C-RADIOv4-H | Vision backbone and derivative-model obligations | [nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H), revision `0057b339059c0b9e1b4ba996f975410ebbfdfcc8` | NVIDIA Open Model License Agreement |
| Hugging Face Transformers | Runtime interfaces and Qwen3.5 implementation | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache License 2.0 |

## License boundaries

- The root `LICENSE` contains the NVIDIA Open Model License Agreement that
  governs the C-RADIOv4-H model and applicable derivative-model obligations.
- Qwen3.5-9B artifacts remain under Apache License 2.0. A copy is provided in
  the GPT-QModel repository root `LICENSE`.
- Standalone-inference source files authored for this repository (for example
  `cradio_config.py` and `cradio_model.py`) are provided under the Apache
  License 2.0, as stated in their file headers.
- Source files carrying their own copyright or license headers remain governed
  by those file-level terms.
- Dependencies installed separately remain subject to their own licenses.

Redistribution must include the NVIDIA Open Model License Agreement and a
Notice file containing: `Licensed by NVIDIA Corporation under the NVIDIA Open
Model License`.

This summary is informational and does not replace the authoritative license
