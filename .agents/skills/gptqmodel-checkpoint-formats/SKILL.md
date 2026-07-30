---
name: gptqmodel-checkpoint-formats
description: Support, convert, or debug quantized checkpoint formats in GPT-QModel. Use for MXFP4, GGUF, safetensors sharding, vLLM/sglang compatibility, EoRA adapters, and format-specific loader/save paths.
---

# GPT-QModel checkpoint formats

Use when the task involves a specific quantized format (MXFP4, GGUF, EoRA, vLLM/sglang, custom safetensors layouts)
rather than a generic `GPTQModel.load`/`save`.

## When to use

- Adding MXFP4 CPU or GPU support (`gptqmodel_ext` or `nn_modules/qlinear` changes).
- Implementing or fixing GGUF loaders (`tests/test_gguf_qlinear_llama.py`, `tests/test_internal_gguf.py`, `tests/test_loader_gguf.py`, `tests/kernels/test_gguf_cpp.py`, `tests/models/test_llama3_2_gguf*.py`).
- Converting a GPT-QModel checkpoint for vLLM/sglang consumption.
- Adding EoRA adapter save/load formats and `marlin_lora` integration.
- Re-quantizing a checkpoint to a uniform `group_size` for a specific backend (e.g., W4G64 for Marlin MoE mega-kernel).

## Key files

- `gptqmodel/utils/model.py` (checkpoint discovery, `get_checkpoints`)
- `gptqmodel/models/writer.py` (`save_quantized`, `save_quantized_embeddings`)
- `gptqmodel/quantization/config.py` (`QuantizeConfig`, `FORMAT`, `group_size`, `dynamic`)
- `gptqmodel/nn_modules/qlinear/marlin.py`, `exllama*.py`, `triton.py` (backend-specific packing)
- `gptqmodel_ext/mxfp4_cpu_kernel.cpp` and `gptqmodel/nn_modules/qlinear/mxfp4_cpu.py` for MXFP4 CPU support
- `hw/komodo.md`, `hw/cannoe.md` for NPU formats
- `tests/test_gguf_qlinear_llama.py`, `tests/test_internal_gguf.py`, `tests/test_loader_gguf.py`, `tests/kernels/test_gguf_cpp.py`, `tests/models/test_llama3_2_gguf*.py`, `tests/test_ovis2_6_moe_support.py`

## Workflow

1. **Identify the format contract.**
   - What tensors are stored (`qweight`, `scales`, `qzeros`, `g_idx`), their shapes, dtypes, and packing layout.
   - What metadata is required (`bits`, `group_size`, `desc_act`, `sym`, `format`, `backend`, `moe_*` fields).
   - What external runtimes need (vLLM, sglang, ExLlama, Marlin, Transformers).

2. **Add a loader/save path without breaking existing formats.**
   - Keep the default `GPTQ`/`GPTQ_V2` path unchanged.
   - Gate the new format on explicit config flags or autodetected metadata.
   - Provide a fallback to the old path if the new format is not fully validated.

3. **Validate conversion round-trips.**
   - Save a model in the new format, reload it, and run a forward pass.
   - Compare against the pre-conversion dense or GPTQ reference on a small model.
   - Test with `tests/test_local_model_paths.py` and `tests/test_save_load_*.py` if they exist.

4. **Document compatibility limits.**
   - If a format needs `group_size` 64/128, `sym=True`, no `adapter`, etc., state it explicitly.
   - Keep unsupported combinations as clear fallback errors, not silent wrong outputs.

## Anti-patterns

- Do not change the default GPTQ pack layout without a format-version bump.
- Do not assume a converted checkpoint loads in upstream GPTQModel if it uses Ultra-only features
  (EoRA, MXFP4, packed-prefill).
- Do not delete original weights during conversion unless the user explicitly opts in.
- Do not leave format-specific code in generic `base.py` paths when a backend/format module is available.
