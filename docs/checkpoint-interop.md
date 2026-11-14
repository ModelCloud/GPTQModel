# Checkpoint interop: save, Marlin `g_idx`, format flags, export, and mixed-bit dequantization

This page describes how GPT-QModel reads third-party checkpoints and how it
saves and exports them without changing the caller's live model. It covers the
interop fixes shipped for issue #4053 items 4-8: `save()` model replacement,
Marlin activation-order indices, `is_marlin_format`, `GPTQModel.export`, and
per-module `dynamic` bit widths in `dequantize_model`.

The rule behind all five flows is the same: the checkpoint's stored metadata
and tensors are authoritative. A runtime layout, a legacy flag, or a global
default must not silently replace them.

## 1. Save keeps the live model

`save_quantized()` needs a shell model that owns the raw checkpoint tensors for
loaded quantized models. It builds that shell with
`get_model_with_quantize()` but keeps it in a local (`model_for_save`).
`self.model` is never replaced, so inference after `save()` still uses the
kernel the caller loaded.

```mermaid
flowchart TD
    A["save_quantized()"] --> B{"load_quantized_model<br/>and format != EXL3?"}
    B -- no --> C["model_for_save = self.model"]
    B -- yes --> D["model_for_save = get_model_with_quantize(qcfg, model_local_path)"]
    D --> E["write config from model_for_save"]
    C --> E
    E --> F["get_state_dict_for_save(model_for_save)"]
    F --> G["write safetensors shards"]
    G --> H["self.model unchanged"]
    H --> I["caller inference keeps live kernel"]
    style D fill:#fff4cc,stroke:#b8860b
    style H fill:#d9f2d9,stroke:#2e7d32
```

Before this change `save_quantized()` assigned the shell to `self.model`. For a
Marlin checkpoint that shell was an fp16 `TorchCpuAvx512Linear` model with no
`post_init`, and the caller's logits moved by relative 1.47 after `save()`.

## 2. Marlin honors a non-contiguous `g_idx`

`MarlinLinear.post_init()` now compares the stored `g_idx` with the contiguous
mapping `arange(K) // group_size`. When `desc_act=False` but the index is
non-contiguous, the module switches to the activation-order path: it sorts the
index with `marlin_sort_g_idx()`, repacks `qweight` with
`g_idx_sort_indices`, and sets `is_k_full=True`. Shapes that cannot run the
activation-order kernel fail closed instead of decoding contiguous groups.

```mermaid
flowchart TD
    A["MarlinLinear.post_init()"] --> B{"self.desc_act?"}
    B -- yes --> F["activation-order path"]
    B -- no --> C{"_g_idx_is_contiguous(g_idx, K, group_size)?"}
    C -- yes --> D["contiguous path: empty g_idx"]
    C -- no --> E{"marlin_is_tile_aligned(N, K)?"}
    E -- no --> X["raise NotImplementedError<br/>load a non-Marlin backend"]
    E -- yes --> G["set desc_act = True<br/>is_k_full = True"]
    G --> F
    F --> H["marlin_sort_g_idx + gptq_marlin_repack"]
    style X fill:#f8d7da,stroke:#b02a37
    style G fill:#fff4cc,stroke:#b8860b
```

The motivating checkpoint, `Qwen2-1.5B-Instruct-GPTQ-4bits-gp32`, declares
`desc_act: false` and stores a block-permuted `g_idx`. Layer-0 `q_proj`
relative error against the Torch decode was 0.13885 before the change and
0.00000 after it.

## 3. `is_marlin_format` accepts `false`

AutoGPTQ and hugging-quants checkpoints persist `is_marlin_format: false`.
Only the removed `true` value requested the legacy Marlin runtime layout, so
`false` (or a string `"false"`) is dropped silently and `true` still raises.

```mermaid
flowchart TD
    A["quantize_config payload"] --> B{"has is_marlin_format?"}
    B -- no --> D["parse normally"]
    B -- yes --> C{"_is_truthy_compat_flag(value)?"}
    C -- true --> X["raise ValueError"]
    C -- false --> E["drop the field"]
    E --> D
    style X fill:#f8d7da,stroke:#b02a37
```

## 4. `export(..., "hf")` routes through the checkpoint dequantizer

`GPTQModel.export()` sends `format="hf"` to
`gptqmodel.utils.model_dequant.dequantize_model()` before selecting a runtime
backend. That decoder handles AWQ GEMM/GEMV layouts, GPTQ, quantized
embeddings, and checkpoints that only ship `quantize_config.json`. The output
is a dense Transformers directory that reloads with
`AutoModelForCausalLM.from_pretrained()`.

```mermaid
flowchart TD
    A["GPTQModel.export(model, target, format)"] --> B{"format == hf?"}
    B -- yes --> C["resolve local snapshot"]
    C --> D["dequantize_model(source, target, trust_remote_code)"]
    D --> E["config.json without quantization_config"]
    E --> F["dense safetensors + copied aux files"]
    B -- no --> G["AutoConfig + backend selection"]
    G --> H["GPTQModel.load"]
    H --> I["format == mlx: convert_gptq_to_mlx_weights"]
    style D fill:#fff4cc,stroke:#b8860b
    style F fill:#d9f2d9,stroke:#2e7d32
```

The old path loaded the model with a runtime backend and called
`nn_modules.qlinear.torch.dequantize_model()`. That function accepted only
`TorchLinear` (so AWQ failed), turned quantized embeddings into `nn.Linear`
(shape error on forward), and hard-coded fp16. The new path never selects a
runtime kernel for `hf`.

## 5. `dequantize_model` resolves per-module bits

`convert_gptq_file()` resolves each module's width through
`dynamic_get(dynamic, prefix, "bits", default)`, mirroring the AWQ path. The
default itself is normalized with `quant_bits_width`, so a string `"4"` is
accepted. `dequantize_model()` also merges a legacy `quantize_config.json`
into `config.json`'s `quantization_config` when the latter is missing fields.

```mermaid
flowchart TD
    A["convert_gptq_file(path, config)"] --> B["default_bits = quant_bits_width(config.bits)"]
    B --> C["for each module prefix"]
    C --> D["bits = _gptq_module_int(config, prefix, bits, default_bits)"]
    D --> E{"bits == 3?"}
    E -- yes --> F["3-bit unpack path"]
    E -- no --> G["standard unpack_rows / unpack_cols"]
    F --> H["fp64 weight reconstruction"]
    G --> H
    H --> I["finalize_for_save(target_dtype)"]
    style D fill:#fff4cc,stroke:#b8860b
```

A mixed 3-bit body with a 4-bit `lm_head` previously decoded every module with
the 3-bit default and raised
`3-bit GPTQ qzeros expects columns divisible by 3`. A 4-bit body with an 8-bit
`lm_head` raised a size mismatch. Both now decode exactly against an fp64
oracle.

## Failure contract

| Situation | Behavior |
| --- | --- |
| `is_marlin_format: true` | `ValueError` (removed layout) |
| `is_marlin_format: false` | accepted and dropped |
| Non-contiguous `g_idx`, `desc_act=False`, tile-aligned | activation-order decode |
| Non-contiguous `g_idx`, `desc_act=False`, misaligned | `NotImplementedError`; use a non-Marlin backend |
| `export("hf")` on AWQ / quantized embeddings / legacy config | dense reloadable directory |
| Unknown/undetectable export format | `ValueError` from `detect_format` |
| `dynamic` bits override | applied per module, not globally |
