---
name: gptqmodel-lazy-load-smoke-testing
description: End-to-end smoke test for GPT-QModel's lazy checkpoint load (LazyTurtle) and first-layer materialization through StageInputsCapture.cache_inputs. Use when verifying lazy-load behavior, first-layer input capture, shell_module_materialize module_path propagation, or CPU-only GPTQ quantization from a tiny fixture.
---

# GPT-QModel lazy-load smoke testing

This skill describes how to exercise the lazy checkpoint path on CPU and verify that the first decoder layer is correctly materialized during `StageInputsCapture.cache_inputs`.

## When to use

- A PR touches `gptqmodel/looper/module_looper.py`, `gptqmodel/looper/stage_inputs_capture.py`, `gptqmodel/models/base.py`, or `gptqmodel/utils/structure.py`.
- You need to verify that `shell_module_materialize` receives a dotted `module_path` such as `model.layers.0` instead of `None` or a class name.
- You want to confirm that `LazyTurtle.materialize_submodule` loads real parameters/buffers (not `meta`) and that forward/generate runs without `AttributeError` or `NaN`.

## Devin Secrets Needed

None for CPU-only smoke testing. If GPU testing is required, set `CUDA_VISIBLE_DEVICES` as appropriate and use `device="cuda"`.

## Environment prerequisites

1. Install system development headers; otherwise torch inductor / `cpp_extension` JIT builds for `pack_block_cpu` fail with `Python.h: No such file or directory`:
   ```bash
   sudo apt-get update -qq && sudo apt-get install -y -qq python3-dev python3.10-dev
   ```
2. Install the CPU torch stack and project dependencies:
   ```bash
   pip install -U pip
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   pip install -r requirements.txt
   pip install -e . --no-build-isolation --no-deps
   ```
3. Verify `gptqmodel` is importable from the repo root:
   ```bash
   python -c "import gptqmodel; print(gptqmodel.__version__)"
   ```

## Minimal fixture

Build a tiny `LlamaForCausalLM` and save it to a temp directory. The fixture should be small enough to quantize in seconds on CPU:

```python
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import torch, os, tempfile

config = LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=1,
    num_attention_heads=4,
    max_position_embeddings=128,
)
model = LlamaForCausalLM(config)
tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
tok_trainer = trainers.WordLevelTrainer(
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"], vocab_size=128
)
tok.train_from_iterator(
    ["the quick brown fox", "lorem ipsum dolor", "hello world"],
    trainer=tok_trainer,
)
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>")

tmp = tempfile.mkdtemp()
model.save_pretrained(tmp)
hf_tokenizer.save_pretrained(tmp)
```

## Lazy-load + input-capture test

```python
from gptqmodel import GPTQModel, BACKEND
from gptqmodel.models.base import QUANT_CONFIG_NAME
from gptqmodel.quantization.config import QuantizeConfig

calibration = [
    {"text": "the quick brown fox jumps over the lazy dog"},
    {"text": "lorem ipsum dolor sit amet"},
    {"text": "hello world this is a test"},
]

qcfg = QuantizeConfig(
    bits=4,
    group_size=32,
    desc_act=False,
    device="cpu",
)
model = GPTQModel.load(
    tmp,
    quantize_config=qcfg,
    backend=BACKEND.TORCH,
)
assert model.turtle_model is not None
```

## Capture the materialization path

Wrap `model.shell_module_materialize` to record the `module_path` passed during `cache_inputs`:

```python
calls = []
orig = model.shell_module_materialize

def capture(*args, **kwargs):
    calls.append((args, kwargs))
    return orig(*args, **kwargs)

model.shell_module_materialize = capture
model.quantize(
    calibration,
    batch_size=1,
    backend=BACKEND.GPTQ_TORCH,
    calibration_data_min_length=1,
)

first_call_args, first_call_kwargs = calls[0]
module_path = first_call_kwargs.get("module_path")
assert module_path == "model.layers.0", module_path
```

## Verify first-layer materialization

After `model.quantize` reaches `cache_inputs`, inspect the first decoder layer:

```python
first_layer = model.model.model.layers[0]
devs = {p.device.type for p in first_layer.parameters()}
assert "meta" not in devs, f"first layer still has meta parameters: {devs}"
assert not any(torch.isnan(p).any() for p in first_layer.parameters())
```

## Save, reload, and generate

```python
quantized_dir = os.path.join(tmp, "quantized")
model.save(quantized_dir)

qmodel = GPTQModel.load(
    quantized_dir,
    backend=BACKEND.GPTQ_TORCH,
    device="cpu",
)
in_ids = torch.tensor([[2, 5, 11]])
out = qmodel.generate(
    input_ids=in_ids,
    attention_mask=torch.ones_like(in_ids),
    max_new_tokens=4,
    do_sample=False,
)
assert out.shape == (1, in_ids.shape[1] + 4)
assert not torch.isnan(out).any()
```

## Expected log signals

- `INFO  Loader: using checkpoint-backed lazy turtle source for /tmp/...`
- `INFO  ModuleLooper: capturing layer inputs from N calibration batches`
- `[test] first materialized module path: 'model.layers.0'`
- `[test] generate output shape: torch.Size([1, 6])`
- `[test] PASS`

## Common issues

- `Python.h: No such file or directory` during `TorchLinear` forward: install `python3-dev`/`python3.10-dev` and rerun.
- `torchao` extension load warnings (`_C_cutlass_90a.so`, `_C_mxfp8...so`) are non-fatal on CPU.
- `Calibration dataset size should be more than 256` warnings are expected for tiny smoke fixtures.
- If `module_path` is `None`, the new resolution logic in `stage_inputs_capture.py` was not reached or `layer_names` was not propagated from `module_looper.py`.

## Regression test reference

- `tests/test_stage_inputs_capture.py` covers the `module_path` resolution paths.
- For end-to-end validation, run `smoke_lazy_load.py` from the repo root.
