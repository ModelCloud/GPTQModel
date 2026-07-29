#!/usr/bin/env python3
"""Quick integration check: build a tiny GPT2 MXFP4 checkpoint and load it with GPTQModel.load(..., backend="mxfp4_cpu")."""

import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, PreTrainedTokenizerFast

import gptqmodel
from gptqmodel.nn_modules.qlinear.mxfp4_cpu import Mxfp4CpuLinear
from gptqmodel.quantization import MXFP4Config


def _replace_linears(module, parent_name=""):
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if name in ("lm_head", "wte"):
            continue
        if isinstance(child, torch.nn.Linear) or child.__class__.__name__ == "Conv1D":
            qlinear = Mxfp4CpuLinear(
                bits=4,
                group_size=-1,
                desc_act=False,
                sym=True,
                in_features=child.weight.shape[0] if child.__class__.__name__ == "Conv1D" else child.in_features,
                out_features=child.weight.shape[1] if child.__class__.__name__ == "Conv1D" else child.out_features,
                bias=child.bias is not None,
                use_vnni=False,
            )
            qlinear.pack_original(child, scales=None, zeros=None)
            setattr(module, name, qlinear)
        else:
            _replace_linears(child, full_name)


def _make_tokenizer(vocab_size: int):
    unk = "<|endoftext|>"
    vocab = {unk: 0}
    for i in range(1, vocab_size):
        vocab[f"t{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=unk))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens([unk])
    tok.add_tokens([f"t{i}" for i in range(1, vocab_size)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=unk,
        pad_token=unk,
        bos_token=unk,
        eos_token=unk,
        model_max_length=128,
    )


def main():
    vocab_size = 1024
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=64,
        n_layer=1,
        n_head=4,
        n_inner=256,
        torch_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.eval()
    _replace_linears(model)

    tmpdir = tempfile.mkdtemp(prefix="mxfp4_cpu_test_")
    model.save_pretrained(tmpdir)
    config.save_pretrained(tmpdir)

    tokenizer = _make_tokenizer(vocab_size)
    tokenizer.save_pretrained(tmpdir)

    qcfg = MXFP4Config()
    with open(Path(tmpdir) / "quantize_config.json", "w", encoding="utf-8") as f:
        json.dump(qcfg.to_dict(), f, indent=2)

    print(f"Saved test checkpoint to {tmpdir}")

    loaded = gptqmodel.GPTQModel.load(
        tmpdir,
        backend="mxfp4_cpu",
        device="cpu",
        dtype="bfloat16",
    )
    print("Loaded model:", type(loaded.model).__name__)

    for name, module in loaded.model.named_modules():
        if isinstance(module, Mxfp4CpuLinear):
            print(f"  {name}: {module.__class__.__name__}")

    tokenizer_loaded = AutoTokenizer.from_pretrained(tmpdir)
    inputs = tokenizer_loaded("t1 t2 t3", return_tensors="pt")
    with torch.inference_mode():
        out = loaded.model(**inputs)
    print("Logits shape:", out.logits.shape)
    print("ok")


if __name__ == "__main__":
    main()
