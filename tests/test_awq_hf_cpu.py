# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""CPU AWQ smoke tests using revision-pinned, trained Hugging Face checkpoints."""

import gc
import json
import time

import pytest
import torch
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig


_MODELS = [
    pytest.param("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", 210, id="dense"),
    pytest.param("FlameF0X/TinyMoE-100m-2x8-retrained", "0c1d097c964b949bc0092fa8e99c88c325421167", 280, id="moe"),
]
_CALIBRATION = [
    "The solar system contains planets that orbit the sun. Scientists observe their motion with telescopes. "
    "Gravity attracts objects with mass, and measurements help researchers test predictions about nature.",
    "A young child walked through the garden and found a small bird beside a tree. She watched quietly "
    "as it spread its wings and flew back to its nest. Then she went home to tell her family.",
    "To write a computer program, first describe the problem and identify the inputs. A function transforms "
    "those inputs into results. Tests compare the results with expected values and help find mistakes.",
    "The baker mixed flour with water and yeast before leaving the dough to rise. After an hour, he shaped "
    "the bread and placed it in a hot oven. The warm loaves were ready for the evening meal.",
]
_HELD_OUT = (
    "Water moves through the environment in a cycle. Heat from the sun causes evaporation from lakes and oceans. "
    "Clouds form when the vapor cools, and rain returns the water to the ground."
)


def _nll(logits, ids):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("repo,revision,expected_linears", _MODELS)
def test_real_checkpoint_awq_cpu_roundtrip(repo, revision, expected_linears, tmp_path, record_property, caplog):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        started = time.perf_counter()
        path = snapshot_download(
            repo, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        )
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        batch = tokenizer(_HELD_OUT, return_tensors="pt")
        baseline = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, attn_implementation="eager", trust_remote_code=False,
        ).eval()
        with torch.inference_mode():
            reference = baseline(**batch, use_cache=False).logits.float()
        assert torch.isfinite(reference).all()
        baseline_nll = _nll(reference, batch["input_ids"])
        del baseline
        gc.collect()

        qcfg = AWQConfig(bits=4, group_size=32, device="cpu", offload_to_disk=False)
        model = GPTQModel.load(
            path, quantize_config=qcfg, backend=BACKEND.TORCH,
            dtype=torch.float16, attn_implementation="eager", trust_remote_code=False,
        )
        calibration = [dict(tokenizer(text, return_tensors="pt")) for text in _CALIBRATION]
        model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
        packed = {name: module for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
        assert len(packed) == expected_linears
        assert all(module.qweight.dtype == torch.int32 for module in packed.values())
        if model.model.config.model_type == "mixtral":
            for layer in model.model.model.layers:
                assert not isinstance(layer.mlp.gate, AwqTorchLinear)
                assert layer.mlp.gate.weight.is_floating_point()
                assert layer.mlp.gate.weight.shape == (8, model.model.config.hidden_size)
                assert len(layer.mlp.experts) == 8
        else:
            assert "configured plain-quantization fallback" not in caplog.text
        with torch.inference_mode():
            actual = model.model(**batch, use_cache=False).logits.float()
            generated = model.generate(**batch, min_new_tokens=4, max_new_tokens=4, do_sample=False, use_cache=False)
        assert torch.isfinite(actual).all()
        relative_rmse = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
        quantized_nll = _nll(actual, batch["input_ids"])
        # Logit RMSE is not portable across CPU microarchitectures: torch dispatches to a
        # capability-specific kernel (SVE128 on aarch64, AVX2/AVX-512 on the x86 ubuntu-24.04
        # runners), so identical deterministic weights round differently. Measured for
        # SmolLM2-135M, torch 2.15.0, threads=2: 0.2073 on aarch64 (bit-identical at 842d666
        # and 89fa9f4) and 0.2582 on x86 CI. 241087a (#4176/#4243) moved this metric when
        # token-row replay began using the retained rows' real rotary positions instead of a
        # synthetic arange: 0.1827 -> 0.2073 RMSE, while the NLL delta improved 0.2259 ->
        # 0.1579. That is a documented correctness fix, not a quality regression, so the bound
        # is realigned to the measured range instead of the pre-fix value. 0.30 keeps ~16%
        # headroom over the worst observed value and still catches a real quality collapse;
        # the NLL-delta bound below is the primary quality gate.
        assert relative_rmse < 0.30
        assert quantized_nll - baseline_nll < 0.30

        destination = tmp_path / "quantized"
        model.save(str(destination))
        assert list(destination.glob("*.safetensors"))
        assert not list(destination.glob("*.bin"))
        assert not list(destination.glob("*.pt"))
        del packed, model
        gc.collect()
        restored = GPTQModel.load(
            str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
            dtype=torch.float16, attn_implementation="eager",
        )
        with torch.inference_mode():
            reloaded = restored.model(**batch, use_cache=False).logits.float()
            regenerated = restored.generate(**batch, min_new_tokens=4, max_new_tokens=4, do_sample=False, use_cache=False)
        torch.testing.assert_close(reloaded, actual, rtol=1e-3, atol=1e-3)
        assert torch.equal(regenerated, generated)
        assert generated.shape[-1] == batch["input_ids"].shape[-1] + 4
        metrics = {
            "repo": repo, "revision": revision, "packed_linears": expected_linears,
            "baseline_nll": baseline_nll, "quantized_nll": quantized_nll,
            "relative_logit_rmse": relative_rmse, "seconds": time.perf_counter() - started,
            "coverage_fallbacks": caplog.text.count("configured plain-quantization fallback"),
        }
        for name, value in metrics.items():
            record_property(name, value)
        print("AWQ_HF_CPU_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
        del restored
    finally:
        torch.set_num_threads(previous_threads)
        gc.collect()
