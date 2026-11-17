# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-interop regressions for issue #4053 (items 1-3).

Every case failed on the parent commit:

* an AutoAWQ checkpoint that declares ``modules_to_not_convert: ["down_proj"]``
  was loaded with a zero-filled AWQ placeholder (NaN output) because the loader
  ignored the key;
* re-saving a loaded v1 checkpoint after setting ``format = gptq_v2`` copied the
  old v1 qzeros under the v2 label;
* ``save_pretrained()`` on a loaded model wrote the kernel's converted v2
  qzeros while ``checkpoint_format`` still said v1.

The synthetic fixtures are CPU-only and use an independent dense oracle for the
AWQ case; the GPTQ cases compare logits before and after the round trip.
"""

import json
import os
import shutil
import types

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig, RTNConfig
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 32
GROUP = 32
BITS = 4


def _save_tokenizer(directory):
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    ).save_pretrained(directory)


def _tiny_llama(seed=0):
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    return config, LlamaForCausalLM(config).half()


def _logits(model):
    with torch.inference_mode():
        return model(input_ids=torch.tensor([[1, 3, 4, 5]]), use_cache=False).logits.float()


def _relative_error(reference, actual):
    return (reference - actual).abs().max().item() / reference.abs().max().item()


def _build_awq_checkpoint(path):
    """Write an AWQ checkpoint whose ``down_proj`` is deliberately dense."""

    os.makedirs(path, exist_ok=True)
    config, dense = _tiny_llama()
    state = {name: value.clone() for name, value in dense.state_dict().items()}
    oracle = LlamaForCausalLM(config).half()
    oracle.load_state_dict(dense.state_dict())

    for name, linear in list(dense.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        if name.endswith("mlp.down_proj"):
            # The producer left this module unquantized; its dense weight must
            # survive the load instead of becoming a zero-filled AWQ placeholder.
            continue
        out_features, in_features = linear.out_features, linear.in_features
        groups = in_features // GROUP
        codes = torch.randint(0, 1 << BITS, (in_features, out_features), dtype=torch.int32)
        zeros = torch.randint(0, 1 << BITS, (groups, out_features), dtype=torch.int32)
        scales = (torch.rand(groups, out_features) * 0.02 + 0.001).half()
        qweight, qzeros = pack_awq(codes, zeros, BITS)
        state[f"{name}.qweight"] = qweight
        state[f"{name}.qzeros"] = qzeros
        state[f"{name}.scales"] = scales
        state.pop(f"{name}.weight", None)

        # Independent dense oracle: (code - zero) * scale, expanded per group.
        expanded_zeros = zeros.repeat_interleave(GROUP, dim=0).to(torch.float32)
        expanded_scales = scales.repeat_interleave(GROUP, dim=0).to(torch.float32)
        weight = (codes.to(torch.float32) - expanded_zeros) * expanded_scales
        oracle.get_submodule(name).weight.data.copy_(weight.t().to(torch.float16))

    save_file(
        {name: value.contiguous() for name, value in state.items()},
        os.path.join(path, "model.safetensors"),
        metadata={"format": "pt"},
    )
    config.architectures = ["LlamaForCausalLM"]
    config.quantization_config = {
        "quant_method": "awq",
        "bits": BITS,
        "group_size": GROUP,
        "zero_point": True,
        "version": "gemm",
        "modules_to_not_convert": ["down_proj"],
    }
    config.save_pretrained(path)
    _save_tokenizer(path)
    return oracle


def _build_gptq_v1_checkpoint(path):
    """Write a v1 GPTQ checkpoint and return its qzeros for byte assertions."""

    os.makedirs(path, exist_ok=True)
    config, model = _tiny_llama()
    qcfg = QuantizeConfig(bits=BITS, group_size=GROUP, desc_act=False, sym=True, format=FORMAT.GPTQ)
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=BITS,
            group_size=GROUP,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
        )
        scales = torch.full((linear.out_features, linear.in_features // GROUP), 0.01)
        zeros = torch.full_like(scales, 8)
        quant.pack_original(linear, scales, zeros)
        model.set_submodule(name, quant)
    convert_gptq_v2_to_v1_format(model, qcfg, TorchLinear)
    # The module was constructed for the v2 export format; after the explicit
    # conversion it holds v1 words and must emit them under the v1 label.
    for module in model.modules():
        if isinstance(module, TorchLinear):
            module._checkpoint_format_on_disk = FORMAT.GPTQ

    state = {name: value.clone() for name, value in model.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(path)
    qcfg.save_pretrained(path)
    _save_tokenizer(path)
    return state


def _load(path, backend=BACKEND.GPTQ_TORCH, device="cpu"):
    return GPTQModel.load(
        path,
        backend=backend,
        device=device,
        dtype=torch.float16,
        local_files_only=True,
    )


def test_awq_modules_to_not_convert_keeps_dense_module(tmp_path):
    source = tmp_path / "awq"
    oracle = _build_awq_checkpoint(str(source))

    model = _load(str(source), backend=BACKEND.AWQ_TORCH)
    down_proj = model.model.model.layers[0].mlp.down_proj

    assert type(down_proj) is torch.nn.Linear
    logits = _logits(model.model)
    assert torch.isfinite(logits).all()
    assert _relative_error(_logits(oracle), logits) < 2e-3


def test_optimum_inclusion_list_restricts_quantization():
    config = QuantizeConfig.from_quant_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "modules_in_block_to_quantize": [["self_attn.q_proj"], ["mlp.down_proj"]],
        }
    )

    assert config.external_module_inclusions() == ("self_attn.q_proj", "mlp.down_proj")


def test_awq_missing_qweight_is_rejected(tmp_path):
    source = tmp_path / "awq_missing"
    _build_awq_checkpoint(str(source))
    state = load_file(str(source / "model.safetensors"))
    state.pop("model.layers.0.self_attn.q_proj.qweight")
    save_file(state, str(source / "model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(ValueError, match="Missing required quantized weights"):
        _load(str(source), backend=BACKEND.AWQ_TORCH)


def test_resave_v1_to_v2_converts_qzeros(tmp_path):
    source = tmp_path / "v1"
    _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source))
    reference = _logits(model.model)

    model.quantize_config.format = FORMAT.GPTQ_V2
    target = tmp_path / "v2"
    model.save(str(target))

    source_qzeros = load_file(str(source / "model.safetensors"))[
        "model.layers.0.self_attn.q_proj.qzeros"
    ]
    saved_qzeros = load_file(str(target / "model.safetensors"))[
        "model.layers.0.self_attn.q_proj.qzeros"
    ]
    # v1 stores (zero_point - 1); v2 stores the logical zero point.
    assert not torch.equal(source_qzeros, saved_qzeros)

    reloaded = _load(str(target))
    assert _relative_error(reference, _logits(reloaded.model)) < 1e-5


def test_resave_refuses_unconvertible_format_change(tmp_path):
    source = tmp_path / "v1"
    _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source))
    model.quantize_config.format = FORMAT.MARLIN

    with pytest.raises(ValueError, match="requires re-quantization"):
        model.save(str(tmp_path / "marlin"))


def test_save_pretrained_emits_checkpoint_qzeros(tmp_path):
    source = tmp_path / "v1"
    state = _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source))
    reference = _logits(model.model)

    target = tmp_path / "hf"
    model.model.save_pretrained(str(target))
    shutil.copy(source / "quantize_config.json", target / "quantize_config.json")
    _save_tokenizer(str(target))

    saved_qzeros = load_file(str(target / "model.safetensors"))[
        "model.layers.0.self_attn.q_proj.qzeros"
    ]
    # The in-memory kernel converted v1 -> v2; the serialized checkpoint must
    # still carry the v1 words named by `checkpoint_format: gptq`.
    assert torch.equal(saved_qzeros, state["model.layers.0.self_attn.q_proj.qzeros"])

    reloaded = _load(str(target))
    assert _relative_error(reference, _logits(reloaded.model)) < 1e-5


def test_save_pretrained_key_set_matches_save(tmp_path):
    """Issue #4152: the HF writer must not add ``*.qzeros_format`` keys.

    ``GPTQModel.save()`` has always skipped the runtime layout marker, but HF
    ``save_pretrained()`` collected the persistent buffer, so the two save
    paths produced different checkpoints: one extra int64 tensor per quantized
    linear. Published checkpoints then carried keys that ``config.json`` and
    the GPTQ/AWQ format do not describe, which strict third-party loaders may
    reject or warn on.
    """

    source = tmp_path / "source"
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.float16).eval().save_pretrained(str(source))
    _save_tokenizer(str(source))

    qcfg = RTNConfig(bits=BITS, group_size=128, offload_to_disk=False, device="cpu")
    model = GPTQModel.load(
        str(source), quantize_config=qcfg, backend=BACKEND.GPTQ_TORCH
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(source))
    calibration = []
    for text in ("the quick brown fox", "a tiny llama calibration sample"):
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    model.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    reference = _logits(model.model)

    save_dir = tmp_path / "save"
    hf_dir = tmp_path / "hf"
    model.save(str(save_dir))
    model.model.save_pretrained(str(hf_dir))
    # ``save_pretrained`` writes tensors and config only; the quantize config
    # and tokenizer come from the save directory (as for any HF re-save).
    shutil.copy(save_dir / "quantize_config.json", hf_dir / "quantize_config.json")
    _save_tokenizer(str(hf_dir))

    def _tensor_keys(directory):
        keys = set()
        for name in os.listdir(directory):
            if name.endswith(".safetensors"):
                keys.update(load_file(os.path.join(directory, name)))
        return keys

    save_keys = _tensor_keys(str(save_dir))
    hf_keys = _tensor_keys(str(hf_dir))

    marker_keys = [key for key in hf_keys if key.endswith(".qzeros_format")]
    assert marker_keys == [], f"save_pretrained wrote layout markers: {marker_keys}"
    assert hf_keys == save_keys, (
        f"save_pretrained differs from save(): extra={sorted(hf_keys - save_keys)}, "
        f"missing={sorted(save_keys - hf_keys)}"
    )

    # Both save paths write the same tensors, so their reloads must agree
    # exactly. The in-memory comparison keeps a small fp16 tolerance: the
    # freshly quantized module and its fp16 payload can round differently on
    # some runtimes (measured 0.0 on CUDA and 4.4e-3 on the ROCm CPU runtime),
    # while the two serialized checkpoints are identical either way.
    reloaded_logits = {}
    for directory, label in ((save_dir, "save"), (hf_dir, "save_pretrained")):
        reloaded = _load(str(directory))
        reloaded_logits[label] = _logits(reloaded.model)
        assert _relative_error(reference, reloaded_logits[label]) < 5e-3, label
    assert torch.equal(reloaded_logits["save"], reloaded_logits["save_pretrained"])


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        (BACKEND.GPTQ_TORCH, "cpu"),
        (BACKEND.GPTQ_TRITON, "cuda"),
    ],
    ids=["gptq_torch", "gptq_triton"],
)
def test_load_state_dict_round_trip_preserves_v1_model(tmp_path, backend, device):
    """`load_state_dict(model.state_dict())` must be an identity on a v1 model.

    `state_dict()` emits checkpoint-layout v1 qzeros, but the Torch/Triton
    kernels decode v2. Copying the words back unchanged shifted every zero
    point by one while `strict=True` reported success.
    """

    if backend == BACKEND.GPTQ_TRITON and (
        not torch.cuda.is_available() or torch.version.hip is not None
    ):
        pytest.skip("gptq_triton requires a CUDA device")

    source = tmp_path / "v1"
    state = _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source), backend=backend, device=device)
    reference = _logits(model.model)

    qzeros_key = "model.layers.0.self_attn.q_proj.qzeros"
    round_trip = model.model.state_dict()
    assert torch.equal(round_trip[qzeros_key].cpu(), state[qzeros_key])

    result = model.model.load_state_dict(round_trip)
    assert not result.missing_keys and not result.unexpected_keys

    module = model.model.model.layers[0].self_attn.q_proj
    assert module.qzero_format() == 2
    assert _relative_error(reference, _logits(model.model)) < 1e-6
    # The live buffer holds v2 words; the next state_dict must still emit the
    # same checkpoint v1 words as the original file.
    assert torch.equal(model.model.state_dict()[qzeros_key].cpu(), state[qzeros_key])


def test_partial_load_state_dict_keeps_runtime_qzeros_layout(tmp_path):
    """A load without `qzeros` must not relabel the live v2 words as v1.

    PEFT adapters and other partial loads carry no quantized tensors. The old
    hook reset `_qzeros_format` from the checkpoint format anyway, so the next
    save wrote v2 words under the v1 label and the reload was garbage.
    """

    source = tmp_path / "v1"
    state = _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source))
    reference = _logits(model.model)

    qzeros_key = "model.layers.0.self_attn.q_proj.qzeros"
    module = model.model.model.layers[0].self_attn.q_proj
    live_words = module.qzeros.clone()
    assert module.qzero_format() == 2

    model.model.load_state_dict({}, strict=False)

    assert module.qzero_format() == 2
    assert torch.equal(module.qzeros, live_words)
    assert _relative_error(reference, _logits(model.model)) < 1e-6
    # The next checkpoint write must still emit the original v1 words.
    assert torch.equal(model.model.state_dict()[qzeros_key], state[qzeros_key])

    target = tmp_path / "hf"
    model.model.save_pretrained(str(target))
    saved_qzeros = load_file(str(target / "model.safetensors"))[qzeros_key]
    assert torch.equal(saved_qzeros, state[qzeros_key])


def test_optimized_torch_linear_model_pickles(tmp_path):
    """`torch.save(model)` must survive `optimize()`'s compiled callable."""

    source = tmp_path / "v1"
    _build_gptq_v1_checkpoint(str(source))
    model = _load(str(source))
    module = model.model.model.layers[0].self_attn.q_proj
    assert module.optimized
    if "dequantize_weight" not in module.__dict__:
        pytest.skip("torch.compile is disabled for this TorchLinear")

    target = tmp_path / "model.pt"
    torch.save(model.model, str(target))
    restored = torch.load(str(target), weights_only=False)

    assert "dequantize_weight" not in restored.model.layers[0].self_attn.q_proj.__dict__
    assert _relative_error(_logits(model.model), _logits(restored)) < 1e-6


def _optimized_pickle_kernel(kind):
    """Build and `optimize()` one CPU kernel that stores a compiled callable."""

    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear

    if kind == "torch":
        return TorchLinear(
            bits=4, group_size=32, sym=True, desc_act=False,
            in_features=128, out_features=64, bias=False,
            pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        )
    if kind == "fp8":
        return TorchFP8Linear(
            bits=8, group_size=-1, sym=True, desc_act=False,
            in_features=128, out_features=64, register_buffers=True,
        )
    if kind == "exl3":
        in_features, out_features, bits = 128, 128, 4
        generator = torch.Generator().manual_seed(731)
        return ExllamaV3TorchLinear.from_tensors(
            in_features=in_features,
            out_features=out_features,
            name="pickle_exl3",
            tensors={
                "trellis": torch.randint(
                    -32768, 32767,
                    (in_features // 16, out_features // 16, bits * 16),
                    dtype=torch.int16, generator=generator,
                ),
                "suh": (torch.rand(in_features, generator=generator) * 0.02 + 0.001).half(),
                "svh": (torch.rand(out_features, generator=generator) * 0.02 + 0.001).half(),
                "bias": torch.zeros(out_features, dtype=torch.float16),
            },
        )
    if kind == "vptq":
        module = VPTQLinear(
            bits=2, in_features=64, out_features=64, bias=False,
            vector_lens=[-1, 4], num_centroids=[-1, 16],
            num_res_centroids=[-1, -1], group_size=64, group_num=-1,
            enable_norm=False, enable_perm=False, dtype=torch.float16,
        )
        module._allocate_buffers(bias=False)
        return module
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["torch", "fp8", "exl3", "vptq"])
def test_optimized_cpu_kernels_pickle_round_trip(tmp_path, kind):
    """#4091 item 6: every kernel that stores a compiled callable must pickle."""

    module = _optimized_pickle_kernel(kind).eval()
    module.optimize()
    if not module.optimized:
        pytest.skip(f"torch.compile is disabled for {type(module).__name__}")

    compiled_attr = {
        "torch": "dequantize_weight",
        "fp8": "_compiled_forward",
        "exl3": "_compiled_forward",
        "vptq": "dequantize",
    }[kind]
    assert compiled_attr in module.__dict__, f"{type(module).__name__} did not compile"

    target = tmp_path / f"{kind}.pt"
    torch.save(module, str(target))
    restored = torch.load(str(target), weights_only=False)

    x = torch.randn(2, module.in_features, dtype=torch.float16)
    with torch.inference_mode():
        eager = restored(x)
        compiled = module(x)
    # The restored module runs eager; compiled and eager differ only by fp16
    # rounding in the fused epilogue.
    torch.testing.assert_close(eager, compiled, rtol=1e-2, atol=1e-5)

    assert restored.optimized is False
    if compiled_attr == "_compiled_forward":
        # The FP8/EXL3 forward reads the attribute directly.
        assert restored._compiled_forward is None
    else:
        assert compiled_attr not in restored.__dict__
        # The eager class method is restored instead of a compiled shadow.
        assert isinstance(getattr(restored, compiled_attr), types.MethodType)


def test_hf_from_pretrained_refuses_packed_gguf(tmp_path):
    """A GPT-QModel GGUF checkpoint must not silently random-init in HF."""

    from transformers import AutoModelForCausalLM

    from gptqmodel._quant_methods import GGUF_EXPORT_QUANT_METHOD

    source = tmp_path / "gguf"
    _build_gptq_v1_checkpoint(str(source))
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {
        "method": "gguf",
        "quant_method": GGUF_EXPORT_QUANT_METHOD,
        "bits": BITS,
    }
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="packed GGUF checkpoints cannot be loaded"):
        AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True)


def _runtime_layout_shell(cls, flag: str):
    """A module flagged as holding an unrecoverable runtime layout."""

    module = object.__new__(cls)
    # Bypass the kernel constructor (it needs a device and real tensors) but
    # still initialize the nn.Module bookkeeping that ``state_dict()`` uses.
    torch.nn.Module.__init__(module)
    module._qzeros_format = 2
    setattr(module, flag, True)
    return module


@pytest.mark.parametrize(
    "quant_method",
    ["bitsandbytes", "exl3", "qqq", "paroquant", "macaq", "nf12", "tt-blockfloat"],
)
def test_hf_from_pretrained_refuses_every_refused_method(tmp_path, quant_method):
    """Plain transformers must raise instead of loading random weights (#4091 item 5)."""

    from transformers import AutoModelForCausalLM

    from gptqmodel._quant_methods import REFUSED_HF_QUANT_METHODS

    assert quant_method in REFUSED_HF_QUANT_METHODS
    source = tmp_path / quant_method
    _build_gptq_v1_checkpoint(str(source))
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {
        "method": quant_method,
        "quant_method": quant_method,
        "bits": BITS,
    }
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError) as excinfo:
        AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True)

    message = str(excinfo.value)
    assert "cannot be loaded by `transformers`" in message
    assert quant_method in message


@pytest.mark.parametrize(
    ("cls", "flag"),
    [
        (MarlinLinear, "_marlin_repacked"),
        (ExllamaV2Linear, "_native_shuffled"),
    ],
    ids=["marlin", "exllamav2"],
)
def test_runtime_layout_state_dict_is_readable_but_checkpoint_writes_refuse(cls, flag):
    """Issue #4091 item 4: PEFT calls ``state_dict()`` and must not fail.

    ``set_peft_model_state_dict`` reads ``model.state_dict()``, so an
    unconditional raise in ``_save_to_state_dict`` broke adapter loading on the
    default CUDA kernel. Only a real checkpoint write must fail closed; a
    read-only collection returns the runtime tensors.
    """

    from gptqmodel.utils.checkpoint_state import checkpoint_serialization

    module = _runtime_layout_shell(cls, flag)

    # Read-only collection: the PEFT path.
    assert module._save_to_state_dict({}, "", False) is None
    assert module.state_dict() == {}

    # A checkpoint write must still refuse instead of publishing the layout.
    with checkpoint_serialization():
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            module._save_to_state_dict({}, "", False)


def _build_gptq_v2_checkpoint(path):
    """Write a ``gptq_v2`` checkpoint of the same weights as the v1 fixture."""

    os.makedirs(path, exist_ok=True)
    config, model = _tiny_llama()
    qcfg = QuantizeConfig(bits=BITS, group_size=GROUP, desc_act=False, sym=True, format=FORMAT.GPTQ_V2)
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=BITS,
            group_size=GROUP,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
        )
        scales = torch.full((linear.out_features, linear.in_features // GROUP), 0.01)
        zeros = torch.full_like(scales, 8)
        quant.pack_original(linear, scales, zeros)
        model.set_submodule(name, quant)

    state = {name: value.clone() for name, value in model.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(path)
    qcfg.save_pretrained(path)
    _save_tokenizer(path)
    return state


@pytest.mark.parametrize(
    ("source_kind", "target_kind"),
    [("v1", "v2"), ("v2", "v1")],
    ids=["v1_into_v2_model", "v2_into_v1_model"],
)
def test_state_dict_copy_between_v1_and_v2_models(tmp_path, source_kind, target_kind):
    """Issue #4091 item 3: a state dict must describe its own qzero layout.

    ``_load_from_state_dict`` used to assume the incoming words matched the
    *target's* ``_checkpoint_format_on_disk``. Nothing recorded the source
    layout, so copying a state dict between a v1 and a v2 model shifted every
    zero point while ``strict=True`` reported success.
    """

    paths = {}
    for kind, builder in (("v1", _build_gptq_v1_checkpoint), ("v2", _build_gptq_v2_checkpoint)):
        directory = tmp_path / kind
        builder(str(directory))
        paths[kind] = str(directory)

    source = _load(paths[source_kind])
    target = _load(paths[target_kind])
    reference = _logits(target.model)

    state = source.model.state_dict()
    assert state._metadata["model.layers.0.self_attn.q_proj"]["qzeros_format"] in (1, 2)

    result = target.model.load_state_dict(state)
    assert not result.missing_keys and not result.unexpected_keys

    # The copy is between models with identical weights, so it must be a no-op
    # on the output regardless of the two on-disk qzero conventions.
    assert _relative_error(reference, _logits(target.model)) < 1e-6

    module = target.model.model.layers[0].self_attn.q_proj
    assert module.qzero_format() == 2
    # The live words must match the source module's logical zero points, not
    # the raw words that were on disk in either checkpoint.
    source_module = source.model.model.layers[0].self_attn.q_proj
    assert torch.equal(module.qzeros, source_module.qzeros)


@pytest.mark.cuda
def test_marlin_state_dict_is_readable_for_peft():
    """Issue #4091 item 4: PEFT loads an adapter by reading ``state_dict()``.

    ``set_peft_model_state_dict`` calls ``model.state_dict()``. Marlin used to
    raise unconditionally once it repacked, so merely *loading* an adapter
    trained on another kernel failed. A read-only collection now returns the
    runtime tensors; only a checkpoint write refuses.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Marlin")

    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.utils.checkpoint_state import checkpoint_serialization
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error
    from gptqmodel.utils.packing import hf_pack_layer, hf_repack_layer

    if not marlin_runtime_available(torch.float16):
        pytest.skip(f"Marlin runtime unavailable: {marlin_runtime_error(torch.float16)}")

    device = "cuda:0"
    linear = torch.nn.Linear(128, 64, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    scales = torch.full((64, 1), 0.01)
    g_idx = torch.arange(128, dtype=torch.int32) // 128

    source = TorchLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    hf_pack_layer(source, linear, scales, None, g_idx, checkpoint_format=FORMAT.GPTQ_V2)
    source.to(device)

    marlin = hf_repack_layer(source, BACKEND.MARLIN, device=device)
    assert isinstance(marlin, MarlinLinear)
    assert marlin._marlin_repacked

    # The PEFT path: read-only collection must succeed and expose the runtime
    # tensors that the kernel actually holds.
    state = marlin.state_dict()
    assert torch.equal(state["qweight"], marlin.qweight)
    assert torch.equal(state["scales"], marlin.scales)

    # A checkpoint write must still fail closed.
    with checkpoint_serialization():
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            marlin.state_dict()
