# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4450.

With the default ``offload_to_disk=True`` every finalized quantized module is
written to the run's offload directory and replaced by meta tensors behind an
accelerate hook. The loop restored only the base modules, so after
``quantize()`` the in-session model re-read its packed weights from disk on
every forward, could not be moved with ``.to()``, and broke with
``FileNotFoundError`` once the run directory finalizer ran at teardown.

The loop must leave the same model ``offload_to_disk=False`` produces: packed
modules in memory on CPU, no offload hooks and no meta tensors.
"""

import gc
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.offload import (
    bind_private_offload_root,
    is_meta_module,
    offload_to_disk,
    restore_disk_offloaded_modules,
)


GROUP = 32
BITS = 4


def _packed(linear: nn.Linear, seed: int) -> TorchLinear:
    """Pack ``linear`` as an asymmetric GPTQ module (v2 qzeros at runtime)."""

    generator = torch.Generator().manual_seed(seed)
    groups = linear.in_features // GROUP
    scales = torch.rand(linear.out_features, groups, generator=generator) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << BITS, (linear.out_features, groups), generator=generator, dtype=torch.int32)
    module = TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=FORMAT.GPTQ,
    )
    module.pack_original(linear, scales, zeros)
    return module


def _quantized_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).half().eval()
    names = []
    for index, (name, linear) in enumerate(list(model.named_modules())):
        if isinstance(linear, nn.Linear) and name != "lm_head":
            model.set_submodule(name, _packed(linear, seed=index))
            names.append(name)
    return model, names


def _offload_like_the_looper(model, names, disk_path):
    """Mirror ``stage_layer`` finalization: force-offload each packed module."""

    for name in names:
        offload_to_disk(model=model, module=model.get_submodule(name), disk_path=disk_path, force=True)
    assert all(is_meta_module(model.get_submodule(name)) for name in names)
    assert all(hasattr(model.get_submodule(name), "_hf_hook") for name in names)


def _logits(model, device="cpu"):
    ids = torch.tensor([[1, 5, 9, 13, 17]], device=device)
    with torch.inference_mode():
        return model(input_ids=ids).logits.float().cpu()


def _assert_plain_cpu_model(model):
    assert not any(hasattr(module, "_hf_hook") for module in model.modules())
    tensors = list(model.parameters()) + list(model.buffers())
    assert not [t for t in tensors if t.device.type != "cpu"]


def _stub_looper(looper_cls, qmodel):
    looper = object.__new__(looper_cls)
    looper.gptq_model = qmodel
    looper._offloaded_base_modules = []
    looper._loop_impl = lambda **_kwargs: "finished"
    return looper


@pytest.mark.parametrize("looper_cls", [ModuleLooper, WeightOnlyLooper])
def test_loop_exit_materializes_disk_offloaded_quantized_modules(tmp_path, looper_cls):
    model, names = _quantized_llama()
    reference = _logits(model)
    _offload_like_the_looper(model, names, str(tmp_path / "offload"))

    qmodel = SimpleNamespace(model=model, get_base_modules=lambda model: [])
    assert _stub_looper(looper_cls, qmodel).loop() == "finished"

    _assert_plain_cpu_model(model)
    for name in names:
        module = model.get_submodule(name)
        # The bundle stores checkpoint (v1) words; the live kernel reads v2.
        assert module.qzero_format() == 2
        assert int(module._buffers["qzeros_format"].item()) == 2
    # The in-session model no longer depends on the bundles at all.
    shutil.rmtree(tmp_path / "offload")
    assert torch.equal(_logits(model), reference)


class _SwitchedKernel(TorchLinear):
    def forward(self, x):
        return torch.full((*x.shape[:-1], self.out_features), 7.0, dtype=x.dtype)


def test_restored_module_dispatches_forward_through_its_class(tmp_path):
    """Hook removal must not pin the pre-offload class's bound forward.

    ``rehome_quantized_kernel`` switches AWQ/QQQ/GGUF modules to a portable
    kernel by changing ``__class__`` after the loop. accelerate restores
    ``module.forward`` as an instance attribute, which kept the CUDA-only AWQ
    GEMM forward running on the CPU kernel's buffers.
    """

    model, names = _quantized_llama()
    _offload_like_the_looper(model, names, str(tmp_path / "offload"))
    restore_disk_offloaded_modules(model, strict=True)

    module = model.get_submodule(names[0])
    assert "forward" not in module.__dict__
    module.__class__ = _SwitchedKernel
    out = module(torch.zeros(2, module.in_features, dtype=torch.float16))
    assert torch.equal(out, torch.full_like(out, 7.0))


class _InferenceModeShellQModel:
    """Weight-only stand-in whose base modules come back as inference tensors.

    ``LazyTurtle`` materializes checkpoint tensors under
    ``torch.inference_mode``; an inference parameter makes ``model.to()``
    raise "Inference tensors do not track version counter".
    """

    def __init__(self, model, source):
        self.model = model
        self.source = source

    def get_base_modules(self, model):
        return ["model.embed_tokens", "model.norm"]

    def shell_module_materialize(self, *, target_submodule, device, module_path=None):
        source = self.source.get_submodule(module_path)
        with torch.inference_mode():
            for name, param in source.named_parameters(recurse=False):
                target_submodule._parameters[name] = nn.Parameter(param.detach().clone(), requires_grad=False)
        self.model.tie_weights()
        return target_submodule


def test_weight_only_materialized_base_modules_stay_movable():
    source, _ = _quantized_llama()
    model, _ = _quantized_llama()
    for name in ("model.embed_tokens", "model.norm"):
        model.get_submodule(name).to("meta")
    model.tie_weights()

    _stub_looper(WeightOnlyLooper, _InferenceModeShellQModel(model, source)).loop()

    assert not any(param.is_inference() for param in model.parameters())
    model.to(torch.float32)
    assert model.get_input_embeddings().weight.dtype == torch.float32
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


class _Owner:
    """Weak-referenceable stand-in for the BaseQModel wrapper."""

    def __init__(self, model, root):
        self.model = model
        self.quantize_config = SimpleNamespace(offload_to_disk=True, offload_to_disk_path=str(root))

    def get_base_modules(self, model):
        return []


def test_model_survives_wrapper_teardown_after_loop(tmp_path):
    model, names = _quantized_llama()
    reference = _logits(model)
    owner = _Owner(model, tmp_path / "root")
    run_root = bind_private_offload_root(owner)
    _offload_like_the_looper(model, names, owner.quantize_config.offload_to_disk_path)
    assert os.listdir(run_root)

    _stub_looper(ModuleLooper, owner).loop()

    hf = owner.model
    del owner
    gc.collect()
    # Teardown still removes the run directory (#4368) ...
    assert not os.path.exists(run_root)
    assert os.listdir(tmp_path / "root") == []
    # ... and the surviving model reads nothing from it (#4450).
    assert torch.equal(_logits(hf), reference)


def test_restore_reports_modules_it_cannot_materialize(tmp_path, monkeypatch):
    model, names = _quantized_llama()
    _offload_like_the_looper(model, names, str(tmp_path / "offload"))
    # A restore that leaves the hook and meta tensors in place must not return
    # a model that silently depends on the run directory.
    monkeypatch.setattr("gptqmodel.utils.offload.undo_offload_to_disk", lambda module, **_kwargs: module)

    with pytest.raises(RuntimeError, match="reload the saved checkpoint"):
        restore_disk_offloaded_modules(model, strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="moving the in-session model needs an accelerator")
def test_restored_model_moves_to_cuda(tmp_path):
    model, names = _quantized_llama()
    reference = _logits(model)
    _offload_like_the_looper(model, names, str(tmp_path / "offload"))

    _stub_looper(ModuleLooper, SimpleNamespace(model=model)).loop()

    model.to("cuda:0")
    torch.testing.assert_close(_logits(model, "cuda:0"), reference, rtol=0, atol=2e-2)


_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny llama calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 3


def _tiny_checkpoint(model_dir):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.float16).eval().save_pretrained(model_dir)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)
    return fast


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stranded-norm path needs an accelerator quantize device")
@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_default_offload_quantize_leaves_a_normal_in_session_model(tmp_path, method):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig, RTNConfig

    model_dir = tmp_path / "source"
    tokenizer = _tiny_checkpoint(model_dir)
    calibration = [
        {key: value for key, value in tokenizer(text, return_tensors="pt").items() if key != "token_type_ids"}
        for text in _CALIBRATION_TEXTS
    ]
    config_cls = RTNConfig if method == "rtn" else GPTQConfig
    qcfg = config_cls(
        bits=4,
        group_size=32,
        offload_to_disk_path=str(tmp_path / "offload"),
        device="cuda:0",
    )
    assert qcfg.offload_to_disk

    qmodel = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    qmodel.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)

    hf = qmodel.model
    run_root = qmodel._private_offload_run_root
    # Weight-only layers used to keep their norms on cuda:0 while the rest of
    # the model was on CPU, and every packed module stayed disk-hooked.
    _assert_plain_cpu_model(hf)
    ids = tokenizer("the quick brown fox", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        in_session = hf(input_ids=ids).logits.float()

    checkpoint = tmp_path / "quantized"
    qmodel.save(str(checkpoint))
    reloaded = GPTQModel.load(str(checkpoint), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    with torch.inference_mode():
        assert torch.equal(reloaded.model(input_ids=ids).logits.float(), in_session)

    del qmodel
    gc.collect()
    assert not os.path.exists(run_root)
    with torch.inference_mode():
        assert torch.equal(hf(input_ids=ids).logits.float(), in_session)
    hf.to("cuda:0")
    with torch.inference_mode():
        torch.testing.assert_close(
            hf(input_ids=ids.to("cuda:0")).logits.float().cpu(), in_session, rtol=0, atol=5e-2,
        )


def test_unquantized_layers_keep_a_version_counter_after_the_cpu_move():
    """#4572: `param.data = cpu_copy` leaves a parameter that cannot be used.

    ``pre_quantize`` materializes unquantized layers (MoE routers, dynamic
    ``-:`` exclusions, norms) under ``inference_mode``; ``post_quantize`` then
    moves them to CPU with ``param.data =``, which clears the inference flag but
    not the unusable state: the parameter still has no version counter, so
    ``F.linear`` raises "Inference tensors do not track version counter".
    """

    from gptqmodel.utils.offload import ensure_version_counters, replace_inference_parameters

    with torch.inference_mode():
        module = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)).half()
    for param in module.parameters():
        param.data = param.data.clone()  # move_module_to_cpu_pinned
    assert all(not param.is_inference() for param in module.parameters())
    for param in module.parameters():
        with pytest.raises(RuntimeError, match="version counter"):
            _ = param._version
    with pytest.raises(RuntimeError, match="version counter"):
        module(torch.ones(2, 8, dtype=torch.float16))

    repaired = ensure_version_counters(module)
    assert repaired
    for param in module.parameters():
        assert param._version == 0
    module(torch.ones(2, 8, dtype=torch.float16))

    # The materialization helper repairs the same state on its own subtree.
    with torch.inference_mode():
        other = nn.Linear(4, 4)
    other.weight.data = other.weight.data.clone()
    replace_inference_parameters(other)
    assert other.weight._version == 0
    other(torch.ones(2, 4))


def test_resumed_packed_module_hook_is_materialized_at_loop_exit(tmp_path):
    """#4562: a resumed meta module needs the offload hook the loop restores.

    The resume path rebuilds packed modules on ``meta`` and writes an
    Accelerate-format index for them. Without a hook the loop-exit restore
    never sees them, so they stay meta and the in-session forward fails
    silently. Attaching the hook makes the #4498 restore materialize them.
    """

    from gptqmodel.utils.offload import attach_disk_offload_hook

    model, names = _quantized_llama()
    name = names[0]
    module = model.get_submodule(name)
    probe = torch.randn(2, module.in_features, dtype=torch.float16)
    with torch.inference_mode():
        expected = module(probe)

    offload_root = tmp_path / "run"
    offload_to_disk(model=model, module=module, disk_path=str(offload_root), force=True)
    assert is_meta_module(module)

    # A fresh model stands in for the resumed process: the packed module is
    # rebuilt on meta with no hook, exactly like `restore_packed_module`.
    resumed, _ = _quantized_llama()
    resumed_module = resumed.get_submodule(name).to("meta")
    resumed.set_submodule(name, resumed_module)
    assert is_meta_module(resumed_module) and not hasattr(resumed_module, "_hf_hook")

    attach_disk_offload_hook(resumed_module, str(offload_root / name))
    assert hasattr(resumed_module, "_hf_hook")
    assert restore_disk_offloaded_modules(resumed, device=torch.device("cpu"), strict=True) == []
    assert not is_meta_module(resumed_module)
    with torch.inference_mode():
        assert torch.equal(resumed_module(probe), expected)


def _tiny_tied_vl():
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

    config = Qwen2VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "tie_word_embeddings": True,
            "eos_token_id": 2,
            "bos_token_id": 1,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "embed_dim": 32,
            "hidden_size": 32,
            "num_heads": 2,
            "in_chans": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        tie_word_embeddings=True,
    )
    return Qwen2VLForConditionalGeneration(config)


def test_tied_vl_lm_head_is_retied_after_base_materialization():
    """#4597: a tied VL head is not a checkpoint tensor; re-tie, don't read it."""

    from gptqmodel.utils.offload import materialize_remaining_meta_modules

    model = _tiny_tied_vl()
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    assert head.weight is embedding.weight

    # The #4498 end state after the lazy shell materialized the base module:
    # the embedding is real, the tie is broken and lm_head is a meta placeholder.
    embedding.weight = nn.Parameter(embedding.weight.detach().clone())
    head.weight = nn.Parameter(torch.empty((0, 0), device="meta"))
    assert any(tensor.is_meta for tensor in model.parameters())

    unresolved = materialize_remaining_meta_modules(SimpleNamespace(model=model))
    assert unresolved == []
    assert head.weight is embedding.weight
    assert not any(tensor.is_meta for tensor in model.parameters())


def test_untied_meta_output_embedding_fails_loudly():
    """#4597: a head that cannot be materialized must raise, not be logged."""

    from gptqmodel.utils.offload import materialize_remaining_meta_modules

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.lm_head.weight = nn.Parameter(torch.empty((0, 0), device="meta"))

    with pytest.raises(RuntimeError, match="remain on the meta device"):
        materialize_remaining_meta_modules(SimpleNamespace(model=model))
