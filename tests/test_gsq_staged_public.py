# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Saved model coverage for the public staged scalar GSQ lifecycle."""

import copy
import pytest
import torch
import math


def test_staged_llama_uses_separate_gptq_train_and_validation_documents(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization import gsq_training as gsq_mod

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).half().eval()
    observed_initializers = []
    original_initializer = gsq_mod.initialize_llama_gptq

    def check_initializer(layer, batches, **kwargs):
        observed_initializers.append(len(batches))
        return original_initializer(layer, batches, **kwargs)

    monkeypatch.setattr(gsq_mod, "initialize_llama_gptq", check_initializer)
    training = [{"input_ids": list(range(1, 17))}, {"input_ids": list(range(17, 33))}]
    gptq = [{"input_ids": list(range(33, 49))}]
    validation = [{"input_ids": list(range(49, 65))}]
    run = quantize_llama_gsq_model(
        model, training, initialization_documents=gptq, validation_documents=validation,
        bits=3, group_size=32,
        gsq=GSQTrainingConfig(enabled=True, epochs=1, qk_steps=1,
                              batch_size=2, microbatch_size=1),
        offload_capture=True, capture_directory=tmp_path / "capture")
    assert run["state"] == "complete"
    assert (run["training_documents"], run["initialization_documents"],
            run["validation_documents"]) == (2, 1, 1)
    assert observed_initializers == [1, 1]
    for name in ("attention", "mlp"):
        stage = run["blocks"][0]["stages"][name]
        assert len(stage["validation_history"]) == 1
        assert math.isfinite(stage["validation_hard_loss_before"])
        assert math.isfinite(stage["validation_hard_loss_after"])
    assert not (tmp_path / "capture" / "layer-00" / "train").exists()
    assert not (tmp_path / "capture" / "layer-00" / "gptq").exists()
    assert not (tmp_path / "capture" / "layer-00" / "validation").exists()
    assert not list((tmp_path / "capture" / "layer-00").glob("gsq-*-*"))


@pytest.mark.parametrize("mode", ["attention", "mlp"])
def test_cached_llama_stage_preserves_loss_and_gradients(mode, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.quantization.gsq_batching import CachedLlamaStageBatches
    from gptqmodel.quantization.gsq_training import (
        LlamaGSQAttentionStage,
        cache_llama_stage_targets,
        reconstruction_cached_llama_mlp_loss,
        reconstruction_stage_loss,
        reconstruction_stage_student_loss,
    )

    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    pristine = model.model.layers[0]
    fitted = copy.deepcopy(pristine)
    names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    teacher_weights = {name + ".weight": pristine.get_submodule(name).weight.detach().clone()
                       for name in names}
    with torch.no_grad():
        for name in names:
            fitted.get_submodule(name).weight.add_(0.01)
    hidden = torch.randn(2, 16, 64)
    positions = torch.arange(16).expand(2, -1)
    kwargs = dict(attention_mask=None, position_ids=positions,
                  position_embeddings=model.model.rotary_emb(hidden, positions), use_cache=False)
    original = [[((hidden, kwargs, None), hidden.numel())]]
    if mode == "mlp":
        short_kwargs = {**kwargs, "position_ids": positions[:1],
                        "position_embeddings": tuple(value[:1] for value in kwargs["position_embeddings"])}
        original[0].append(((hidden[:1], short_kwargs, None), hidden[:1].numel()))
    stage = LlamaGSQAttentionStage(fitted) if mode == "attention" else fitted
    selected = names[2:] if mode == "attention" else ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    def weights():
        return {name + ".weight": stage.get_submodule(name).weight.detach().clone().requires_grad_()
                for name in selected}

    old_weights = weights()
    old_loss = reconstruction_stage_loss(stage, (hidden,), kwargs, student_weights=old_weights,
                                         teacher_weights=teacher_weights)
    old_loss.backward()
    paths = cache_llama_stage_targets(stage, original, teacher_weights, mode=mode, directory=tmp_path)
    cached = CachedLlamaStageBatches(original, paths, mode=mode, device="cpu")
    if mode == "mlp":
        assert cached[0][1][0][0].shape[0] == 1
    new_weights = weights()
    if mode == "attention":
        inputs, cached_kwargs, mask, teacher = cached[0][0][0]
        new_loss = reconstruction_stage_student_loss(stage, (inputs,), cached_kwargs,
                                                      student_weights=new_weights, teacher=teacher,
                                                      output_mask=mask)
    else:
        new_loss = reconstruction_cached_llama_mlp_loss(stage, cached[0][0][0], new_weights)
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    for name in old_weights:
        torch.testing.assert_close(new_weights[name].grad, old_weights[name].grad, rtol=0, atol=0)


def test_cached_mlp_prefetch_preserves_shuffled_optimizer_order(tmp_path):
    from gptqmodel.quantization.gsq_batching import CachedLlamaStageBatches, iter_gsq_training_batches

    source = [[((None, None, None), 8)] for _ in range(3)]
    files = []
    for index in range(3):
        residual = torch.full((1, 2, 4), float(index))
        teacher = residual+1
        path = tmp_path / f"{index}.pt"
        torch.save(torch.stack((residual, teacher)).unsqueeze(0), path)
        files.append(path)
    cached = CachedLlamaStageBatches(source, files, mode="mlp", device="cpu")
    observed = [(index, microbatches[0][0][0][0, 0, 0].item(), microbatches[0][1])
                for index, microbatches in iter_gsq_training_batches(cached, [2, 0, 1])]
    assert observed == [(2, 2., 8), (0, 0., 8), (1, 1., 8)]


def test_cached_mlp_accepts_short_final_microbatch(tmp_path):
    from gptqmodel.quantization.gsq_batching import CachedLlamaStageBatches

    source = [[((None, None, None), 16), ((None, None, None), 8)]]
    complete = torch.zeros(2, 2, 2, 4)
    short = torch.ones(2, 1, 2, 4)
    path = tmp_path / "ragged.pt"
    torch.save((complete, short), path)
    cached = CachedLlamaStageBatches(source, [path], mode="mlp", device="cpu")
    batches = cached[0]
    assert [count for _, count in batches] == [16, 8]
    assert [batch[0].shape[0] for batch, _ in batches] == [2, 1]


@pytest.mark.parametrize("method,bits,initializer", [
    ("gptq", 2, "gptq"), ("gptq", 3, "gptq"), ("gptq", 4, "gptq"),
    ("gptq", 4, "rtn"), ("awq", 4, "awq"),
])
def test_staged_llama_checkpoint_roundtrip(tmp_path, monkeypatch, method, bits, initializer):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig, FORMAT, GPTQConfig
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    attention_names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    dense_attention = [
        {f"{name}.weight": layer.get_submodule(name).weight.detach().clone() for name in attention_names}
        for layer in model.model.layers
    ]
    from gptqmodel.quantization import gsq_training as gsq_mod

    original_loss = gsq_mod.reconstruction_stage_loss
    checked_stages = set()

    def check_dense_teacher(stage, args, kwargs, *, student_weights, teacher_weights=None, output_mask=None):
        stage_name = "attention" if len(student_weights) == 2 else "mlp"
        assert stage_name in ("attention", "mlp")
        assert teacher_weights is not None
        assert set(teacher_weights) == set(dense_attention[0])
        if method == "gptq":
            assert any(all(torch.equal(teacher_weights[name], layer[name]) for name in layer)
                       for layer in dense_attention)
        checked_stages.add(stage_name)
        return original_loss(stage, args, kwargs, student_weights=student_weights,
                             teacher_weights=teacher_weights, output_mask=output_mask)

    monkeypatch.setattr(gsq_mod, "reconstruction_stage_loss", check_dense_teacher)
    if method == "awq":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        model.model.rotary_emb = LlamaRotaryEmbedding(config)
    model.save_pretrained(tmp_path / "dense")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="0")),
        unk_token="0", pad_token="0")
    training = dict(enabled=True, initializer=initializer, epochs=1, qk_steps=2,
                    batch_size=1, microbatch_size=1)
    if method == "gptq":
        quant_config = GPTQConfig(bits=bits, group_size=32, format=FORMAT.GPTQ_V2,
                                  act_group_aware=False, device=DEVICE.CPU, offload_to_disk=False,
                                  gsq_training=training)
    else:
        quant_config = AWQConfig(bits=4, group_size=32, sym=False, device=DEVICE.CPU,
                                 offload_to_disk=False, gsq_training=training)
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=quant_config,
                          tokenizer=tokenizer, model_local_path=str(tmp_path / "dense"))
    wrapper.quantize([{"input_ids": list(range(1, 17))}], backend=BACKEND.TORCH,
                     calibration_data_min_length=1)
    assert checked_stages == {"attention", "mlp"}
    assert wrapper.quantized
    assert wrapper.quantize_config.gsq_training.enabled
    packed_type = TorchLinear if method == "gptq" else AwqTorchLinear
    for layer in wrapper.model.model.layers:
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert isinstance(layer.get_submodule(name), packed_type)
    if method == "awq":
        assert len(quant_config.meta["gsq_training_runs"]) == 2
        runs = quant_config.meta["gsq_training_runs"]
    else:
        assert len(wrapper.gsq_training_run["blocks"]) == 2
        runs = wrapper.gsq_training_run["blocks"]
    for block in runs:
        assert set(block["stages"]) == {"self_attn.q_proj", "self_attn.k_proj", "attention", "mlp"}
        for stage in block["stages"].values():
            assert math.isfinite(stage["hard_loss_before"])
            assert math.isfinite(stage["hard_loss_after"])

    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(ids, use_cache=False).logits
    wrapper.save(str(tmp_path / "quantized"))
    loaded = GPTQModel.load(str(tmp_path / "quantized"), backend=BACKEND.TORCH,
                            device="cpu", dtype=torch.float16, attn_implementation="eager")
    assert loaded.quantize_config.gsq_training.to_dict() == wrapper.quantize_config.gsq_training.to_dict()
    with torch.no_grad():
        after = loaded.model(ids, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
