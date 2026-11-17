import pytest
import torch

from gptqmodel.models.definitions.deepseek_vl_v2 import DeepSeekVLV2QModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.deepseek_vl_v2.configuration_deepseek_vl_v2 import DeepseekVLV2Config
from gptqmodel.models.ports.deepseek_vl_v2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM


def tiny_config(use_mla=False, **overrides):
    language = {
        "vocab_size": 48,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_group": 2,
        "topk_group": 1,
        "topk_method": "greedy",
        "kv_lora_rank": 8,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "max_position_embeddings": 32,
        "use_mla": use_mla,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    language.update(overrides.pop("language", {}))
    return DeepseekVLV2Config(
        language_config=language,
        vision_config={
            "image_size": 8,
            "patch_size": 2,
            "width": 8,
            "layers": 1,
            "heads": 2,
            "mlp_ratio": 2,
            "global_pool": "",
            "weight_init": "skip",
        },
        projector_config={"input_dim": 8, "n_embed": 16, "depth": 2, "downsample_ratio": 2},
        candidate_resolutions=((8, 8), (16, 8), (8, 16)),
        **overrides,
    )


def test_definition_owns_local_model_and_processor():
    assert isinstance(DeepSeekVLV2QModel.loader, LocalModelLoader)
    assert not DeepSeekVLV2QModel.require_trust_remote_code
    assert DeepSeekVLV2QModel.loader.processor_class.endswith(".DeepseekVLV2Processor")


@pytest.mark.parametrize("mla", [False, True])
def test_tiny_forward(mla):
    model = DeepseekVLV2ForCausalLM(tiny_config(mla)).eval()
    with torch.no_grad():
        result = model(torch.tensor([[1, 4, 5, 6]]), use_cache=True)
    assert result.logits.shape == (1, 4, 48)
    assert torch.isfinite(result.logits).all()


def tokenizer_fixture(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<image>",
        "<unk>",
        "<|User|>",
        "<|Assistant|>",
        "describe",
        "this",
        "image",
        "hello",
        "world",
    ]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<image>", "<|User|>", "<|Assistant|>"],
    )
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.mark.parametrize(
    "mla,q_lora,method",
    [(False, None, "greedy"), (True, None, "greedy"), (True, 8, "group_limited_greedy"), (True, 8, "noaux_tc")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cache_generation_and_safetensors_roundtrip(tmp_path, mla, q_lora, method, dtype):
    torch.manual_seed(123)
    config = tiny_config(
        mla, language={"q_lora_rank": q_lora, "topk_method": method, "norm_topk_prob": True, "scoring_func": "sigmoid"}
    )
    model = DeepseekVLV2ForCausalLM(config).eval().to(dtype)
    ids = torch.tensor([[0, 1, 8, 9, 6], [1, 7, 9, 5, 8]])
    mask = (ids != 0).long()
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        prefix = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        assert prefix.past_key_values.get_seq_length() == 3
        continuation = model(ids[:, 3:], attention_mask=mask, past_key_values=prefix.past_key_values)
        assert continuation.past_key_values.get_seq_length() == 5
        tolerance = 2e-3 if dtype != torch.float32 else 2e-6
        torch.testing.assert_close(continuation.logits, full[:, 3:], atol=tolerance, rtol=tolerance)
        assert torch.equal(
            model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False),
            model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False),
        )
        beam = model.generate(ids[:1], attention_mask=mask[:1], max_new_tokens=2, num_beams=2)
        assert beam.shape[0] == 1
        model.save_pretrained(tmp_path)
        restored = DeepSeekVLV2QModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
        assert set(restored.state_dict()) == set(model.state_dict())
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(restored(ids, attention_mask=mask, use_cache=False).logits, full, atol=0, rtol=0)
        assert list(tmp_path.glob("*.safetensors"))
        assert not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_poisoned_checkpoint_cannot_replace_packaged_components(tmp_path, monkeypatch, trust, bypass):
    import json

    from gptqmodel import GPTQModel
    from gptqmodel.models.auto import check_and_get_model_definition
    from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code

    model = DeepseekVLV2ForCausalLM(tiny_config()).eval()
    model.save_pretrained(tmp_path)
    tokenizer = tokenizer_fixture(tmp_path)
    poison = {
        "AutoConfig": "poison.EvilConfig",
        "AutoModelForCausalLM": "poison.Evil",
        "AutoProcessor": "poison.EvilProcessor",
    }
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data["auto_map"] = poison
    data["language_config"]["auto_map"] = poison
    config_path.write_text(json.dumps(data))
    (tmp_path / "processor_config.json").write_text(json.dumps({"auto_map": poison}))
    (tmp_path / "poison.py").write_text("raise AssertionError('checkpoint code executed')")
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is DeepSeekVLV2QModel
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is DeepseekVLV2Config
    loaded = DeepSeekVLV2QModel.loader.from_pretrained(tmp_path, trust_remote_code=trust).eval()
    assert type(loaded) is DeepseekVLV2ForCausalLM
    processor = DeepSeekVLV2QModel.loader.load_processor(
        tmp_path, config=config, tokenizer=tokenizer, trust_remote_code=trust
    )
    assert type(processor).__module__.startswith("gptqmodel.")
    public = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, dtype=torch.float32)
    assert type(public.model) is DeepseekVLV2ForCausalLM
    assert type(public.processor) is type(processor)
    assert public.processor.tokenizer is public.tokenizer.tokenizer
    for key in public.model.state_dict():
        if key.startswith("language.model.layers."):
            owner = key.rsplit(".", 1)[0].split(".", 4)[4]
            assert public.module_plan().find(owner) is not None
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(torch.tensor([[1, 5]])).logits, model(torch.tensor([[1, 5]])).logits, atol=0, rtol=0
        )
        torch.testing.assert_close(
            public.model(torch.tensor([[1, 5]])).logits, model(torch.tensor([[1, 5]])).logits, atol=0, rtol=0
        )


@pytest.mark.parametrize("mla", [False, True])
@pytest.mark.parametrize("tile_tag", ["1D", "2D"])
@pytest.mark.parametrize("position", ["head", "tail"])
def test_processor_image_forward_cache_generation_and_calibration(tmp_path, mla, tile_tag, position):
    import json

    import numpy as np
    from PIL import Image

    model = DeepseekVLV2ForCausalLM(tiny_config(mla, tile_tag=tile_tag, global_view_pos=position)).eval()
    tokenizer = tokenizer_fixture(tmp_path)
    (tmp_path / "processor_config.json").write_text(json.dumps({"mask_prompt": False}))
    processor = DeepSeekVLV2QModel.loader.load_processor(tmp_path, config=model.config, tokenizer=tokenizer)
    image = Image.fromarray(np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3))
    conversations = [
        {"role": "<|User|>", "content": "<image> describe this image", "images": [image]},
        {"role": "<|Assistant|>", "content": "hello world"},
    ]
    batch = processor(conversations=conversations, images=[image])
    assert batch.images_spatial_crop.tolist() == [[[2, 1]]]
    assert batch.images.shape == (1, 3, 3, 8, 8)
    data = {key: value for key, value in vars(batch).items() if torch.is_tensor(value)}
    with torch.no_grad():
        embeds = model.prepare_inputs_embeds(**data)
        expected = model.language(
            inputs_embeds=embeds, attention_mask=batch.attention_mask, labels=batch.labels, use_cache=False
        )
        actual = model(**data, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        assert torch.isfinite(actual.loss)
        changed = dict(data, images=1.0 - data["images"])
        assert not torch.allclose(actual.logits, model(**changed, use_cache=False).logits)
        generated = model.generate(
            **{key: value for key, value in data.items() if key != "labels"}, max_new_tokens=2, do_sample=False
        )
        embedded = model.language.generate(
            inputs_embeds=embeds, attention_mask=batch.attention_mask, max_new_tokens=2, do_sample=False
        )
        assert torch.equal(generated[:, -2:], embedded[:, -2:])
        empty = model.language(batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        assert not torch.allclose(empty.logits, actual.logits)
    qmodel = object.__new__(DeepSeekVLV2QModel)
    torch.nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.processor = processor
    calibration = qmodel.prepare_dataset([conversations])
    assert len(calibration) == 1
    torch.testing.assert_close(calibration[0]["input_ids"], batch.input_ids)


@pytest.mark.parametrize("mla", [False, True])
def test_failed_media_and_mask_input_can_retry_without_model_mutation(mla):
    model = DeepseekVLV2ForCausalLM(tiny_config(mla)).eval()
    ids = torch.tensor([[1, 4, 5, 6]])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="require"):
        model(ids, images=torch.randn(1, 2, 3, 8, 8))
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids, attention_mask=torch.ones(1, 3))
    with torch.no_grad():
        fresh = model(ids, use_cache=False).logits
        embeds, cache = model.incremental_prefilling(input_ids=ids, chunk_size=2)
        continued = model(inputs_embeds=embeds[:, -1:], past_key_values=cache)
        torch.testing.assert_close(continued.logits, fresh[:, -1:], atol=1e-6, rtol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_text_only_processor_roundtrip_and_bad_image_retry(tmp_path):
    import json

    from gptqmodel.models.ports.deepseek_vl_v2.processing_deepseek_vl_v2 import DeepseekVLV2Processor

    model = DeepseekVLV2ForCausalLM(tiny_config()).eval()
    tokenizer = tokenizer_fixture(tmp_path)
    model.config.save_pretrained(tmp_path)
    (tmp_path / "processor_config.json").write_text(json.dumps({"mask_prompt": False}))
    processor = DeepseekVLV2Processor.from_pretrained(tmp_path, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="image"):
        processor(prompt="<image> describe this image")
    batch = processor(prompt="hello world")
    assert not batch.images_seq_mask.any()
    with torch.no_grad():
        direct = model(batch.input_ids, attention_mask=batch.attention_mask).logits
        from_images = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            images=batch.images,
            images_seq_mask=batch.images_seq_mask,
            images_spatial_crop=batch.images_spatial_crop,
        ).logits
        torch.testing.assert_close(direct, from_images)
    processor.save_pretrained(tmp_path)
    reloaded = DeepseekVLV2Processor.from_pretrained(tmp_path)
    assert torch.equal(reloaded(prompt="hello world").input_ids, batch.input_ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_attention_matches_native_llama_oracle(dtype):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

    model = DeepseekVLV2ForCausalLM(tiny_config()).eval().to(dtype)
    attention = model.language.model.layers[0].self_attn
    config = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    native = LlamaAttention(config, 0).eval().to(dtype)
    native.load_state_dict(attention.state_dict(), strict=True)
    hidden = torch.randn(2, 7, 16, dtype=dtype)
    positions = torch.arange(3, 10).unsqueeze(0)
    mask = torch.zeros(2, 1, 7, 7, dtype=dtype).masked_fill(
        torch.ones(7, 7, dtype=torch.bool).triu(1), torch.finfo(dtype).min
    )
    with torch.no_grad():
        expected = native(
            hidden, position_embeddings=LlamaRotaryEmbedding(config)(hidden, positions), attention_mask=mask
        )[0]
        actual = attention(hidden, attention_mask=mask, position_ids=positions)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("q_lora", [None, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mla_matches_pinned_absorbed_weight_equations(q_lora, dtype):
    import math

    import torch.nn.functional as F

    # Reference: modeling_deepseek.py DeepseekV2Attention.forward at HF cca6d50b.
    # Deliberately evaluate the absorbed-weight formulation, not local helpers.
    torch.manual_seed(67)
    config = tiny_config(
        True,
        language={
            "q_lora_rank": q_lora,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
        },
    )
    attn = DeepseekVLV2ForCausalLM(config).eval().to(dtype).language.model.layers[0].self_attn
    hidden = torch.randn(2, 7, 16, dtype=dtype)
    positions = torch.arange(40, 47).unsqueeze(0)
    mask = torch.zeros(2, 1, 7, 7, dtype=dtype).masked_fill(
        torch.ones(7, 7, dtype=torch.bool).triu(1), torch.finfo(dtype).min
    )

    def norm(x, layer):
        return (x.float() * (x.float().square().mean(-1, keepdim=True) + layer.variance_epsilon).rsqrt()).to(
            dtype
        ) * layer.weight

    def rope(x):
        x = x.reshape(*x.shape[:-1], 2, 2).transpose(-1, -2).reshape_as(x)
        # For dim=4, beta_fast=32/beta_slow=1 and original window 32,
        # YaRN's correction ramp is [0, 1], so inv_freq=[1, .01/4].
        angles = positions[..., None].float() * torch.tensor([1.0, 0.01 / 4.0])
        angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)
        rotated = torch.cat([-x[..., 2:], x[..., :2]], -1)
        return x * angles.cos().to(dtype) + rotated * angles.sin().to(dtype)

    with torch.no_grad():
        query = (
            F.linear(hidden, attn.q_proj.weight)
            if q_lora is None
            else F.linear(norm(F.linear(hidden, attn.q_a_proj.weight), attn.q_a_layernorm), attn.q_b_proj.weight)
        )
        q_plain, q_rope = query.reshape(2, 7, 4, 8).transpose(1, 2).split([4, 4], -1)
        latent, k_rope = F.linear(hidden, attn.kv_a_proj_with_mqa.weight).split([8, 4], -1)
        latent = norm(latent, attn.kv_a_layernorm)
        k_rope = rope(k_rope.unsqueeze(1))
        q_rope = rope(q_rope)
        absorb = attn.kv_b_proj.weight.reshape(4, 8, 8)
        key_weight, value_weight = absorb.split([4, 4], dim=1)
        q_absorbed = torch.matmul(q_plain, key_weight)
        scores = q_rope @ k_rope.mT + q_absorbed @ latent.unsqueeze(1).mT
        scores *= 8**-0.5 * (1 + 0.1 * math.log(4)) ** 2
        probabilities = (scores + mask).softmax(-1, dtype=torch.float32).to(dtype)
        output = (probabilities @ latent.unsqueeze(1)) @ value_weight.mT
        expected = F.linear(output.transpose(1, 2).reshape(2, 7, -1), attn.o_proj.weight)
        actual = attn(hidden, attention_mask=mask, position_ids=positions)[0]
        tolerance = 2e-8 if dtype == torch.float32 else 8e-4
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.02 if dtype != torch.float32 else 1e-5)


@pytest.mark.parametrize("mla", [False, True])
@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy", "noaux_tc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("disable_feature_fallback", [False, True])
def test_awq_plain_policy_packs_or_rejects_missing_features(
    mla, method, dtype, disable_feature_fallback, tmp_path
):
    from types import SimpleNamespace

    from safetensors.torch import load_file, save_file

    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.loop_processor import CalibrationConfig
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.models.module_plan import ModulePlan
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    config = tiny_config(
        mla,
        language={
            "hidden_size": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 8,
            "topk_method": method,
        },
    )
    config.projector_config.n_embed = 32
    torch.manual_seed(113)
    model = DeepseekVLV2ForCausalLM(config).eval().to(dtype)
    wrapper = SimpleNamespace(
        model=model, qlinear_kernel=AwqTorchLinear, rotary_embedding=None, lm_head="language.lm_head"
    )
    wrapper.awq_get_modules_for_scaling = DeepSeekVLV2QModel.awq_get_modules_for_scaling.__get__(wrapper)
    options = {"fallback": None} if disable_feature_fallback else {}
    qcfg = AWQConfig(device="cpu", group_size=-1, bits=4, sym=False, **options)
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=wrapper,
        model=model,
        calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    plan = ModulePlan.from_tree(DeepSeekVLV2QModel.module_tree)
    ids = torch.tensor([[1, 5, 7, 9, 8, 11, 6]])
    excluded = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if "layernorm" in key or ".gate." in key or "embed_tokens" in key or ".norm." in key
    }
    for index, layer in enumerate(model.language.model.layers):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        named = {}
        for name, module in layer.named_modules():
            spec = plan.find(name)
            if not isinstance(module, torch.nn.Linear) or spec is None or not spec.quantize:
                continue
            full = f"language.model.layers.{index}.{name}"
            named[full] = NamedModule(module, name, full, index)
            # Empty features are intentional: this definition rejects scaling,
            # so the actual AWQ fallback must quantize without any fold/search.
        state.modules = dict(named)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        if disable_feature_fallback:
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with pytest.raises(RuntimeError, match="missing activation features.*fallback disabled"):
                processor._quantize_layer(index, state)
            assert not state.quantized
            for key, value in before.items():
                assert torch.equal(model.state_dict()[key], value)
            return
        processor._quantize_layer(index, state)
        assert state.quantized
        assert named
        assert all("q_zeros" in entry.state for entry in named.values())
        with torch.no_grad():
            before_pack = model(ids, use_cache=False).logits
        for entry in named.values():
            processor.pack_module(entry)
            assert isinstance(model.get_submodule(entry.full_name), AwqTorchLinear)
        with torch.no_grad():
            after_pack = model(ids, use_cache=False).logits
        # Packing stores scales at FP16; this compares to the exact staged run.
        torch.testing.assert_close(after_pack, before_pack, atol=3e-4, rtol=0.003)
    for key, value in excluded.items():
        assert torch.equal(model.state_dict()[key], value)
    with torch.inference_mode():
        assert torch.equal(
            model.generate(ids, max_new_tokens=2),
            model.generate(ids, max_new_tokens=2, use_cache=False),
        )
    packed = {key: value.clone() for key, value in model.state_dict().items()}
    save_file(packed, tmp_path / "packed.safetensors")
    with torch.inference_mode():
        model.load_state_dict(load_file(tmp_path / "packed.safetensors"), strict=True)
        torch.testing.assert_close(model(ids, use_cache=False).logits, after_pack, atol=0, rtol=0)


@pytest.mark.parametrize("mla", [False, True])
def test_dense_only_tied_biased_config_outputs_and_save(tmp_path, mla):
    config = tiny_config(
        mla,
        language={
            "n_routed_experts": None,
            "n_shared_experts": None,
            "attention_bias": True,
            "tie_word_embeddings": True,
            "output_attentions": True,
            "output_hidden_states": True,
        },
    )
    model = DeepseekVLV2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        output = model(ids, use_cache=False)
    assert len(output.attentions) == 2
    assert len(output.hidden_states) == 3
    assert model.language.lm_head.weight is model.language.model.embed_tokens.weight
    model.save_pretrained(tmp_path)
    restored = DeepSeekVLV2QModel.loader.from_pretrained(tmp_path).eval()
    assert restored.language.lm_head.weight is restored.language.model.embed_tokens.weight
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, output.logits, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy", "noaux_tc"])
def test_routed_and_shared_moe_matches_independent_full_consumer_sum(method):
    import torch.nn.functional as F

    config = tiny_config(
        True,
        language={
            "topk_method": method,
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.7,
        },
    )
    moe = DeepseekVLV2ForCausalLM(config).eval().language.model.layers[1].mlp
    hidden = torch.randn(2, 7, 16)
    # Deliberately assign varied router/correction scores, so routing is not a
    # degenerate all-experts/tied-score fixture.
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.15)
        if method == "noaux_tc":
            moe.gate.e_score_correction_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
        indices, weights, _ = moe.gate(hidden)
        assert indices.unique().numel() > 2
        expected = torch.zeros_like(hidden)

        def mlp(x, expert):
            return F.linear(
                F.silu(F.linear(x, expert.gate_proj.weight)) * F.linear(x, expert.up_proj.weight),
                expert.down_proj.weight,
            )

        for batch in range(2):
            for token in range(7):
                for slot in range(2):
                    expected[batch, token] += weights[batch * 7 + token, slot] * mlp(
                        hidden[batch, token], moe.experts[indices[batch * 7 + token, slot]]
                    )
        expected += mlp(hidden, moe.shared_experts)
        torch.testing.assert_close(moe(hidden), expected, rtol=1e-5, atol=1e-8)


def test_vision_attention_matches_independent_softmax_equations():
    import torch.nn.functional as F

    model = DeepseekVLV2ForCausalLM(tiny_config()).eval()
    attn = model.vision.blocks[0].attn
    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        q, k, v = F.linear(x, attn.qkv.weight, attn.qkv.bias).reshape(2, 16, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        expected = ((q @ k.mT * 0.5).softmax(-1) @ v).transpose(1, 2).reshape(2, 16, 8)
        expected = F.linear(expected, attn.proj.weight, attn.proj.bias)
        torch.testing.assert_close(attn(x), expected, rtol=1e-6, atol=1e-7)
