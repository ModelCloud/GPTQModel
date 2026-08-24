from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions import unlimited_ocr as unlimited_ocr_module
from gptqmodel.models.definitions.unlimited_ocr import UnlimitedOCRQModel
from gptqmodel.utils.hf import normalize_hf_config_compat
from gptqmodel.utils.model import MODALITY


def test_unlimited_ocr_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="unlimited-ocr")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert (
        auto.check_and_get_model_definition(
            "/tmp/unlimited-ocr", trust_remote_code=True
        )
        is UnlimitedOCRQModel
    )


def test_unlimited_ocr_module_tree_expands_mha_and_moe_paths():
    layer_modules = UnlimitedOCRQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert UnlimitedOCRQModel.loader.__name__ == "AutoModel"
    assert UnlimitedOCRQModel.__bases__ == (BaseQModel,)
    assert UnlimitedOCRQModel.require_trust_remote_code is True
    assert UnlimitedOCRQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert UnlimitedOCRQModel.layer_modules_strict is False
    assert UnlimitedOCRQModel.pre_lm_head_norm_module == "model.norm"
    assert UnlimitedOCRQModel.modules_with_direct_meta_tensors == ["model"]
    assert UnlimitedOCRQModel.extract_layers_node() == ["model.layers"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules


def test_unlimited_ocr_restores_remote_parent_config_defaults():
    config = SimpleNamespace(model_type="unlimited-ocr", sliding_window_size=128)

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.attention_bias is False
    assert config.attention_dropout == 0.0
    assert config.ep_size == 1
    assert config.hidden_act == "silu"
    assert config.moe_layer_freq == 1
    assert config.norm_topk_prob is False
    assert config.rms_norm_eps == 1e-6
    assert config.rope_scaling is None
    assert config.rope_theta == 10000.0
    assert config.scoring_func == "softmax"
    assert config.sliding_window == 128
    assert config.use_cache is True


def test_unlimited_ocr_config_compat_preserves_checkpoint_values():
    config = SimpleNamespace(
        attention_dropout=0.25,
        model_type="unlimited-ocr",
        rope_theta=500000.0,
        sliding_window=64,
        sliding_window_size=128,
    )

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.attention_dropout == 0.25
    assert config.rope_theta == 500000.0
    assert config.sliding_window == 64


def test_unlimited_ocr_keeps_multimodal_modules_in_base_dtype():
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.layers = nn.ModuleList([nn.Identity()])
    model.model.norm = nn.LayerNorm(4)
    model.model.sam_model = nn.Linear(4, 4)
    model.model.vision_model = nn.Linear(4, 4)
    model.model.projector = nn.Linear(4, 4)
    model.model.image_newline = nn.Parameter(torch.randn(4))
    model.model.view_seperator = nn.Parameter(torch.randn(4))

    base_modules = set(UnlimitedOCRQModel.get_base_modules(model))

    assert "model.embed_tokens" in base_modules
    assert "model.norm" in base_modules
    assert "model.sam_model" in base_modules
    assert "model.vision_model" in base_modules
    assert "model.projector" in base_modules
    assert UnlimitedOCRQModel.get_modules_with_direct_meta_tensors(model) == ["model"]


def test_unlimited_ocr_prepares_remote_image_layout(monkeypatch):
    class FakeImageTransform:
        def __init__(self, **kwargs):
            del kwargs

        def __call__(self, image):
            return torch.zeros(3, image.height, image.width)

    fake_remote_module = SimpleNamespace(
        BasicImageTransform=FakeImageTransform,
        dynamic_preprocess=lambda image, image_size: ([image], (1, 1)),
    )
    monkeypatch.setattr(
        unlimited_ocr_module, "import_module", lambda name: fake_remote_module
    )
    monkeypatch.setattr(
        unlimited_ocr_module,
        "fetch_image",
        lambda sample: Image.new("RGB", (32, 16), color="white"),
    )

    tokenizer = SimpleNamespace(
        bos_token_id=0,
        convert_tokens_to_ids=lambda token: 128815 if token == "<image>" else -1,
        encode=lambda text, add_special_tokens=False: [7] if text else [],
    )
    qmodel = object.__new__(UnlimitedOCRQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = nn.Module()
    qmodel.tokenizer = SimpleNamespace(tokenizer=tokenizer)

    prepared = qmodel._prepare_image_sample(
        {"image": "unused", "text": "<image>\nFree OCR."}
    )

    assert prepared["input_ids"].shape == (1, 275)
    assert prepared["images_seq_mask"].sum().item() == 273
    assert prepared["images"][0][0].shape == (1, 3, 1024, 1024)
    assert prepared["images"][0][1].shape == (1, 3, 1024, 1024)
    assert prepared["images"][0][0].dtype == torch.bfloat16
    assert prepared["images_spatial_crop"].tolist() == [[1, 1]]


def test_unlimited_ocr_move_input_capture_casts_nested_images_to_vision_dtype():
    model = nn.Module()
    model.model = nn.Module()
    model.model.sam_model = nn.Linear(4, 4).to(dtype=torch.bfloat16)

    qmodel = object.__new__(UnlimitedOCRQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model

    example = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "images": [
            (
                torch.ones(1, 3, 4, 4, dtype=torch.float32),
                torch.ones(1, 3, 4, 4, dtype=torch.float32),
            )
        ],
        "images_seq_mask": torch.zeros(1, 4, dtype=torch.bool),
        "images_spatial_crop": torch.ones(1, 2, dtype=torch.long),
    }

    moved = qmodel.move_input_capture_example(example, torch.device("cpu"))

    assert moved["images"][0][0].dtype == torch.bfloat16
    assert moved["images"][0][1].dtype == torch.bfloat16


def test_unlimited_ocr_restores_checkpoint_omitted_vision_position_ids():
    core_model = nn.Module()
    core_model.vision_model = nn.Module()
    core_model.vision_model.embeddings = nn.Module()
    core_model.vision_model.embeddings.position_embedding = nn.Embedding(17, 4)

    UnlimitedOCRQModel._restore_vision_position_ids(core_model)

    embeddings = core_model.vision_model.embeddings
    assert embeddings.position_ids.tolist() == [list(range(17))]
    assert "position_ids" in embeddings._non_persistent_buffers_set


def test_unlimited_ocr_moves_direct_multimodal_parameters():
    model = nn.Module()
    model.model = nn.Module()
    model.model.image_newline = nn.Parameter(torch.randn(4), requires_grad=False)
    model.model.view_seperator = nn.Parameter(torch.randn(4))

    qmodel = object.__new__(UnlimitedOCRQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model

    original_image_newline = model.model.image_newline
    qmodel._move_direct_parameters(torch.device("meta"))

    assert model.model.image_newline is not original_image_newline
    assert model.model.image_newline.device.type == "meta"
    assert model.model.image_newline.requires_grad is False
    assert model.model.view_seperator.device.type == "meta"
