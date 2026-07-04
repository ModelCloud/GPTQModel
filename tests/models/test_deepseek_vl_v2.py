# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import os.path
from types import SimpleNamespace

os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")

from gptqmodel.models.definitions.deepseek_vl_v2 import DeepSeekVLV2QModel  # noqa: E402
from model_test import ModelTest  # noqa: E402
from ovis import image_to_test_dataset  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


def test_deepseek_vl_v2_pre_quantize_hook_materializes_base_modules():
    model = nn.Module()
    model.language = nn.Module()
    model.language.model = nn.Module()
    model.language.model.embed_tokens = nn.Embedding(4, 4)
    model.language.model.norm = nn.LayerNorm(4)
    model.vision = nn.Linear(4, 4)
    model.projector = nn.Linear(4, 4)

    qmodel = object.__new__(DeepSeekVLV2QModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.turtle_model = None
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    qmodel._direct_parameter_names = ()
    materialized = []

    def shell_module_materialize(module, device):
        materialized.append((module, device))
        return module

    qmodel.shell_module_materialize = shell_module_materialize

    qmodel.pre_quantize_generate_hook_start()

    assert materialized == [
        (model.language.model.embed_tokens, torch.device("cpu")),
        (model.language.model.norm, torch.device("cpu")),
        (model.vision, torch.device("cpu")),
        (model.projector, torch.device("cpu")),
    ]


def test_prepare_deepseek_vl_v2_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("image-url", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_deepseek_vl_v2_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_deepseek_vl_v2_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        [
            {
                "role": "<|User|>",
                "content": "<image>\ngenerate a caption for this image",
                "images": ["image-url"],
            },
            {"role": "<|Assistant|>", "content": "caption"},
        ]
    ]


class TestDeepSeekVLV2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/deepseek-vl2-tiny" # "Isotr0py/deepseek-vl2-tiny"
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4309, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4113, "floor_pct": 0.2},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_deepseek_vl_v2(self):
        self.quantize_and_evaluate()
        # if not torch.cuda.is_available():
        #     self.skipTest("DeepSeek VL2 quantization test requires CUDA.")
        #
        # with self.model_compat_test_context():
        #     model, _tokenizer, processor = self.quantModel(
        #         self.NATIVE_MODEL_ID,
        #         trust_remote_code=self.TRUST_REMOTE_CODE,
        #         dtype=self.TORCH_DTYPE,
        #         call_perform_post_quant_validation=False,
        #     )
        #
        # image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        # image = Image.open(image_path).convert("RGB")
        # conversation = [
        #     {
        #         "role": "<|User|>",
        #         "content": "<image>\nWhat does this picture show?",
        #         "images": [image],
        #     },
        #     {"role": "<|Assistant|>", "content": ""},
        # ]
        # inputs = processor(
        #     conversations=conversation,
        #     images=[image],
        #     force_batchify=True,
        #     system_prompt="",
        # ).to(model.device)
        #
        # tokenizer = processor.tokenizer
        # inputs_embeds = model.model.prepare_inputs_embeds(**inputs.__dict__)
        # outputs = model.model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=inputs.attention_mask,
        #     pad_token_id=tokenizer.eos_token_id,
        #     bos_token_id=tokenizer.bos_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     max_new_tokens=128,
        #     do_sample=False,
        #     use_cache=True,
        # )
        # output_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print("output_text:", output_text)
        #
        # self.assertIn("snow", output_text.lower())
        # self.check_kernel(model, self.KERNEL_INFERENCE)
