# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

from gptqmodel.models._const import DEVICE


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from q4_reference import get_diff  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.utils.importer import select_quant_linear  # noqa: E402
from gptqmodel.utils.model import gptqmodel_post_init  # noqa: E402


GENERATE_EVAL_SIZE = 100

class TestsQ4ExllamaV2(unittest.TestCase):
    def test_exllamav2(self):
        group_size = 128

        m = 1
        k = 1024
        n = 1024
        device = torch.device("cuda:0")
        pack_dtype = torch.int32

        linear_class = select_quant_linear(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            backend=BACKEND.EXLLAMA_V2,
            format=FORMAT.GPTQ,
            quant_method=METHOD.GPTQ,
            pack_dtype=pack_dtype,
            device=DEVICE.CUDA,
        )

        linear = linear_class(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=k,
            out_features=n,
            bias=False,
            pack_dtype=pack_dtype,
            backend=BACKEND.EXLLAMA_V2,
        )

        self.assertTrue(isinstance(linear, ExllamaV2Linear))

        # Deterministic nibble codes. The previous fixture used
        # ``torch.randint`` with a fixed seed, whose stream is torch-version
        # dependent, so the baked reference no longer described the inputs.
        rows = torch.arange(k, dtype=torch.int64).reshape(k // 8, 8, 1)
        cols = torch.arange(n, dtype=torch.int64).reshape(1, 1, n)
        codes = ((rows * 7 + cols * 3) % 16).to(torch.int32)
        qweight = torch.zeros(k // 8, n, dtype=torch.int32)
        for lane in range(8):
            qweight.bitwise_or_(codes[:, lane, :] << (4 * lane))

        linear.qweight = qweight
        linear.scales = torch.full_like(linear.scales, 0.002)
        linear.qzeros.zero_()
        # One zero point per nibble (v2 layout: the stored word is the zero).
        linear.qzeros += 0b00010001000100010001000100010001
        linear.qzero_format(2)

        # ``gptqmodel_post_init`` shuffles ``qweight`` in place into ExLlamaV2's
        # native runtime layout, so keep the checkpoint-layout buffers for the
        # independent reference.
        checkpoint_qweight = linear.qweight.detach().clone()
        checkpoint_qzeros = linear.qzeros.detach().clone()
        checkpoint_scales = linear.scales.detach().clone()

        linear = linear.eval()
        linear = linear.to(device)

        linear = gptqmodel_post_init(linear, use_act_order=False)

        torch.manual_seed(42)
        inp = torch.rand(1, m, k, dtype=torch.float16).to(device)

        with torch.inference_mode():
            res = linear(inp)[0][0]

        # Independent fp64 dequantization of the exact buffers above. This does
        # not call any gptqmodel kernel or dequantizer.
        def unpack_rows(qw, bits=4):
            factor = 32 // bits
            kf, width = qw.shape
            out = torch.zeros(kf * factor, width, dtype=torch.int64)
            words = qw.to(torch.int64) & 0xFFFFFFFF
            for lane in range(factor):
                out[lane::factor] = (words >> (bits * lane)) & 0xF
            return out

        def unpack_cols(qz, bits=4):
            factor = 32 // bits
            groups, width = qz.shape
            out = torch.zeros(groups, width * factor, dtype=torch.int64)
            words = qz.to(torch.int64) & 0xFFFFFFFF
            for lane in range(factor):
                out[:, lane::factor] = (words >> (bits * lane)) & 0xF
            return out

        q = unpack_rows(checkpoint_qweight.cpu()).to(torch.float64)
        z = unpack_cols(checkpoint_qzeros.cpu()).to(torch.float64)
        scales = checkpoint_scales.cpu().to(torch.float64)
        weight = torch.zeros_like(q)
        groups = z.shape[0]
        rows_per_group = k // groups
        for group in range(groups):
            span = slice(group * rows_per_group, (group + 1) * rows_per_group)
            weight[span, :] = (q[span, :] - z[group:group + 1, :]) * scales[group:group + 1, :]

        reference = (inp[0][0].cpu().to(torch.float64) @ weight).to(torch.float32)

        self.assertTrue(
            torch.allclose(res.float().cpu(), reference, rtol=3e-5, atol=2e-2),
            get_diff(res.cpu(), reference),
        )

    def test_generation_desc_act_false(self):
        prompt = "I am in Paris and"
        torch.device("cuda:0")

        reference_output = "<s> I am in Paris and I am in love with you.\n\nScene 2:\n\n(The stage is now dark, but the audience can see the characters walking around the stage.)\n\n(The stage is now lit up, but the audience can only see the characters' silhouettes.)\n\n("

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        model_q = GPTQModel.load(model_id, device="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        predicted_text = ModelTest.generate_stable_with_limit(
            model_q,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        self.assertEqual(predicted_text[:GENERATE_EVAL_SIZE], reference_output[:GENERATE_EVAL_SIZE])

    def test_generation_desc_act_true(self):
        prompt = "The capital of France is"
        torch.device("cuda:0")

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        revision = "desc_act_true"

        model_q = GPTQModel.load(
            model_id,
            rivision=revision,
            device="cuda:0",
            backend=BACKEND.EXLLAMA_V2,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        predicted_text = ModelTest.generate_stable_with_limit(
            model_q,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        print("predicted_text", predicted_text)
        assert "paris" in predicted_text.lower() or "city" in predicted_text.lower()

    def test_exllama_v2_buffer_size(self):
        # prompt = "I'm in Paris and" * 450
        prompt = "I'm in Paris and" * 500
        device = torch.device("cuda:0")

        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        # revision = "desc_act_true"

        model_q = GPTQModel.from_quantized(
            model_id,
            # revision=revision,
            device="cuda:0",
            backend=BACKEND.EXLLAMA_V2,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        self.assertGreater(inp["input_ids"].shape[1], 2048)  # 2048 is the default max_input_length for LLama

        _ = model_q.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
