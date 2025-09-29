# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import unittest  # noqa: E402

from parameterized import parameterized  # noqa: E402

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402


class TestTokenicer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
        self.model = GPTQModel.load(self.pretrained_model_id, quantize_config=QuantizeConfig(bits=4, group_size=128))
        self.tokenizer = self.model.tokenizer
        self.example = 'Test Case String'
        self.expect_input_ids = [2271, 11538, 923]

    def test_tokenicer_func(self):
        input_ids = self.tokenizer(self.example)['input_ids']
        self.assertEqual(
            input_ids,
            self.expect_input_ids,
            msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
        )

    @parameterized.expand(
        [
            ('eos_token', "<|im_end|>"),
            ('pad_token', "<|fim_pad|>"),
            ('vocab_size', 151643)
        ]
    )
    def test_tokenicer_property(self, property, expect_token):
        if property == 'eos_token':
            result = self.tokenizer.eos_token
        elif property == 'pad_token':
            result = self.tokenizer.pad_token
        elif property == 'vocab_size':
            result = self.tokenizer.vocab_size

        self.assertEqual(
            result,
            expect_token,
            msg=f"Expected property result='{expect_token}' but got '{result}'."
        )

    def test_tokenicer_encode(self):
         input_ids = self.tokenizer.encode(self.example, add_special_tokens=False)
         self.assertEqual(
             input_ids,
             self.expect_input_ids,
             msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
         )

    def test_tokenicer_decode(self):
        example = self.tokenizer.decode(self.expect_input_ids, skip_special_tokens=True)
        self.assertEqual(
            self.example,
            example,
            msg=f"Expected example='{self.example}' but got '{example}'."
        )
