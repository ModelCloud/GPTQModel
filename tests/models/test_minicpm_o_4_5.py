# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path
import unittest
from importlib.metadata import PackageNotFoundError, version

import torch
from model_test import ModelTest
from packaging.version import Version
from PIL import Image

from gptqmodel.models.definitions.minicpm_o import MiniCPMOQModel


class TestMiniCPMO4_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-o-4_5"
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        required = {
            "audioread": Version("3.1.0"),
            "librosa": Version("0.11.0"),
            "av": Version("16.0.1"),
        }
        for pkg, minimum in required.items():
            try:
                installed = Version(version(pkg))
            except PackageNotFoundError:
                raise unittest.SkipTest(f"MiniCPM-o requires {pkg}>={minimum}")
            if installed < minimum:
                raise unittest.SkipTest(f"MiniCPM-o requires {pkg}>={minimum}, found {installed}")

    def test_minicpm_o_4_5(self):
        model, tokenizer, processor = self.quantModel(
            self.NATIVE_MODEL_ID,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            batch_size=1,
        )

        image = Image.open('/root/projects/GPTQModel/tests/models/ovis/10016.jpg').convert('RGB')

        # First round chat
        question = "What is the landform in the picture?"
        msgs = [{'role': 'user', 'content': [image, question]}]

        answer = model.chat(
            msgs=msgs,
        )

        generated_text = ""
        for new_text in answer:
            generated_text += new_text
            print(new_text, flush=True, end='')
