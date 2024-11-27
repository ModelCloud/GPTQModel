# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin_inference import MarlinInferenceQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQ4Marlin(unittest.TestCase):

    @parameterized.expand(
        [
            # act_order==False, group_size=128
            ("TheBloke/Llama-2-7B-GPTQ", "main",
             "<s> I am in Paris and I am in love. everybody knows that.\n"
             "I am in Paris and I am in love.\n"
             "I am in Paris and I am in love. everybody knows that.\n"
             "I am in Paris and I am in love. everybody knows that.\n"
             "I am in Paris and I am in love"),

            # act_order==True, group_size=128
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "main",
             "<s> I am in Paris and I am so excited to be here. I am here for the first time in my life and I am so grateful for this opportunity. I am here to learn and to grow and to meet new people and to experience new things. I am here to see the Eiffel Tower and to walk along"),
            # act_order==True, group_size=64
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-64g-actorder_True",
             "<s> I am in Paris and I am so happy to be here. I have been here for 10 years and I have never been happier. I have been here for 10 years and I have never been happier. I have been here for 10 years and I have never been happier. I"),
            # act_order==True, group_size=32
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-4bit-32g-actorder_True",
             "<s> I am in Paris and I am in love with you.\n"
             "\n"
             "Scene 2:\n"
             "\n"
             "(The stage is now dark, with only the sound of the rain falling on the windowpane. The lights come up on a young couple, JESSICA and JASON, sitting on a park ben"),

            # # 8-bit, act_order==True, group_size=channelwise
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit--1g-actorder_True",
             "<s> I am in Paris and I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy"),
            # # 8-bit, act_order==True, group_size=128
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-128g-actorder_True",
             "<s> I am in Paris and I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy to be here. I am so happy"),
            # # 8-bit, act_order==True, group_size=32
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", "gptq-8bit-32g-actorder_True",
             "<s> I am in Paris and I am looking for a good restaurant for a special occasion. Can you recommend any restaurants in Paris that are known for their specialties? I am looking for something unique and special. Please let me know if you have any recommendations."),

            # # 4-bit, act_order==True, group_size=128
            ("TechxGenus/gemma-1.1-2b-it-GPTQ", "main",
             "<bos>I am in Paris and I am looking for a good bakery with fresh bread.\n"
             "\n"
             "**What are some good bakeries in Paris with fresh bread?**\n"
             "\n"
             "**Bonus:** Any recommendations for specific types of bread they specialize in?\n"
             "\n"
             "**Additional Information:**\n"
             "\n"
             "* I am open to both traditional bakeries and newer, trendy")
        ]
    )
    def test_generation(self, model_id, revision, reference_output):
        prompt = "I am in Paris and"
        device = torch.device("cuda:0")

        try:
            model_q = GPTQModel.load(model_id, revision=revision, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        has_marlin = False
        for _, module in model_q.named_modules():
            linear = MarlinQuantLinear if model_q.quantize_config.format == FORMAT.MARLIN else MarlinInferenceQuantLinear
            if isinstance(module, linear):
                has_marlin = True
                break
        self.assertTrue(has_marlin)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertEqual(predicted_text[30], reference_output[30])

    def test_bias(self):
        # TheBloke/Llama-2-7B-Chat-GPTQ has bias, but they are all zeros, use a checkpoint which really uses bias.
        model_id = "/monster/data/model/starcoderbase-1b-GPTQ"
        try:
            model_q = GPTQModel.load(model_id, device="cuda:0", backend=BACKEND.MARLIN)
        except ValueError as e:
            raise e

        for _, param in model_q.named_parameters():
            self.assertNotEqual(param.device, torch.device("meta"))

        for _, param in model_q.named_buffers():
            self.assertNotEqual(param.device, torch.device("meta"))

        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_proj.bias) > 0)
        self.assertTrue(torch.count_nonzero(model_q.model.transformer.h[0].attn.c_attn.bias) > 0)

        model_id = "/monster/data/model/starcoderbase-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "Today I am in Paris and"
        inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        res = model_q.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

        predicted_text = tokenizer.decode(res[0])

        self.assertIn("Today I am in Paris and I am a student of", predicted_text)

    def test_outfeatures_288(self):
        """
        "ModelCloud/tinyllama-15M-stories" should be loaded with BACKEND.MARLIN after quantization using MarlinQuantLinear.
        MarlinInferenceQuantLinear does not support outfeatures = 288, because 288 is not divisible by 64.
        """
        model_id = "/monster/data/model/tinyllama-15M-stories"
        format=FORMAT.GPTQ
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=format,
            desc_act=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [tokenizer(example["text"]) for example in traindata.select(range(1024))]

        model = GPTQModel.load(
            model_id,
            quantize_config=quantize_config,
        )
        model.quantize(calibration_dataset, batch_size=256)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )

            del model

            model = GPTQModel.load(
                tmp_dir,
                device_map="auto",
                backend=BACKEND.MARLIN,
            )

            for _, submodule in model.named_modules():
                if isinstance(submodule, MarlinQuantLinear):
                    break
            else:
                raise ValueError(
                    "Did not find a MarlinQuantLinear layer")

            prompt = "I am in Paris and"
            device = torch.device("cuda:0")

            inp = tokenizer(prompt, return_tensors="pt").to(device)

            res = model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

            predicted_text = tokenizer.decode(res[0])
            print(f"predicted_text: {predicted_text}")

            self.assertTrue(predicted_text.startswith("<s> I am in Paris and I"))

