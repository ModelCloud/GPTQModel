# -- do not touch
import os
import tempfile

from transformers import AutoTokenizer

from gptqmodel.nn_modules.qlinear import BaseQuantLinear

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import gc  # noqa: E402
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from datasets import load_dataset  # noqa: E402


class TestLoadVLLM(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if importlib.util.find_spec("flashinfer") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flashinfer", "-i",
                                   f"https://flashinfer.ai/whl/cu{torch.version.cuda.replace('.', '')}/torch{'.'.join(torch.__version__.split('.')[:2])}"])

        if importlib.util.find_spec("vllm") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])

        from vllm import SamplingParams  # noqa: E402
        self.MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.SHARDED_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-sharded"
        self.prompts = [
            "The capital of France is",
        ]
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16, top_k=1)

    def release_vllm_model(self):
        from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402

        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

    def test_load_vllm(self):
        model = GPTQModel.load(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.2,
        )

        tokenizer = model.get_tokenizer()

        outputs = model.generate(
            prompts=self.prompts,
            sampling_params=self.sampling_params,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(self.prompts[0]):]
        print(f"Prompt: {self.prompts!r}, Generated text: {generated_text!r}")
        self.assertEquals(generated_text, " Paris.\n\n2. The capital of the United States is Washington, D")

        outputs = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
            max_length=16,
            top_k=1,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(self.prompts[0]):]
        print(f"Prompt: {self.prompts!r}, Generated text: {generated_text!r}")
        self.assertEquals(generated_text, " Paris.\n\n2. The capital of the United States is Washington, D")

        del model
        self.release_vllm_model()

    def test_load_shared_vllm(self):
        model = GPTQModel.load(
            self.SHARDED_MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.2,
        )
        tokenizer = model.get_tokenizer()
        outputs = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(self.prompts[0]):]
        print(f"Prompt: {self.prompts!r}, Generated text: {generated_text!r}")
        self.assertEquals(generated_text, " Paris, which is also known as the city of love.")

        del model
        self.release_vllm_model()

    def test_dynamic(self):
        NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(NATIVE_MODEL_ID, use_fast=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [tokenizer(example["text"]) for example in traindata.select(range(1024))]

        # support dynamic override of bits, group_size, desc_act, sym for each layer/module match
        #
        dynamic = {
            # `.*\.` matches the layers_node prefix
            # layer index start at 0
            r"-:model\.layers\.0\..*": {},  # skip 0 layers
            r".*\.18\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 18 gate and up module
            r".*\.19\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 19 gate and up module
            r".*\.20\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 20 gate and up module
            r".*\.21\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 21 gate and up module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic=dynamic,
            group_size=128,
        )
        model = GPTQModel.load(
            NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        model.quantize(calibration_dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model

            model = GPTQModel.load(
                tmp_dir,
                device="cuda:0",
                backend=BACKEND.VLLM,
                gpu_memory_utilization=0.2,
            )

            tokenizer = model.get_tokenizer()

            for name, submodule in model.named_modules():
                if name == 'model.model.layers.0.self_attn.q_proj' and isinstance(submodule,
                                                                                  BaseQuantLinear):  # module 0 was skipped
                    raise ValueError("first layer should be native module")

            outputs = model.generate(
                prompts=self.prompts,
                temperature=0.8,
                top_p=0.95,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(self.prompts[0]):]
            print(f"Prompt: {self.prompts!r}, Generated text: {generated_text!r}")
            self.assertEquals(generated_text,
                              " Paris, which is also the country's largest city.")

            del model
            self.release_vllm_model()
