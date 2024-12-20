import os.path
import tempfile

import torch
from model_test import ModelTest
from PIL import Image

from gptqmodel import GPTQModel


class TestOvis1_6_Llama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis1.6-Llama3.2-3B"

    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = False
    BATCH_SIZE = 1

    def test_ovis_1_6(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                           torch_dtype=self.TORCH_DTYPE, multimodal_max_length=8192, batch_size=1)

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        # enter image path and prompt
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path)
        text = "What does this picture show?"
        query = f'<image>\n{text}'

        # format conversation
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        # generate output
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": 1024,
                "do_sample": False,
                "top_p": None,
                "top_k": None,
                "temperature": None,
                "repetition_penalty": None,
                "eos_token_id": model.generation_config.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True
            }
            output_ids = \
                model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

            print(f'Output:\n{output}')

            self.assertIn("snow", output.lower())
