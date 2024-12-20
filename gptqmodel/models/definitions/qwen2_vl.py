from typing import Dict

from qwen_vl_utils import process_vision_info

from transformers import AutoModelForVision2Seq, Qwen2VLProcessor

from ..base import BaseGPTQModel
from ...utils.calibration import batched
from ...utils.model import MODALITY


class Qwen2VLGPTQ(BaseGPTQModel):
    loader = AutoModelForVision2Seq

    base_modules = ["model.embed_tokens", "model.norm"]

    layers_node = "model.layers"
    layer_type = "Qwen2VLDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    quant_override_files = {
        "preprocessor_config.json": {
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [
                0.48145466,
                0.4578275,
                0.40821073
            ],
            "image_processor_type": "Qwen2VLImageProcessor",
            "image_std": [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            "max_pixels": 1003520,
            "merge_size": 2,
            "min_pixels": 3136,
            "patch_size": 14,
            "processor_class": "Qwen2VLProcessor",
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {
                "max_pixels": 1003520,
                "min_pixels": 3136
            },
            "temporal_patch_size": 2,
            "vision_token_id": 151654
        }
    }

    def preprocess_inputs(self, sample: Dict) -> Dict:
        return sample

    def prepare_dataset(
            self,
            calibration_dataset,
            batch_size: int = 1,
            tokenizer=None, ):
        processor = Qwen2VLProcessor.from_pretrained(self.model_id_or_path)
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_inputs):
            text = processor.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(batch)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            calib_data.append(inputs)
        del processor
        return calib_data
