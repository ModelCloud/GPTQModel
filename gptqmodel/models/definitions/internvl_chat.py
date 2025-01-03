from typing import Dict

import torch

from transformers import AutoTokenizer
from ..base import BaseGPTQModel
from ...utils.calibration import batched
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from ...utils.image import fetch_image
from ...utils.model import MODALITY

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLChatGPTQ(BaseGPTQModel):
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    require_pkgs_version = ["transformers<=4.44.2", "timm>=1.0.12", "torchvision>=0.20.1"]

    base_modules = ["language_model.model.tok_embeddings", "language_model.model.norm"]

    layers_node = "language_model.model.layers"
    layer_type = "InternLM2DecoderLayer"
    layer_modules = [
        ["attention.wqkv", "attention.wo"],

        ["feed_forward.w1", "feed_forward.w3"],
        ["feed_forward.w2"],
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    def preprocess_dataset(self, sample: Dict) -> Dict:
        template = self.model.conv_template
        template.append_message(template.roles[0], sample["question"])
        template.append_message(template.roles[1], sample["answer"])
        query = template.get_prompt()

        pixel_values = load_image(fetch_image(sample), max_num=12).to(torch.bfloat16)
        num_patches = pixel_values.size(0)
        image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + self.IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)
        return {
            "query": query,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

    def prepare_dataset(
            self,
            calibration_dataset,
            batch_size: int = 1,
            tokenizer=None, ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

        tokenizer.padding_side = 'left'

        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            queries, pixel_values, image_flags = tuple(
                [instance[key] for instance in batch] for key in ("query", "pixel_values", "image_flags"))
            model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']

            pixel_values = torch.cat(pixel_values, dim=0)
            image_flags = torch.cat(image_flags, dim=0)

            calib_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_flags": image_flags,
            })
        return calib_data
