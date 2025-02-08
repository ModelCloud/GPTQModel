# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gptqmodel.models import OvisGPTQ, Qwen2VLGPTQ


def format_ovis_dataset(image, assistant):
    return {
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a detailed description of this image, do not forget about the texts on it if they exist. Also, do not forget to mention the type / style of the image. No bullet points. When writing descriptions, prioritize clarity and direct observation over embellishment or interpretation.\nDon't forget these rules:\n1. **Be Direct and Concise**: Provide straightforward descriptions without adding interpretative or speculative elements.\n2. **Use Segmented Details**: Break down details about different elements of an image into distinct sentences, focusing on one aspect at a time.\n3. **Maintain a Descriptive Focus**: Prioritize purely visible elements of the image, avoiding conclusions or inferences.\n4. **Follow a Logical Structure**: Begin with the central figure or subject and expand outward, detailing its appearance before addressing the surrounding setting.\n5. **Avoid Juxtaposition**: Do not use comparison or contrast language; keep the description purely factual.\n6. **Incorporate Specificity**: Mention age, gender, race, and specific brands or notable features when present, and clearly identify the medium if it's discernible."
            },
            {
                "from": "gpt",
                "value": assistant
            }
        ]
    }


def format_qwen2_vl_dataset(image, assistant):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "generate a caption for this image"},
            ],
        },
        {"role": "assistant", "content": assistant},
    ]


def prepare_dataset(format_func, n_sample: int = 20) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]"
    )
    return [
        format_func(sample["url"], sample["caption"])
        for sample in dataset
    ]


def get_calib_dataset(model):
    if isinstance(model, OvisGPTQ):
        return prepare_dataset(format_ovis_dataset, n_sample=20)

    if isinstance(model, Qwen2VLGPTQ):
        return prepare_dataset(format_qwen2_vl_dataset, n_sample=1)

    raise NotImplementedError(f"Unsupported MODEL: {model.__class__}")
