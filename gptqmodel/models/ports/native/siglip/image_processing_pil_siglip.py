# Vendored from huggingface/transformers v5.23.0 (Apache-2.0) by GPT-QModel.
# Do not edit by hand except to keep it on the 5.23 ABI; see docs/transformers5-model-ports.md.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for SigLIP."""

from transformers.image_processing_backends import PilBackend
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from transformers.utils import auto_docstring


@auto_docstring(custom_intro="Constructs a SigLIP image processor.")
class SiglipImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True


__all__ = ["SiglipImageProcessorPil"]
