# Copyright 2026 Agnes AI. All rights reserved.
# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
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
"""Image processor for Agnes 3.0 Flash: Qwen2-VL dynamic-resolution patching."""

from collections.abc import Iterable

from transformers.image_utils import SizeDict
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring

from ..native.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    Qwen2VLImageProcessorKwargs,
    smart_resize,
)


AgnesImageProcessorKwargs = Qwen2VLImageProcessorKwargs


def _resolve_pixel_size(size, min_pixels=None, max_pixels=None):
    """Resolve modern size plus legacy per-bound overrides without mutation."""
    if isinstance(size, dict):
        size = SizeDict(**size)
    shortest = min_pixels if min_pixels is not None else size.shortest_edge
    longest = max_pixels if max_pixels is not None else size.longest_edge
    if not shortest or not longest:
        raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
    return SizeDict(shortest_edge=shortest, longest_edge=longest)


@auto_docstring
class AgnesImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16

    def __init__(self, **kwargs: Unpack[AgnesImageProcessorKwargs]):
        size = dict(self.size if kwargs.get("size") is None else kwargs["size"])
        kwargs["size"] = size
        if "shortest_edge" not in size and kwargs.get("min_pixels") is None or \
                "longest_edge" not in size and kwargs.get("max_pixels") is None:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        # Native __init__ maps legacy min/max_pixels onto the copied size dict.
        super().__init__(**kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        # Per-call min/max_pixels stay supported without the native deprecation path.
        kwargs = super()._standardize_kwargs(size=self.size if size is None else size, **kwargs)
        kwargs["size"] = _resolve_pixel_size(kwargs["size"], min_pixels, max_pixels)
        return kwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Patch count for an image, honouring `size` and `do_resize` overrides."""
        images_kwargs = images_kwargs or {}
        size = self._standardize_kwargs(**images_kwargs)["size"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        new_h, new_w = height, width
        if images_kwargs.get("do_resize", self.do_resize):
            new_h, new_w = smart_resize(height, width, patch_size * merge_size,
                                        min_pixels=size.shortest_edge, max_pixels=size.longest_edge)
        return (new_h // patch_size) * (new_w // patch_size)


__all__ = ["AgnesImageProcessor"]
