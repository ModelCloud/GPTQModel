# Copyright 2023 Haotian Liu
# SPDX-License-Identifier: Apache-2.0
# Adapted from Apple's pinned LLaVA implementation; see NOTICE.

import ast

import torch


IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


def select_best_resolution(original_size, possible_resolutions):
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    resolutions = ast.literal_eval(possible_resolutions) if isinstance(possible_resolutions, str) else possible_resolutions
    if not resolutions or any(w <= 0 or h <= 0 for w, h in resolutions):
        raise ValueError("image_grid_pinpoints must contain positive resolutions")
    best, effective_best, wasted_best = None, -1, float("inf")
    for w, h in resolutions:
        scale = min(w / width, h / height)
        effective = min(int(width * scale) * int(height * scale), width * height)
        wasted = w * h - effective
        if effective > effective_best or (effective == effective_best and wasted < wasted_best):
            best, effective_best, wasted_best = (w, h), effective, wasted
    return best


def unpad_image(tensor, original_size):
    width, height = original_size
    current_h, current_w = tensor.shape[1:]
    if width / height > current_w / current_h:
        padding = (current_h - int(height * current_w / width)) // 2
        return tensor[:, padding:current_h - padding, :]
    padding = (current_w - int(width * current_h / height)) // 2
    return tensor[:, :, padding:current_w - padding]


class LlavaMultimodalMixin:
    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def encode_images(self, images):
        return self.model.mm_projector(self.get_vision_tower()(images))

    def _image_features(self, images, image_sizes):
        if not isinstance(images, (list, tuple)) and images.ndim == 4:
            return self.encode_images(images)
        crops = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
        if not crops or any(image.ndim != 4 or image.shape[0] == 0 for image in crops):
            raise ValueError("Images must contain nonempty CHW images or NCHW crop groups")
        features = self.encode_images(torch.cat(crops))
        features = features.split([image.shape[0] for image in crops])
        merge = self.config.mm_patch_merge_type
        if merge == "flat":
            return [image.flatten(0, 1) for image in features]
        merged = []
        for index, image in enumerate(features):
            base = image[0]
            if image.shape[0] > 1:
                if getattr(self.config, "image_aspect_ratio", None) != "anyres" or image_sizes is None:
                    raise ValueError("Spatial multicrop merging requires anyres metadata and image_sizes")
                side = self.get_vision_tower().num_patches_per_side
                if base.shape[0] != side * side:
                    raise ValueError("Spatial merging requires a square patch-only vision grid")
                size = self.get_vision_tower().input_image_size
                w, h = select_best_resolution(image_sizes[index], self.config.image_grid_pinpoints)
                if w % size or h % size or (w // size) * (h // size) != image.shape[0] - 1:
                    raise ValueError("Image crops do not match image_grid_pinpoints")
                grid = image[1:].reshape(h // size, w // size, side, side, -1)
                if "unpad" in merge:
                    grid = grid.permute(4, 0, 2, 1, 3).flatten(1, 2).flatten(2, 3)
                    grid = unpad_image(grid, image_sizes[index])
                    newline = self.model.image_newline[:, None, None].expand(*grid.shape[:-1], 1)
                    grid = torch.cat((grid, newline.to(grid)), -1).flatten(1, 2).transpose(0, 1)
                else:
                    grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 3)
                image = torch.cat((base, grid))
            else:
                image = base
                if "unpad" in merge:
                    image = torch.cat((image, self.model.image_newline[None].to(image)))
            merged.append(image)
        return merged

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask,
                                             past_key_values, labels, images, image_sizes=None):
        cached = past_key_values is not None and past_key_values.get_seq_length() > 0
        if images is None or self.get_vision_tower() is None or cached:
            if input_ids is not None and (input_ids == IMAGE_TOKEN_INDEX).any():
                raise ValueError("Image placeholders require uncached images and a vision tower")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("Images require two-dimensional input_ids")
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError("Training image-start/end adapters is not supported")
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if mask.shape != input_ids.shape:
            raise ValueError("The uncached attention_mask must match input_ids")
        token_rows = [row[keep] for row, keep in zip(input_ids, mask)]
        if any(row.numel() == 0 for row in token_rows):
            raise ValueError("An image prompt cannot be all padding")
        required = sum(max(1, int((row == IMAGE_TOKEN_INDEX).sum())) for row in token_rows)
        # As in upstream, an image-free batch row consumes one dummy feature group.
        if len(images) != required:
            raise ValueError(f"Image feature groups mismatch: expected {required}, got {len(images)}")
        features = self._image_features(images, image_sizes)
        source_labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels
        label_rows = [row[keep] for row, keep in zip(source_labels, mask)]
        embeddings, targets, image_index = [], [], 0
        for tokens, target in zip(token_rows, label_rows):
            locations = torch.where(tokens == IMAGE_TOKEN_INDEX)[0].tolist()
            bounds = [-1] + locations + [tokens.numel()]
            parts = [tokens[start + 1:end] for start, end in zip(bounds[:-1], bounds[1:])]
            embeds = self.model.embed_tokens(torch.cat(parts))
            parts = embeds.split([part.numel() for part in parts])
            row_embeddings, row_labels = [], []
            for index, part in enumerate(parts):
                row_embeddings.append(part)
                row_labels.append(target[bounds[index] + 1:bounds[index + 1]])
                if index < len(locations):
                    image = features[image_index].to(part)
                    image_index += 1
                    row_embeddings.append(image)
                    row_labels.append(torch.full((image.shape[0],), IGNORE_INDEX, device=target.device, dtype=target.dtype))
            if not locations:
                image_index += 1
            limit = getattr(self.config, "tokenizer_model_max_length", None)
            embeddings.append(torch.cat(row_embeddings)[:limit])
            targets.append(torch.cat(row_labels)[:limit])
        length = max(row.shape[0] for row in embeddings)
        if not length:
            raise ValueError("Image expansion produced an empty sequence")
        padded = embeddings[0].new_zeros((len(embeddings), length, embeddings[0].shape[-1]))
        new_mask = mask.new_zeros((len(embeddings), length))
        positions = input_ids.new_zeros((len(embeddings), length))
        new_labels = source_labels.new_full((len(embeddings), length), IGNORE_INDEX)
        for index, (embedding, target) in enumerate(zip(embeddings, targets)):
            n = embedding.shape[0]
            section = slice(length - n, length) if self.config.tokenizer_padding_side == "left" else slice(0, n)
            padded[index, section] = embedding
            new_labels[index, section] = target
            new_mask[index, section] = True
            positions[index, section] = torch.arange(n, device=positions.device)
        return None, positions, new_mask, past_key_values, padded, None if labels is None else new_labels
