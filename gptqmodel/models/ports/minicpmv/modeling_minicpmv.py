# SPDX-FileCopyrightText: 2024 OpenBMB
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Legacy MiniCPM-V wrapper on Transformers 5 native language decoders.

See PROVENANCE.json for pinned implementations and README.md for support limits.
No checkpoint Python or AutoModel/AutoProcessor discovery occurs in this module.
"""

from copy import deepcopy

import torch
from torch import nn
from transformers import LlamaForCausalLM, PreTrainedModel, Qwen2ForCausalLM, Qwen3ForCausalLM
from transformers.generation import GenerationMixin

from .configuration_minicpm import MiniCPMVConfig
from .legacy_decoder import MiniCPMForCausalLM
from .resampler import Resampler
from .vision import SiglipVisionTransformer


class MiniCPMV(PreTrainedModel, GenerationMixin):
    config_class = MiniCPMVConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["VisionLayer", "LlamaDecoderLayer", "Qwen2DecoderLayer", "Qwen3DecoderLayer"]
    _tied_weights_keys = {"llm.lm_head.weight": "llm.model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        if config._attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError("MiniCPM-V local port qualifies eager and SDPA attention only")
        decoder = config.decoder_config()
        cls = (Qwen3ForCausalLM if config.version == 4.5 else Qwen2ForCausalLM if config.version == 2.6
               else LlamaForCausalLM if config.version >= 2.5 else MiniCPMForCausalLM)
        self.llm = cls(decoder)
        if config.version >= 2.5:
            config.vision_config._attn_implementation = config._attn_implementation
            self.vpm = SiglipVisionTransformer(config.vision_config, config.drop_vision_last_layer)
        else:
            # A fixed, inspected timm constructor; checkpoint strings cannot
            # select a Hub model, external registry, or downloaded implementation.
            try:
                from timm.models.vision_transformer import vit_so400m_patch14_siglip_384
            except ImportError as exc:
                raise ImportError(
                    "Legacy MiniCPM-V vision requires timm; install with `pip install 'gptqmodel[vision]'` "
                    "using a torchvision build compatible with your PyTorch installation."
                ) from exc

            if config.vision_encoder != "vit_so400m_patch14_siglip_384.webli":
                raise ValueError(f"Unqualified MiniCPM-V vision encoder: {config.vision_encoder}")
            options = getattr(config, "legacy_vision_config", {})
            self.vpm = vit_so400m_patch14_siglip_384(
                pretrained=False, num_classes=0, dynamic_img_size=True, dynamic_img_pad=True, **options,
            )
            self.vpm.attn_pool = nn.Identity()
            if config.drop_vision_last_layer:
                self.vpm.blocks = self.vpm.blocks[:-1]
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = config.hidden_size
        self.resampler = Resampler(config, self.vision_dim)
        self.processor = None
        self.terminators = (["<|im_end|>", "<|endoftext|>"] if config.version in (2.6, 4.5)
                            else ["<|eot_id|>", "</s>"] if config.version == 2.5 else ["<|im_end|>", "</s>"])
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Resampler):
            nn.init.trunc_normal_(module.query, std=0.02)
            nn.init.normal_(module.proj, std=self.config.hidden_size ** -0.5)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.llm.set_output_embeddings(value)

    def get_decoder(self):
        return self.llm

    def set_decoder(self, decoder):
        self.llm = decoder

    def _vision_features(self, pixels, sizes, temporal_ids):
        weight = self.get_input_embeddings().weight
        if self.config.version < 2.5:
            outputs = []
            for crop in pixels:
                crop = crop.to(device=weight.device, dtype=weight.dtype)
                h, w = crop.shape[-2:]
                x = self.vpm.forward_features(crop[None])
                x = x[:, self.vpm.num_prefix_tokens:]
                outputs.append(self.resampler(x, torch.tensor([[h // self.config.patch_size, w // self.config.patch_size]])))
            return torch.cat(outputs)
        strips = [crop.flatten(end_dim=1).transpose(0, 1) for crop in pixels]
        sizes = torch.as_tensor(sizes, dtype=torch.int32)
        if sizes.shape != (len(pixels), 2) or bool((sizes <= 0).any()):
            raise ValueError("tgt_sizes must contain positive height/width for every crop")
        for strip, size in zip(strips, sizes):
            if strip.shape[0] != int(size.prod()) * self.config.patch_size:
                raise ValueError("Image patch strip does not match tgt_sizes")
        # Version 2.5 used upstream Idefics2 on the flat strip, without tgt_sizes.
        # Its published batched mask marks the full padded strip valid. Preserve
        # this historical behavior instead of silently changing trained numerics.
        if self.config.version == 2.5 and not self.config.batch_vision_input:
            outputs = []
            for crop, size in zip(strips, sizes):
                px = crop.transpose(0, 1).reshape(1, 3, self.config.patch_size, -1).to(weight)
                x = self.vpm(px).last_hidden_state
                outputs.append(self.resampler(x, size[None]))
            return torch.cat(outputs)
        px = nn.utils.rnn.pad_sequence(strips, batch_first=True)
        px = px.transpose(1, 2).reshape(len(pixels), 3, self.config.patch_size, -1).to(weight)
        length = px.shape[-1] // self.config.patch_size
        mask = torch.arange(length, device=weight.device)[None, None] < sizes.prod(-1).to(weight.device)[:, None, None]
        if self.config.version == 2.5:
            mask = torch.ones_like(mask)
        features = []
        for start in range(0, len(pixels), self.config.vision_batch_size):
            stop = start + self.config.vision_batch_size
            features.append(self.vpm(px[start:stop], patch_attention_mask=mask[start:stop],
                                     tgt_sizes=None if self.config.version == 2.5 else sizes[start:stop]).last_hidden_state)
        return self.resampler(torch.cat(features), sizes, temporal_ids)

    def get_vllm_embedding(self, data):
        ids = data["input_ids"]
        embeddings = self.get_input_embeddings()(ids)
        if self.config.version < 2.5:
            embeddings = embeddings * self.config.scale_emb
        states = data.get("vision_hidden_states")
        if states is None:
            pixel_rows = data.get("pixel_values")
            if pixel_rows is None:
                pixel_rows = [[] for _ in ids]
            if len(pixel_rows) != len(ids):
                raise ValueError("pixel_values batch differs from input_ids")
            temporal_rows = data.get("temporal_ids")
            size_rows = data.get("tgt_sizes")
            states = []
            # Keep each sample's pooling groups independent; do not partition
            # pooled outputs by the original number of frames.
            for i, crops in enumerate(pixel_rows):
                if not len(crops):
                    states.append([])
                    continue
                groups = temporal_rows[i] if temporal_rows is not None else None
                sizes = size_rows[i] if size_rows is not None else None
                states.append(self._vision_features(crops, sizes, groups))
        if len(states) != len(ids):
            raise ValueError("vision_hidden_states batch differs from input_ids")
        bounds = data.get("image_bound")
        result = embeddings.clone()
        for i, state in enumerate(states):
            if not len(state):
                continue
            if bounds is None:
                raise ValueError("Image features require image_bound")
            indices = []
            for start, stop in bounds[i]:
                start, stop = int(start), int(stop)
                if start < 0 or stop > ids.shape[1] or start >= stop:
                    raise ValueError("Invalid image_bound interval")
                indices.extend(range(start, stop))
            if len(set(indices)) != len(indices) or len(indices) != state.numel() // self.embed_dim:
                raise ValueError("image_bound must match every resampled query exactly once")
            result[i, indices] = state.reshape(-1, self.embed_dim).to(result)
        return result, states

    def forward(self, data=None, input_ids=None, attention_mask=None, position_ids=None,
                pixel_values=None, tgt_sizes=None, image_bound=None, temporal_ids=None,
                vision_hidden_states=None, inputs_embeds=None, **kwargs):
        if isinstance(data, torch.Tensor):
            input_ids = data
        elif data is not None:
            data = dict(data)
            input_ids = data.get("input_ids", input_ids)
            position_ids = data.get("position_ids", position_ids)
            attention_mask = data.get("attention_mask", attention_mask)
        if data is None or isinstance(data, torch.Tensor):
            data = {"input_ids": input_ids, "pixel_values": pixel_values, "tgt_sizes": tgt_sizes,
                        "image_bound": image_bound, "temporal_ids": temporal_ids}
            if vision_hidden_states is not None:
                data["vision_hidden_states"] = vision_hidden_states
        if inputs_embeds is None:
            inputs_embeds, _ = self.get_vllm_embedding(data)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    @torch.inference_mode()
    def generate(self, input_ids=None, pixel_values=None, tgt_sizes=None, image_bound=None, temporal_ids=None,
                 attention_mask=None, tokenizer=None, vision_hidden_states=None,
                 return_vision_hidden_states=False, stream=False, decode_text=False, **kwargs):
        data = {"input_ids": input_ids, "pixel_values": pixel_values, "tgt_sizes": tgt_sizes,
                    "image_bound": image_bound, "temporal_ids": temporal_ids}
        if vision_hidden_states is not None:
            data["vision_hidden_states"] = vision_hidden_states
        embeds, states = self.get_vllm_embedding(data)
        if tokenizer is not None:
            terminators = [tokenizer.convert_tokens_to_ids(t) for t in self.terminators]
            if self.config.version <= 2.5:
                terminators.insert(0, tokenizer.eos_token_id)
            terminators = [i for i in terminators if isinstance(i, int) and i != tokenizer.unk_token_id]
            if terminators:
                kwargs.setdefault("eos_token_id", terminators)
        result = self.llm.generate(inputs_embeds=embeds, attention_mask=attention_mask, **kwargs)
        if decode_text or stream:
            if tokenizer is None:
                raise ValueError("Text decoding requires the caller's normalized tokenizer")
            result = tokenizer.batch_decode(result, skip_special_tokens=True)
        if stream:
            if len(result) != 1:
                raise ValueError("MiniCPM-V buffered streaming requires one conversation")
            result = iter(result)
        return (result, states) if return_vision_hidden_states else result

    def chat(self, image=None, msgs=None, tokenizer=None, processor=None, vision_hidden_states=None,
             max_new_tokens=2048, min_new_tokens=0, sampling=True, max_inp_length=16384,
             system_prompt="", stream=False, max_slice_nums=None, use_image_id=None,
             temporal_ids=None, enable_thinking=False, **kwargs):
        from PIL import Image

        from .processing_minicpmv import MiniCPMVProcessor

        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        processor = processor or self.processor
        if processor is None:
            processor = MiniCPMVProcessor.from_pretrained(self.config._name_or_path, config=self.config, tokenizer=tokenizer)
        batched = isinstance(msgs[0], list)
        rows = msgs if batched else [msgs]
        if batched and image is not None:
            raise ValueError("Batched chat requires images inside each message")
        prompts, images = [], []
        for row in rows:
            row = deepcopy(row)
            if image is not None:
                row[0]["content"] = [image, row[0]["content"]]
            media = []
            for msg in row:
                content = msg["content"]
                if isinstance(content, str):
                    continue
                chunks = []
                for item in content:
                    if isinstance(item, Image.Image):
                        media.append(item)
                        chunks.append("(<image>./</image>)")
                    elif isinstance(item, str):
                        chunks.append(item)
                    else:
                        raise ValueError("MiniCPM-V chat accepts text and PIL images")
                msg["content"] = "\n".join(chunks)
            if system_prompt:
                row.insert(0, {"role": "system", "content": system_prompt})
            prompts.append(processor.tokenizer.apply_chat_template(row, tokenize=False, add_generation_prompt=True,
                                                                    enable_thinking=enable_thinking))
            images.append(media)
        inputs = processor(prompts, images, return_tensors="pt", max_slice_nums=max_slice_nums,
                           use_image_id=use_image_id, temporal_ids=temporal_ids, max_length=max_inp_length).to(self.device)
        inputs.pop("image_sizes", None)
        kwargs.setdefault("do_sample", sampling)
        kwargs.setdefault("min_new_tokens", min_new_tokens)
        result = self.generate(**inputs, tokenizer=processor.tokenizer, vision_hidden_states=vision_hidden_states,
                               max_new_tokens=max_new_tokens, decode_text=True, stream=stream, **kwargs)
        if stream:
            return result
        prefill = "<think>\n" if enable_thinking else ""
        return [prefill + x for x in result] if batched else prefill + result[0]
