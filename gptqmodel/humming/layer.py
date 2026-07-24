# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import json
import os
import re
from typing import Any

import torch

from . import dtypes
from .config import GemmType, LayerConfig
from .forward import (
    humming_forward,
    may_hadamard_quant_input,
    may_quant_input,
)
from .schema import (
    BaseInputSchema,
    BaseWeightSchema,
    HummingInputSchema,
    HummingWeightSchema,
)
from .transform import (
    check_and_pad_tensors,
    prepare_layer_config,
    process_fused_e8m0_scale,
    process_int_weight_scale,
    transform_humming_tensors,
)
from .tune import get_heuristics_config


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class HummingLayerMeta(LayerConfig):
    sublayer_name: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._config_str = self.to_str()

    @classmethod
    def from_layer_config(cls, config: LayerConfig, sublayer_name: str = "") -> "HummingLayerMeta":
        values = {f.name: getattr(config, f.name) for f in dataclasses.fields(LayerConfig) if f.init}
        return cls(**values, sublayer_name=sublayer_name)

    @property
    def name_prefix(self):
        return self.sublayer_name + "_" if self.sublayer_name else ""

    @property
    def weight_name(self):
        return self.name_prefix + "weight"

    @property
    def zero_point_name(self):
        return self.name_prefix + "zero_point"

    @property
    def weight_scale_name(self):
        return self.name_prefix + "weight_scale"

    @property
    def weight_scale_2_name(self):
        return self.name_prefix + "weight_scale_2"

    @property
    def bias_name(self):
        return self.name_prefix + "bias"


class HummingLayerMethod:
    completed_layer_configs: set[tuple[HummingLayerMeta, tuple[str, ...]]] = set()

    @classmethod
    def _get_meta(cls, layer: torch.nn.Module, sublayer_name: str = "") -> HummingLayerMeta:
        metas = getattr(layer, "humming_metas", None)
        assert isinstance(metas, dict), "call prepare_layer_meta() before this"
        return metas[sublayer_name]

    @classmethod
    def may_set_param(cls, layer: torch.nn.Module, name: str, tensor: torch.Tensor | None):
        if tensor is None:
            return
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))

    @classmethod
    def prepare_layer_meta(
        cls,
        layer: torch.nn.Module,
        shape_n: int,
        shape_k: int,
        weight_schema: HummingWeightSchema,
        input_schema: HummingInputSchema | None = None,
        num_experts: int | None = None,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        has_bias: bool = False,
        torch_dtype: torch.dtype | None = None,
        sublayer_name: str = "",
    ) -> HummingLayerMeta:
        config = prepare_layer_config(
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            num_experts=num_experts,
            pad_n_to_multiple=pad_n_to_multiple,
            pad_k_to_multiple=pad_k_to_multiple,
            has_bias=has_bias,
            torch_dtype=torch_dtype,
        )
        meta = HummingLayerMeta.from_layer_config(config, sublayer_name)

        if not isinstance(getattr(layer, "humming_metas", None), dict):
            layer.humming_metas = {}  # type: ignore[assignment]
        layer.humming_metas[sublayer_name] = meta
        return meta

    @classmethod
    def check_and_pad_tensors(cls, meta: HummingLayerMeta, tensors: dict[str, torch.Tensor]):
        return check_and_pad_tensors(meta, tensors)

    @classmethod
    def may_process_int_weight_scale(
        cls,
        meta: HummingLayerMeta,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return process_int_weight_scale(meta, weight_scale=weight_scale, weight_scale_2=weight_scale_2)

    @classmethod
    def may_process_fused_e8m0_scale(
        cls,
        meta: HummingLayerMeta,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return process_fused_e8m0_scale(
            meta, weight=weight, weight_scale=weight_scale, weight_scale_2=weight_scale_2
        )

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer: torch.nn.Module,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        use_m_major_input_scale: bool = False,
        gemm_type: GemmType | str = GemmType.DENSE,
        sublayer_name: str = "",
    ) -> list[Any]:
        return get_heuristics_config(
            layer_config=cls._get_meta(layer, sublayer_name),
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
            gemm_type=gemm_type,
        )

    @classmethod
    def transform_humming_layer(
        cls,
        layer: torch.nn.Module,
        sublayer_name: str = "",
        already_padded: bool = False,
    ):
        meta = cls._get_meta(layer, sublayer_name)
        prefix = meta.name_prefix
        tensors = {
            key.removeprefix(prefix): value
            for key, value in layer.state_dict().items()
            if key.startswith(prefix)
        }

        outputs = transform_humming_tensors(
            meta,
            tensors,
            already_padded=already_padded,
        )

        for key, name in [
            ("weight", meta.weight_name),
            ("weight_scale", meta.weight_scale_name),
            ("zero_point", meta.zero_point_name),
            ("weight_scale_2", meta.weight_scale_2_name),
            ("bias", meta.bias_name),
        ]:
            cls.may_set_param(layer, name, outputs.get(key))

    @classmethod
    def may_quant_input(
        cls,
        layer: torch.nn.Module,
        inputs: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        sublayer_name: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return may_quant_input(
            cls._get_meta(layer, sublayer_name),
            inputs=inputs,
            input_scale=input_scale,
            quanted_input=quanted_input,
        )

    @classmethod
    def may_hadamard_quant_input(
        cls,
        layer: torch.nn.Module,
        inputs: torch.Tensor,
        hadamard_block_size: int | None = None,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        m_major_scale: bool = False,
        sublayer_name: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return may_hadamard_quant_input(
            cls._get_meta(layer, sublayer_name),
            inputs=inputs,
            hadamard_block_size=hadamard_block_size,
            input_scale=input_scale,
            quanted_input=quanted_input,
            m_major_scale=m_major_scale,
        )

    @classmethod
    def forward_layer(
        cls,
        layer: torch.nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
        sublayer_name: str = "",
        hadamard_block_size: int | None = None,
    ):
        meta = cls._get_meta(layer, sublayer_name)
        return humming_forward(
            meta,
            inputs=inputs,
            weight=getattr(layer, meta.weight_name),
            weight_scale=getattr(layer, meta.weight_scale_name, None),
            zero_point=getattr(layer, meta.zero_point_name, None),
            bias=getattr(layer, meta.bias_name, None),
            weight_scale_2=getattr(layer, meta.weight_scale_2_name, None),
            outputs=outputs,
            input_scale=input_scale,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            locks=getattr(layer, "locks", None),
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
            hadamard_block_size=hadamard_block_size,
        )


class HummingMethod(HummingLayerMethod):
    pass


@dataclasses.dataclass(repr=False, eq=False)
class HummingLayer(torch.nn.Module):
    shape_n: int
    shape_k: int
    weight_config: BaseWeightSchema | dict[str, Any]
    input_config: BaseInputSchema | dict[str, Any] | None = None
    pad_n_to_multiple: int = 1
    pad_k_to_multiple: int = 1
    num_experts: int | None = None
    has_bias: bool = False
    torch_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        super().__init__()

        self.humming_config: LayerConfig | None = None
        self._humming_metas: dict[str, HummingLayerMeta] = {}

        if self.torch_dtype is None:
            self.torch_dtype = torch.get_default_dtype()
            if self.torch_dtype not in [torch.float16, torch.bfloat16]:
                self.torch_dtype = (
                    torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                )
        assert self.torch_dtype in [torch.float16, torch.bfloat16], self.torch_dtype

        self.input_config = self.input_config or {}

        if isinstance(self.input_config, dict):
            if "quant_method" not in self.input_config:
                self.input_config["quant_method"] = "humming"
            if "dtype" not in self.input_config:
                self.input_config["dtype"] = dtypes.DataType.from_torch_dtype(self.torch_dtype)
        if isinstance(self.weight_config, dict) and "quant_method" not in self.weight_config:
            self.weight_config["quant_method"] = "humming"

        self.input_schema: BaseInputSchema = (
            self.input_config
            if isinstance(self.input_config, BaseInputSchema)
            else BaseInputSchema.from_config(self.input_config)
        )

        self.weight_schema: BaseWeightSchema = (
            self.weight_config
            if isinstance(self.weight_config, BaseWeightSchema)
            else BaseWeightSchema.from_config(self.weight_config)
        )

        tensors_attrs = self.weight_schema.get_tensors_attrs(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            param_dtype=self.torch_dtype,
            num_experts=self.num_experts,
            has_bias=self.has_bias,
        )

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            param = torch.nn.Parameter(tensor, requires_grad=False)
            for key, value in attrs.items():
                if key not in ["shape", "dtype"]:
                    setattr(param, key, value)
            setattr(self, name, param)

        # Stream-K synchronization state must not be shared between layers:
        # separate layers may execute concurrently on different CUDA streams.
        locks = torch.zeros((1024,), dtype=torch.int32, device=torch.cuda.current_device())
        self.register_buffer("locks", locks)

    @property
    def humming_metas(self) -> dict[str, HummingLayerMeta]:
        if not self._humming_metas and self.humming_config is not None:
            self._humming_metas = {"": HummingLayerMeta.from_layer_config(self.humming_config)}
        return self._humming_metas

    @humming_metas.setter
    def humming_metas(self, value: dict[str, HummingLayerMeta]) -> None:
        self._humming_metas = value

    @staticmethod
    def filter_tensors(tensors: dict[str, torch.Tensor], prefix: str = "") -> dict[str, torch.Tensor]:
        tensors_new = {}
        for key in tensors:
            if key.startswith(prefix):
                key_new = key.removeprefix(prefix).lstrip(".")
                tensors_new[key_new] = tensors[key]
        return tensors_new

    def load_from_unquantized(self, tensor: torch.Tensor):
        assert isinstance(self.weight_schema, HummingWeightSchema)
        assert tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
        expected_shape: tuple[int, ...] = (self.shape_n, self.shape_k)
        if self.num_experts is not None and self.num_experts != 0:
            expected_shape = (self.num_experts,) + expected_shape
        assert tensor.shape == expected_shape

        assert self.torch_dtype is not None
        tensors = HummingWeightSchema.quant_tensor(tensor, self.weight_schema, self.torch_dtype)
        self.load_from_tensors(tensors)

    def load_from_tensors(self, tensors: dict[str, torch.Tensor], prefix: str = ""):
        tensors = self.filter_tensors(tensors, prefix)
        self.load_state_dict(tensors, strict=False)

    def load_from_safetensors(self, name: str, prefix: str = ""):
        assert os.path.exists(name)
        import safetensors.torch

        if os.path.isfile(name):
            tensors = safetensors.torch.load_file(name)
            return self.load_from_tensors(tensors, prefix)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            return self.load_from_safetensors(filename, prefix)

        assert os.path.exists(index_filename)
        with open(index_filename, "r") as f:
            index_data = json.load(f)
        loaded_filenames = set()
        for key, filename in index_data["weight_map"].items():
            filename = os.path.join(name, filename)
            if filename in loaded_filenames:
                continue
            if key.startswith(prefix):
                self.load_from_safetensors(filename, prefix)
                loaded_filenames.add(filename)

    @classmethod
    def from_safetensors(
        cls,
        name: str,
        prefix: str = "",
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        torch_dtype: torch.dtype | None = None,
    ):
        assert os.path.isdir(name)
        import safetensors.torch

        config_filename = os.path.join(name, "config.json")
        with open(config_filename, "r") as f:
            config = json.load(f)
            if torch_dtype is None and config.get("torch_dtype", "") == "float16":
                torch_dtype = torch.float16

            assert "quantization_config" in config, "not a quantization model"
            config = config["quantization_config"]

        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        for key in keys:
            ignore_layers = config.get(key, []) or []
            assert not any(x in prefix for x in ignore_layers), f"layer {prefix} is unquantized"

        layer_config = config.copy()
        for regex in config.get("dynamic", {}):
            if regex[:1] != "-":
                assert not re.match(regex[2:], prefix), f"layer {prefix} is unquantized"
            elif re.match(regex[2:], prefix):
                layer_config.update(config["dynamic"][regex])
                break

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            target_group_config = None
            for group_config in config["config_groups"].values():
                if "Linear" in group_config["targets"]:
                    target_group_config = group_config["weights"].copy()
                    break
            assert target_group_config is not None, f"layer {prefix} is unquantized"
            target_group_config["quant_method"] = config["quant_method"]
            if "format" in config:
                target_group_config["format"] = config["format"]
            if "quant_algo" in config:
                target_group_config["quant_algo"] = config["quant_algo"]
            layer_config = target_group_config

        schema = BaseWeightSchema.from_config(layer_config)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            tensors = safetensors.torch.load_file(filename)
            tensors = cls.filter_tensors(tensors, prefix)
        else:
            assert os.path.exists(index_filename)
            with open(index_filename, "r") as f:
                index_data = json.load(f)
            loaded_filenames = set()
            tensors = {}
            for key, filename in index_data["weight_map"].items():
                filename = os.path.join(name, filename)
                if filename in loaded_filenames:
                    continue
                if key.startswith(prefix):
                    tensors2 = safetensors.torch.load_file(filename)
                    tensors.update(cls.filter_tensors(tensors2, prefix))
                    loaded_filenames.add(filename)

        shape_n, shape_k, num_experts, has_bias = schema.infer_shape(tensors)

        layer = cls(
            shape_n=shape_n,
            shape_k=shape_k,
            weight_config=schema,
            num_experts=num_experts or 0,
            pad_n_to_multiple=pad_n_to_multiple,
            pad_k_to_multiple=pad_k_to_multiple,
            has_bias=has_bias,
            torch_dtype=torch_dtype,
        )

        layer.load_from_tensors(tensors)
        return layer

    def transform(self):
        if not isinstance(self.weight_schema, HummingWeightSchema):
            assert self.torch_dtype is not None
            self.weight_schema, tensors = self.weight_schema.convert_humming(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            self.input_schema, _ = self.input_schema.convert_humming(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            for name, _ in list(self.named_parameters()):
                delattr(self, name)

            for name, tensor in tensors.items():
                param = torch.nn.Parameter(tensor, requires_grad=False)
                setattr(self, name, param)

        assert isinstance(self.input_schema, HummingInputSchema)
        assert isinstance(self.weight_schema, HummingWeightSchema)
        self.humming_config = prepare_layer_config(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            num_experts=self.num_experts,
            pad_n_to_multiple=self.pad_n_to_multiple,
            pad_k_to_multiple=self.pad_k_to_multiple,
            torch_dtype=self.torch_dtype,
            has_bias=self.has_bias,
        )
        self._humming_metas = {}

        tensors = {
            name: tensor
            for name in ["weight", "weight_scale", "zero_point", "bias", "weight_scale_2"]
            if (tensor := getattr(self, name, None)) is not None
        }
        tensors = transform_humming_tensors(self.humming_config, tensors)
        for name, tensor in tensors.items():
            setattr(self, name, torch.nn.Parameter(tensor, requires_grad=False))

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
        hadamard_block_size: int | None = None,
    ) -> torch.Tensor:
        assert self.humming_config is not None, "call transform() before forward()"
        return humming_forward(
            self.humming_config,
            inputs=inputs,
            weight=self.weight,
            weight_scale=getattr(self, "weight_scale", None),
            zero_point=getattr(self, "zero_point", None),
            bias=getattr(self, "bias", None),
            weight_scale_2=getattr(self, "weight_scale_2", None),
            outputs=outputs,
            input_scale=input_scale,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            locks=self.locks,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
            hadamard_block_size=hadamard_block_size,
        )
