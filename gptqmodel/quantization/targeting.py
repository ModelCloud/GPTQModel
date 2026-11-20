# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Resolve solver/format requests against declared and live model structure."""

import copy
import fnmatch
import json
import re
from dataclasses import asdict, dataclass

import torch

from . import formats, solvers
from .config import QuantizeConfig


@dataclass(frozen=True)
class Matcher:
    kind: str
    pattern: str = ""
    projections: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in ("all", "glob", "regex", "attention", "moe_experts"):
            raise ValueError(f"Unknown matcher kind: {self.kind}")
        if self.kind in ("glob", "regex") and not self.pattern:
            raise ValueError("A name matcher needs a nonempty pattern")
        if self.kind == "regex":
            re.compile(self.pattern)
        if self.kind == "attention" and (
            not self.projections or any(role not in ("q", "k", "v", "o") for role in self.projections)
        ):
            raise ValueError("Attention projections must select q, k, v, or o")

    @classmethod
    def glob(cls, pattern):
        return cls("glob", pattern=pattern)

    @classmethod
    def regex(cls, pattern):
        return cls("regex", pattern=pattern)

    @classmethod
    def attention(cls, *, projections=("q", "k", "v")):
        return cls("attention", projections=tuple(projections))

    @classmethod
    def moe_experts(cls):
        return cls("moe_experts")

    def matches(self, name, relative, module_plan):
        if self.kind == "all":
            return True
        if self.kind == "glob":
            return fnmatch.fnmatchcase(name, self.pattern)
        if self.kind == "regex":
            return re.fullmatch(self.pattern, name) is not None
        if self.kind == "moe_experts":
            return module_plan.is_expert(relative)
        roles = set(module_plan.roles(relative))
        return bool(roles & set(self.projections)) and not module_plan.is_moe(relative)


class MatchedQuantizeConfig(QuantizeConfig):
    _matched_solver_config = True

    def __init__(self, *, matcher, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(matcher, Matcher):
            raise TypeError("matcher must be a Matcher")
        self.matcher = matcher

    def to_dict(self):
        return {**super().to_dict(), "matcher": asdict(self.matcher)}


@dataclass(frozen=True)
class Assignment:
    name: str
    layer: int
    relative: str
    group: tuple[str, int]
    config: QuantizeConfig


@dataclass(frozen=True)
class SolverFormatPlan:
    assignments: tuple[Assignment, ...]
    errors: tuple[str, ...]
    layer_prefix: str = ""

    @property
    def success(self):
        return not self.errors

    def __bool__(self):
        return self.success

    def summary(self):
        lines = ["Quantization plan: " + ("SUPPORTED" if self.success else "UNSUPPORTED")]
        lines.extend(f"{entry.name}: {json.dumps(entry.config.to_dict(), sort_keys=True)}"
                     for entry in self.assignments)
        lines.extend(f"ERROR: {error}" for error in self.errors)
        lines.append(f"{len(self.assignments)} targeted modules; unmatched modules unchanged")
        return "\n".join(lines)


def replay_buffer_errors(model):
    return [f"{name}.{key}: nonpersistent floating buffers must retain native FP32 precision; load with dtype instead of casting the model"
            for name, module in model.named_modules() for key in module._non_persistent_buffers_set
            if (buffer := module._buffers.get(key)) is not None
            and buffer.is_floating_point() and buffer.dtype != torch.float32]


def resolve_plan(model, config):
    errors, assignments = [], []
    configs = list(config) if isinstance(config, (list, tuple)) else [config]
    if not configs:
        errors.append("An empty target list is not a quantization request")
    valid = all(isinstance(cfg, QuantizeConfig) and hasattr(cfg, "solver") for cfg in configs)
    if not valid:
        return SolverFormatPlan((), ("config must contain solver/format QuantizeConfig objects",))
    if isinstance(config, (list, tuple)) and any(not isinstance(cfg, MatchedQuantizeConfig) for cfg in configs):
        errors.append("Lists require MatchedQuantizeConfig entries; there is no implicit catch-all priority")
    prefixes = model.extract_layers_node()
    if len(prefixes) != 1:
        return SolverFormatPlan((), ("CPU solver/format replay requires one declared decoder stack",))
    prefix = prefixes[0]
    if model.quantized or getattr(model, "_solver_format_partial", False):
        errors.append("Model is already quantized or has an incomplete quantization; reload its dense source")
    if model.turtle_model is not None:
        errors.append("Load without quantize_config for solver/format quantization; lazy source replay is not qualified")
    if model.trust_remote_code:
        errors.append("Remote-code model lifecycles are not qualified for solver/format replay")
    errors.extend(replay_buffer_errors(model.model))
    if getattr(model, "_auto_fusion", False) or getattr(model, "_fusion_plan", None) is not None:
        errors.append("Fused model lifecycles are not qualified for solver/format replay")
    declared = type(model).__dict__.get("solver_format_support", ())
    for index, cfg in enumerate(configs):
        signature = (cfg.solver.name, cfg.format.name, "cpu")
        if signature not in declared:
            errors.append(f"Target {index}: {signature} lifecycle is not qualified for {type(model).__name__}")
        if not isinstance(cfg.solver, solvers.GPTQ) or not isinstance(cfg.format, formats.MXFP4):
            errors.append(f"Target {index}: this CPU lifecycle implements GPTQ -> MXFP4 only")
    if any(tensor.device.type != "cpu" for tensor in (*model.model.parameters(), *model.model.buffers())):
        errors.append("The solver/format CPU lifecycle requires all model tensors on CPU")
    matched = [0] * len(configs)
    module_plan = model.module_plan()
    layers = model.model.get_submodule(prefix)
    for layer_index, layer in enumerate(layers):
        for relative, module in layer.named_modules():
            spec = module_plan.find(relative)
            if spec is None or not spec.quantize or not isinstance(module, torch.nn.Linear):
                continue
            name = f"{prefix}.{layer_index}.{relative}"
            hits = []
            for index, cfg in enumerate(configs):
                matcher = getattr(cfg, "matcher", Matcher("all"))
                if matcher.matches(name, relative, module_plan):
                    matched[index] += 1
                    hits.append(index)
            if len(hits) > 1:
                errors.append(f"Overlapping targets {hits}: {name}")
            for index in hits:
                if module.in_features % 32:
                    errors.append(f"{name}: MXFP4 requires in_features divisible by 32")
                if module_plan.is_moe(relative):
                    errors.append(f"{name}: routed/shared MoE calibration replay is not qualified")
                numbers = [int(flag) for flag in spec.flags if flag.isdigit()]
                group = (relative.split(".")[0], numbers[0] if numbers else 0)
                cfg = configs[index]
                resolved = QuantizeConfig(solver=copy.deepcopy(cfg.solver), format=copy.deepcopy(cfg.format))
                assignments.append(Assignment(name, layer_index, relative, group, resolved))
    errors.extend(f"Target {index} matches no eligible modules" for index, count in enumerate(matched) if not count)
    return SolverFormatPlan(tuple(assignments), tuple(dict.fromkeys(errors)), prefix)
