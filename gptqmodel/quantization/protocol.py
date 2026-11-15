# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pcre

from .config import FORMAT, METHOD, GGUFBits, GGUFConfig, QuantizeConfig, SmoothMAD
from .norm_tuning import norm_type_of


@dataclass(frozen=True)
class OperationSpec:
    method: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantizeSpec:
    method: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportSpec:
    format: Optional[str] = None
    variant: Optional[str] = None
    impl: Optional[str] = None
    version: Optional[int | str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    mode: Optional[str] = None
    prepare: tuple[OperationSpec, ...] = ()
    quantize: Optional[QuantizeSpec] = None
    export: Optional[ExportSpec] = None


@dataclass(frozen=True)
class MatchSpec:
    pattern: str
    include: bool = True

    @property
    def modifier(self) -> str:
        return "+" if self.include else "-"

    def matches(self, module_name: str) -> bool:
        return _pattern_matches(self.pattern, module_name)


@dataclass(frozen=True)
class Rule:
    match: tuple[MatchSpec, ...]
    aliases: dict[str, Any] | None = None
    actions: tuple[OperationSpec, ...] = ()
    stop: bool = False
    weight: Optional[TargetSpec] = None
    input: Optional[TargetSpec] = None
    output: Optional[TargetSpec] = None
    kv_cache: Optional[TargetSpec] = None

    def matches(self, module_name: str) -> bool:
        includes = tuple(selector for selector in self.match if selector.include)
        excludes = tuple(selector for selector in self.match if not selector.include)
        if not includes:
            return False
        if not any(selector.matches(module_name) for selector in includes):
            return False
        return not any(selector.matches(module_name) for selector in excludes)


@dataclass(frozen=True)
class Stage:
    name: str
    rules: tuple[Rule, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    version: int
    stages: tuple[Stage, ...]


class ProtocolActionError(RuntimeError):
    """A rule-scoped protocol action could not be resolved or applied."""


@dataclass(frozen=True)
class ActionContext:
    """Everything one registered action needs to operate on a live model."""

    model: Any
    rule: Rule
    stage: str
    matched: tuple[tuple[str, Any], ...]
    calibration: Optional[Callable[[], None]]


ActionHandler = Callable[[ActionContext, Mapping[str, Any]], None]


@dataclass(frozen=True)
class CompiledAction:
    """An ``OperationSpec`` resolved against the action registry."""

    method: str
    handler: ActionHandler
    args: dict[str, Any]


# Keyed by ``OperationSpec.method``. Compilation resolves every action through
# this registry; execution resolves the same keys again in ``apply_plan_actions``.
ACTION_REGISTRY: dict[str, ActionHandler] = {}


def register_action(method: str, handler: ActionHandler) -> None:
    """Register one rule-scoped action handler under ``method``."""

    key = str(method).strip().lower()
    if not key:
        raise ValueError("Protocol action method must not be empty.")
    ACTION_REGISTRY[key] = handler


def known_action_methods() -> tuple[str, ...]:
    """Registered action methods, sorted for stable error messages."""

    return tuple(sorted(ACTION_REGISTRY))


def compile_rule_actions(actions: tuple[OperationSpec, ...]) -> tuple[CompiledAction, ...]:
    """Resolve rule ``actions`` against the registry.

    Unknown methods fail closed with the list of known methods. Compilation does
    not touch a model; ``apply_plan_actions`` executes the resolved handlers.
    """

    return tuple(_compile_action(action) for action in actions)


def _compile_action(action: OperationSpec) -> CompiledAction:
    method = str(action.method).strip().lower()
    handler = ACTION_REGISTRY.get(method)
    if handler is None:
        known = ", ".join(f"`{name}`" for name in known_action_methods()) or "none"
        raise NotImplementedError(
            f"Unsupported protocol action `{action.method}`. Known actions: {known}."
        )
    return CompiledAction(method=method, handler=handler, args=dict(action.args))


def skip() -> dict[str, str]:
    return {"method": "skip"}


def compile_protocol(source: Any) -> ExecutionPlan:
    payload = _normalize_root(source)
    version = int(payload.get("version", 2))
    if version != 2:
        raise ValueError(f"Unsupported quantization protocol version: {version}.")

    stages = tuple(_normalize_stage(stage) for stage in payload.get("stages", ()))
    if not stages:
        raise ValueError("Quantization protocol must define at least one stage.")

    return ExecutionPlan(version=version, stages=stages)


def compile_protocol_yaml_text(text: str) -> ExecutionPlan:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise ModuleNotFoundError("PyYAML is required to parse protocol YAML.") from exc

    payload = yaml.safe_load(text)
    return compile_protocol(payload)


def compile_protocol_yaml_file(path: str | Path) -> ExecutionPlan:
    protocol_path = Path(path)
    return compile_protocol_yaml_text(protocol_path.read_text())


def compile_plan_to_quantize_config(plan: ExecutionPlan):
    if len(plan.stages) != 1:
        raise NotImplementedError("Initial protocol implementation supports exactly one stage for config compilation.")

    stage = plan.stages[0]
    if len(stage.rules) != 1:
        raise NotImplementedError("Initial protocol implementation supports exactly one rule for config compilation.")

    # A directly constructed ``ExecutionPlan`` (as in the documented Python DSL)
    # can still carry a string ``match`` or dict targets; normalize the rule so
    # compilation and action resolution see the same typed shape as
    # ``compile_protocol`` produces.
    rule = _normalize_rule(stage.rules[0])
    if rule.aliases:
        raise NotImplementedError("Initial protocol implementation does not support aliases during config compilation.")
    if rule.actions:
        # Actions execute as a separate stage (``apply_plan_actions``); the
        # compiler only resolves their methods so an unknown action still fails
        # loudly here, before any model or calibration work starts.
        compile_rule_actions(rule.actions)
    if rule.stop:
        raise NotImplementedError("Initial protocol implementation does not support stop during config compilation.")
    if rule.input is not None or rule.output is not None or rule.kv_cache is not None:
        raise NotImplementedError("Initial protocol implementation only supports weight-target compilation.")
    if rule.weight is None:
        raise ValueError("Initial protocol implementation requires a `weight` target.")

    return _compile_weight_target(rule.weight, matchers=rule.match)


def compile_protocol_to_quantize_config(source: Any):
    return compile_plan_to_quantize_config(compile_protocol(source))


def compile_protocol_yaml_to_quantize_config(text: str):
    return compile_plan_to_quantize_config(compile_protocol_yaml_text(text))


def apply_plan_actions(
    model: Any,
    plan: Any,
    *,
    calibration: Optional[Callable[[], None]] = None,
) -> None:
    """Run rule-scoped protocol ``actions`` against ``model``.

    This is the separate action stage the protocol documents: config
    compilation stays pure, and this function resolves each rule's matches on
    the live model before running the registered handlers in stage/rule order.

    ``calibration`` must be a zero-argument callable that runs one or more
    forward passes through ``model`` (the model root). Actions that need
    activation statistics, such as ``smoothquant``, raise
    :class:`ProtocolActionError` when it is omitted.
    """

    resolved = _normalize_plan(plan)
    named_modules = tuple(model.named_modules())
    for stage in resolved.stages:
        for rule in stage.rules:
            if not rule.actions:
                continue
            compiled = compile_rule_actions(rule.actions)
            matched = tuple((name, module) for name, module in named_modules if name and rule.matches(name))
            if not matched:
                raise ProtocolActionError(
                    f"Protocol action rule {_rule_label(rule)} in stage `{stage.name}` matched no modules."
                )
            context = ActionContext(
                model=model,
                rule=rule,
                stage=stage.name,
                matched=matched,
                calibration=calibration,
            )
            for action in compiled:
                action.handler(context, action.args)


def _normalize_plan(plan: Any) -> ExecutionPlan:
    if isinstance(plan, ExecutionPlan):
        return ExecutionPlan(
            version=plan.version,
            stages=tuple(_normalize_stage(stage) for stage in plan.stages),
        )
    return compile_protocol(plan)


def _rule_label(rule: Rule) -> str:
    selectors = []
    for selector in rule.match:
        selectors.append(f"{'' if selector.include else '-:'}{selector.pattern}")
    return "[" + ", ".join(selectors) + "]"


def _apply_smoothquant_action(context: ActionContext, args: Mapping[str, Any]) -> None:
    import torch

    from .awq.quantize.scale import is_foldable_source, scale_ln_fcs

    unknown = set(args) - {"alpha"}
    if unknown:
        raise ProtocolActionError(
            f"smoothquant does not accept argument(s) {sorted(unknown)}; only `alpha` is supported."
        )
    try:
        alpha = float(args.get("alpha", 0.5))
    except (TypeError, ValueError) as exc:
        raise ProtocolActionError(
            f"smoothquant `alpha` must be a float, got {args.get('alpha')!r}."
        ) from exc
    if not 0.0 <= alpha <= 1.0:
        raise ProtocolActionError(f"smoothquant `alpha` must be in [0, 1], got {alpha}.")

    if context.calibration is None:
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(context.rule)} requires calibration activations; "
            "call apply_plan_actions(model, plan, calibration=callable)."
        )

    matched_modules = dict(context.matched)
    for name, module in context.matched:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(context.rule)} matched `{name}` "
                f"({type(module).__name__}), which has no 2-D weight to scale."
            )

    modules = dict(context.model.named_modules())
    state = _capture_smoothquant_statistics(context.model, context.calibration)
    groups = _resolve_smoothquant_groups(state, matched_modules, context.rule)

    scales_by_group = {}
    for norm_name, consumer_names in groups.items():
        norm = modules[norm_name]
        if not is_foldable_source(norm):
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(context.rule)} resolved `{norm_name}` "
                f"({type(norm).__name__}) as the preceding norm, but it has no foldable gain."
            )
        consumers = _validate_smoothquant_consumers(
            state, modules, consumer_names, norm_name=norm_name, rule=context.rule
        )
        scales_by_group[norm_name] = (
            norm,
            [module for _, module in consumers],
            _smoothquant_scales(state, consumers, alpha=alpha, rule=context.rule, norm_name=norm_name),
        )
        norm_width = int(norm.weight.numel())
        scales_width = int(scales_by_group[norm_name][2].numel())
        if norm_width != scales_width:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(context.rule)} computed {scales_width} channel scales "
                f"for `{norm_name}` with {norm_width} gain values."
            )

    with torch.no_grad():
        for norm_name, (norm, consumers, scales) in scales_by_group.items():
            # ``scale_ln_fcs`` divides the norm gain/bias by the scale and
            # multiplies every consumer's input columns by it, so the block
            # output is preserved for every consumer of that norm output.
            scale_ln_fcs(norm, consumers, scales)


def _capture_smoothquant_statistics(model: Any, calibration: Callable[[], None]) -> dict[str, Any]:
    """Capture first-pass data flow plus per-channel activation maxima.

    Only pointer metadata is retained for data-flow resolution; the per-channel
    maxima are small ``[in_features]`` tensors. No activation tensor is kept
    after calibration, so this does not create a dense activation cache.
    """

    import torch

    modules = dict(model.named_modules())
    weighted = {
        name: module
        for name, module in modules.items()
        if name and isinstance(getattr(module, "weight", None), torch.Tensor)
    }
    norms = {
        name: module
        for name, module in weighted.items()
        if norm_type_of(module) is not None and module.weight.ndim == 1
    }

    state = {
        "first_pass": 0,
        "seen_forward": False,
        "input_ptrs": {name: set() for name in weighted},
        "output_ptrs": {name: set() for name in norms},
        "x_max": dict.fromkeys(weighted),
    }
    handles = []

    def _root_pre_hook(module, args):
        if state["seen_forward"]:
            state["first_pass"] += 1
        state["seen_forward"] = True

    def _input_hook(name):
        def hook(module, args, kwargs):
            activation = args[0] if args else kwargs.get("input")
            if not isinstance(activation, torch.Tensor) or not activation.is_floating_point():
                return None
            if activation.ndim == 0:
                return None
            if state["first_pass"] == 0:
                state["input_ptrs"][name].add(activation.data_ptr())
            flat = activation.detach().reshape(-1, activation.shape[-1]).to(torch.float32)
            current = flat.abs().amax(dim=0)
            previous = state["x_max"][name]
            state["x_max"][name] = current if previous is None else torch.maximum(previous, current)
            return None

        return hook

    def _output_hook(name):
        def hook(module, args, output):
            if state["first_pass"] == 0 and isinstance(output, torch.Tensor) and output.is_floating_point():
                state["output_ptrs"][name].add(output.data_ptr())
            return None

        return hook

    handles.append(model.register_forward_pre_hook(_root_pre_hook))
    for name, module in weighted.items():
        handles.append(module.register_forward_pre_hook(_input_hook(name), with_kwargs=True))
    for name, module in norms.items():
        handles.append(module.register_forward_hook(_output_hook(name)))
    try:
        calibration()
    finally:
        for handle in handles:
            handle.remove()
    return state


def _resolve_smoothquant_groups(
    state: dict[str, Any],
    matched_modules: dict[str, Any],
    rule: Rule,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in matched_modules:
        ptrs = state["input_ptrs"][name]
        if len(ptrs) != 1:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} matched `{name}` but its input did not "
                f"resolve to exactly one tensor in the first calibration pass ({len(ptrs)} distinct inputs)."
            )
        ptr = next(iter(ptrs))
        predecessors = [norm_name for norm_name, output_ptrs in state["output_ptrs"].items() if ptr in output_ptrs]
        if len(predecessors) != 1:
            if not predecessors:
                detail = "no norm output feeds it"
            else:
                detail = f"multiple norm outputs feed it: {sorted(predecessors)}"
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} matched `{name}` but {detail}; "
                "v1 folds only the canonical norm -> linear pair."
            )
        groups.setdefault(predecessors[0], []).append(name)

    for norm_name, consumers in groups.items():
        norm_ptrs = state["output_ptrs"][norm_name]
        for name, input_ptrs in state["input_ptrs"].items():
            if name in consumers:
                continue
            if input_ptrs & norm_ptrs:
                consumers.append(name)
    return groups


def _validate_smoothquant_consumers(
    state: dict[str, Any],
    modules: dict[str, Any],
    consumer_names: list[str],
    *,
    norm_name: str,
    rule: Rule,
) -> list[tuple[str, Any]]:
    import torch

    consumers = []
    seen_weights = set()
    for name in consumer_names:
        module = modules[name]
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} resolved `{norm_name}` as the preceding norm "
                f"for `{name}`, but `{name}` ({type(module).__name__}) has no 2-D weight; "
                "v1 cannot fold that consumer."
            )
        if len(state["input_ptrs"][name]) != 1:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} found `{name}` consuming `{norm_name}` "
                "with more than one distinct input in the first calibration pass; "
                "the fold would not be well-defined."
            )
        ptr = weight.data_ptr()
        if ptr in seen_weights:
            continue
        seen_weights.add(ptr)
        consumers.append((name, module))
    return consumers


def _smoothquant_scales(
    state: dict[str, Any],
    consumers: list[tuple[str, Any]],
    *,
    alpha: float,
    rule: Rule,
    norm_name: str,
) -> Any:
    import torch

    x_max = None
    w_max = None
    for name, module in consumers:
        activation_max = state["x_max"].get(name)
        if activation_max is None:
            continue
        if int(module.weight.shape[1]) != int(activation_max.numel()):
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} found `{name}` with "
                f"{module.weight.shape[1]} input columns but {activation_max.numel()} activation channels."
            )
        x_max = activation_max if x_max is None else torch.maximum(x_max, activation_max)
        weight_max = module.weight.detach().abs().amax(dim=0).to(torch.float32)
        w_max = weight_max if w_max is None else torch.maximum(w_max, weight_max)
    if x_max is None or w_max is None:
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} collected no calibration activations for `{norm_name}`."
        )
    if x_max.numel() != w_max.numel():
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} found mismatched channel counts under `{norm_name}`: "
            f"activation {x_max.numel()}, weight {w_max.numel()}."
        )
    eps = torch.finfo(torch.float32).eps
    return (x_max.clamp_min(eps).pow(alpha) / w_max.clamp_min(eps).pow(1.0 - alpha)).clamp_min(eps)


register_action("smoothquant", _apply_smoothquant_action)


def _normalize_root(source: Any) -> dict[str, Any]:
    if isinstance(source, ExecutionPlan):
        return {
            "version": source.version,
            "stages": list(source.stages),
        }
    if isinstance(source, Mapping):
        return dict(source)
    if is_dataclass(source):
        return {
            "version": getattr(source, "version"),
            "stages": getattr(source, "stages"),
        }
    raise TypeError(
        "Quantization protocol root must be a mapping or dataclass-like object with `version` and `stages`."
    )


def _normalize_stage(source: Any) -> Stage:
    if isinstance(source, Stage):
        return Stage(
            name=source.name,
            rules=tuple(_normalize_rule(rule) for rule in source.rules),
        )
    data = _coerce_mapping(source, context="stage")
    name = data.get("name")
    if not name:
        raise ValueError("Stage requires a non-empty `name`.")
    rules = tuple(_normalize_rule(rule) for rule in data.get("rules", ()))
    if not rules:
        raise ValueError(f"Stage `{name}` must define at least one rule.")
    return Stage(name=str(name), rules=rules)


def _normalize_rule(source: Any) -> Rule:
    if isinstance(source, Rule):
        return Rule(
            match=_normalize_match(source.match),
            aliases=_copy_optional_mapping(source.aliases),
            actions=tuple(_normalize_operation(action) for action in source.actions),
            stop=bool(source.stop),
            weight=_normalize_target(source.weight),
            input=_normalize_target(source.input),
            output=_normalize_target(source.output),
            kv_cache=_normalize_target(source.kv_cache),
        )

    data = _coerce_mapping(source, context="rule")
    match = data.get("match")
    if not match:
        raise ValueError("Rule requires a non-empty `match`.")

    return Rule(
        match=_normalize_match(match),
        aliases=_copy_optional_mapping(data.get("aliases")),
        actions=tuple(_normalize_operation(action) for action in data.get("actions", ())),
        stop=bool(data.get("stop", False)),
        weight=_normalize_target(data.get("weight")),
        input=_normalize_target(data.get("input")),
        output=_normalize_target(data.get("output")),
        kv_cache=_normalize_target(data.get("kv_cache")),
    )


def _normalize_target(source: Any) -> Optional[TargetSpec]:
    if source is None:
        return None
    if isinstance(source, TargetSpec):
        return TargetSpec(
            mode=source.mode,
            prepare=tuple(_normalize_operation(op) for op in source.prepare),
            quantize=_normalize_quantize(source.quantize),
            export=_normalize_export(source.export),
        )

    data = _coerce_mapping(source, context="target")
    return TargetSpec(
        mode=data.get("mode"),
        prepare=tuple(_normalize_operation(op) for op in data.get("prepare", ()) or ()),
        quantize=_normalize_quantize(data.get("quantize")),
        export=_normalize_export(data.get("export")),
    )


def _normalize_match(source: Any) -> tuple[MatchSpec, ...]:
    if isinstance(source, MatchSpec):
        return (MatchSpec(pattern=source.pattern, include=bool(source.include)),)
    if isinstance(source, str):
        return (_normalize_match_selector(source),)
    if isinstance(source, (tuple, list)):
        selectors = tuple(_normalize_match_selector(item) for item in source)
        if not selectors:
            raise ValueError("Rule `match` list must not be empty.")
        return selectors
    raise TypeError("Rule `match` must be a string, MatchSpec, or a list/tuple of selector strings.")


def _normalize_match_selector(source: Any) -> MatchSpec:
    if isinstance(source, MatchSpec):
        return MatchSpec(pattern=source.pattern, include=bool(source.include))
    if not isinstance(source, str):
        raise TypeError("Match selector must be a string or MatchSpec.")

    selector = source.strip()
    if not selector:
        raise ValueError("Match selector must not be empty.")

    include = True
    if selector.startswith("+:"):
        selector = selector[2:].strip()
    elif selector.startswith("-:"):
        include = False
        selector = selector[2:].strip()

    if not selector:
        raise ValueError("Match selector pattern must not be empty.")

    return MatchSpec(pattern=selector, include=include)


def _normalize_operation(source: Any) -> OperationSpec:
    if isinstance(source, OperationSpec):
        return OperationSpec(method=source.method, args=dict(source.args))
    if isinstance(source, str):
        return OperationSpec(method=source)
    data = _coerce_mapping(source, context="operation")
    method = data.get("method")
    if not method:
        raise ValueError("Operation requires a non-empty `method`.")
    args = {key: value for key, value in data.items() if key != "method"}
    return OperationSpec(method=str(method), args=args)


def _normalize_quantize(source: Any) -> Optional[QuantizeSpec]:
    if source is None:
        return None
    if isinstance(source, QuantizeSpec):
        return QuantizeSpec(method=source.method, args=dict(source.args))
    if isinstance(source, str):
        return QuantizeSpec(method=source)
    data = _coerce_mapping(source, context="quantize")
    method = data.get("method")
    args = {key: value for key, value in data.items() if key != "method"}
    return QuantizeSpec(method=str(method) if method is not None else None, args=args)


def _normalize_export(source: Any) -> Optional[ExportSpec]:
    if source is None:
        return None
    if isinstance(source, ExportSpec):
        return ExportSpec(
            format=source.format,
            variant=source.variant,
            impl=source.impl,
            version=source.version,
            options=dict(source.options),
        )
    if isinstance(source, str):
        return ExportSpec(format=source)
    data = _coerce_mapping(source, context="export")
    options = dict(data.get("options", {}) or {})
    return ExportSpec(
        format=data.get("format"),
        variant=data.get("variant"),
        impl=data.get("impl"),
        version=data.get("version"),
        options=options,
    )


def _coerce_mapping(source: Any, *, context: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if is_dataclass(source):
        return {field_name: getattr(source, field_name) for field_name in source.__dataclass_fields__}
    raise TypeError(f"Quantization protocol {context} must be provided as a mapping or dataclass.")


def _copy_optional_mapping(source: Any) -> dict[str, Any] | None:
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise TypeError("Rule `aliases` must be a mapping when provided.")
    return dict(source)


def _compile_weight_target(weight: TargetSpec, *, matchers: tuple[MatchSpec, ...]):
    if weight.mode not in {None, "merge"}:
        raise NotImplementedError("Initial protocol compiler supports only the default target merge mode.")

    quantize = weight.quantize
    if quantize is None or not quantize.method:
        raise ValueError("Weight target requires `weight.quantize.method`.")

    method = str(quantize.method).strip().lower()
    if method == METHOD.GGUF.value:
        return _compile_gguf_weight_target(weight, matchers=matchers)
    if method in {METHOD.GPTQ.value, METHOD.AWQ.value}:
        return _compile_quantize_config_weight_target(weight, matchers=matchers, method=METHOD(method))
    raise NotImplementedError(
        "Initial protocol compiler supports only `weight.quantize.method` in {\"gguf\", \"gptq\", \"awq\"}."
    )


def _compile_gguf_weight_target(weight: TargetSpec, *, matchers: tuple[MatchSpec, ...]) -> GGUFConfig:
    if not _supports_initial_weight_match_compilation(matchers):
        raise NotImplementedError(
            "Initial GGUF protocol compiler supports only `match=\"*\"` or `match=[\"*\", \"-:...\"]`."
        )

    quantize = weight.quantize
    if quantize is None:
        raise ValueError("GGUF weight target requires `weight.quantize`.")
    if quantize.method != "gguf":
        raise NotImplementedError(
            "Initial GGUF compiler supports only `weight.quantize.method = \"gguf\"`."
        )

    bits = quantize.args.get("bits")
    if bits is None:
        raise ValueError("GGUF weight target requires `weight.quantize.bits`.")

    export = weight.export
    if export is not None and export.format not in {None, "gguf"}:
        raise NotImplementedError("Initial GGUF compiler supports only `weight.export.format = \"gguf\"`.")

    smoother = _compile_supported_smoother(weight.prepare)
    gguf_format = _resolve_gguf_public_format(bits=bits, export=export)
    dynamic = _compile_negative_match_dynamic(matchers)
    return GGUFConfig(bits=bits, format=gguf_format, smoother=smoother, dynamic=dynamic)


def _compile_quantize_config_weight_target(weight: TargetSpec, *, matchers: tuple[MatchSpec, ...], method: METHOD):
    if not _supports_initial_weight_match_compilation(matchers):
        raise NotImplementedError(
            f"Initial {method.value.upper()} protocol compiler supports only `match=\"*\"` or `match=[\"*\", \"-:...\"]`."
        )
    if weight.prepare:
        raise NotImplementedError(
            f"Initial {method.value.upper()} protocol compiler does not yet support `weight.prepare`."
        )

    quantize = weight.quantize
    if quantize is None:
        raise ValueError(f"{method.value.upper()} weight target requires `weight.quantize`.")
    if quantize.method != method.value:
        raise NotImplementedError(
            f"Initial {method.value.upper()} compiler supports only `weight.quantize.method = \"{method.value}\"`."
        )

    bits = quantize.args.get("bits")
    if bits is None:
        raise ValueError(f"{method.value.upper()} weight target requires `weight.quantize.bits`.")

    export_format = _resolve_export_format(method=method, export=weight.export)
    dynamic = _compile_negative_match_dynamic(matchers)
    group_size = quantize.args.get("group_size", 128)
    sym = bool(quantize.args.get("sym", True))

    kwargs = {
        "method": method,
        "format": export_format,
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "dynamic": dynamic,
    }

    if "desc_act" in quantize.args or method == METHOD.GPTQ:
        kwargs["desc_act"] = bool(quantize.args.get("desc_act", False))

    if method == METHOD.GPTQ:
        if "act_group_aware" in quantize.args:
            kwargs["act_group_aware"] = bool(quantize.args["act_group_aware"])

    return QuantizeConfig(**kwargs)


def _compile_supported_smoother(prepare: tuple[OperationSpec, ...]) -> Optional[SmoothMAD]:
    if not prepare:
        return None
    if len(prepare) != 1:
        raise NotImplementedError("Initial GGUF compiler supports at most one weight.prepare operation.")

    op = prepare[0]
    if op.method not in {"smooth.mad", "smoother"}:
        raise NotImplementedError(
            "Initial GGUF compiler supports only `smooth.mad` in `weight.prepare`."
        )
    k = op.args.get("k")
    if k is None:
        smooth_payload = op.args.get("smooth")
        if isinstance(smooth_payload, Mapping):
            if smooth_payload.get("type") not in {None, "mad"}:
                raise NotImplementedError("Initial GGUF compiler supports only MAD smoothers.")
            k = smooth_payload.get("k")
    return SmoothMAD(k=2.75 if k is None else float(k))


def _resolve_gguf_public_format(bits: Any, export: Optional[ExportSpec]) -> Optional[str]:
    variant = export.variant if export is not None else None

    if isinstance(bits, str):
        normalized = bits.strip().lower().replace("-", "_")
        if normalized and not normalized.isdigit():
            bits_spec = GGUFBits.from_string(normalized)
            public_format = bits_spec.to_public_format()
            if variant is not None and variant != public_format:
                raise ValueError(
                    f"GGUF protocol uses incompatible bits/export variant combination: bits={bits}, export.variant={variant}."
                )
            return variant or public_format

    return variant


def _is_global_match(matchers: tuple[MatchSpec, ...]) -> bool:
    return len(matchers) == 1 and matchers[0].include and matchers[0].pattern == "*"


def _supports_initial_weight_match_compilation(matchers: tuple[MatchSpec, ...]) -> bool:
    includes = tuple(selector for selector in matchers if selector.include)
    return bool(includes) and all(selector.pattern == "*" for selector in includes)


def _compile_negative_match_dynamic(matchers: tuple[MatchSpec, ...]) -> Optional[dict[str, dict[str, Any]]]:
    excludes = tuple(selector for selector in matchers if not selector.include)
    if not excludes:
        return None
    return {f"-:{selector.pattern}": {} for selector in excludes}


def _resolve_export_format(method: METHOD, export: Optional[ExportSpec]) -> FORMAT:
    if method == METHOD.GPTQ:
        if export is None:
            return FORMAT.GPTQ
        if export.format not in {None, METHOD.GPTQ.value}:
            raise NotImplementedError("Initial GPTQ compiler supports only `weight.export.format = \"gptq\"`.")
        variant = str(export.variant or FORMAT.GPTQ.value).strip().lower().replace("-", "_")
        mapping = {
            FORMAT.GPTQ.value: FORMAT.GPTQ,
            FORMAT.GPTQ_V2.value: FORMAT.GPTQ_V2,
            FORMAT.GPTQ_P.value: FORMAT.GPTQ_P,
            FORMAT.MARLIN.value: FORMAT.MARLIN,
            FORMAT.BITBLAS.value: FORMAT.BITBLAS,
        }
        if variant not in mapping:
            raise NotImplementedError(f"Unsupported GPTQ export variant: `{variant}`.")
        return mapping[variant]

    if method == METHOD.AWQ:
        if export is None:
            return FORMAT.GEMM
        if export.format not in {None, METHOD.AWQ.value}:
            raise NotImplementedError("Initial AWQ compiler supports only `weight.export.format = \"awq\"`.")
        variant = str(export.variant or FORMAT.GEMM.value).strip().lower().replace("-", "_")
        mapping = {
            FORMAT.GEMM.value: FORMAT.GEMM,
            FORMAT.GEMV.value: FORMAT.GEMV,
            FORMAT.GEMV_FAST.value: FORMAT.GEMV_FAST,
            "gemvfast": FORMAT.GEMV_FAST,
            FORMAT.LLM_AWQ.value.replace("-", "_"): FORMAT.LLM_AWQ,
            FORMAT.LLM_AWQ.value: FORMAT.LLM_AWQ,
            FORMAT.MARLIN.value: FORMAT.MARLIN,
        }
        if variant not in mapping:
            raise NotImplementedError(f"Unsupported AWQ export variant: `{variant}`.")
        return mapping[variant]

    raise NotImplementedError(f"Unsupported export method resolution for `{method}`.")


def _pattern_matches(pattern: str, module_name: str) -> bool:
    if pattern == "*":
        return True
    return pcre.compile(pattern).search(module_name) is not None
