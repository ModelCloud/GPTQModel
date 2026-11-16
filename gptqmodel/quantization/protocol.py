# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import weakref
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


@dataclass
class ActionReport:
    """Outcome of one ``apply_plan_actions`` call.

    ``skipped`` is not silent: every module a rule matched but could not fold is
    recorded with the reason, and a rule that resolves nothing raises.
    """

    applied: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def record_applied(self, stage: str, rule: Rule, method: str) -> None:
        self.applied.append(f"{stage}:{_rule_label(rule)}:{method}")

    def record_skipped(self, module: str, reason: str) -> None:
        self.skipped.append((module, reason))


@dataclass(frozen=True)
class ActionContext:
    """Everything one registered action needs to operate on a live model."""

    model: Any
    rule: Rule
    stage: str
    matched: tuple[tuple[str, Any], ...]
    calibration: Optional[Callable[[], None]]
    report: ActionReport


# A handler returns an undo callable, or ``None`` when it changed nothing. The
# undo restores the model so a later failure cannot leave a half-applied plan.
Undo = Callable[[], None]
ActionHandler = Callable[[ActionContext, Mapping[str, Any]], Optional[Undo]]
ActionValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ActionSpec:
    handler: ActionHandler
    validate: ActionValidator


@dataclass(frozen=True)
class CompiledAction:
    """An ``OperationSpec`` resolved against the action registry."""

    method: str
    handler: ActionHandler
    validate: ActionValidator
    args: dict[str, Any]


# Keyed by ``OperationSpec.method``. Compilation resolves every action through
# this registry; execution resolves the same keys again in ``apply_plan_actions``.
ACTION_REGISTRY: dict[str, ActionSpec] = {}


def _accept_any_args(args: Mapping[str, Any]) -> None:
    return None


def register_action(
    method: str,
    handler: ActionHandler,
    *,
    validate: Optional[ActionValidator] = None,
) -> None:
    """Register one rule-scoped action handler under ``method``.

    ``validate`` runs at compilation time, before any model exists, so an
    invalid argument fails the plan instead of a half-applied fold.
    """

    key = str(method).strip().lower()
    if not key:
        raise ValueError("Protocol action method must not be empty.")
    ACTION_REGISTRY[key] = ActionSpec(handler=handler, validate=validate or _accept_any_args)


def known_action_methods() -> tuple[str, ...]:
    """Registered action methods, sorted for stable error messages."""

    return tuple(sorted(ACTION_REGISTRY))


def compile_rule_actions(actions: tuple[OperationSpec, ...]) -> tuple[CompiledAction, ...]:
    """Resolve rule ``actions`` against the registry.

    Unknown methods fail closed with the list of known methods, and each method
    validates its arguments. Compilation does not touch a model;
    ``apply_plan_actions`` executes the resolved handlers.
    """

    return tuple(_compile_action(action) for action in actions)


def _compile_action(action: OperationSpec) -> CompiledAction:
    method = str(action.method).strip().lower()
    spec = ACTION_REGISTRY.get(method)
    if spec is None:
        known = ", ".join(f"`{name}`" for name in known_action_methods()) or "none"
        raise NotImplementedError(f"Unsupported protocol action `{action.method}`. Known actions: {known}.")
    args = dict(action.args)
    spec.validate(args)
    return CompiledAction(method=method, handler=spec.handler, validate=spec.validate, args=args)


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
    """Compile the plan's single `weight` target into a ``QuantizeConfig``.

    Action-only stages and rules are accepted, and their action methods and
    arguments are validated, but they are not executed here: actions run in
    ``apply_plan_actions``. That keeps this function pure while still refusing
    an unknown method or an invalid argument. ``compile_and_apply_plan`` runs
    both steps for one plan.
    """

    # A directly constructed ``ExecutionPlan`` (as in the documented Python DSL)
    # can still carry a string ``match`` or dict targets; normalize the plan so
    # compilation and action resolution see the same typed shape as
    # ``compile_protocol`` produces.
    resolved = _normalize_plan(plan)

    config_rules = []
    for stage in resolved.stages:
        for rule in stage.rules:
            if rule.aliases:
                raise NotImplementedError(
                    "Initial protocol implementation does not support aliases during config compilation."
                )
            if rule.stop:
                raise NotImplementedError(
                    "Initial protocol implementation does not support stop during config compilation."
                )
            if rule.actions:
                # Validated, never dropped: an unknown method or a bad argument
                # fails the plan before any model or calibration work starts.
                compile_rule_actions(rule.actions)
            if rule.weight is None:
                if rule.input is not None or rule.output is not None or rule.kv_cache is not None:
                    raise NotImplementedError(
                        "Initial protocol implementation only supports weight-target compilation."
                    )
                continue
            config_rules.append(rule)

    if not config_rules:
        raise ValueError("Initial protocol implementation requires a `weight` target.")
    if len(config_rules) > 1:
        raise NotImplementedError(
            "Initial protocol implementation supports exactly one rule with a `weight` target "
            "for config compilation."
        )

    rule = config_rules[0]
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
) -> ActionReport:
    """Run rule-scoped protocol ``actions`` against ``model``.

    This is the separate action stage the protocol documents: config
    compilation stays pure, and this function resolves each rule's matches on
    the live model before running the registered handlers in stage/rule order.

    Every rule's action methods, arguments, and matches are resolved before the
    first handler runs, so an unknown action or an empty match cannot leave an
    earlier stage applied. A handler that fails after mutating the model rolls
    itself back, and a failure in a later rule rolls back every earlier rule.

    ``calibration`` must be a zero-argument callable that runs one or more
    forward passes through ``model`` (the model root). Actions that need
    activation statistics, such as ``smoothquant``, raise
    :class:`ProtocolActionError` when it is omitted.
    """

    resolved = _normalize_plan(plan)
    named_modules = tuple(model.named_modules())

    executions = []
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
            executions.append((stage.name, rule, compiled, matched))

    report = ActionReport()
    undos: list[Undo] = []
    try:
        for stage_name, rule, compiled, matched in executions:
            context = ActionContext(
                model=model,
                rule=rule,
                stage=stage_name,
                matched=matched,
                calibration=calibration,
                report=report,
            )
            for action in compiled:
                undo = action.handler(context, action.args)
                if undo is not None:
                    undos.append(undo)
                report.record_applied(stage_name, rule, action.method)
    except BaseException:
        for undo in reversed(undos):
            undo()
        raise
    return report


def _normalize_plan(plan: Any) -> ExecutionPlan:
    if isinstance(plan, ExecutionPlan):
        return ExecutionPlan(
            version=plan.version,
            stages=tuple(_normalize_stage(stage) for stage in plan.stages),
        )
    return compile_protocol(plan)


def compile_and_apply_plan(
    model: Any,
    source: Any,
    *,
    calibration: Optional[Callable[[], None]] = None,
) -> tuple[Any, ActionReport]:
    """Compile one plan to a ``QuantizeConfig`` and apply its actions.

    One plan can therefore carry both steps the protocol documents: an
    action-only stage that folds norms, and the weight stage that supplies the
    quantizer configuration.
    """

    plan = source if isinstance(source, ExecutionPlan) else compile_protocol(source)
    config = compile_plan_to_quantize_config(plan)
    report = apply_plan_actions(model, plan, calibration=calibration)
    return config, report


def _rule_label(rule: Rule) -> str:
    selectors = []
    for selector in rule.match:
        selectors.append(f"{'' if selector.include else '-:'}{selector.pattern}")
    return "[" + ", ".join(selectors) + "]"


_SMOOTHQUANT_METHOD = "smoothquant"
_SMOOTHQUANT_DEFAULT_ALPHA = 0.5
# Verification bound for a fold, expressed as a multiple of the model dtype's
# relative precision at the observed output magnitude. Rescaling a weight by one
# ulp of the storage dtype is unavoidable, so a low-precision fold moves the
# output by a few eps; a missed consumer moves it by a fraction of the norm
# output, which is far larger. Measured on Qwen2.5-0.5B (q/k/v, alpha=0.5):
# fp32 4.4e-5, fp16 8.6e-2, bf16 6.9e-1 absolute, i.e. 2e-6, 3.9e-3 and 3.2e-2
# of max|logits| - all below 8*eps of their dtype. The smallest observed
# corruption (a missed MoE router) is 1.6e-1 of max|logits|, above the bound.
_FOLD_EPS_MULTIPLIER = 8.0
_FOLD_ATOL = 1e-3


def smoothquant(alpha: float = _SMOOTHQUANT_DEFAULT_ALPHA) -> OperationSpec:
    """Build the ``smoothquant`` rule action used throughout the protocol docs."""

    return OperationSpec(method=_SMOOTHQUANT_METHOD, args={"alpha": alpha})


def _validate_smoothquant_args(args: Mapping[str, Any]) -> None:
    unknown = set(args) - {"alpha"}
    if unknown:
        raise ProtocolActionError(
            f"smoothquant does not accept argument(s) {sorted(unknown)}; only `alpha` is supported."
        )
    value = args.get("alpha", _SMOOTHQUANT_DEFAULT_ALPHA)
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolActionError(f"smoothquant `alpha` must be a float, got {value!r}.") from exc
    if math.isnan(alpha) or not 0.0 <= alpha <= 1.0:
        raise ProtocolActionError(f"smoothquant `alpha` must be in [0, 1], got {alpha}.")


@dataclass
class _Capture:
    """First-pass data-flow trace and per-channel activation maxima."""

    x_max: dict[str, Any] = field(default_factory=dict)
    predecessors: dict[str, set[str]] = field(default_factory=dict)
    call_args: Optional[tuple[tuple[Any, ...], dict[str, Any]]] = None
    call_output: Any = None


def _capture_smoothquant_statistics(model: Any, calibration: Callable[[], None]) -> _Capture:
    """Record the first pass's norm -> consumer graph plus per-channel maxima.

    Identity is resolved with ``weakref`` rather than a raw ``data_ptr()``: the
    allocator reuses a freed address, so a pointer recorded for one layer's norm
    output also matched inputs in other layers. A weak reference dereferences to
    ``None`` once the tensor is gone, so a recycled address can never match, and
    no activation is pinned for the duration of the run.

    Only ``[in_features]`` maxima are retained. No activation tensor survives the
    call, so this creates no dense activation cache.
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

    capture = _Capture()
    capture.x_max = dict.fromkeys(weighted)
    capture.predecessors = {name: set() for name in modules if name}

    # id() -> [(norm name, weakref)] for the first pass only.
    norm_refs: dict[int, list[tuple[str, Any]]] = {}
    state = {"pass": 0}
    mask_holder: dict[str, Any] = {}
    handles = []

    def _root_pre_hook(module, args, kwargs):
        state["pass"] += 1
        if state["pass"] != 1:
            return None
        capture.call_args = (args, kwargs)
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask_holder["mask"] = mask
        return None

    def _root_post_hook(module, args, output):
        if state["pass"] == 1:
            capture.call_output = output
        return None

    def _accumulate_max(name, activation):
        flat = activation.detach()
        if flat.ndim > 2:
            mask = mask_holder.get("mask")
            if mask is not None and tuple(mask.shape) == tuple(flat.shape[:2]):
                keep = mask.reshape(-1).bool()
                flat = flat.reshape(-1, flat.shape[-1])[keep]
            else:
                flat = flat.reshape(-1, flat.shape[-1])
        else:
            flat = flat.reshape(-1, flat.shape[-1])
        if flat.numel() == 0:
            return
        current = flat.abs().amax(dim=0).to(torch.float32)
        previous = capture.x_max[name]
        capture.x_max[name] = current if previous is None else torch.maximum(previous, current)

    def _input_hook(name, accumulate):
        def hook(module, args, kwargs):
            activation = args[0] if args else kwargs.get("input")
            if accumulate and isinstance(activation, torch.Tensor) and activation.is_floating_point():
                if activation.ndim >= 2:
                    _accumulate_max(name, activation)
            if state["pass"] != 1:
                return None
            # Identity is checked against every tensor argument, not only the
            # first one: transformers modules are usually called with
            # `hidden_states=` as a keyword.
            for tensor in (*args, *kwargs.values()):
                if not isinstance(tensor, torch.Tensor):
                    continue
                refs = norm_refs.get(id(tensor))
                if refs:
                    live = {norm_name for norm_name, ref in refs if ref() is tensor}
                    if live:
                        capture.predecessors[name].update(live)
            return None

        return hook

    def _output_hook(name):
        def hook(module, args, output):
            if state["pass"] != 1 or not isinstance(output, torch.Tensor):
                return None
            norm_refs.setdefault(id(output), []).append((name, weakref.ref(output)))
            return None

        return hook

    handles.append(model.register_forward_pre_hook(_root_pre_hook, with_kwargs=True))
    handles.append(model.register_forward_hook(_root_post_hook))
    # Hook every module, not only the weighted ones: a consumer of the norm
    # output that has no 2-D weight (a fused expert block, a functional wrapper)
    # is exactly the case that used to corrupt the output silently.
    for name, module in modules.items():
        if not name:
            continue
        handles.append(module.register_forward_pre_hook(_input_hook(name, name in weighted), with_kwargs=True))
    for name, module in norms.items():
        handles.append(module.register_forward_hook(_output_hook(name)))
    try:
        calibration()
    finally:
        for handle in handles:
            handle.remove()
    return capture


def _output_tensor(output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, Mapping):
        for key in ("logits", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _snapshot_parameters(parameters: list[Any]) -> list[tuple[Any, Any]]:
    import torch

    unique: dict[int, Any] = {}
    for parameter in parameters:
        unique.setdefault(id(parameter), parameter)
    with torch.no_grad():
        return [(parameter, parameter.detach().clone()) for parameter in unique.values()]


def _restore_parameters(snapshot: list[tuple[Any, Any]]) -> None:
    import torch

    with torch.no_grad():
        for parameter, saved in snapshot:
            parameter.data.copy_(saved)


def _assert_fold_preserves_output(
    model: Any,
    capture: _Capture,
    *,
    rule: Rule,
    dtype: Any = None,
    folds: Optional[list[str]] = None,
) -> None:
    """Replay the captured calibration call and require the same output.

    A fold is a gauge: the norm and every consumer of its output are rescaled so
    the model computes the same function. Any consumer the resolver could not
    see - a functional residual, a fused expert parameter, a tied head - shows up
    here as an output change, and the caller then rolls the fold back.
    """

    import torch

    if capture.call_args is None:
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} ran calibration without a model forward pass; "
            "the fold cannot be verified."
        )
    reference = _output_tensor(capture.call_output)
    if reference is None:
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} cannot verify the fold: the model output is not a tensor."
        )
    if not bool(torch.isfinite(reference).all()):
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} cannot verify the fold: the reference output is not finite."
        )

    args, kwargs = capture.call_args
    with torch.no_grad():
        replayed = model(*args, **kwargs)
    actual = _output_tensor(replayed)
    if actual is None or tuple(actual.shape) != tuple(reference.shape):
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} cannot verify the fold: the replayed output shape changed."
        )

    reference_f64 = reference.to(torch.float64)
    actual_f64 = actual.to(torch.float64)
    diff = (actual_f64 - reference_f64).abs()
    dtype = reference.dtype if dtype is None else dtype
    try:
        eps = float(torch.finfo(dtype).eps)
    except TypeError:  # non-floating output dtype: fall back to the output's own
        eps = float(torch.finfo(reference.dtype).eps)
    tolerance = _FOLD_ATOL + _FOLD_EPS_MULTIPLIER * eps * reference_f64.abs().max()
    excess = diff - tolerance
    if not bool((excess > 0).any()):
        return
    agreement = (reference_f64.argmax(dim=-1) == actual_f64.argmax(dim=-1)).to(torch.float64).mean()
    if folds is None:
        raise ProtocolActionError(
            f"smoothquant rule {_rule_label(rule)} cannot verify a fold: replaying the calibration call "
            f"changed the output (max |diff| = {diff.max().item():.6g}). "
            "The forward pass is not deterministic; run the model in eval() mode."
        )
    raise ProtocolActionError(
        f"smoothquant rule {_rule_label(rule)} folded {sorted(folds)} but the model output changed: "
        f"max |diff| = {diff.max().item():.6g} (limit {tolerance.item():.6g}), "
        f"argmax agreement = {agreement.item():.3f}. "
        "A consumer of the folded norm output is not a module the resolver can see "
        "(functional residual, fused expert parameter, or a weight shared with a non-consumer); "
        "the fold was rolled back."
    )


def _assert_consumer_weights_exclusive(
    modules: dict[str, Any],
    consumers: list[tuple[str, Any]],
    norm_name: str,
    rule: Rule,
) -> None:
    """Refuse to scale a weight that another, unscaled module also owns."""

    import torch

    owners: dict[tuple[int, int, tuple[int, ...]], list[str]] = {}
    for name, module in modules.items():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        key = (weight.untyped_storage().data_ptr(), weight.storage_offset(), tuple(weight.shape))
        owners.setdefault(key, []).append(name)

    consumer_ids = {id(module) for _, module in consumers}
    for name, module in consumers:
        weight = module.weight
        key = (weight.untyped_storage().data_ptr(), weight.storage_offset(), tuple(weight.shape))
        shared = [other for other in owners.get(key, ()) if id(modules[other]) not in consumer_ids]
        if shared:
            raise ProtocolActionError(
                f"smoothquant rule {_rule_label(rule)} refuses to scale `{name}`: its weight is shared with "
                f"{sorted(shared)}, which does not consume `{norm_name}`. Scaling one would change the other."
            )


def _apply_smoothquant_action(context: ActionContext, args: Mapping[str, Any]) -> Optional[Undo]:
    import torch

    from .awq.quantize.scale import is_foldable_source, scale_ln_fcs

    _validate_smoothquant_args(args)
    alpha = float(args.get("alpha", _SMOOTHQUANT_DEFAULT_ALPHA))
    label = _rule_label(context.rule)

    if context.calibration is None:
        raise ProtocolActionError(
            f"smoothquant rule {label} requires calibration activations; "
            "call apply_plan_actions(model, plan, calibration=callable)."
        )

    foldable = []
    for name, module in context.matched:
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            foldable.append((name, module))
        else:
            context.report.record_skipped(name, f"{type(module).__name__} has no 2-D weight")
    if not foldable:
        example = context.matched[0][0] if context.matched else "<none>"
        raise ProtocolActionError(
            f"smoothquant rule {label} matched no module with a 2-D weight; "
            f"the first match is `{example}` ({type(context.matched[0][1]).__name__}). "
            "Match the linear that consumes the norm, not its container."
        )

    modules = dict(context.model.named_modules())
    capture = _capture_smoothquant_statistics(context.model, context.calibration)

    groups: dict[str, list[str]] = {}
    unresolved: list[tuple[str, list[str]]] = []
    for name, _module in foldable:
        predecessors = sorted(capture.predecessors.get(name, set()))
        if len(predecessors) == 1:
            groups.setdefault(predecessors[0], []).append(name)
        else:
            unresolved.append((name, predecessors))

    unresolved_details = []
    for name, predecessors in unresolved:
        if not predecessors:
            reason = "no gain norm output feeds it"
        else:
            reason = f"several gain norm outputs feed it: {predecessors}"
        context.report.record_skipped(name, reason)
        unresolved_details.append((name, reason))

    if not groups:
        detail = "; ".join(f"`{name}`: {reason}" for name, reason in unresolved_details)
        raise ProtocolActionError(
            f"smoothquant rule {label} resolved no foldable norm -> linear pair ({detail}). "
            "v1 folds only the canonical norm -> linear pair."
        )

    plans = []
    for norm_name, consumer_names in groups.items():
        norm = modules[norm_name]
        if not is_foldable_source(norm):
            raise ProtocolActionError(
                f"smoothquant rule {label} resolved `{norm_name}` ({type(norm).__name__}) "
                "as the preceding norm, but it has no foldable gain."
            )
        for name, predecessors in capture.predecessors.items():
            if norm_name in predecessors and name not in consumer_names:
                consumer_names.append(name)
        consumers = []
        for name in consumer_names:
            module = modules[name]
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                consumers.append((name, module))
                continue
            # A container that only forwards the norm output to consumers below
            # it (an attention or MLP block) is transparent: its children apply
            # the scale. Anything else cannot be rescaled, and folding the norm
            # would change its input, so the fold is refused.
            if any(other.startswith(f"{name}.") for other in consumer_names if other != name):
                context.report.record_skipped(name, "container forwards the norm output to scaled consumers")
                continue
            raise ProtocolActionError(
                f"smoothquant rule {label} resolved `{norm_name}` as the preceding norm for `{name}`, "
                f"but `{name}` ({type(module).__name__}) has no 2-D weight and no scaled child; "
                "v1 cannot fold that consumer."
            )
        if not consumers:
            raise ProtocolActionError(
                f"smoothquant rule {label} resolved `{norm_name}` but no consumer of its output has a 2-D weight."
            )
        _assert_consumer_weights_exclusive(modules, consumers, norm_name, context.rule)
        scales = _smoothquant_scales(capture, consumers, alpha=alpha, rule=context.rule, norm_name=norm_name)
        if int(scales.numel()) != int(norm.weight.numel()):
            raise ProtocolActionError(
                f"smoothquant rule {label} computed {scales.numel()} channel scales for `{norm_name}` "
                f"with {norm.weight.numel()} gain values."
            )
        plans.append((norm_name, norm, consumers, scales))

    parameters: list[Any] = []
    for _norm_name, norm, consumers, _scales in plans:
        parameters.append(norm.weight)
        if getattr(norm, "bias", None) is not None:
            parameters.append(norm.bias)
        parameters.extend(module.weight for _name, module in consumers)
    snapshot = _snapshot_parameters(parameters)

    # Determinism precondition: the fold is verified by replaying the captured
    # calibration call, which only proves anything if the model is deterministic.
    fold_dtype = plans[0][1].weight.dtype
    _assert_fold_preserves_output(context.model, capture, rule=context.rule, dtype=fold_dtype)

    with torch.no_grad():
        for norm_name, norm, consumers, scales in plans:
            # ``scale_ln_fcs`` divides the norm gain/bias and multiplies every
            # consumer's input columns, so the block output is preserved.
            scale_ln_fcs(norm, [module for _name, module in consumers], scales)

    try:
        _assert_fold_preserves_output(
            context.model,
            capture,
            rule=context.rule,
            dtype=fold_dtype,
            folds=[norm_name for norm_name, _norm, _consumers, _scales in plans],
        )
    except ProtocolActionError:
        _restore_parameters(snapshot)
        raise

    return lambda: _restore_parameters(snapshot)


def _smoothquant_scales(
    capture: _Capture,
    consumers: list[tuple[str, Any]],
    *,
    alpha: float,
    rule: Rule,
    norm_name: str,
) -> Any:
    import torch

    label = _rule_label(rule)
    x_max = None
    w_max = None
    for name, module in consumers:
        activation_max = capture.x_max.get(name)
        if activation_max is None:
            continue
        if int(module.weight.shape[1]) != int(activation_max.numel()):
            raise ProtocolActionError(
                f"smoothquant rule {label} found `{name}` with {module.weight.shape[1]} input columns "
                f"but {activation_max.numel()} activation channels."
            )
        x_max = activation_max if x_max is None else torch.maximum(x_max, activation_max)
        weight_max = module.weight.detach().abs().amax(dim=0).to(torch.float32)
        w_max = weight_max if w_max is None else torch.maximum(w_max, weight_max)
    if x_max is None or w_max is None:
        raise ProtocolActionError(
            f"smoothquant rule {label} collected no calibration activations for `{norm_name}`."
        )
    if x_max.numel() != w_max.numel():
        raise ProtocolActionError(
            f"smoothquant rule {label} found mismatched channel counts under `{norm_name}`: "
            f"activation {x_max.numel()}, weight {w_max.numel()}."
        )
    if not bool(torch.isfinite(x_max).all()) or not bool(torch.isfinite(w_max).all()):
        # An fp16 activation can overflow to inf. Reject before any parameter is
        # divided, so a non-finite scale cannot leave the model partly modified.
        raise ProtocolActionError(
            f"smoothquant rule {label} measured a non-finite activation or weight maximum under `{norm_name}`; "
            "the fold was not applied."
        )
    eps = torch.finfo(torch.float32).eps
    scales = (x_max.clamp_min(eps).pow(alpha) / w_max.clamp_min(eps).pow(1.0 - alpha)).clamp_min(eps)
    if not bool(torch.isfinite(scales).all()):
        raise ProtocolActionError(
            f"smoothquant rule {label} computed non-finite channel scales for `{norm_name}`; the fold was not applied."
        )
    return scales


register_action(_SMOOTHQUANT_METHOD, _apply_smoothquant_action, validate=_validate_smoothquant_args)


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
    compiled: dict[str, dict[str, Any]] = {}
    for selector in excludes:
        pattern = selector.pattern
        if not pattern.startswith(("^", ".*")):
            pattern = f".*(?:{pattern})"
        compiled[f"-:{pattern}"] = {}
    return compiled


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
