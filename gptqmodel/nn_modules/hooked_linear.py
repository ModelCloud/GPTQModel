# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import copy
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn

from ..utils.env import env_flag
from ..utils.logger import setup_logger


log = setup_logger()
MODULE_TRACE = env_flag("GPTQMODEL_FORWARD_TRACE", False)


class StopForward(Exception):
    """Signal an intentional early stop of the forward pass."""
    pass

STOP_FORWARD_EXCEPTION = StopForward("Forwarding stopped")

def _trace_describe(value: Any) -> str:
    if torch.is_tensor(value):
        device = value.device
        device_str = f"{device.type}:{device.index}" if device.index is not None else device.type
        return f"Tensor(shape={tuple(value.shape)}, device={device_str}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(map(str, keys[:3]))
        if len(keys) > 3:
            preview += ", ..."
        return f"dict(keys=[{preview}])"
    if value is None:
        return "None"
    return type(value).__name__


def _trace_log_pre(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    original_device: torch.device,
    target_device: torch.device,
) -> Optional[Tuple[str, float]]:
    if not MODULE_TRACE:
        return None
    label = getattr(module, "_hooked_name", module.__class__.__name__)
    log.info(
        "HookTrace:start module=%s cls=%s training=%s orig_device=%s target_device=%s input=%s extras=%s kwargs=%s",
        label,
        module.__class__.__name__,
        module.training,
        original_device,
        target_device,
        _trace_describe(input_tensor),
        [_trace_describe(arg) for arg in args] if args else [],
        {k: _trace_describe(v) for k, v in kwargs.items()} if kwargs else {},
    )
    return label, time.perf_counter()


def _trace_log_post(ctx: Optional[Tuple[str, float]], output: Any) -> None:
    if ctx is None:
        return
    label, start_ts = ctx
    duration = time.perf_counter() - start_ts if start_ts is not None else float("nan")
    log.info(
        "HookTrace:end module=%s duration=%.4fs output=%s",
        label,
        duration,
        _trace_describe(output),
    )


def _trace_log_exception(ctx: Optional[Tuple[str, float]]) -> None:
    if ctx is None:
        return
    label, start_ts = ctx
    duration = time.perf_counter() - start_ts if start_ts is not None else float("nan")
    log.exception(
        "HookTrace:error module=%s duration=%.4fs",
        label,
        duration,
    )


def _filter_args_for_signature(
    forward_fn: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Drop positional/keyword arguments that are unsupported by a function signature."""
    try:
        signature = inspect.signature(forward_fn)
    except (TypeError, ValueError):
        return args, kwargs

    params = list(signature.parameters.values())
    if not params:
        return (), {}

    allows_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    allows_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    filtered_args = args
    if not allows_var_positional and args:
        positional_slots = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        extra_allowed = max(0, len(positional_slots) - 1)
        filtered_args = args[:extra_allowed]

    if allows_var_keyword or not kwargs:
        return filtered_args, kwargs

    allowed_keyword_names = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keyword_names}
    return filtered_args, filtered_kwargs


def _call_with_optional_args(
    forward_fn: Callable[..., Any],
    input_tensor: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Invoke a forward function while gracefully handling unexpected args/kwargs."""
    if not args and not kwargs:
        return forward_fn(input_tensor)

    try:
        return forward_fn(input_tensor, *args, **kwargs)
    except TypeError as first_error:
        filtered_args, filtered_kwargs = _filter_args_for_signature(forward_fn, args, kwargs)
        if (filtered_args, filtered_kwargs) != (args, kwargs):
            try:
                return forward_fn(input_tensor, *filtered_args, **filtered_kwargs)
            except TypeError:
                pass
        try:
            return forward_fn(input_tensor)
        except TypeError:
            raise first_error


def _call_module_with_optional_args(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Invoke a module while tolerating unsupported args/kwargs."""
    if not args and not kwargs:
        return module(input_tensor)

    try:
        return module(input_tensor, *args, **kwargs)
    except TypeError as first_error:
        filtered_args, filtered_kwargs = _filter_args_for_signature(module.forward, args, kwargs)
        if (filtered_args, filtered_kwargs) != (args, kwargs):
            try:
                return module(input_tensor, *filtered_args, **filtered_kwargs)
            except TypeError:
                pass
        try:
            return module(input_tensor)
        except TypeError:
            raise first_error


def _call_wrapped_forward(
    wrapped_module: Optional[torch.nn.Module],
    fallback_forward: Callable[..., Any],
    input_tensor: torch.Tensor,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Execute a wrapped module if available, otherwise fall back to a provided forward."""
    if wrapped_module is not None:
        try:
            return _call_module_with_optional_args(wrapped_module, input_tensor, args, kwargs)
        except TypeError as module_error:
            try:
                return _call_with_optional_args(fallback_forward, input_tensor, args, kwargs)
            except TypeError:
                raise module_error

    return _call_with_optional_args(fallback_forward, input_tensor, args, kwargs)


def _move_output_to_device(output: Any, device: torch.device) -> Any:
    """Recursively move tensors within an output structure to the requested device."""
    if isinstance(output, torch.Tensor):
        return output.to(device=device)
    if isinstance(output, tuple):
        return tuple(_move_output_to_device(item, device) for item in output)
    if isinstance(output, list):
        return [_move_output_to_device(item, device) for item in output]
    if isinstance(output, dict):
        return {key: _move_output_to_device(value, device) for key, value in output.items()}
    return output

# Models using conv1d: gpt2
class HookedConv1D(transformers.Conv1D):
    def __init__(self, nf: int, nx: int) -> None:
        torch.nn.Module.__init__(self)
        self.nf = nf
        self.nx = nx
        self.forward_hook = None
        self.forward_hook_last = False
        self._forward_source: Optional[torch.nn.Module] = None

    @staticmethod
    def from_conv1d(m: transformers.Conv1D):
        assert isinstance(m, transformers.Conv1D)
        custom = HookedConv1D(m.nf, m.nx)
        custom.weight = m.weight
        custom.bias = m.bias
        custom._forward_source = m
        existing_name = getattr(m, "_hooked_name", getattr(m, "full_name", None))
        if existing_name:
            setattr(custom, "_hooked_name", existing_name)
        return custom

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_device = x.device
        target_device = self.weight.data.device
        trace_ctx = _trace_log_pre(self, x, args, kwargs, original_device, target_device)
        if original_device != target_device:
            x = x.to(device=target_device)
        if self._forward_source is not None:
            self._forward_source.training = self.training
        try:
            output = _call_wrapped_forward(self._forward_source, super().forward, x, args, kwargs)
        except StopForward:
            if MODULE_TRACE:
                label = getattr(self, "_hooked_name", self.__class__.__name__)
                log.info("HookTrace:stop module=%s reason=StopForward", label)
            raise
        except Exception:
            if MODULE_TRACE:
                _trace_log_exception(trace_ctx)
            raise

        if self.forward_hook:
            self.forward_hook(self, (x,) + args, output)
            if self.forward_hook_last:
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)

        if original_device != target_device:
            output = _move_output_to_device(output, original_device)
        if MODULE_TRACE:
            _trace_log_post(trace_ctx, output)
        return output

class HookedConv1d(torch.nn.Conv1d):
    def __init__(
        self,
    ) -> None:
        # in_channels,
        # out_channels,
        # kernel_size,
        # stride,
        # padding,
        # dilation,
        # groups,
        # padding_mode,
        torch.nn.Module.__init__(self)
        # TODO: call super().__init__() is too slow, need to find a better way
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride,
        #     padding,
        #     dilation,
        #     groups,
        #     padding_mode,
        # )
        self.forward_hook = None
        self.forward_hook_last = False
        self._forward_source: Optional[torch.nn.Module] = None

    @staticmethod
    def from_conv1d(m: torch.nn.Conv1d):
        assert isinstance(m, torch.nn.Conv1d)

        custom = HookedConv1d()
        custom.in_channels = m.in_channels
        custom.out_channels = m.out_channels
        custom.kernel_size = m.kernel_size
        custom.stride = m.stride
        custom.padding = m.padding
        custom.dilation = m.dilation
        custom.transposed = m.transposed
        custom.output_padding = m.output_padding
        custom.groups = m.groups
        custom.padding_mode = m.padding_mode

        custom.weight = m.weight
        custom.bias = m.bias
        custom._forward_source = m
        existing_name = getattr(m, "_hooked_name", getattr(m, "full_name", None))
        if existing_name:
            setattr(custom, "_hooked_name", existing_name)
        return custom

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_device = x.device
        target_device = self.weight.data.device
        trace_ctx = _trace_log_pre(self, x, args, kwargs, original_device, target_device)
        if original_device != target_device:
            x = x.to(device=target_device)
        if self._forward_source is not None:
            self._forward_source.training = self.training
        try:
            output = _call_wrapped_forward(self._forward_source, super().forward, x, args, kwargs)
        except StopForward:
            if MODULE_TRACE:
                label = getattr(self, "_hooked_name", self.__class__.__name__)
                log.info("HookTrace:stop module=%s reason=StopForward", label)
            raise
        except Exception:
            if MODULE_TRACE:
                _trace_log_exception(trace_ctx)
            raise
        if self.forward_hook:
            self.forward_hook(self, (x,) + args, output)
            if self.forward_hook_last:
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)
        if original_device != target_device:
            output = _move_output_to_device(output, original_device)
        if MODULE_TRACE:
            _trace_log_post(trace_ctx, output)
        return output

# Models using conv2d: ovis
class HookedConv2d(torch.nn.Conv2d):
    def __init__(
        self,
    ) -> None:
        # in_channels,
        # out_channels,
        # kernel_size,
        # stride,
        # padding,
        # dilation,
        # groups,
        # padding_mode,
        torch.nn.Module.__init__(self)
        # TODO: call super().__init__() is too slow, need to find a better way
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride,
        #     padding,
        #     dilation,
        #     groups,
        #     padding_mode,
        # )
        self.forward_hook = None
        self.forward_hook_last = False
        self._forward_source: Optional[torch.nn.Module] = None

    @staticmethod
    def from_conv2d(m: torch.nn.Conv2d):
        assert isinstance(m, torch.nn.Conv2d)

        custom = HookedConv2d()
        custom.in_channels = m.in_channels
        custom.out_channels = m.out_channels
        custom.kernel_size = m.kernel_size
        custom.stride = m.stride
        custom.padding = m.padding
        custom.dilation = m.dilation
        custom.transposed = m.transposed
        custom.output_padding = m.output_padding
        custom.groups = m.groups
        custom.padding_mode = m.padding_mode

        custom.weight = m.weight
        custom.bias = m.bias
        custom._forward_source = m
        existing_name = getattr(m, "_hooked_name", getattr(m, "full_name", None))
        if existing_name:
            setattr(custom, "_hooked_name", existing_name)
        return custom

    @torch.inference_mode()
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_device = input.device
        target_device = self.weight.data.device
        trace_ctx = _trace_log_pre(self, input, args, kwargs, original_device, target_device)
        if original_device != target_device:
            input = input.to(device=target_device)
        if self._forward_source is not None:
            self._forward_source.training = self.training
        try:
            output = _call_wrapped_forward(self._forward_source, super().forward, input, args, kwargs)
        except StopForward:
            if MODULE_TRACE:
                label = getattr(self, "_hooked_name", self.__class__.__name__)
                log.info("HookTrace:stop module=%s reason=StopForward", label)
            raise
        except Exception:
            if MODULE_TRACE:
                _trace_log_exception(trace_ctx)
            raise
        if self.forward_hook:
            self.forward_hook(self, (input,) + args, output)
            if self.forward_hook_last:
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)
        if original_device != target_device:
            output = _move_output_to_device(output, original_device)
        if MODULE_TRACE:
            _trace_log_post(trace_ctx, output)
        return output

# Models using transformers.conv1d: gpt2
class HookedTransformerConv1D(transformers.Conv1D):
    def __init__(self, nf: int, nx: int) -> None:
        torch.nn.Module.__init__(self)
        self.nf = nf
        self.nx = nx
        self.forward_hook = None
        self.forward_hook_last = False
        self._forward_source: Optional[torch.nn.Module] = None

    @staticmethod
    def from_conv1d(conv1d: transformers.Conv1D):
        custom = HookedTransformerConv1D(conv1d.nf, conv1d.nx)
        custom.weight = conv1d.weight
        custom.bias = conv1d.bias
        custom._forward_source = conv1d
        existing_name = getattr(conv1d, "_hooked_name", getattr(conv1d, "full_name", None))
        if existing_name:
            setattr(custom, "_hooked_name", existing_name)
        return custom

    @torch.inference_mode()
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_device = input.device
        target_device = self.weight.data.device
        trace_ctx = _trace_log_pre(self, input, args, kwargs, original_device, target_device)
        if original_device != target_device:
            input = input.to(device=target_device)
        if self._forward_source is not None:
            self._forward_source.training = self.training
        try:
            output = _call_wrapped_forward(self._forward_source, super().forward, input, args, kwargs)
        except StopForward:
            if MODULE_TRACE:
                label = getattr(self, "_hooked_name", self.__class__.__name__)
                log.info("HookTrace:stop module=%s reason=StopForward", label)
            raise
        except Exception:
            if MODULE_TRACE:
                _trace_log_exception(trace_ctx)
            raise
        if self.forward_hook:
            self.forward_hook(self, (input,) + args, output)
            if self.forward_hook_last:
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)
        if original_device != target_device:
            output = _move_output_to_device(output, original_device)
        if MODULE_TRACE:
            _trace_log_post(trace_ctx, output)
        return output

class HookedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        # avoid calling super().__init__() as it would allocate memory based on in/out features
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

        self.forward_hook = None
        self.forward_hook_last = False
        self._forward_source: Optional[torch.nn.Module] = None

    @staticmethod
    def from_linear(linear: torch.nn.Linear):
        custom_linear = HookedLinear(linear.in_features, linear.out_features)
        custom_linear.weight = linear.weight
        custom_linear.bias = linear.bias
        custom_linear._forward_source = linear
        existing_name = getattr(linear, "_hooked_name", getattr(linear, "full_name", None))
        if existing_name:
            setattr(custom_linear, "_hooked_name", existing_name)
        return custom_linear

    @torch.inference_mode()
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_device = input.device
        target_device = self.weight.data.device
        trace_ctx = _trace_log_pre(self, input, args, kwargs, original_device, target_device)
        if original_device != target_device:
            input = input.to(device=target_device)
        if self._forward_source is not None:
            self._forward_source.training = self.training
        try:
            output = _call_wrapped_forward(self._forward_source, super().forward, input, args, kwargs)
        except StopForward:
            if MODULE_TRACE:
                label = getattr(self, "_hooked_name", self.__class__.__name__)
                log.info("HookTrace:stop module=%s reason=StopForward", label)
            raise
        except Exception:
            if MODULE_TRACE:
                _trace_log_exception(trace_ctx)
            raise
        if self.forward_hook:
            self.forward_hook(self, (input,) + args, output)
            if self.forward_hook_last:
                raise STOP_FORWARD_EXCEPTION.with_traceback(None)
        if original_device != target_device:
            output = _move_output_to_device(output, original_device)
        if MODULE_TRACE:
            _trace_log_post(trace_ctx, output)
        return output


def _replace_module(
    module,
    child,
    name,
    level: int = 0,
    debug: bool = False,
    path: str = "",
) -> bool:
    level_indent = "---" * level
    instance_type = type(child)
    if debug:
        log.info(f"{level_indent} Hook: {instance_type.__name__}: {name}")

    full_name = path or name

    if isinstance(child, torch.nn.Linear):
        hooked = HookedLinear.from_linear(child)
        setattr(hooked, "_hooked_name", full_name)
        setattr(module, name, hooked)
    elif isinstance(child, transformers.Conv1D):
        hooked = HookedTransformerConv1D.from_conv1d(child)
        setattr(hooked, "_hooked_name", full_name)
        setattr(module, name, hooked)
    elif isinstance(child, nn.Conv1d):
        hooked = HookedConv1d.from_conv1d(child)
        setattr(hooked, "_hooked_name", full_name)
        setattr(module, name, hooked)
    elif isinstance(child, nn.Conv2d):
        hooked = HookedConv2d.from_conv2d(child)
        setattr(hooked, "_hooked_name", full_name)
        setattr(module, name, hooked)
    else:
        if debug:
            log.error(f"{level_indent} Hook: execute_replace but layer skipped due to type not supported: {name}")
        return False

    return True


def replace_module_with_hooked_legacy(
    module,
    level: int = 0,
    quant_lm_head: bool = False,
    prefix: str = "",
):
    # if level == 0:
    #     log.info("Hooked Modules: Using legacy based config for targeting of modules")

    for name, child in module.named_children():
        if not quant_lm_head and hasattr(module, "get_output_embeddings") and child == module.get_output_embeddings():
            continue

        child_path = f"{prefix}.{name}" if prefix else name

        if not _replace_module(module, child, name, level, quant_lm_head, path=child_path):
            replace_module_with_hooked_legacy(
                child,
                level=level+1,
                quant_lm_head=quant_lm_head,
                prefix=child_path,
            )

# deprecated features
def replace_module_with_hooked_tree(
    module,
    tree: Union[List, Dict] = [],
    level: int = 0,
    debug: bool = False,
    prefix: str = "",
):
    if level == 0:
        log.info("Hooked Modules: Using tree based config for accurate targeting of modules")

    tree = copy.copy(tree) # defensive copy

    # tuple represents targeted modules
    execute_replace = isinstance(tree, Tuple)
    # level indent
    level_indent = "---" * level

    for name, child in module.named_children():
        if debug:
            log.info(f"{level_indent} child name: {name}")
        child_path = f"{prefix}.{name}" if prefix else name
        if execute_replace:
            if name in tree:
                # do replace if name in tree and tree is a tuple
                _replace_module(module, child, name, level, debug, path=child_path)
            else:
                if debug:
                    log.warn(f"{level_indent} Hook: execute_replace but layer skipped due to not targeted: {name}")
        else:
            if isinstance(tree, Dict):
                if name in tree:
                    if isinstance(tree[name], Tuple) and name in tree[name]:
                        # do replace if name in tree and tree is a tuple
                        _replace_module(module, child, name, level, debug, path=child_path)
                    elif isinstance(tree[name], (Dict, Tuple, List)):
                        # follow tree node if name in tree and tree is a dict
                        if debug:
                            log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")

                        replace_module_with_hooked_tree(
                            child,
                            tree=tree[name],
                            level=level+1,
                            debug=debug,
                            prefix=child_path,
                        )
                    else:
                        if debug:
                            log.warn(f"{level_indent} Hook: skipped unknown tree node dict: {name} vs tree: {tree}")

                elif "#" in tree and name.isdigit():
                    if debug:
                        log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                    replace_module_with_hooked_tree(
                        child,
                        tree=tree["#"],
                        level=level+1,
                        debug=debug,
                        prefix=child_path,
                    )
                else:
                    if debug:
                        log.warn(f"{level_indent} Hook: skipped unknown tree node dict: {name} vs tree: {tree}")
            elif isinstance(tree, List):
                next_node = tree[0]
                if isinstance(next_node, Dict):
                    if name in next_node:
                        if debug:
                            log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                        replace_module_with_hooked_tree(
                            child,
                            tree=next_node[name],
                            level=level + 1,
                            debug=debug,
                            prefix=child_path,
                        )
                    elif name.isdigit and "#" in next_node:
                        if debug:
                            log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                        replace_module_with_hooked_tree(
                            child,
                            tree=next_node["#"],
                            level=level + 1,
                            debug=debug,
                            prefix=child_path,
                        )
                    else:
                        if debug:
                            log.warn(
                            f"{level_indent} Hook: skipped unknown tree node dict: {name} vs next_node: {next_node}")
                elif name == next_node or (next_node == "#" and name.isdigit()):
                    if debug:
                        log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                    replace_module_with_hooked_tree(
                        child,
                        tree=tree[1:],
                        level=level + 1,
                        debug=debug,
                        prefix=child_path,
                    )
                else:
                    if debug:
                        log.warn(f"{level_indent} Hook: skipped unknown tree node list: {name} vs next_node: {next_node}")

            # if len(tree) > 0:
            #     # list or dict
            #     node = next(tree_iter)
            #     log.info(f"{level_indent} next node: {node}")
            #
            #     # matched user designed tree node/path
            #     if name == node or (node == "#" and name.isdigit()):
            #         log.info(f"{level_indent} Hook: follow tree node: {node} -> nest into {name}")
            #         replace_module_with_hooked(child, tree=tree[1:] if isinstance(tree, List) else tree[name], level=level + 1, debug=debug)
            #     # # list: simple node
            #     # elif isinstance(node, List):
            #     #     if name in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node list: {node} -> {name} -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=tree[1:], level=level + 1, debug=debug)
            #     #     elif name.isdigit() and "#" in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node list: {node} -> # -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=tree[1:], level=level + 1, debug=debug)
            #     #     else:
            #     #         log.warn(f"{level_indent} Hook: stopped at unknown tree node list: {node} -> {name}")
            #     # # dict: nested node with final modules larget as value [list]
            #     # elif isinstance(node, Dict):
            #     #     if name in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node dict: {node} -> {name} -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=node[name], level=level + 1, debug=debug)
            #     #     elif name.isdigit() and "#" in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node dict: {node} -> # -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=node["#"], level=level + 1, debug=debug)
            #     #     else:
            #     #         log.warn(f"{level_indent} Hook: stopped at unknown tree node dict: {node} -> {name}")
            #     else:
            #         log.warn(f"{level_indent} Hook: skipped at unknown tree node: {name}")
            # else:
            #     log.warn(f"{level_indent} Hook: follow naively -> nest into {name}")
            #     replace_module_with_hooked(child, [], level=level + 1, debug=debug)
