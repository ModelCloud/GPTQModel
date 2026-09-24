# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""
MoE (Mixture of Experts) lifecycle hooks system.

This module provides a base class for model-specific MoE lifecycle hooks that allow
customization of MoE forward passes and routing logic during quantization.
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..nn_modules.hooked_linear import StopForward
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.model import move_to
from ..utils.python import has_gil_disabled
from ..utils.torch import torch_sync
from .moe_capture_streams import RoutedMoECaptureStreamAttachment
from .moe_input_replay import RoutedMoEInputReplayAttachment


log = setup_logger()
_ROUTED_CAPTURE_STREAMS = RoutedMoECaptureStreamAttachment()


def _moe_parallel_input_capture_eligible(quantize_config: Any) -> bool:
    """Return whether default-on MoE input capture can safely run in parallel."""

    moe_config = getattr(quantize_config, "moe", None)
    execution = getattr(moe_config, "execution", None)
    return bool(
        getattr(execution, "parallel_input_capture", True)
        and has_gil_disabled()
        and torch.cuda.device_count() > 1
    )


def _get_module_by_relative_path(parent: nn.Module, relative_path: str) -> Optional[nn.Module]:
    """
    Get a submodule from a parent module by relative path.

    Args:
        parent: The parent module (e.g., a layer replica)
        relative_path: Dot-separated path to the submodule (e.g., 'mlp.experts.0.gate_proj')

    Returns:
        The submodule if found, None otherwise
    """
    if not relative_path:
        return parent

    parts = relative_path.split('.')
    current = parent

    for i, part in enumerate(parts):
        path_so_far = '.'.join(parts[:i + 1])
        if hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, '__getitem__') and part.isdigit():
            # Handle indexed access for nn.ModuleList or similar
            try:
                current = current[int(part)]
            except (IndexError, KeyError) as e:
                raise ValueError(f"[MoE PATH] Failed indexing '{part}': {e}")
        else:
            # List available attributes for debugging
            attrs = [a for a in dir(current) if not a.startswith('_')][:20]
            raise ValueError(
                f"[MoE PATH] Failed at '{part}' ({path_so_far}), current={type(current).__name__}, attrs={attrs}")

    return current


class MoELifecycleHooks:
    """
    Base class for model-specific MoE lifecycle hooks.

    This class provides customization points for MoE-specific operations during quantization.
    Subclasses should override these methods to provide model-specific implementations.

    Main use case: Forward whole calibration dataset to all experts instead of only routed ones.
    """

    # List of possible expert block names that models can override
    expert_block_names = ['experts']

    # List of possible shared expert block names that models can override
    shared_expert_block_names = ['shared_experts', 'shared_expert']

    def get_moe_block(self, layer_module: nn.Module, model_class: type) -> Optional[nn.Module]:
        """
        Extract the MoE block from a layer module using the :moe flag from module_tree.

        Args:
            layer_module: The layer module (e.g., DecoderLayer)
            model_class: The model class (to access module_tree)
        Returns:
            The MoE block module, or None if not found

        Example:
            For GLM-4, module_tree has "mlp:moe", so this returns layer_module.mlp
            For MiniMax-M2, module_tree has "block_sparse_moe:moe", so this returns layer_module.block_sparse_moe
        """
        # Get MoE module name from model definition
        moe_module_name = model_class.get_moe_module_name()

        if not moe_module_name:
            model_name = model_class.__name__ if isinstance(model_class, type) else type(model_class).__name__
            log.error(f"No :moe flag found in module_tree for {model_name}")
            return None

        # Get the module by name
        moe_block = getattr(layer_module, moe_module_name[0], None)

        return moe_block

    def get_moe_block_for_subset(
        self,
        layer_module: nn.Module,
        model_class: type,
        current_subset: Optional[Dict[str, Any]] = None,
    ) -> Optional[nn.Module]:
        """Resolve the MoE root for a quantization subset.

        The default delegates to the original two-argument hook so existing
        model-specific ``get_moe_block`` overrides remain compatible. Models
        with multiple expert families can override this method.
        """
        return self.get_moe_block(layer_module, model_class)

    def get_experts_module(self, moe_block: nn.Module, model_class: type) -> Optional[nn.Module]:
        """
        Extract experts module from MoE block by checking common attribute names.

        Args:
            moe_block: The MoE block module
            model_class: The model class (for compatibility, unused)

        Returns:
            The experts module, or None if not found

        Example:
            Returns moe_block.experts if it exists
        """
        # Use the helper to get the attribute name
        name = self.get_experts_module_name(moe_block)
        if name:
            experts_module = getattr(moe_block, name)
            return experts_module

        return None

    def get_experts_module_name(self, moe_block: nn.Module) -> Optional[str]:
        """
        Get the attribute name for experts module if it exists.

        Args:
            moe_block: The MoE block module

        Returns:
            The attribute name ('experts', 'expert_list', 'expert_modules', etc.) or None

        Example:
            name = hooks.get_experts_module_name(moe_block)
            if name:
                experts = getattr(moe_block, name)
        """
        # Try expert container attribute names from the class attribute
        for name in self.expert_block_names:
            if hasattr(moe_block, name):
                return name
        return None

    def get_shared_experts_module_name(self, moe_block: nn.Module) -> Optional[str]:
        """
        Get the attribute name for shared experts module if it exists.

        Args:
            moe_block: The MoE block module

        Returns:
            The attribute name ('shared_experts', 'shared_expert', etc.) or None

        Example:
            name = hooks.get_shared_experts_module_name(moe_block)
            if name:
                shared_experts = getattr(moe_block, name)
        """
        # Try shared expert container attribute names from the class attribute
        for name in self.shared_expert_block_names:
            if hasattr(moe_block, name):
                return name
        return None

    def get_shared_experts_module(self, moe_block: nn.Module, model_class: type) -> Optional[nn.Module]:
        """
        Extract shared experts module from MoE block by checking common attribute names.

        Args:
            moe_block: The MoE block module
            model_class: The model class (for compatibility, unused)

        Returns:
            The shared experts module, or None if not found (shared experts are optional)

        Example:
            Returns moe_block.shared_experts or moe_block.shared_expert if either exists
        """
        # Use the helper to get the attribute name
        name = self.get_shared_experts_module_name(moe_block)
        if name:
            shared_experts_module = getattr(moe_block, name)
            return shared_experts_module

        # This is normal - not all MoE models have shared experts
        return None

    def get_subset_execution_order(
            self,
            ordered_module_names: list[str],
            moe_block_prefix: Optional[str],
            experts_attr_name: Optional[str],
            shared_expert_attr_name: Optional[str],
    ) -> list[str]:
        """
        Infer shared-expert vs routed-expert replay order from explicit subset names.

        The module tree is designed to mirror forward execution order, and
        subset planning preserves that ordered module list explicitly.
        Replay should therefore follow the first occurrence of each MoE family
        in `ordered_module_names` instead of relying on dict iteration or
        model-specific hardcoded ordering.
        """
        if not ordered_module_names or moe_block_prefix is None:
            return []

        order = []
        expert_prefix = f"{moe_block_prefix}.{experts_attr_name}." if experts_attr_name else None
        shared_prefix = f"{moe_block_prefix}.{shared_expert_attr_name}." if shared_expert_attr_name else None

        for key in ordered_module_names:
            if shared_prefix and key.startswith(shared_prefix):
                if "shared" not in order:
                    order.append("shared")
                continue
            if expert_prefix and key.startswith(expert_prefix):
                if "experts" not in order:
                    order.append("experts")

        return order

    def forward_to_all_experts(
            self,
            moe_block: nn.Module,
            hidden_states: torch.Tensor,
            processor: Any,
            subset: Dict[str, Any],
            ordered_module_names: Optional[list[str]],
            original_forward: callable,
            model_class: type,
            module_looper: Any,
            moe_block_prefix: Optional[str] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward hidden states to all experts, bypassing routing.
        Subclasses should implement this with model-specific logic.

        Args:
            moe_block: The MoE block module
            hidden_states: Input tensor
            processor: The quantization processor
            subset: Dict[str, NamedModule] being calibrated
            original_forward: Original forward function
            model_class: The model class
            moe_block_prefix: Optional prefix for MoE block modules (optimized parameter)
            module_looper: ModuleLooper instance for TLS-based hooks pausing
            **kwargs: Additional arguments

        Returns:
            Output tensor from routed forward pass
        """
        log.error(
            f"forward_to_all_experts not implemented for {type(moe_block).__name__}. "
            f"Falling back to normal forward. Override this method "
            f"for model-specific implementation."
        )
        return moe_block(hidden_states, **kwargs)


class ExpertProjectionMoELifecycleHooks(MoELifecycleHooks):
    """
    Base MoE lifecycle hooks for expert architectures with 3 projections.

    This class extracts the common logic and allows subclasses to configure
    the attribute names used for the three projections.

    Expert forward pattern: output = down_proj(act_fn(gate_proj(x)) * up_proj(x))

    Subclasses should set:
    - gate_proj_name: Name of gate/w1 projection
    - up_proj_name: Name of up/w3 projection
    - down_proj_name: Name of down/w2 projection
    """

    # Subclasses should override these
    gate_proj_name: str = None
    up_proj_name: str = None
    down_proj_name: str = None

    def __init__(self, gate_proj_name: str = None, up_proj_name: str = None, down_proj_name: str = None):
        """
        Initialize with custom projection names.

        Args:
            gate_proj_name: Name of gate projection (e.g., "gate_proj" or "w1")
            up_proj_name: Name of up projection (e.g., "up_proj" or "w3")
            down_proj_name: Name of down projection (e.g., "down_proj" or "w2")
        """
        if gate_proj_name is not None:
            self.gate_proj_name = gate_proj_name
        if up_proj_name is not None:
            self.up_proj_name = up_proj_name
        if down_proj_name is not None:
            self.down_proj_name = down_proj_name

        # Validate that names are set
        if not all([self.gate_proj_name, self.up_proj_name, self.down_proj_name]):
            raise ValueError(
                f"Projection names must be set either as class attributes or constructor parameters. "
                f"Got: gate={self.gate_proj_name}, up={self.up_proj_name}, down={self.down_proj_name}"
            )
        self.input_replay = RoutedMoEInputReplayAttachment()

    def prepare_input_replay(self, subset, batch_count: int) -> None:
        self.input_replay.prepare_subset(subset, batch_count)

    def take_input_replay(self, batch_index: int) -> Optional[torch.Tensor]:
        return self.input_replay.take(batch_index)

    def apply_expert_activation(self, experts_module, expert, gate_out, up_out):
        """Apply the model's fused expert gate when it exposes one."""

        fused_gate = getattr(experts_module, "_apply_gate", None)
        if callable(fused_gate):
            return fused_gate(torch.cat([gate_out, up_out], dim=-1))
        if hasattr(expert, "act_fn"):
            return expert.act_fn(gate_out) * up_out
        return torch.nn.functional.silu(gate_out) * up_out

    def _extract_moe_block_prefix(self, subset: Dict[str, Any], moe_block: nn.Module) -> Optional[str]:
        """
        Extract moe_block_prefix from subset keys.

        Args:
            subset: Dict[str, NamedModule] being calibrated
            moe_block: The MoE block module

        Returns:
            The moe_block_prefix string, or None if not found
        """
        if not subset:
            return None

        experts_attr_name = self.get_experts_module_name(moe_block)
        shared_expert_attr_name = self.get_shared_experts_module_name(moe_block)

        for key in subset.keys():
            if experts_attr_name and f".{experts_attr_name}." in key:
                return key.split(f".{experts_attr_name}.")[0]
            if shared_expert_attr_name and f".{shared_expert_attr_name}." in key:
                return key.split(f".{shared_expert_attr_name}.")[0]

        return None

    def forward_to_all_experts(
            self,
            moe_block: nn.Module,
            hidden_states: torch.Tensor,
            processor: Any,
            subset: Dict[str, Any],
            ordered_module_names: Optional[list[str]],
            original_forward: callable,
            model_class: type,
            module_looper: Any,  # Required for TLS-based hooks pausing
            moe_block_prefix: Optional[str] = None,
            replica_module: Optional[nn.Module] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward to all experts using configurable projection names.

        Args:
            moe_block: The MoE block module
            hidden_states: Input tensor
            processor: The quantization processor (needed for hook pausing)
            subset: Dict[str, NamedModule] containing modules currently being calibrated
            original_forward: Original forward function to call for final output
            model_class: The model class (to access module_tree)
            replica_module: Optional replica of the layer module. When provided, modules
                           are looked up from the replica instead of using subset directly.
                           This is needed for multi-GPU parallel execution where the replica
                           is on a different device than the original modules in subset.
            **kwargs: Additional arguments (attention_mask, etc.)

        This implementation replays shared-expert and routed-expert paths in the
        order they appear in the subset/module tree, then calls the original
        routed forward for the final output.
        """
        if not processor or not original_forward:
            error_msg = "Missing processor or original_forward"
            log.error(error_msg)
            raise ValueError(error_msg)

        if moe_block_prefix is None:
            # moe_block_prefix is None fallback to original forward
            # this is normal for example glm4_moe has 1-3 :moe layers without experts
            return original_forward(hidden_states, **kwargs)

        root_recorder = getattr(processor, "record_moe_root_input_feature", None)
        if callable(root_recorder):
            root_recorder(moe_block_prefix, hidden_states)

        expert_count = 0
        stop_forward_raised = False
        proj_names = [self.gate_proj_name, self.up_proj_name, self.down_proj_name]
        input_only_capture = bool(getattr(processor, "moe_input_capture_without_forward", False))
        self.input_replay.retain(processor.current_batch_index(), hidden_states)

        def get_callable_module(key: str):
            """
            Get the callable module for a given subset key.

            When replica_module is provided, resolves the module from the replica
            using the key as a relative path. This ensures forward passes happen
            on the correct device for multi-GPU execution.

            Falls back to subset[key] when replica is not provided or lookup fails.
            """
            # The key is already a relative path (e.g., "mlp.experts.0.gate_proj")
            # Use it directly to look up the module in the replica
            if replica_module is not None:
                replica_submodule = _get_module_by_relative_path(replica_module, key)
                if replica_submodule is not None:
                    return replica_submodule
                else:
                    raise ValueError(f"[MoE DEBUG] replica_submodule is None for key={key}")

            # Fallback to using subset (original behavior for single-GPU)
            subset_module = subset.get(key)
            return subset_module

        def capture_input_or_forward(key: str, module_input: torch.Tensor):
            """Run an input-only quantization hook without computing an unused projection output."""

            callable_module = get_callable_module(key)
            raw_module = getattr(callable_module, "module", callable_module)
            capture_hook = getattr(raw_module, "forward_hook", None)
            if not input_only_capture or not callable(capture_hook):
                return callable_module(module_input)

            timer = getattr(getattr(module_looper, "gptq_model", None), "quant_region_timer", None)
            started_at = time.perf_counter() if timer is not None else None
            try:
                capture_hook(raw_module, (module_input,), None)
                if getattr(raw_module, "forward_hook_last", False):
                    raise StopForward()
            finally:
                if timer is not None and started_at is not None:
                    timer.record(
                        "moe_input_capture_direct",
                        time.perf_counter() - started_at,
                        source=key,
                    )
            return None

        # Get experts modules and shared expert attribute name
        experts_module = self.get_experts_module(moe_block, model_class)
        shared_experts_module = self.get_shared_experts_module(moe_block, model_class)
        shared_expert_attr_name = self.get_shared_experts_module_name(
            moe_block)  # e.g., "shared_experts" or "shared_expert"
        experts_attr_name = self.get_experts_module_name(moe_block)  # e.g., "experts", "expert_list", "expert_modules"

        # Check which shared_expert projections are in subset using detected attribute name
        has_shared_experts = False
        if shared_experts_module is not None and shared_expert_attr_name and moe_block_prefix:
            # Use the attribute name we already detected (e.g., "shared_experts" or "shared_expert")
            for name in proj_names:
                key = f"{moe_block_prefix}.{shared_expert_attr_name}.{name}"
                if key in subset:
                    has_shared_experts = True
                    break

        # Check if any expert projections are in subset
        # only part of expert projections might be loaded,
        # so we need to check all subset keys instead of just the first expert
        has_expert_projs = False
        if experts_module is not None and hasattr(experts_module, '__iter__') and len(
                experts_module) > 0 and experts_attr_name and moe_block_prefix:
            # Check all subset keys for any expert projections
            expert_prefix = f"{moe_block_prefix}.{experts_attr_name}."
            for key in subset.keys():
                if key.startswith(expert_prefix):
                    # Extract the expert index and projection name from the key
                    parts = key[len(expert_prefix):].split('.')
                    if len(parts) >= 2 and parts[1] in proj_names:
                        has_expert_projs = True
                        break

        def run_shared_experts():
            nonlocal expert_count, stop_forward_raised
            if not has_shared_experts or not shared_expert_attr_name:
                return
            try:
                shared_expert_module = getattr(moe_block, shared_expert_attr_name)
                shared_expert_device = get_device(shared_expert_module)
                shared_expert_module(move_to(hidden_states, shared_expert_device))
                expert_count += 1
            except StopForward:
                stop_forward_raised = True

        def run_routed_experts():
            nonlocal expert_count, stop_forward_raised
            if not has_expert_projs or experts_module is None or not hasattr(experts_module, '__iter__') or not experts_attr_name:
                return

            # Keep the [B, S, H] shape so the keep-mask applied by
            # pre_process_fwd_hook drops padding positions; otherwise a flattened
            # view would include padded tokens and inflate expert sample counts
            # to batch_size * seq_len instead of the number of valid tokens.
            hidden_states_for_experts = hidden_states
            shared_input_captures = {}
            expert_inputs_by_device = {}
            expert_replay_ops = {}

            parallel_down_groups = {}
            quantize_config = getattr(getattr(module_looper, "gptq_model", None), "quantize_config", None)
            parallel_down_enabled = bool(
                input_only_capture and _moe_parallel_input_capture_eligible(quantize_config)
            )
            moe_execution = getattr(getattr(quantize_config, "moe", None), "execution", None)
            capture_stream_count = int(getattr(moe_execution, "parallel_input_capture_streams", 2))
            if parallel_down_enabled:
                for expert_idx, expert in enumerate(experts_module):
                    down_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.down_proj_name}"
                    if down_key not in subset:
                        continue
                    gate_module_ref = getattr(expert, self.gate_proj_name, None)
                    expert_device = get_device(gate_module_ref) if gate_module_ref is not None else get_device(expert)
                    parallel_down_groups.setdefault(expert_device, []).append((expert, down_key))

                parallel_down_enabled = bool(
                    len(parallel_down_groups) > 1
                    and all(device.type == "cuda" for device in parallel_down_groups)
                )

            def expert_input_for(device: torch.device) -> torch.Tensor:
                """Reuse one hidden-state transfer for every expert on a device."""

                cache_key = str(device)
                cached = expert_inputs_by_device.get(cache_key)
                if cached is None:
                    cached = move_to(hidden_states_for_experts, device)
                    expert_inputs_by_device[cache_key] = cached
                return cached

            def capture_down_input(expert: nn.Module, down_key: str, expert_input: torch.Tensor) -> bool:
                """Build one expert intermediate and capture it for its down projection."""

                try:
                    replay_op = expert_replay_ops.get(down_key)
                    if replay_op is None and down_key not in expert_replay_ops:
                        named_module = subset[down_key]
                        flags = getattr(named_module, "state", {}).get("module_tree_flags", frozenset())
                        declarations = sorted(
                            (kind, flag.split("=", 1)[1])
                            for flag in flags
                            for kind in ("expert_activation", "expert_forward", "expert_gate")
                            if flag.startswith(f"{kind}=") and flag.split("=", 1)[1]
                        )
                        if len(declarations) > 1:
                            raise RuntimeError(
                                f"MoE module_tree declares multiple expert replay methods for {down_key}: "
                                f"{declarations}"
                            )
                        if declarations:
                            kind, method_path = declarations[0]
                            owner_name, separator, relative_path = method_path.partition(".")
                            owners = {"expert": expert, "experts": experts_module}
                            owner = owners.get(owner_name)
                            if owner is None or not separator or not relative_path:
                                raise RuntimeError(
                                    f"MoE module_tree declaration {kind}={method_path} for {down_key} must start "
                                    "with the explicit owner `expert.` or `experts.`"
                                )
                            replay_fn = owner
                            for attribute in relative_path.split("."):
                                replay_fn = getattr(replay_fn, attribute, None)
                                if replay_fn is None:
                                    break
                            if not callable(replay_fn):
                                raise RuntimeError(
                                    f"MoE module_tree declares {kind}={method_path} for {down_key}, but it does "
                                    "not resolve to a callable"
                                )
                            replay_op = (kind, replay_fn)
                        expert_replay_ops[down_key] = replay_op

                    if replay_op is None:
                        # Without an exact module-tree declaration, execute the
                        # model's expert forward and let the down-projection hook
                        # capture its real input. This can compute an unused down
                        # GEMM, but never guesses activation or gating semantics.
                        try:
                            expert(expert_input)
                        except NotImplementedError as exc:
                            raise RuntimeError(
                                f"Cannot reconstruct the down-projection input for {down_key}: the expert has no "
                                "forward and its module_tree node does not declare an exact expert replay method"
                            ) from exc
                        return False

                    replay_kind, replay_fn = replay_op
                    if replay_kind == "expert_forward":
                        # Some experts do not implement the standard gated MLP
                        # equation for every configuration. Their module tree
                        # explicitly selects exact forward replay instead.
                        replay_fn(expert_input)
                        return False

                    gate_module = getattr(expert, self.gate_proj_name)
                    up_module = getattr(expert, self.up_proj_name)
                    gate_out = gate_module(expert_input)
                    up_out = up_module(expert_input)
                    if replay_kind == "expert_activation":
                        intermediate = replay_fn(gate_out) * up_out
                    else:
                        intermediate = replay_fn(torch.cat([gate_out, up_out], dim=-1))
                    del gate_out, up_out

                    capture_input_or_forward(down_key, intermediate)
                    del intermediate
                except StopForward:
                    return True
                return False

            for expert_idx, expert in enumerate(experts_module):
                gate_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.gate_proj_name}"
                up_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.up_proj_name}"
                down_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.down_proj_name}"

                if gate_key not in subset and up_key not in subset and down_key not in subset:
                    continue

                if down_key in subset and parallel_down_enabled:
                    continue

                input_only_gate_up = input_only_capture and down_key not in subset
                if input_only_gate_up:
                    # Direct hooks only consume the input tensor; an unloaded
                    # projection shell may still live on meta and must not
                    # determine the activation device.
                    expert_input = hidden_states_for_experts
                else:
                    gate_module_ref = getattr(expert, self.gate_proj_name, None)
                    expert_device = get_device(gate_module_ref) if gate_module_ref is not None else get_device(expert)
                    expert_input = expert_input_for(expert_device)

                try:
                    if down_key in subset:
                        stop_forward_raised |= capture_down_input(expert, down_key, expert_input)
                        expert_count += 1
                    else:
                        called_any = False
                        if gate_key in subset:
                            group_key = processor.moe_shared_input_group_key(gate_key) if input_only_capture else None
                            group_capture = shared_input_captures.get(group_key) if group_key is not None else None
                            if group_capture is None:
                                group_capture = {
                                    "source_name": gate_key,
                                    "follower_names": [],
                                }
                                if group_key is not None:
                                    shared_input_captures[group_key] = group_capture
                                capture_input_or_forward(gate_key, expert_input)
                            else:
                                group_capture["follower_names"].append(gate_key)
                                callable_module = get_callable_module(gate_key)
                                raw_module = getattr(callable_module, "module", callable_module)
                                stop_forward_raised |= bool(getattr(raw_module, "forward_hook_last", False))
                            called_any = True
                        if up_key in subset:
                            group_key = processor.moe_shared_input_group_key(up_key) if input_only_capture else None
                            group_capture = shared_input_captures.get(group_key) if group_key is not None else None
                            if group_capture is None:
                                group_capture = {
                                    "source_name": up_key,
                                    "follower_names": [],
                                }
                                if group_key is not None:
                                    shared_input_captures[group_key] = group_capture
                                capture_input_or_forward(up_key, expert_input)
                            else:
                                group_capture["follower_names"].append(up_key)
                                callable_module = get_callable_module(up_key)
                                raw_module = getattr(callable_module, "module", callable_module)
                                stop_forward_raised |= bool(getattr(raw_module, "forward_hook_last", False))
                            called_any = True
                        if called_any:
                            expert_count += 1

                except StopForward:
                    stop_forward_raised = True

            if parallel_down_enabled:
                parent_mask = module_looper._get_processor_mask(processor)
                parent_batch_index = processor.current_batch_index()

                # Materialize one immutable activation copy per device before
                # worker launch. Each worker then preserves expert order on its
                # own CUDA stream while independent devices run concurrently.
                for device in parallel_down_groups:
                    expert_input_for(device)

                def capture_device_group(device: torch.device, entries) -> tuple[int, bool]:
                    from ..quantization.gptq import defer_hessian_sync

                    module_looper._set_processor_mask(processor, parent_mask)
                    processor._set_current_batch_index(parent_batch_index)
                    local_stop = False
                    group_started_at = time.perf_counter()
                    try:
                        expert_input = expert_inputs_by_device[str(device)]
                        with defer_hessian_sync():
                            results = _ROUTED_CAPTURE_STREAMS.run(
                                device=device,
                                entries=entries,
                                stream_count=capture_stream_count,
                                launch=lambda entry: capture_down_input(entry[0], entry[1], expert_input),
                            )
                            local_stop = any(results)
                    finally:
                        # Each ThreadX CUDA worker owns a stream. Complete all
                        # ordered Hessian updates once per device group before
                        # later quant workers may consume them on other streams.
                        torch_sync(device=device)
                        timer = getattr(getattr(module_looper, "gptq_model", None), "quant_region_timer", None)
                        if timer is not None:
                            timer.record(
                                "moe_capture_device_group",
                                time.perf_counter() - group_started_at,
                                source=f"{device};streams={capture_stream_count}",
                            )
                        module_looper._set_processor_mask(processor, None)
                        processor._set_current_batch_index(None)
                    return len(entries), local_stop

                # Lifecycle work must use the process-wide threadx pool so it
                # inherits stable device contexts, worker warmups, accounting,
                # and GC coordination. The enclosing bypass forward is serial
                # whenever expert placement spans devices, so these submissions
                # cannot wait on their own outer device worker.
                from .. import DEVICE_THREAD_POOL

                futures = [
                    DEVICE_THREAD_POOL.submit(device, capture_device_group, device, entries)
                    for device, entries in parallel_down_groups.items()
                ]
                for future in futures:
                    captured_count, local_stop = future.result()
                    expert_count += captured_count
                    stop_forward_raised |= local_stop

            for group_capture in shared_input_captures.values():
                processor.record_moe_shared_input_followers(**group_capture)

            expert_inputs_by_device.clear()

        execution_order = self.get_subset_execution_order(
            ordered_module_names=ordered_module_names or [],
            moe_block_prefix=moe_block_prefix,
            experts_attr_name=experts_attr_name,
            shared_expert_attr_name=shared_expert_attr_name,
        )
        if has_shared_experts and "shared" not in execution_order:
            execution_order.append("shared")
        if has_expert_projs and "experts" not in execution_order:
            execution_order.append("experts")

        for group_name in execution_order:
            if group_name == "shared":
                run_shared_experts()
            elif group_name == "experts":
                run_routed_experts()

        if stop_forward_raised:
            # Re-raise StopForward if it was caught
            raise StopForward()

        # After forcing all experts to see the data for calibration,
        # call the original forward to get proper output.
        # Only pause hooks if we actually forwarded through experts (expert_count > 0).
        # For layers without experts (e.g., Layer 0 in GLM-4), hooks must remain active
        # to collect calibration data from the standard MLP.
        if expert_count > 0:
            # Pause hooks to avoid double-counting (already collected from expert calls)
            # Use TLS-based pausing for thread safety in parallel execution (GIL-free safe)
            module_looper._set_processor_hooks_paused(processor, True)
            try:
                result = original_forward(hidden_states, **kwargs)
            finally:
                module_looper._set_processor_hooks_paused(processor, False)
        else:
            # No experts forwarded, let hooks fire normally for standard MLP
            result = original_forward(hidden_states, **kwargs)

        return result


class GateUpDownMoELifecycleHooks(ExpertProjectionMoELifecycleHooks):
    """
    MoE lifecycle hooks for models using gate_proj/up_proj/down_proj naming.

    Used by: GLM-4, Qwen2-MoE, Mixtral, Phi-3 MoE, most Llama-based MoE models
    """
    gate_proj_name = "gate_proj"
    up_proj_name = "up_proj"
    down_proj_name = "down_proj"


class W1W3W2MoELifecycleHooks(ExpertProjectionMoELifecycleHooks):
    """
    MoE lifecycle hooks for models using w1/w3/w2 naming.

    Used by: Some architectures that explicitly name layers w1, w3, w2 (e.g., MiniMax-M2)
    """
    gate_proj_name = "w1"
    up_proj_name = "w3"
    down_proj_name = "w2"
