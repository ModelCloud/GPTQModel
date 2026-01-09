# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""
MoE (Mixture of Experts) lifecycle hooks system.

This module provides a base class for model-specific MoE lifecycle hooks that allow
customization of MoE forward passes and routing logic during quantization.
"""

from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from ..nn_modules.hooked_linear import StopForward
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.model import move_to

log = setup_logger()


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

        if moe_module_name is None:
            log.error(f"No :moe flag found in module_tree for {model_class.__name__}")
            return None

        # Get the module by name
        moe_block = getattr(layer_module, moe_module_name, None)

        return moe_block

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

    def forward_to_all_experts(
            self,
            moe_block: nn.Module,
            hidden_states: torch.Tensor,
            processor: Any,
            subset: Dict[str, Any],
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

        This implementation:
        1. Forwards to shared_experts (if present and in subset)
        2. Forwards to all individual experts (only those with modules in subset)
        3. Returns normal routed forward output (not averaged)
        """
        import torch.nn.functional as F

        if not processor or not original_forward:
            error_msg = "Missing processor or original_forward"
            log.error(error_msg)
            raise ValueError(error_msg)

        if moe_block_prefix is None:
            # moe_block_prefix is None fallback to original forward
            # this is normal for example glm4_moe has 1-3 :moe layers without experts
            return original_forward(hidden_states, **kwargs)

        expert_count = 0
        stop_forward_raised = False
        proj_names = [self.gate_proj_name, self.up_proj_name, self.down_proj_name]

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

        # Forward to shared experts if they exist AND if any of their modules are in subset
        # Simply call the shared expert module's forward - hooks will fire for projections in subset
        if has_shared_experts and shared_expert_attr_name:
            try:
                # Get the shared expert module and call its forward
                shared_expert_module = getattr(moe_block, shared_expert_attr_name)
                # Ensure input is on correct device
                shared_expert_device = get_device(shared_expert_module)
                shared_expert_module(move_to(hidden_states, shared_expert_device))
                expert_count += 1
            except StopForward:
                stop_forward_raised = True

        # Forward to all individual experts
        if has_expert_projs and experts_module is not None and hasattr(experts_module,
                                                                       '__iter__') and experts_attr_name:
            # Reshape hidden_states from [B, S, H] to [B*S, H] for expert modules
            if hidden_states.dim() == 3:
                hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            else:
                hidden_states_2d = hidden_states

            for expert_idx, expert in enumerate(experts_module):
                # Construct keys for this expert's projections using the detected attribute name
                gate_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.gate_proj_name}"
                up_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.up_proj_name}"
                down_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.down_proj_name}"

                # Skip if none of this expert's projections are in subset
                if gate_key not in subset and up_key not in subset and down_key not in subset:
                    continue

                # Determine device for this expert
                # Use gate_proj as reference since it's typically present
                expert_device = None
                gate_module_ref = getattr(expert, self.gate_proj_name, None)
                if gate_module_ref is not None:
                    expert_device = get_device(gate_module_ref)
                else:
                    expert_device = get_device(expert)

                # Move input to expert device
                expert_input = move_to(hidden_states_2d, expert_device)

                try:
                    # Strategy: If down_proj is in subset, compute intermediate on-the-fly
                    # Note: When down is in subset, gate/up are NOT in subset (separate subset groups)
                    # We need to compute gate/up outputs to create the intermediate for down
                    if down_key in subset:
                        # Get gate/up modules directly from expert via getattr
                        # This returns the UNWRAPPED modules (not NamedModule wrappers)
                        # Hooks are only registered on NamedModule wrappers in subset,
                        # so calling unwrapped modules means NO hooks fire automatically.
                        gate_module = getattr(expert, self.gate_proj_name)
                        up_module = getattr(expert, self.up_proj_name)

                        # Compute intermediate (no hooks fire since modules are unwrapped)
                        # Ensure modules are called with input on correct device
                        # gate_module/up_module are submodules of expert, so they are on expert_device
                        gate_out = gate_module(expert_input)
                        up_out = up_module(expert_input)

                        # Compute intermediate using expert's activation function
                        if hasattr(expert, 'act_fn'):
                            intermediate = expert.act_fn(gate_out) * up_out
                        else:
                            intermediate = F.silu(gate_out) * up_out
                        del gate_out, up_out

                        # Call down_proj via wrapper (or replica module) with hooks enabled for activation collection
                        # Module returned by get_callable_module IS on expert_device (if replica is correct)
                        # Intermediate is on expert_device
                        get_callable_module(down_key)(intermediate)
                        del intermediate
                        expert_count += 1
                    else:
                        # For gate_proj/up_proj in subset, just call them directly via wrappers (or replica modules)
                        called_any = False
                        if gate_key in subset:
                            get_callable_module(gate_key)(expert_input)
                            called_any = True
                        if up_key in subset:
                            get_callable_module(up_key)(expert_input)
                            called_any = True
                        if called_any:
                            expert_count += 1

                except StopForward:
                    stop_forward_raised = True
                finally:
                    # Promptly release tensor copy to free VRAM
                    del expert_input

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