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

from ..utils.logger import setup_logger

log = setup_logger()


class MoELifecycleHooks:
    """
    Base class for model-specific MoE lifecycle hooks.
    
    This class provides customization points for MoE-specific operations during quantization.
    Subclasses should override these methods to provide model-specific implementations.
    
    Main use case: Forward whole calibration dataset to all experts instead of only routed ones.
    """
    
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
            log.info(f"[MOEDEBUG] No :moe flag found in module_tree for {model_class.__name__}")
            return None
        
        # Get the module by name
        moe_block = getattr(layer_module, moe_module_name, None)
        
        if moe_block is None:
            log.info(
                f"[MOEDEBUG] MoE module '{moe_module_name}' not found in layer {type(layer_module).__name__}. "
                f"This is normal for layers without MoE (layer_modules_strict=False)"
            )
        
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
        # Try common expert container attribute names
        for name in ['experts', 'expert_list', 'expert_modules']:
            experts_module = getattr(moe_block, name, None)
            if experts_module is not None:
                log.info(f"[MOEDEBUG] Found experts module: {type(moe_block).__name__}.{name}")
                return experts_module
        
        log.info(f"[MOEDEBUG] No experts module found in MoE block {type(moe_block).__name__}")
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
        # Try both singular and plural naming
        for name in ['shared_experts', 'shared_expert']:
            shared_experts_module = getattr(moe_block, name, None)
            if shared_experts_module is not None:
                log.info(f"[MOEDEBUG] Found shared experts module: {type(moe_block).__name__}.{name}")
                return shared_experts_module
        
        # This is normal - not all MoE models have shared experts
        return None
    
    def forward_to_all_experts(
        self,
        moe_block: nn.Module,
        hidden_states: torch.Tensor,
        processor: Any,
        subset: Dict[str, Any],
        subset_modules: Set[Any],  # Precomputed set(subset.values()) for performance
        original_forward: callable,
        model_class: type,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward hidden states to all experts, bypassing routing.
        
        This is the core method for the "forward whole dataset to each expert" feature.
        It should forward the input to all experts for activation collection, then
        return the output from the original routed forward pass.
        
        Args:
            moe_block: The MoE block module
            hidden_states: Input tensor
            processor: The quantization processor (needed for hook pausing)
            subset: Dict[str, NamedModule] containing modules currently being calibrated
            original_forward: Original forward function to call for final output
            model_class: The model class (to access module_tree)
            **kwargs: Additional arguments (attention_mask, etc.)
        
        Returns:
            Output tensor from original_forward (with proper routing)
        
        Note:
            This method should be overridden by model-specific implementations.
            The default implementation logs an error and falls back to normal forward.
        """
        log.error(
            f"[MOEDEBUG] forward_to_all_experts not implemented for {type(moe_block).__name__}. "
            f"Falling back to normal forward pass. Override MoELifecycleHooks.forward_to_all_experts() "
            f"for model-specific implementation."
        )
        
        # Fallback to normal forward
        return moe_block(hidden_states, **kwargs)
    
    def count_expert_activations(
        self,
        moe_block: nn.Module,
        expert_counts: Optional[Dict[int, int]] = None
    ) -> Dict[int, int]:
        """
        Count how many times each expert was activated.
        
        Args:
            moe_block: The MoE block module
            expert_counts: Existing counts dictionary to update (optional)
        
        Returns:
            Dictionary mapping expert index to activation count
        
        Note:
            This is used for debugging/logging purposes to verify that
            all experts receive calibration data.
        """
        if expert_counts is None:
            expert_counts = {}
        
        # Default implementation: no-op
        # Model-specific implementations can track expert usage
        return expert_counts


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
    
    
    def forward_to_all_experts(
        self,
        moe_block: nn.Module,
        hidden_states: torch.Tensor,
        processor: Any,
        subset: Dict[str, Any],
        subset_modules: Set[Any],  # Precomputed set(subset.values()) for performance
        original_forward: callable,
        model_class: type,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward to all experts using configurable projection names.
        
        Args:
            moe_block: The MoE block module
            hidden_states: Input tensor
            processor: The quantization processor (needed for hook pausing)
            subset: Dict[str, NamedModule] containing modules currently being calibrated
            subset_modules: Precomputed set(subset.values()) for O(1) lookup (avoids recomputing per batch)
            original_forward: Original forward function to call for final output
            model_class: The model class (to access module_tree)
            **kwargs: Additional arguments (attention_mask, etc.)
        
        This implementation:
        1. Forwards to shared_experts (if present and in subset)
        2. Forwards to all individual experts (only those with modules in subset)
        3. Returns normal routed forward output (not averaged)
        """
        import torch.nn.functional as F
        
        if not processor or not original_forward:
            log.warning("[MOEDEBUG] Missing processor or original_forward, falling back to normal forward")
            return moe_block(hidden_states, **kwargs)
        
        # Get experts modules using flag-based detection
        experts_module = self.get_experts_module(moe_block, model_class)
        shared_experts_module = self.get_shared_experts_module(moe_block, model_class)
        
        expert_count = 0
        stop_forward_raised = False
        proj_names = [self.gate_proj_name, self.up_proj_name, self.down_proj_name]
        
        # Check if shared_experts modules are in subset by checking dictionary keys
        shared_experts_in_subset = False
        if shared_experts_module is not None:
            for name in proj_names:
                if hasattr(shared_experts_module, name):
                    module = getattr(shared_experts_module, name)
                    if module in subset_modules:
                        shared_experts_in_subset = True
                        break
        
        # Pre-check which projection modules are in subset (all experts have same structure)
        expert_proj_names_in_subset = set()  # Projection names like 'gate_proj', 'down_proj'
        if experts_module is not None and hasattr(experts_module, '__iter__') and len(experts_module) > 0:
            first_expert = experts_module[0]
            for name in proj_names:
                if hasattr(first_expert, name):
                    module = getattr(first_expert, name)
                    if module in subset_modules:
                        expert_proj_names_in_subset.add(name)
        
        # Optimization: If ONLY shared_experts in subset (no individual expert modules),
        # skip manual forwarding and let original_forward handle it with hooks enabled.
        # This avoids computing shared_experts twice.
        if shared_experts_in_subset and not expert_proj_names_in_subset:
            log.info(f"[MOEDEBUG] Only shared_experts in subset, using original_forward with hooks enabled")
            return original_forward(hidden_states, **kwargs)
        
        # Forward to shared experts if they exist AND if any of their modules are in subset
        # (This path is only reached if individual expert modules are also in subset)
        if shared_experts_module is not None and shared_experts_in_subset:
            try:
                if hasattr(shared_experts_module, self.gate_proj_name) and hasattr(shared_experts_module, self.up_proj_name):
                    # Pause hooks to compute intermediate
                    with processor._hook_state_lock:
                        processor.hooks_paused = True
                    try:
                        gate_out = getattr(shared_experts_module, self.gate_proj_name)(hidden_states)
                        up_out = getattr(shared_experts_module, self.up_proj_name)(hidden_states)
                        intermediate = F.silu(gate_out) * up_out
                    finally:
                        with processor._hook_state_lock:
                            processor.hooks_paused = False
                    
                    # Call down_proj with hooks enabled for activation collection
                    getattr(shared_experts_module, self.down_proj_name)(intermediate)
                    expert_count += 1
                    log.info(f"[MOEDEBUG] Forwarded to shared_experts")
            except Exception as e:
                if "StopForward" in str(type(e).__name__):
                    stop_forward_raised = True
                else:
                    log.info(f"[MOEDEBUG] Error forwarding to shared_experts: {e}")
        
        
        # Forward to all individual experts
        if expert_proj_names_in_subset and experts_module is not None and hasattr(experts_module, '__iter__'):
            # Reshape hidden_states from [B, S, H] to [B*S, H] for expert modules
            if hidden_states.dim() == 3:
                hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            else:
                hidden_states_2d = hidden_states
            
            for idx, expert in enumerate(experts_module):
                try:
                    # Strategy: If down_proj is in subset, compute intermediate on-the-fly
                    if self.down_proj_name in expert_proj_names_in_subset and hasattr(expert, self.down_proj_name):
                        intermediate = None
                        with processor._hook_state_lock:
                            processor.hooks_paused = True
                        try:
                            gate_out = None
                            up_out = None
                            
                            if hasattr(expert, self.gate_proj_name):
                                gate_out = getattr(expert, self.gate_proj_name)(hidden_states_2d)
                            if hasattr(expert, self.up_proj_name):
                                up_out = getattr(expert, self.up_proj_name)(hidden_states_2d)
                            
                            if gate_out is not None and up_out is not None:
                                # Compute intermediate
                                if hasattr(expert, 'act_fn'):
                                    intermediate = expert.act_fn(gate_out) * up_out
                                else:
                                    intermediate = F.silu(gate_out) * up_out
                        finally:
                            with processor._hook_state_lock:
                                processor.hooks_paused = False
                        
                        # Call down_proj with hooks enabled
                        if intermediate is not None:
                            getattr(expert, self.down_proj_name)(intermediate)
                            expert_count += 1
                    else:
                        # For gate_proj/up_proj, just call them directly (hooks enabled)
                        if self.gate_proj_name in expert_proj_names_in_subset and hasattr(expert, self.gate_proj_name):
                            getattr(expert, self.gate_proj_name)(hidden_states_2d)
                        if self.up_proj_name in expert_proj_names_in_subset and hasattr(expert, self.up_proj_name):
                            getattr(expert, self.up_proj_name)(hidden_states_2d)
                        expert_count += 1
                
                except Exception as e:
                    if "StopForward" in str(type(e).__name__):
                        stop_forward_raised = True
                    else:
                        log.info(f"[MOEDEBUG] Error forwarding to expert {idx}: {e}")
        
        log.info(f"[MOEDEBUG] Forwarded to {expert_count} experts for subset calibration")
        
        if stop_forward_raised:
            # Re-raise StopForward if it was caught
            from ..looper.loop_processor import StopForward  # Local import to avoid circular dependency
            raise StopForward()
        
        # After forcing all experts to see the data for calibration,
        # call the original forward to get proper output.
        # Only pause hooks if we actually forwarded through experts (expert_count > 0).
        # For layers without experts (e.g., Layer 0 in GLM-4), hooks must remain active
        # to collect calibration data from the standard MLP.
        if expert_count > 0:
            # Pause hooks to avoid double-counting (already collected from expert calls)
            with processor._hook_state_lock:
                processor.hooks_paused = True
            try:
                result = original_forward(hidden_states, **kwargs)
            finally:
                with processor._hook_state_lock:
                    processor.hooks_paused = False
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
