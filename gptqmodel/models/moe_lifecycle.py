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
        # Use the helper to get the attribute name
        name = self.get_experts_module_name(moe_block)
        if name:
            experts_module = getattr(moe_block, name)
            log.info(f"[MOEDEBUG] Found experts module: {type(moe_block).__name__}.{name}")
            return experts_module
        
        log.info(f"[MOEDEBUG] No experts module found in MoE block {type(moe_block).__name__}")
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
        # Try common expert container attribute names
        for name in ['experts', 'expert_list', 'expert_modules']:
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
        # Try both singular and plural naming
        for name in ['shared_experts', 'shared_expert']:
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
            f"[MOEDEBUG] forward_to_all_experts not implemented for {type(moe_block).__name__}. "
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
            **kwargs: Additional arguments (attention_mask, etc.)
        
        This implementation:
        1. Forwards to shared_experts (if present and in subset)
        2. Forwards to all individual experts (only those with modules in subset)
        3. Returns normal routed forward output (not averaged)
        """
        import torch.nn.functional as F
        
        if not processor or not original_forward:
            error_msg = "[MOEDEBUG] Missing processor or original_forward - this is an invalid operation that should never happen"
            log.error(error_msg)
            raise ValueError(error_msg)
        
        if moe_block_prefix is None:
            log.info(f"[MOEDEBUG] moe_block_prefix is None fallback to original forward")
            return original_forward(hidden_states, **kwargs)

        expert_count = 0
        stop_forward_raised = False
        proj_names = [self.gate_proj_name, self.up_proj_name, self.down_proj_name]
        
        # Get experts modules and shared expert attribute name
        experts_module = self.get_experts_module(moe_block, model_class)
        shared_experts_module = self.get_shared_experts_module(moe_block, model_class)
        shared_expert_attr_name = self.get_shared_experts_module_name(moe_block)  # e.g., "shared_experts" or "shared_expert"
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
        # With VRAMStrategy.BALANCED, only part of expert projections might be loaded,
        # so we need to check all subset keys instead of just the first expert
        has_expert_projs = False
        if experts_module is not None and hasattr(experts_module, '__iter__') and len(experts_module) > 0 and experts_attr_name and moe_block_prefix:
            # Check all subset keys for any expert projections
            expert_prefix = f"{moe_block_prefix}.{experts_attr_name}."
            for key in subset.keys():
                if key.startswith(expert_prefix):
                    # Extract the expert index and projection name from the key
                    parts = key[len(expert_prefix):].split('.')
                    if len(parts) >= 2 and parts[1] in proj_names:
                        has_expert_projs = True
                        break
        
        log.info(f"[MOEDEBUG] has_shared_experts: {has_shared_experts}, has_expert_projs: {has_expert_projs}")
        
        # Forward to shared experts if they exist AND if any of their modules are in subset
        # Simply call the shared expert module's forward - hooks will fire for projections in subset
        if has_shared_experts and shared_expert_attr_name:
            log.info(f"[MOEDEBUG] Forwarding to {shared_expert_attr_name}")
            try:
                # Get the shared expert module and call its forward
                shared_expert_module = getattr(moe_block, shared_expert_attr_name)
                result = shared_expert_module(hidden_states)
                expert_count += 1
                log.info(f"[MOEDEBUG] {shared_expert_attr_name}: Forward complete")
            except Exception as e:
                if "StopForward" in str(type(e).__name__):
                    stop_forward_raised = True
                    log.info(f"[MOEDEBUG] {shared_expert_attr_name}: StopForward raised")
                else:
                    log.info(f"[MOEDEBUG] {shared_expert_attr_name}: Error: {e}")
        
        # Forward to all individual experts
        if has_expert_projs and experts_module is not None and hasattr(experts_module, '__iter__') and experts_attr_name:
            # Reshape hidden_states from [B, S, H] to [B*S, H] for expert modules
            if hidden_states.dim() == 3:
                hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            else:
                hidden_states_2d = hidden_states
            
            log.info(f"[MOEDEBUG] Starting to forward to {len(experts_module)} experts")
            for expert_idx, expert in enumerate(experts_module):
                # Construct keys for this expert's projections using the detected attribute name
                gate_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.gate_proj_name}"
                up_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.up_proj_name}"
                down_key = f"{moe_block_prefix}.{experts_attr_name}.{expert_idx}.{self.down_proj_name}"
                
                # Skip if none of this expert's projections are in subset
                if gate_key not in subset and up_key not in subset and down_key not in subset:
                    continue
                
                log.info(f"[MOEDEBUG] Processing expert {expert_idx}")
                try:
                    # Strategy: If down_proj is in subset, compute intermediate on-the-fly
                    # Note: When down is in subset, gate/up are NOT in subset (separate subset groups)
                    # We need to compute gate/up outputs to create the intermediate for down
                    if down_key in subset:
                        log.info(f"[MOEDEBUG] Expert {expert_idx}: Has down_proj in subset")
                        
                        # Get gate/up modules directly from expert via getattr
                        # This returns the UNWRAPPED modules (not NamedModule wrappers)
                        # Hooks are only registered on NamedModule wrappers in subset,
                        # so calling unwrapped modules means NO hooks fire automatically.
                        gate_module = getattr(expert, self.gate_proj_name)
                        up_module = getattr(expert, self.up_proj_name)
                        
                        # Compute intermediate (no hooks fire since modules are unwrapped)
                        gate_out = gate_module(hidden_states_2d)
                        up_out = up_module(hidden_states_2d)
                        
                        # Compute intermediate using expert's activation function
                        if hasattr(expert, 'act_fn'):
                            intermediate = expert.act_fn(gate_out) * up_out
                        else:
                            intermediate = F.silu(gate_out) * up_out
                        
                        # Call down_proj via wrapper with hooks enabled for activation collection
                        down_result = subset[down_key](intermediate)
                        expert_count += 1
                        log.info(f"[MOEDEBUG] Expert {expert_idx}: Forwarded (intermediate + down_proj)")
                    else:
                        # For gate_proj/up_proj in subset, just call them directly via wrappers
                        log.info(f"[MOEDEBUG] Expert {expert_idx}: Using direct projection calls")
                        called_any = False
                        if gate_key in subset:
                            gate_result = subset[gate_key](hidden_states_2d)
                            called_any = True
                        if up_key in subset:
                            up_result = subset[up_key](hidden_states_2d)
                            called_any = True
                        if called_any:
                            expert_count += 1
                            log.info(f"[MOEDEBUG] Expert {expert_idx}: Forwarded (direct)")
                
                except Exception as e:
                    if "StopForward" in str(type(e).__name__):
                        stop_forward_raised = True
                        log.info(f"[MOEDEBUG] Expert {expert_idx}: StopForward raised")
                    else:
                        log.info(f"[MOEDEBUG] Expert {expert_idx}: Error: {e}")
        
        log.info(f"[MOEDEBUG] Forwarded to {expert_count} experts for subset calibration")
        
        if stop_forward_raised:
            # Re-raise StopForward if it was caught
            from ..nn_modules.hooked_linear import StopForward  # Local import to avoid circular dependency
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
