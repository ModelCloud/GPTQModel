"""Bounded, one-tensor-at-a-time V4.1 source access.

Decoded block loading is for numerical diagnostics and trellis source weights.
Supplying native_kernels retains packed source linears and their activation
quantization. The mtp namespace loads only dSpark core blocks, not its input
projection, shared embedding or auxiliary output heads.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .model_dequant import dequantize_deepseek_v4_fp4_expert


def source_layer_key(layer, runtime_key, namespace="layers"):
    if namespace not in ("layers", "mtp") or type(layer) is not int or layer < 0:
        raise ValueError("invalid V4.1 source namespace/layer")
    key = runtime_key
    for family, native in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        if key.startswith(family + "."):
            return f"{namespace}.{layer}.{native}_{key.split('.', 1)[1]}"
    replacements = (
        ("input_layernorm.", "attn_norm."), ("post_attention_layernorm.", "ffn_norm."),
        ("self_attn.", "attn."), ("mlp.", "ffn."),
        ("attn.sinks", "attn.attn_sink"),
        (".q_a_proj.", ".wq_a."), (".q_a_norm.", ".q_norm."),
        (".q_b_proj.", ".wq_b."), (".kv_proj.", ".wkv."),
        (".o_a_proj.", ".wo_a."), (".o_b_proj.", ".wo_b."),
        (".compressor.gate_proj.", ".compressor.wgate."),
        (".compressor.kv_norm.", ".compressor.norm."),
        (".indexer.k_proj.", ".indexer.wk."),
        (".gate.e_score_correction_bias_vl", ".gate.bias_vl"),
        (".gate.e_score_correction_bias", ".gate.bias"),
        (".gate_proj.", ".w1."), (".up_proj.", ".w3."), (".down_proj.", ".w2."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return f"{namespace}.{layer}.{key}"


class V41Source:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        if self.config.get("model_type") != "deepseek_v41":
            raise ValueError("expected DeepSeek V4.1 source config")
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self._projection_buffers = {}
        for key in self.weight_map:
            if ".ffn.experts." in key:
                prefix, _, suffix = key.rpartition(".")
                self._projection_buffers.setdefault(prefix, set()).add(suffix)
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not (self.snapshot / shard).is_file():
                raise ValueError(f"invalid or missing V4.1 shard: {shard}")

    def tensor(self, name, device="cpu"):
        if ".engram.embed." in name:
            raise ValueError("PLE payloads must use MappedPLETable, never full-tensor loading")
        with safe_open(self.snapshot / self.weight_map[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).to(device=device, copy=True)

    def decoded(self, name, device="cpu", dtype=torch.bfloat16):
        if name.endswith(".weight") and name.removesuffix(".weight") + ".trellis" in self.weight_map:
            from ..exllamav3.modules.quant.exl3_lib.quantize import reconstruct_exl3_tensors

            packed = self.packed_projection(name.removesuffix(".weight"), device)
            # Serialized EXL3 reconstruction is [in, out], whereas source
            # linear parameters use [out, in]. This is dense reference replay,
            # not a claim that the fused serving kernel has been qualified.
            value = reconstruct_exl3_tensors(packed, device=device, dtype=dtype).T.contiguous()
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite reloaded EXL3 weight")
            return value
        value = self.tensor(name, device)
        scale_name = name.removesuffix(".weight") + ".scale"
        if not name.endswith(".weight") or scale_name not in self.weight_map:
            return value
        scale = self.tensor(scale_name, device)
        if value.dtype == torch.int8:
            if not re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name):
                raise ValueError(f"unexpected packed INT8 source: {name}")
            return dequantize_deepseek_v4_fp4_expert(value, scale, target_dtype=dtype)
        if value.dtype != torch.float8_e4m3fn or value.ndim != 2:
            raise ValueError(f"unsupported scaled source tensor: {name}")
        rows, cols = value.shape
        if tuple(scale.shape) != ((rows + 31) // 32, (cols + 31) // 32):
            raise ValueError(f"invalid block-32 FP8 scales: {name}")
        result = torch.empty(value.shape, dtype=dtype, device=device)
        # Keep FP32 work bounded to 32 rows instead of expanding the whole matrix.
        for start in range(0, rows, 32):
            expanded = scale[start // 32].float().repeat_interleave(32)[:cols]
            result[start:start + 32] = value[start:start + 32].float() * expanded
        return result

    def packed_projection(self, name, device="cpu"):
        """Read only a routed EXL3 projection from standard indexed shards."""
        match = re.fullmatch(r"(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])", name)
        if match is None:
            raise ValueError("EXL3 reload is restricted to routed V4.1 projections")
        namespace, layer, expert, projection = match.groups()
        layers, experts = (40, 384) if namespace == "layers" else (3, 128)
        if int(layer) >= layers or int(expert) >= experts:
            raise ValueError("EXL3 projection outside V4.1 architecture")
        suffixes = self._projection_buffers.get(name, set())
        if suffixes != {"trellis", "suh", "svh", "mcg"}:
            raise ValueError("EXL3 projection must contain exactly four packed buffers")
        packed = {suffix: self.tensor(name + "." + suffix, device) for suffix in sorted(suffixes)}
        inputs, outputs = (2304, 5120) if projection == "w2" else (5120, 2304)
        trellis, suh, svh, mcg = (packed[key] for key in ("trellis", "suh", "svh", "mcg"))
        if (trellis.dtype != torch.int16 or trellis.ndim != 3
                or tuple(trellis.shape[:2]) != (inputs // 16, outputs // 16)
                or trellis.shape[-1] not in (48, 64)
                or suh.dtype != torch.float16 or tuple(suh.shape) != (inputs,)
                or svh.dtype != torch.float16 or tuple(svh.shape) != (outputs,)
                or mcg.dtype != torch.int32 or mcg.numel() != 1):
            raise ValueError("EXL3 packed geometry/dtype differs from V4.1 projection")
        if not torch.isfinite(suh).all() or not torch.isfinite(svh).all():
            raise ValueError("nonfinite reloaded EXL3 scales")
        return packed

    def block_config(self, namespace="layers"):
        from transformers import DeepseekV41Config
        config = DeepseekV41Config.from_dict(self.config).get_text_config()
        if namespace == "mtp":
            config.num_hidden_layers = config.num_nextn_predict_layers
            config.n_routed_experts = config.dspark_n_routed_experts
            config.num_experts_per_tok = config.dspark_num_experts_per_tok
            config.compress_ratios = [0] * config.num_hidden_layers
            config.kv_source_layer_ids = []
            config.index_source_layer_ids = []
            config.candidate_source_layer_id = -1
            config.engram_layer_ids = []
        elif namespace != "layers":
            raise ValueError("invalid V4.1 block namespace")
        config._experts_implementation = "eager"
        config._attn_implementation = "eager"
        return config

    def load_dspark_input(self, device="cuda:0", *, native_kernels, embedding=None):
        """Load only dSpark's input adapter; optionally share an existing embedding."""
        from ..models.definitions.deepseek_v41 import DeepSeekV41RMSNorm
        from .v41_dspark import V41DSparkInput
        from .v41_native import V41NativeLinear

        config = self.block_config()
        if embedding is None:
            weight = self.tensor("embed.weight", device)
            if weight.dtype != torch.bfloat16:
                raise ValueError("unexpected source token embedding dtype")
            embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
        name = "mtp.0.main_proj"
        if name + ".scale" in self.weight_map:
            projection = V41NativeLinear(self.tensor(name + ".weight", device),
                                         self.tensor(name + ".scale", device), native_kernels)
        else:
            weight = self.tensor(name + ".weight", device)
            projection = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
            projection.weight = torch.nn.Parameter(weight, requires_grad=False)
        norm = DeepSeekV41RMSNorm(self.tensor("mtp.0.main_norm.weight", device), config.rms_norm_eps)
        return V41DSparkInput(embedding, projection, norm, config).eval()

    def load_main_input(self, hash_state, device="cuda:0"):
        """Use an already-bound, attested hash state; never initialize a tokenizer."""
        from ..models.definitions.deepseek_v41 import DeepSeekV41MappedEmbedding, DeepSeekV41RotaryEmbedding
        from .v41_inputs import V41MainInput

        config = self.block_config()
        if hash_state.token_map.numel() != config.vocab_size:
            raise ValueError("main input requires the attested full token map")
        if tuple(hash_state.layout.layer_ids) != tuple(config.engram_layer_ids):
            raise ValueError("hash-state PLE layer layout differs from source")
        weight = self.tensor("embed.weight", device)
        if weight.dtype != torch.bfloat16:
            raise ValueError("unexpected source token embedding dtype")
        embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
        tables = {}
        try:
            for layer in config.engram_layer_ids:
                prefix = f"layers.{layer}.engram.embed"
                tables[str(layer)] = DeepSeekV41MappedEmbedding(
                    self.snapshot / self.weight_map[prefix + ".weight"], prefix,
                    scale_path=self.snapshot / self.weight_map[prefix + ".scale"])
            with torch.device(device):
                rotary = DeepSeekV41RotaryEmbedding(config)
            return V41MainInput(embedding, hash_state.to(device), rotary, tables, config).eval()
        except BaseException:
            for table in tables.values():
                table.close()
            raise

    @torch.no_grad()
    def load_decoded_block(self, layer_index, device="cuda:0", *, native_kernels=None, namespace="layers"):
        from ..models.definitions.deepseek_v41 import DeepSeekV41DecoderLayer, DeepSeekV41Experts

        config = self.block_config(namespace)
        if type(layer_index) is not int or not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError("block index outside source config")
        if namespace == "mtp" and native_kernels is None:
            raise ValueError("dSpark requires its native noncausal draft attention")
        with torch.device("meta"):
            block = DeepSeekV41DecoderLayer(config, layer_index)
            block.mlp.experts = DeepSeekV41Experts.from_fused(block.mlp.experts)
        expected = block.state_dict()
        for key, template in expected.items():
            source_key = source_layer_key(layer_index, key, namespace)
            scale_key = source_key.removesuffix(".weight") + ".scale"
            if (native_kernels is not None and key.endswith(".weight")
                    and scale_key in self.weight_map and ".o_a_proj." not in key):
                from .v41_native import V41NativeLinear

                parent_key = key.removesuffix(".weight")
                owner_key, _, leaf = parent_key.rpartition(".")
                owner = block.get_submodule(owner_key) if owner_key else block
                linear = V41NativeLinear(self.tensor(source_key, device),
                                         self.tensor(scale_key, device), native_kernels)
                if (linear.out_features, linear.in_features) != tuple(template.shape):
                    raise ValueError(f"native V4.1 shape mismatch: {source_key}")
                setattr(owner, leaf, linear)
                continue
            value = self.decoded(source_key, device)
            if value.shape != template.shape:
                raise ValueError(f"V4.1 shape mismatch: {source_key}: {value.shape} != {template.shape}")
            parent_key, _, leaf = key.rpartition(".")
            parent = block.get_submodule(parent_key) if parent_key else block
            if leaf in parent._parameters:
                setattr(parent, leaf, torch.nn.Parameter(value, requires_grad=False))
            else:
                parent.register_buffer(leaf, value)
        # Rotary embedding constants are nonpersistent buffers initialized on meta.
        # Recreate these tiny modules on the execution device from their config.
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41RotaryEmbedding

        from ..models.definitions.deepseek_v41 import DeepSeekV41RotaryEmbedding
        for name, module in list(block.named_modules()):
            if isinstance(module, DeepseekV41RotaryEmbedding):
                parent_key, _, leaf = name.rpartition(".")
                parent = block.get_submodule(parent_key) if parent_key else block
                with torch.device(device):
                    setattr(parent, leaf, DeepSeekV41RotaryEmbedding(config))
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41RMSNorm

        from ..models.definitions.deepseek_v41 import DeepSeekV41RMSNorm

        for name, module in list(block.named_modules()):
            if isinstance(module, DeepseekV41RMSNorm):
                parent_key, _, leaf = name.rpartition(".")
                parent = block.get_submodule(parent_key) if parent_key else block
                setattr(parent, leaf, DeepSeekV41RMSNorm(module.weight.detach(), module.variance_epsilon))
        if any(tensor.is_meta for tensor in list(block.parameters()) + list(block.buffers())):
            raise ValueError("V4.1 block retains uninitialized meta tensors")
        if native_kernels is not None:
            block.self_attn.v41_source_kernels = native_kernels
            from .v41_native import V41NativeAttention, V41NativeIndexer
            block.self_attn.__class__ = V41NativeAttention
            if namespace == "mtp":
                from .v41_native import V41NativeDSparkAttention
                block.self_attn.__class__ = V41NativeDSparkAttention
                with torch.device(device):
                    block.self_attn.compress_rotary = DeepSeekV41RotaryEmbedding(config)
            if block.self_attn.indexer is not None:
                block.self_attn.indexer.__class__ = V41NativeIndexer
                block.self_attn.indexer.v41_source_kernels = native_kernels
            from ..models.definitions.deepseek_v41 import DeepSeekV41HyperConnection
            block.attn_hc = DeepSeekV41HyperConnection(block.attn_hc, native_kernels)
            block.ffn_hc = DeepSeekV41HyperConnection(block.ffn_hc, native_kernels)
        return block.eval()
