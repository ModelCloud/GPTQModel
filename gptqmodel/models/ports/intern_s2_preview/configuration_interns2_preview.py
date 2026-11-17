# Adapted for package-owned Transformers 5 loading by ModelCloud.ai, 2026.
# Pinned upstream and changes: see PROVENANCE.md.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


class InternS2PreviewVisionConfig(PreTrainedConfig):
    model_type = "intern_s2_preview"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class InternS2PreviewTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS2PreviewTextModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.

    ```python
    >>> from transformers import InternS2PreviewTextModel, InternS2PreviewTextConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration =  InternS2PreviewTextConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = InternS2PreviewTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # NOTE: `model_type` is kept as `qwen3_5_moe_text` because transformers hardcodes weight-renaming logic keyed
    # on model_type (e.g. `model_dtype`); reusing the parent's value ensures correct weight loading via
    # `AutoModelForCausalLM.from_pretrained`.
    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.validate_layer_type()

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)


class InternS2PreviewTimeSeriesConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS2PreviewTimeSeriesModel`]. It is used to instantiate a
    InternS2PreviewTimeSeries model according to the specified arguments, defining the model architecture.

    Args:
        ts_adapt_in_dim (`int`, *optional*, defaults to 256):
            The input dimension of the time series adapter.
        ts_adapt_out_dim (`int`, *optional*, defaults to 1024):
            The output dimension of the time series adapter.
        ts_hidden_dim (`int`, *optional*, defaults to 1024):
            The hidden dimension of the time series model.
        ts_cnn_channels (`list[int]`, *optional*, defaults to [1, 32, 64, 128, 128]):
            The channels of the time series CNN.
        ts_cnn_kernel_sizes (`list[int]`, *optional*, defaults to [3, 5, 5, 5]):
            The kernel sizes of the time series CNN.
        ts_cnn_strides (`list[int]`, *optional*, defaults to [2, 4, 4, 5]):
            The strides of the time series CNN.
        ts_cnn_paddings (`list[int]`, *optional*, defaults to [1, 2, 2, 2]):
            The paddings of the time series CNN.
        ts_concat_subsampling_in_channels (`int`, *optional*, defaults to 128):
            The input channels of the time series concat subsampling.
        ts_concat_subsampling_concat_size (`int`, *optional*, defaults to 2):
            The concat size of the time series concat subsampling.
        **super_kwargs:
            Additional keyword arguments passed along to the base class `WhisperConfig`.
    """

    model_type = "interns2_preview_time_series"
    base_config_key = "ts_config"

    def __init__(
        self,
        activation_dropout: float = 0.0,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        d_model: int = 768,
        dropout: float = 0.0,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 3072,
        encoder_layerdrop: float = 0.0,
        encoder_layers: int = 17,
        max_source_positions: int = 1500,
        num_mel_bins: int = 80,
        out_hidden_size: int = 2048,
        scale_embedding: bool = False,
        ts_adapt_in_dim: int = 256,
        ts_adapt_out_dim: int = 1024,
        ts_hidden_dim: int = 1024,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.d_model = d_model
        self.dropout = dropout
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.encoder_layers = encoder_layers
        self.max_source_positions = max_source_positions
        self.num_mel_bins = num_mel_bins
        self.out_hidden_size = out_hidden_size
        self.scale_embedding = scale_embedding
        self.ts_adapt_in_dim = ts_adapt_in_dim
        self.ts_adapt_out_dim = ts_adapt_out_dim
        self.ts_hidden_dim = ts_hidden_dim

        assert self.ts_adapt_out_dim == self.ts_hidden_dim, "ts_adapt_out_dim should be equal to ts_hidden_dim"


class InternS2PreviewConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS2PreviewModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 248056):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 248057):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import InternS2PreviewForConditionalGeneration, InternS2PreviewConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration = InternS2PreviewConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = InternS2PreviewForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "intern_s2_preview"
    sub_configs = {
        "vision_config": InternS2PreviewVisionConfig,
        "text_config": InternS2PreviewTextConfig,
        "ts_config": InternS2PreviewTimeSeriesConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        ts_config=None,
        ts_token_id=248093,
        ts_start_id=248091,
        ts_end_id=248092,
        **kwargs,
    ):
        if isinstance(ts_config, dict):
            self.ts_config = self.sub_configs["ts_config"](**ts_config)
        elif ts_config is None:
            self.ts_config = self.sub_configs["ts_config"]()
        else:
            self.ts_config = ts_config

        self.ts_token_id = ts_token_id
        self.ts_start_id = ts_start_id
        self.ts_end_id = ts_end_id
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)
        self.architectures = ["InternS2PreviewForConditionalGeneration"]


__all__ = ["InternS2PreviewConfig", "InternS2PreviewTextConfig"]
