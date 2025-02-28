import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional, Set, Union

from ..adapter.remote import resolve_path
from ..utils.logger import setup_logger

log = setup_logger()

@dataclass
class LoraConfig():
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool` | `Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the LoRA rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged. Pass `'loftq'` to use LoftQ initialization. Passing
            `'eva'` results in a data-driven initialization of <ahref='https://arxiv.org/abs/2410.07170' >Explained
            Variance Adaptation</a>. EVA initalizes LoRA based on the SVD of layer input activations and achieves SOTA
            performance due to its ability to adapt to the finetuning data. Pass `'olora'` to use OLoRA initialization.
            Passing `'pissa'` results in the initialization of <ahref='https://arxiv.org/abs/2404.02948' >Principal
            Singular values and Singular vectors Adaptation (PiSSA)</a>, which converges more rapidly than LoRA and
            ultimately achieves superior performance. Moreover, PiSSA reduces the quantization error compared to QLoRA,
            leading to further enhancements. Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA
            initialization, where `[number of iters]` indicates the number of subspace iterations to perform FSVD, and
            must be a nonnegative integer. When `[number of iters]` is set to 16, it can complete the initialization of
            a 7B model within seconds, and the training effect is approximately equivalent to using SVD.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create LoRA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply LoRA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        loftq_config (`Optional[LoftQConfig]`):
            The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights
            and initialize Lora layers. Also pass `init_lora_weights='loftq'`. Note that you should not pass a
            quantized model in this case, as LoftQ will quantize the model itself.
        eva_config (`Optional[EvaConfig]`):
            The configuration of EVA. At a minimum the dataset argument needs to be set (use the same dataset as for
            finetuning).
        use_dora (`bool`):
            Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the weights
            into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is
            handled by a separate learnable parameter. This can improve the performance of LoRA especially at low
            ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger overhead than pure
            LoRA, so it is recommended to merge weights for inference. For more information, see
            https://arxiv.org/abs/2402.09353.
        layer_replication (`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate LoRA adapters attached to them.
        runtime_config (`LoraRuntimeConfig`):
            Runtime configurations (which are not saved or restored).
        lora_bias (`bool`):
            Defaults to `False`. Whether to enable the bias term for the LoRA B parameter. Typically, this should be
            disabled. The main use case for this is when the LoRA weights were extracted from fully fine-tuned
            parameters so the bias of those parameters can be taken into account.
    """
    base_model_name_or_path: str = field(default="", metadata={"help": "base_model_name_or_path"}) # required by hf auto api

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Lora."},
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: (
        bool | Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]
    ) = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing `'True'` (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing `'gaussian'` results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to `'False'` leads to completely random initialization and *is discouraged.*"
                "Pass `'eva'` results in a data-driven initialization of Explained Variance Adaptation."
                "Passing `'olora'` results in OLoRA initialization."
                "Passing `'pissa'` results in PiSSA initialization."
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, "
                "where [number of iters] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    # dict type is used when loading config.json
    loftq_config: Union[Any, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    eva_config: Optional[Any] = field(
        default=None,
        metadata={
            "help": (
                "The configuration of EVA. If this is passed, then EVA will be used to intialize the LoRA layers. "
                "Also set `init_lora_weights='eva'` in this case. "
            )
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable <a href='https://arxiv.org/abs/2402.09353'>'Weight-Decomposed Low-Rank Adaptation' (DoRA)</a>. This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        },
    )
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )
    lora_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable the bias term for the LoRA B parameter. Typically, this should be disabled. The "
                "main use case for this is when the LoRA weights were extracted from fully fine-tuned parameters so "
                "the bias of those parameters can be taken into account."
            )
        },
    )

    # from PeftConfigMixin
    task_type: Optional[str] = field(default="CAUSAL_LM", metadata={"help": "The type of task."})
    peft_type: Optional[str] = field(default="LORA", metadata={"help": "The type of PEFT model."})
    auto_mapping: Optional[dict] = field(
        default=None, metadata={"help": "An auto mapping dict to help retrieve the base model class if needed."}
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """

        kv = asdict(self)

        # remove all None valued Keys and values that are empty (str, list, dict, set
        removed_keys = []
        for k, v in kv.items():
            if v in [None, {}, []]:
                removed_keys.append(k)
            # FIX set is not serializable by json
            elif isinstance(v, Set):
                kv[k] = list(v)

        for k in removed_keys:
            kv.pop(k)

        return kv

    def save_pretrained(self, save_dir: str):
        from ..adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME

        log.info(f"Adapter: Saving EoRA/Lora config to -> `{save_dir}`")

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, HF_ADAPTER_CONFIG_FILE_NAME), "w", encoding="utf-8") as f:
            d = self.to_dict()
            json_str = json.dumps(d, indent=2)
            log.info(f"Saved Adapter Config: \n{json_str}")
            f.write(json_str)

    def __post_init__(self):
        self.peft_type = "LORA"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`Adapter: layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("Adapter: `layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("Adapter: When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.use_dora and self.megatron_config:
            raise ValueError("Adapter: DoRA is not supported")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            raise ValueError("Adapter: LoftQ is not supported")
        elif self.loftq_config:
            raise ValueError("Adapter: LoftQ is not supported")
        elif self.init_lora_weights == "eva" and self.eva_config is None:
            raise ValueError("Adapter: Eva is not supported")
        elif self.init_lora_weights != "eva" and self.eva_config is not None:
            raise ValueError("Adapter: Eva is not supported")

        if self.lora_bias:
            raise ValueError("Adapter: Lora bias is not supported")

        # Using post training conversion of modified base weights to restore their initial values (PiSSA, OLoRA) cannot
        # be correctly done when using rslora + rank_pattern/alpha_pattern. We can't really know if the user intends
        # this when they'll eventually call save_pretrained (i.e. if they'll pass
        # path_initial_model_for_weight_conversionl). Therefore, we only warn but don't raise an error here.
        if (
            self.use_rslora
            and (self.rank_pattern or self.alpha_pattern)
            and (
                (isinstance(self.init_lora_weights, str) and (self.init_lora_weights.startswith("pissa")))
                or (self.init_lora_weights == "olora")
            )
        ):
            raise ValueError("Adapter: RSLora is not supported")

    @classmethod
    def from_pretrained(cls, path: str, filename: str):
        resolved_path = resolve_path(path=path, filename=filename)
        with open(resolved_path, "r") as file:
            config_dict = json.load(file)

        # Remove any keys that are not part of the LoraConfig dataclass fields or empty
        valid_fields = {field.name for field in fields(cls)}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_fields and v not in [None, "", [], {}, ()]}

        return cls(**config_dict)