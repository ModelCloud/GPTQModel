from transformers import GenerationConfig, PreTrainedModel

from gptqmodel.utils.logger import setup_logger

logger = setup_logger()

# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
# and the `from_config` api does not auto-load the config from `generation_config.json`
def autofix_hf_model_loading_generation_config(model: PreTrainedModel, path:str):
    if model.can_generate():
        logger.info(f"Model: Loaded `generation_config`: {model.generation_config}")
        try:
            cfg = GenerationConfig.from_pretrained(pretrained_model_name=path)
            if cfg != model.generation_config:
                model.generation_config = cfg
                logger.info(f"Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`.")
            else:
                pass
                #logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
        except Exception as e:
            logger.info("Model: `generation_config.json` not found. Skipped checking.")

def autofix_hf_model_config(model: PreTrainedModel):
    if model.can_generate():
        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        # print(f"After autofix_hf_model_config: {model.generation_config}")

def autofix_hf_generation_config(cfg: GenerationConfig):
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if hasattr(cfg, "temperature") and cfg.temperature is not None and cfg.temperature != 1.0:
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if hasattr(cfg, "typical_p") and cfg.typical_p is not None and cfg.typical_p != 1.0:
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None):
            errors += 1
        if hasattr(cfg, "epsilon_cutoff") and cfg.epsilon_cutoff is not None and cfg.epsilon_cutoff != 0.0:
            errors += 1
        if hasattr(cfg, "eta_cutoff") and cfg.eta_cutoff is not None and cfg.eta_cutoff != 0.0:
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            logger.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")

