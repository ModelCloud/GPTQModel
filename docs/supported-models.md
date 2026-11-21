# Supported models

[Project overview](../README.md) · [Hardware and backend support](support-matrix.md)

The table mirrors every explicit registration in `gptqmodel.models.auto.MODEL_MAP`, including text-model and backward-compatibility aliases. Qwen 3.5 registrations require Transformers 5.2.0 or newer.

<!-- model-types:start -->
| Model family | Registered Transformers `model_type` values |
|---|---|
| A.X-K2 | `axk2` |
| AfMoE / Trinity | `afmoe` |
| Agnes 3.0 Flash | `agnes` |
| Apertus 1 / 1.5 | `apertus`, `apertus1p5`, `apertus1p5_text` |
| Baichuan | `baichuan` |
| Bailing MoE / Hybrid (LING / RING) | `bailing_moe`, `bailing_hybrid` |
| Bloom | `bloom` |
| Brumby | `brumby` |
| ChatGLM | `chatglm` |
| CodeGen | `codegen` |
| Cohere 1 / 2 / 2 MoE / Compass (North Mini / Micro Vision) | `cohere`, `cohere2`, `cohere2_moe`, `cohere_compass` |
| DBRX / DBRX Converted | `dbrx`, `dbrx_converted` |
| DeciLM | `deci` |
| Decision Transformer | `decision_transformer` |
| DeepSeek V2 / V3 / V3.2 / V4 / VL / VL2 / OCR2 | `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`, `deepseek_vl`, `deepseek_vl_v2`, `deepseek_ocr2` |
| KRAFTON Raon-1 mini | `raon1_mini` |
| DiffusionGemma | `diffusion_gemma` |
| Dots1 / Dots3 Note | `dots1`, `dots3_note` |
| Dream | `dream` |
| ERNIE 4.5 / MoE / VL MoE | `ernie4_5`, `ernie4_5_moe`, `ernie4_5_moe_vl`, `ernie4_5_vl_moe` |
| EXAONE 3 / 4 / 4.5 | `exaone`, `exaone4`, `exaone4_5`, `exaone4_5_text` |
| Falcon / Falcon H1 / Falcon Mamba | `falcon`, `falcon_h1`, `falcon_mamba` |
| RecursiveSI Flint | `flint` |
| Gemma 1-4 / 3n / Unified | `gemma`, `gemma2`, `gemma3`, `gemma3_text`, `gemma3n`, `gemma3n_text`, `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` |
| GLM / GLM4 / GLM4V / GLM5 / OCR / ASR | `glm`, `glm4`, `glm4_moe`, `glm4_moe_lite`, `glm4v`, `glm4v_moe`, `glm4v_moe_text`, `glm5_next`, `glm_moe_dsa`, `glm_ocr`, `glmasr` |
| GPT-2 | `gpt2` |
| GPT BigCode | `gpt_bigcode` |
| GPT-Neo / GPT-NeoX | `gpt_neo`, `gpt_neox` |
| GPT-OSS | `gpt_oss` |
| GPT-J | `gptj` |
| Granite / Granite SWA / Granite MoE / Granite MoE SWA / Granite MoE Hybrid | `granite`, `granite_swa`, `granitemoe`, `granitemoe_swa`, `granitemoehybrid` |
| GRIN-MoE | `grinmoe` |
| Helium | `helium` |
| HRM | `hrm_text` |
| Hunyuan V1 / VL / OCR | `hunyuan_v1_dense`, `hunyuan_v1_moe`, `hunyuan_vl` |
| HY-V3 | `hy_v3` |
| Hymba | `hymba` |
| Inkling | `inkling_mm_model` |
| Inkling 2 Mini | `inkling2_loop` |
| Instella | `instella` |
| Intern S1 / S2 Preview | `interns1`, `intern_s2_preview` |
| InternLM 1 / 2 / 2.5 | `internlm`, `internlm2` |
| InternVL 3 / InternVL Chat | `internvl`, `internvl_chat` |
| Jamba / Jamba2 | `jamba` |
| K2-Horizon (Dense / MoVA) | `k2_horizon` |
| Kimi K2 / K2.5 | `kimi_k2`, `kimi_k25` |
| Kimi K3 / Linear (bounded native-source calibration) | `kimi_k3`, `kimi_linear` |
| Klear | `klear` |
| Laguna | `laguna` |
| LFM2 / LFM2 MoE / LFM2-VL | `lfm2`, `lfm2_moe`, `lfm2_vl` |
| LFM3 / LFM3 Omni / LFM3-VL | `lfm3`, `lfm3_omni`, `lfm3_vl` |
| LLaDA2 MoE | `llada2_moe` |
| Llama 1-4 / TinyLlama / Nemotron Ultra | `llama`, `llama4`, `llama4_text` |
| Llama 3.2 VL | `mllama`, `mllama_text_model` |
| LLaVA-1.5 | `llava` |
| FastVLM / LLaVA-Qwen2 | `llava_qwen2` |
| LocateAnything | `locateanything` |
| LongCat Flash | `longcat_flash` |
| LongLLaMA | `longllama` |
| Mage-VL | `mage_vl` |
| Marin | `marin` |
| MiMo / MiMo V2 / MiMo V2 Flash | `mimo`, `mimo_v2`, `mimo_v2_flash` |
| MiniCPM / MiniCPM3 / MiniCPM-O / MiniCPM-V | `minicpm`, `minicpm3`, `minicpmo`, `minicpmv`, `minicpmv4_6` |
| MiniMax M2 / M3-VL | `minimax`, `minimax_m2`, `minimax_m3_vl` |
| Mistral / Mistral3 / Ministral3 | `mistral`, `mistral3`, `ministral3` |
| Mixtral | `mixtral` |
| MobileLLM | `mobilellm` |
| MOSS | `moss` |
| MPT | `mpt` |
| Muse Glimmer | `muse_glimmer` |
| Nanbeige 4.2 | `nanbeige` |
| Nemotron NAS / H / H Puzzle / Omni / Labs Diffusion | `nemotron-nas`, `nemotron_h`, `nemotron_h_puzzle`, `nemotronh_nano_omni_reasoning_v3`, `nemotron_labs_diffusion` |
| OLMo / OLMo 2 / 3 / Hybrid / OLMoE | `olmo`, `olmo2`, `olmo3`, `olmo_hybrid`, `olmoe` |
| OpenVLA | `openvla` |
| OPT | `opt` |
| Ouro | `ouro` |
| Ovis 1.6 / 2 / 2.5 / 2.6 MoE / 2.6 Next | `ovis`, `ovis2`, `ovis2_5`, `ovis2_6_moe`, `ovis2_6_next` |
| PanGu-α | `gpt_pangu` |
| Phi 1-4 / Phi MoE | `phi`, `phi3`, `phi4mm`, `phimoe` |
| Qwen 1-4 / 3.5 / 3.6 / 3.8 / MoE / Next | `qwen`, `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen4_exp`, `qwen4_exp_text` |
| Qwen 2 / 2.5 / 3 VL / 3 VL MoE | `qwen2_vl`, `qwen2_vl_text`, `qwen2_5_vl`, `qwen2_5_vl_text`, `qwen3_vl`, `qwen3_vl_moe` |
| Qwen 2.5 / 3 Omni | `qwen2_5_omni`, `qwen3_omni_moe` |
| Qwen2-Audio | `qwen2_audio` |
| Qwen-Drive 1.0 | `qwen_drive` |
| RefinedWeb | `refinedWeb`, `refinedWebModel` |
| [Ricursive Ripple](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) ([cached generation](ricursive-cached-generation.md)) | `ricursive` |
| Seed-OSS | `seed_oss` |
| [SenseNova-Loop 9B](https://github.com/OpenSenseNova/SenseNova-Loop) (`trust_remote_code`) | `sensenova_loop` |
| SmolLM3 | `smollm3` |
| SmolVLM | `smolvlm` |
| Solar Open / Open 2 | `solar_open`, `solar_open2` |
| Spark 2.5 | `spark2_5` |
| StableLM | `stablelm`, `stablelm_epoch` |
| StarCoder2 | `starcoder2` |
| TeleChat2 | `telechat` |
| Unlimited-OCR | `unlimited_ocr`, `unlimited-ocr` |
| Voxtral | `voxtral` |
| [Xing4.0 (29B-A4B)](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | `xing4_0` |
| XVERSE | `xverse` |
| Yi | `yi` |
| Zamba / Zamba2 | `zamba`, `zamba2` |
| ZDTaichu 5.0 | `zdtaichu5_0` |
<!-- model-types:end -->

Kimi K3 / Linear quantization requires the bounded native-source runner to
preserve AttnRes prefix/residual state; the ordinary `GPTQModel.quantize()`
looper rejects it. See the [Kimi K3 calibration contract](kimi-k3-native-source.md).

Qwen2.5-Omni supports quantizing its text output head with `lm_head=True`.
The head is `thinker.lm_head`; GPTQ defaults it to 8-bit, group size 32.
To use a 4-bit head, configure an explicit dynamic override:

```python
from gptqmodel import QuantizeConfig

quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    lm_head=True,
    dynamic={r"thinker\.lm_head$": {"bits": 4, "group_size": 32, "desc_act": False}},
)
```

This option quantizes the thinker's text head; the talker and audio decoder
retain their original weights. Tied output-head weights must be untied first.
