lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 1