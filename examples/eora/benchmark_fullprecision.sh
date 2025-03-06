## meta-llama/Llama-3.2-1B
## meta-llama/Llama-3.2-3B
## meta-llama/Meta-Llama-3-8B
## meta-llama/Llama-3.1-8B
## meta-llama/Meta-Llama-3-70B

lm_eval --model hf \
    --tasks mmlu,arc_challenge \
    --model_args pretrained=meta-llama/Meta-Llama-3-70B,parallelize=True \
    --batch_size 1 \
    --seed 1234|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/full_precision/Meta-Llama-3-70B.txt


lm_eval --model hf \
    --tasks mmlu,arc_challenge \
    --model_args pretrained=meta-llama/Llama-3.2-1B \
    --batch_size 1 \
    --seed 1234|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/full_precision/Llama-3.2-1B.txt


lm_eval --model hf \
    --tasks mmlu,arc_challenge \
    --model_args pretrained=meta-llama/Llama-3.2-3B \
    --batch_size 1 \
    --seed 1234|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/full_precision/Llama-3.2-3B.txt


lm_eval --model hf \
    --tasks mmlu,arc_challenge \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --batch_size 1 \
    --seed 1234|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/full_precision/Llama-3.1-8B.txt



python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/quantized_weight/Llama-3.2-3BB.txt

python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.1-8B_4bits_128group_size|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/quantized_weight/Llama-3.1-8B.txt

python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Meta-Llama-3-70B_4bits_128group_size|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/quantized_weight/Meta-Llama-3-70B.txt

