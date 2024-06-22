
# prompt 128 batch 1

# v1
python data_generate.py --disable_exllamav2

# v2
python data_generate.py --disable_exllama


# marlin
python data_generate.py --disable_exllama --disable_exllamav2 --use_marlin


# prompt 512 batch 1

# v1
python data_generate.py --disable_exllamav2 --prompt_length 512

# v2
python data_generate.py --disable_exllama --prompt_length 512

# marlin
python data_generate.py --disable_exllama --disable_exllamav2 --use_marlin --prompt_length 512


# prompt 128 batch 4

# v1
python data_generate.py --disable_exllamav2 --batch 4

# v2
python data_generate.py --disable_exllama --batch 4


# marlin
python data_generate.py --disable_exllama --disable_exllamav2 --use_marlin --batch 4


# prompt 512 batch 4

 v1
python data_generate.py --disable_exllamav2 --prompt_length 512 --batch 4

# v2
python data_generate.py --disable_exllama --prompt_length 512 --batch 4

# marlin
python data_generate.py --disable_exllama --disable_exllamav2 --use_marlin --prompt_length 512 --batch 4

