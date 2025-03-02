# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
from typing import Dict, Optional, Sequence

## This is the oldway of constructing the calibration dataset
import numpy as np
import torch
import transformers


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
def get_mathqa_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata_mathqa = load_dataset('math_qa', split='train')
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)

    import random
    random.seed(seed)
    trainloader = []
    mathqa_namsples = int(20)
    print(f"mathqa_namsples {mathqa_namsples}")
    i = 0
    for _ in range(mathqa_namsples):

        cur_len = 0
        input = ""
        while cur_len < seqlen:
            doc = traindata_mathqa[i]
            cur_input = "Question: " + doc["Problem"] + " Choices: " + doc["options"] + ". Rationale: " + doc["Rationale"] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            i += 1

        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    c4_nsamples = nsamples - mathqa_namsples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_arc_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata_arc_easy = load_dataset('ai2_arc', 'ARC-Easy', split='train')
    traindata_arc_challenge = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)


    import random
    random.seed(seed)
    trainloader = []
    arc_e_namsples = int(20)
    print(f"arc_e_namsples {arc_e_namsples}")
    i = 0
    for _ in range(arc_e_namsples):
        
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_easy[i]['choices']['label'].index(traindata_arc_easy[i]['answerKey'])
            cur_input = traindata_arc_easy[i]['question'] +" "+ traindata_arc_easy[i]['choices']['text'][answer] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            i += 1
        
        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))


    arc_c_namsples = int(10)
    print(f"arc_c_namsples {arc_c_namsples}")
    i = 0
    for _ in range(arc_c_namsples):
        
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_challenge[i]['choices']['label'].index(traindata_arc_challenge[i]['answerKey'])
            cur_input = traindata_arc_challenge[i]['question'] +" "+ traindata_arc_challenge[i]['choices']['text'][answer] + ". "
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            i += 1

        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))


    # traindata = load_dataset("json", data_files=f"{c4_data}/c4-train.json")['train']
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    c4_nsamples = nsamples - arc_c_namsples - arc_e_namsples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # print(len(traindata[i]['text']))
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        # print(f"inp {inp.shape}")
        trainloader.append((inp, tar))

    return trainloader

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def get_loaders(
    data_name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if type(data_name) == list:
        raise NotImplementedError
    else:
        if 'wikitext2' in data_name:
            return get_wikitext2(nsamples, seed, seqlen, model)
        if "mathqa" in data_name:
            return get_mathqa_c4(nsamples, seed, seqlen, model)
        if "arc" in data_name:
            return get_arc_c4(nsamples, seed, seqlen, model)

    
    