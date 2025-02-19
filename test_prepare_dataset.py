
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


def question_answering_format(question, answer):
    
    return f"Question: {question}\nAnswer: {answer}"

## An example of using ARC for construting the EoRA calibration set

def construct_c4(nsamples):
    calibration_dataset = load_dataset(
      "allenai/c4",
      data_files="en/c4-train.00001-of-01024.json.gz",
      split="train"
    ).select(range(1024))["text"]
    return calibration_dataset

def construct_ARC(nsamples): 
    arc_easy_calibration_dataset = load_dataset('ai2_arc', 'ARC-Easy', split='train').select(range(nsamples))
    arc_challenge_calibration_dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='train').select(range(nsamples))
    dataset = []

    for example in arc_easy_calibration_dataset:
        answer = example['choices']['text'][example['choices']['label'].index(example['answerKey'])]
        question = example['question']
        dataset.append(question_answering_format(question=question,answer=answer))
    
    for example in arc_challenge_calibration_dataset:
        answer = example['choices']['text'][example['choices']['label'].index(example['answerKey'])]
        question = example['question']
        dataset.append(question_answering_format(question=question,answer=answer))

    ## we recommend also include some examples from C4 to avoid overfitting to the downstream data
    c4_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(nsamples))["text"]
   
    return dataset + c4_dataset


# arc_calibration_dataset = construct_ARC(1024)
# print(len(arc_calibration_dataset))
# print(arc_calibration_dataset[-1])

# c4_calibrarion_dataset = construct_c4(1024)

# model_id = "meta-llama/Llama-3.2-1B"
# quant_config = QuantizeConfig(bits=4, group_size=128)
# model = GPTQModel.load(model_id, quant_config)

# ## tokenizer for testing
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_id)

# prepare_dataset = model.prepare_dataset(c4_calibrarion_dataset)


# inputs = tokenizer(c4_calibrarion_dataset[0], return_tensors="pt")
# print(inputs['input_ids'].shape)

# print(prepare_dataset[0]['input_ids'].shape)