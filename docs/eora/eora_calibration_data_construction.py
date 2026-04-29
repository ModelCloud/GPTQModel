# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from datasets import load_dataset


def question_answering_format(question, answer):

    return f"Question: {question}\nAnswer: {answer}"

def multiple_choices_question_answering_format(question, choices, answer):
    return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}"

## An example of using ARC for construting the EoRA calibration set

def construct_c4():
    calibration_dataset = load_dataset(
      "allenai/c4",
      data_files="en/c4-train.00001-of-01024.json.gz",
      split="train", download_mode="force_redownload"
    ).select(range(1024))["text"]
    return calibration_dataset

def construct_ARC():
    nsamples = 1024
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

def construct_mmlu():

    mmlu_calibration_dataset = load_dataset('cais/mmlu', 'all', split='validation')
    dataset = []
    for example in mmlu_calibration_dataset:
        question = example['question']
        choices = example['choices']
        answer = ['A','B','C','D'][example['answer']]
        dataset.append(multiple_choices_question_answering_format(question, choices, answer))

    ## we recommend also include some examples from C4 to avoid overfitting to the downstream data
    c4_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(1024))["text"]

    return dataset + c4_dataset
