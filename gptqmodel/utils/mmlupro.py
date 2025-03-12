import csv
import json
import os
import random
import re
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils.logger import setup_logger

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048
stop_string = "Question:"

log = setup_logger()

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(model_name, selected_subjects):
    scoring_method = "CoT"
    if isinstance(selected_subjects, list):
        selected_subjects = ",".join(selected_subjects)
    subjects = selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = "The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with 'the answer is (X)' where X is the correct letter choice.\n\n\n"
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        log.info("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(model, tokenizer, inference_batchs, batch_size):
    response_batch = []
    pred_batch = []

    dataloader = DataLoader(inference_batchs, batch_size=batch_size)
    pb = log.pb(dataloader).title("Inference Progress:")

    for batch in pb:
        input_tensor = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_model_length, padding_side='left').to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor["input_ids"],
                tokenizer=tokenizer,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                stop_strings=[stop_string]
            )

        generated_texts = tokenizer.batch_decode(
            outputs[:, input_tensor["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results = [generated_text.replace(stop_string, "").strip() for generated_text in generated_texts]

        for generated_text in results:
            response_batch.append(generated_text)
            pred = extract_answer(generated_text)
            pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, ntrain, batch_size):
    global choices
    log.info("evaluating " + subject)
    inference_batches = []
    for i in range(len(test_df)):
        k = ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(model, tokenizer, inference_batches, batch_size)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    log.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def mmlupro(model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            ntrain: int = 5,
            selected_subjects: str = "all",
            save_dir: str = "results",
            global_record_file: str="eval_record_collection.csv",
            batch_size: int = 1,
            seed: int = 12345):
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    model_name = os.path.basename(model.config.name_or_path)
    save_result_dir = os.path.join(
        save_dir, "/".join(args_generate_path(model_name, selected_subjects))
    )
    file_prefix = "-".join(args_generate_path(model_name, selected_subjects))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(save_dir, "summary", file_name)
    os.makedirs(os.path.join(save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        args_selected = selected_subjects.split(",")
        selected_subjects = []
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)

    log.info("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")

    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))

        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path, ntrain, batch_size)

        log.info(f"subject: {subject}, acc: {acc}, corr_count: {corr_count}, wrong_count: {wrong_count}")

        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))

    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))

    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(model_name, selected_subjects) + [time_str, weighted_acc]
        writer.writerow(record)

    with open(os.path.join(summary_path), "r", encoding="utf-8") as file:
        summary = file.read()

    return summary

