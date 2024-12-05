import json
from typing import Optional
import os

try:
    from evalplus.evaluate import evaluate
except BaseException:
    raise ValueError("evalplus is not installed. Please install via `pip install gptqmodel[evalplus]`.")


def evalplus(
    model: str,
    dataset: str,
    batch: int = 1,
    trust_remote_code: bool = False,
):
    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"

    evaluate(dataset=dataset, model=model, backend="gptqmodel", bs=batch, trust_remote_code=trust_remote_code, greedy=True)

    result_path = model.strip("./").replace("/", "--") + f"_gptqmodel_temp_0.0_eval_results.json"
    result_path = os.path.join("evalplus_results", dataset, result_path)

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"No such file: {result_path}")

    try:
        with open(result_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON: {result_path}")

    try:
        pass_at_k = data["pass_at_k"]
        base = float(pass_at_k["base"]["pass@1"])
        plus = float(pass_at_k["plus"]["pass@1"])

        base_formatted = format(base, ".3f")
        plus_formatted = format(plus, ".3f")
    except KeyError as e:
        raise ValueError(f"Required key not found in JSON: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Data format error: {str(e)}")

    return base_formatted, plus_formatted, result_path


    