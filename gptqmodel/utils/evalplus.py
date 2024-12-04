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
    backend: str = "gptqmodel",
    temperature: float = 0.0,
    parallel: Optional[int] = None,
):
    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"

    evaluate(dataset=dataset, parallel=parallel, model=model, temperature=temperature, backend=backend)

    result_path = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}_eval_results.json"
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
        print(f"Base pass@1: {base_formatted}, Plus pass@1: {plus_formatted}")
    except KeyError as e:
        raise ValueError(f"Required key not found in JSON: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Data format error: {str(e)}")

    return base_formatted, plus_formatted, result_path


    