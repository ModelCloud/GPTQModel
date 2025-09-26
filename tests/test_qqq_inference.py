from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

eval_results = GPTQModel.eval("HandH1998/QQQ-Llama-3-8b-g128",
                                 framework=EVAL.LM_EVAL,
                                 tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])

print(f"{eval_results}")