# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL


eval_results = GPTQModel.eval("HandH1998/QQQ-Llama-3-8b-g128",
                                 framework=EVAL.LM_EVAL,
                                 tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])

print(f"{eval_results}")
