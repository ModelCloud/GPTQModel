# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from tests.eval import resolve_eval_metric_alias


def test_resolve_eval_metric_alias_handles_evalution_mmlu_names():
    metrics = {"acc,ll": 0.4, "acc,ll_avg": 0.5}

    assert resolve_eval_metric_alias("acc", metrics) == "acc,ll"
    assert resolve_eval_metric_alias("acc,none", metrics) == "acc,ll"
    assert resolve_eval_metric_alias("acc_norm", metrics) == "acc,ll_avg"
    assert resolve_eval_metric_alias("acc_norm,none", metrics) == "acc,ll_avg"

