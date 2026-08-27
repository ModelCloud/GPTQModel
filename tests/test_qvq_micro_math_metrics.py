from __future__ import annotations

from scripts.qvq_evaluate import _micro_answer, _micro_equal, build_parser


def test_micro_answer_prefers_gsm_delimiter_and_handles_commas() -> None:
    assert _micro_answer("work 2 + 2 = 4\n#### 1,000") == "1000"
    assert _micro_answer("The answer is 42") == "42"


def test_micro_answer_numeric_equivalence() -> None:
    assert _micro_equal("1,000", "$1000")
    assert _micro_equal("4.0", "4")
    assert not _micro_equal("4", "5")
    assert not _micro_equal(None, "5")


def test_micro_math_parser_requires_and_retains_output_path() -> None:
    args = build_parser().parse_args(
        [
            "micro_math",
            "--dense-model", "/dense",
            "--checkpoint", "/checkpoint",
            "--dataset", "/dataset.jsonl",
            "--manifest", "/manifest.json",
            "--output", "/result.json",
        ]
    )
    assert args.output.name == "result.json"
