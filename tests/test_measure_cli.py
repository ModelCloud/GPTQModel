# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""`python -m gptqmodel.measure` end to end on a tiny checkpoint pair."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel.measure.__main__ import build_parser, main


TEXT = " ".join(f"token{i}" for i in range(64))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("measure_cli") / "tiny"
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(root)
    tokenizer = Tokenizer(WordLevel(vocab={f"token{i}": i for i in range(64)} | {"<pad>": 64},
                                    unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="token0", eos_token="token1",
    ).save_pretrained(root)
    text_file = root / "sample.txt"
    text_file.write_text(TEXT)
    return root, text_file


def test_parser_defaults():
    args = build_parser().parse_args(["--model", "a", "--reference", "b"])
    assert args.batch == 1 and args.max_batches == 4 and args.positions == 64
    assert args.fail_on == "fail"


def test_cli_measures_all_metrics_when_none_are_selected(checkpoint, capsys):
    root, text_file = checkpoint

    report_file = root.parent / "all_metrics.json"
    code = main([
        "--model", str(root), "--reference", str(root),
        "--text", str(text_file), "--json-out", str(report_file),
        "--device", "cpu",
    ])

    assert code == 0
    payload = json.loads(report_file.read_text())
    assert set(payload["values"]) == {"PPL", "KLD", "RMSE", "TOP_K"}
    assert payload["values"]["KLD"] == 0.0
    assert payload["values"]["RMSE"] == 0.0
    assert payload["values"]["TOP_K"]["1"] == 1.0


def test_cli_compares_a_checkpoint_with_itself(checkpoint, capsys):
    root, text_file = checkpoint

    report_file = root.parent / "report.json"
    code = main([
        "--model", str(root), "--reference", str(root),
        "--text", str(text_file), "--json-out", str(report_file),
        "--device", "cpu", "--ppl", "--kld", "--topk", "1",
    ])

    assert code == 0
    payload = json.loads(report_file.read_text())
    assert "PPL" in capsys.readouterr().out
    assert payload["values"]["KLD"] == 0.0
    assert payload["values"]["TOP_K"]["1"] == 1.0


def test_cli_returns_2_when_the_measurement_cannot_run(tmp_path, capsys):
    missing = tmp_path / "missing"
    code = main(["--model", str(missing), "--reference", str(missing), "--text", str(missing)])

    assert code == 2
    assert "measurement failed" in capsys.readouterr().err


def test_cli_fails_on_a_broken_grade(checkpoint, capsys):
    root, text_file = checkpoint

    code = main([
        "--model", str(root), "--reference", str(root),
        "--text", str(text_file), "--device", "cpu", "--kld", "--kld-pass", "<0",
    ])

    # KLD is exactly 0, so "<0" cannot hold: the run must report a failure.
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_parser_rejects_bad_numbers_and_bounds():
    for argv in (
        ["--model", "a", "--reference", "b", "--batch", "0"],
        ["--model", "a", "--reference", "b", "--positions", "0"],
        ["--model", "a", "--reference", "b", "--window", "1"],
        ["--model", "a", "--reference", "b", "--stride", "-1"],
        ["--model", "a", "--reference", "b", "--topk", "0,5"],
        ["--model", "a", "--reference", "b", "--topk", "one"],
        ["--model", "a", "--reference", "b", "--kld-pass", "0.05"],
    ):
        with pytest.raises(SystemExit) as raised:
            build_parser().parse_args(argv)
        assert raised.value.code == 2


def test_bad_device_is_rejected_before_any_model_load(tmp_path, capsys):
    missing = tmp_path / "missing"

    code = main(["--model", str(missing), "--reference", str(missing), "--device", "cuda:999"])

    assert code == 2
    err = capsys.readouterr().err
    assert "asks for GPU 999" in err
    # The error names the flag, so it is actionable without reading the source.
    assert "--device" in err


def test_conflicting_json_flags_are_rejected(checkpoint, capsys):
    root, text_file = checkpoint

    code = main([
        "--model", str(root), "--reference", str(root), "--text", str(text_file),
        "--json", "--json-out", str(root.parent / "r.json"),
    ])

    assert code == 2
    assert "pick one" in capsys.readouterr().err


def test_the_table_is_rendered_through_logbar(checkpoint, capsys):
    root, text_file = checkpoint

    code = main([
        "--model", str(root), "--reference", str(root), "--text", str(text_file),
        "--device", "cpu", "--ppl",
    ])

    assert code == 0
    table = capsys.readouterr().out
    # LogBar's simple table has a header and a rule line; the verdict follows.
    assert "metric" in table and "value" in table
    assert "result:" in table
