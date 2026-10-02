# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.models.w4a_calibration_data import (
    EXCLUSIONS,
    QuestionExclusionIndex,
    _wikitext_articles,
    file_digest,
    load_calibration_artifact,
    save_artifact,
    select_records,
    text_digest,
)


@pytest.mark.parametrize("candidate", [
    "Some prose. How many apples does Alice have? More prose.",
    "HOW\nMANY   APPLES DOES ALICE HAVE",
    "Ｈｏｗ many apples, does Alice have!",
])
def test_excludes_normalized_question_inside_text(candidate):
    index = QuestionExclusionIndex([{"id": "gsm/test/0", "question": "How many apples does Alice have?"}])
    assert index.match(candidate) == "gsm/test/0"


def test_excludes_partial_long_question():
    tokens = [f"word{i}" for i in range(30)]
    index = QuestionExclusionIndex([{"id": "mmlu/test/0", "question": " ".join(tokens)}])
    assert index.match("Unrelated context " + " ".join(tokens[5:18])) == "mmlu/test/0"
    assert index.match(" ".join(tokens[:12])) is None


def test_word_boundaries_do_not_match_substrings():
    index = QuestionExclusionIndex([{"id": "test/0", "question": "cat"}])
    assert index.match("catalog") is None


@pytest.mark.parametrize("questions", [[], [{"id": "test/0", "question": "?!"}],
    [{"id": "test/0", "question": "one"}, {"id": "test/0", "question": "two"}]])
def test_requires_nonempty_unique_questions(questions):
    with pytest.raises(ValueError):
        QuestionExclusionIndex(questions)


def _candidates():
    for i in range(1000):
        yield {"article_id": f"article{i}", "source_rows": [i],
               "text": f"A separate encyclopedia article describes music and composers number {i}."}


def _coverage():
    coverage, questions = [], []
    for dataset, config, revision, splits in EXCLUSIONS:
        for split in splits:
            coverage.append({"dataset": dataset, "config": config, "revision": revision,
                             "split": split, "rows": 1})
            questions.append({"id": f"{dataset}/{config}/{split}/0",
                              "question": "How many oranges remain after the farmer sells fifteen?"})
    return coverage, questions


@pytest.fixture
def artifact(tmp_path):
    coverage, questions = _coverage()
    records, audit = select_records(_candidates(), QuestionExclusionIndex(questions),
                                    fit_rows=4, selection_rows=2, seed=787)
    output = tmp_path / "calibration"
    save_artifact(output, records, questions, coverage, audit, seed=787, max_words=4096)
    return output


def _edit_manifest(path, edit):
    manifest = json.loads((path / "manifest.json").read_text())
    edit(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))


def _edit_samples(path, edit):
    rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    edit(rows)
    (path / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    _edit_manifest(path, lambda value: value["files"].update({"samples.jsonl": file_digest(path / "samples.jsonl")}))


def test_round_trip_checks_fit_and_selection(artifact):
    partitions, manifest = load_calibration_artifact(artifact)
    assert {key: len(value) for key, value in partitions.items()} == {"fit": 4, "selection": 2}
    assert manifest["manifest_sha256"] == file_digest(artifact / "manifest.json")
    assert not ({row["article_id"] for row in partitions["fit"]}
                & {row["article_id"] for row in partitions["selection"]})


def test_procedural_arithmetic_has_exact_independent_answers():
    import random
    from fractions import Fraction

    from tests.models.w4a_arithmetic_calibration import KINDS, problem

    for seed in range(50):
        for kind in KINDS:
            row = problem(kind, random.Random(seed))
            p = [Fraction(value) for value in row['operands']]
            expected = {
                'inventory': lambda: p[0] + p[1] - p[2],
                'bundles': lambda: p[0] * p[1] - p[2],
                'shares': lambda: (p[0] - p[1]) / p[2],
                'production': lambda: p[0] * p[1] + p[0] * p[2],
                'discount': lambda: p[0] * (100 - p[1]) / 100,
                'ratio': lambda: p[2] * p[1] / (p[0] + p[1]),
                'change': lambda: p[2] - p[1] * p[0],
                'distance': lambda: p[0] - p[1] - p[2],
            }[kind]()
            assert expected.denominator == 1 and expected.numerator == row['answer']
            assert row['answer'] > 0
            assert str(row['answer']) in row['reasoning']


def test_procedural_articles_are_reproducible_and_cover_every_family():
    from tests.models.w4a_arithmetic_calibration import KINDS, article

    a = article(9431, 7)
    assert a == article(9431, 7)
    assert a['text'] != article(9432, 7)['text']
    assert a['text'] != article(9431, 8)['text']
    assert {p['kind'] for p in a['problems']} == set(KINDS)
    assert len(a['problems']) == 12
    for p in a['problems']:
        assert p['question'] in a['text'] and p['reasoning'] in a['text']


def test_procedural_artifact_uses_the_same_exclusion_and_partition_contract(artifact, tmp_path):
    from tests.models.w4a_arithmetic_calibration import article, prepare
    from tests.models.w4a_calibration_data import ARITHMETIC_CORPUS

    output = tmp_path / 'arithmetic'
    prepare(output, artifact, fit_rows=4, selection_rows=2, seed=9431, examples=8)
    partitions, manifest = load_calibration_artifact(output)
    assert manifest['corpus'] == ARITHMETIC_CORPUS
    assert manifest['audit']['accepted'] == {'fit': 4, 'selection': 2}
    assert manifest['audit']['accepted_overlap_count'] == 0
    assert manifest['files']['generator.py'] == file_digest(output / 'generator.py')
    for records in partitions.values():
        for row in records:
            original = article(row['generator_seed'], row['source_rows'][0], 8)
            assert row['text'] == original['text'] and row['problems'] == original['problems']
    (output / 'generator.py').write_text('changed generator source')
    with pytest.raises(ValueError, match='hash mismatch: generator.py'):
        load_calibration_artifact(output)


def test_generated_question_is_excluded_by_the_common_index():
    from tests.models.w4a_arithmetic_calibration import article

    record = article(9431, 0)
    index = QuestionExclusionIndex([{'id': 'excluded/0', 'question': record['problems'][0]['question']}])
    assert index.match(record['text']) == 'excluded/0'


def test_generated_questions_cannot_cross_article_partitions():
    from tests.models.w4a_arithmetic_calibration import article
    from tests.models.w4a_calibration_data import article_partition

    questions = {'fit': set(), 'selection': set()}
    for index in range(256):
        row = article(9431, index)
        role = article_partition(row['article_id'], 9431)
        for p in row['problems']:
            identity = text_digest(p['question'])
            assert article_partition(identity, 9431) == role
            questions[role].add(identity)
    assert questions['fit'] and questions['selection']
    assert not questions['fit'] & questions['selection']


def test_generated_loader_rejects_cross_partition_questions_with_updated_hashes(artifact, tmp_path):
    from tests.models.w4a_arithmetic_calibration import prepare

    output = tmp_path / 'arithmetic_cross_partition'
    prepare(output, artifact, fit_rows=4, selection_rows=2, seed=9431, examples=8)
    def edit(rows):
        source = next(row for row in rows if row['partition'] == 'fit')
        target = next(row for row in rows if row['partition'] == 'selection')
        target['problems'][0] = source['problems'][0]
        target['text'] = '\n\n'.join(f'Question: {p["question"]}\nAnswer: {p["reasoning"]} '
                                    f'The result is {p["answer"]}.' for p in target['problems'])
        target['text_sha256'] = text_digest(target['text'])
    _edit_samples(output, edit)
    with pytest.raises(ValueError, match='Arithmetic question belongs to a different partition'):
        load_calibration_artifact(output)


def test_generated_artifact_requires_source_snapshot(tmp_path):
    from tests.models.w4a_calibration_data import ARITHMETIC_CORPUS

    with pytest.raises(ValueError, match='generator source snapshot'):
        save_artifact(tmp_path / 'missing', [], [], [], {}, seed=1, max_words=1,
                      corpus=ARITHMETIC_CORPUS)
    assert not (tmp_path / 'missing').exists()


def test_rejects_unchecked_legacy_data(tmp_path):
    path = tmp_path / "old.parquet"
    path.write_bytes(b"unchecked")
    with pytest.raises(ValueError, match="verified dataset directory"):
        load_calibration_artifact(path)


def test_rejects_file_changes(artifact):
    with (artifact / "samples.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calibration_artifact(artifact)


def test_rejects_missing_benchmark(artifact):
    _edit_manifest(artifact, lambda value: value["evaluation_coverage"].pop())
    with pytest.raises(ValueError, match="coverage"):
        load_calibration_artifact(artifact)


def test_evaluation_registry_is_covered_by_the_exclusion_registry():
    from tests.models.w4a_calibration_data import (
        evaluated_dataset_configs,
        excluded_dataset_configs,
    )

    covered, required = excluded_dataset_configs(), evaluated_dataset_configs()
    assert required, "the quality harness must declare at least one evaluation task"
    assert required <= covered, sorted(required - covered)
    # The benchmarks named in the W4A acceptance contract, including the
    # MMLU-Pro lane the harness does not yet score, must stay excluded.
    for pair in [("madrylab/gsm8k-platinum", "main"), ("openai/gsm8k", "main"),
                 ("TIGER-Lab/MMLU-Pro", "default"), ("allenai/ai2_arc", "ARC-Challenge")]:
        assert pair in covered


def test_require_evaluation_exclusions_fails_closed_on_uncovered_benchmark():
    from tests.models.w4a_calibration_data import require_evaluation_exclusions

    require_evaluation_exclusions([("madrylab/gsm8k-platinum", "main")], source="test")
    with pytest.raises(ValueError, match="does not cover"):
        require_evaluation_exclusions([("new/benchmark", "main")], source="test")


def test_load_rejects_artifact_that_misses_a_requested_evaluation(artifact):
    from tests.models.w4a_calibration_data import evaluated_dataset_configs

    load_calibration_artifact(artifact, required_evaluations=evaluated_dataset_configs())
    with pytest.raises(ValueError, match="does not cover"):
        load_calibration_artifact(artifact, required_evaluations=[("new/benchmark", "main")])


def test_load_default_asserts_harness_registry(artifact, monkeypatch):
    # A caller that omits required_evaluations must still fail closed once the
    # quality harness scores a benchmark the exclusion registry does not cover.
    from tests.models import w4a_quality_regression

    extended = dict(w4a_quality_regression.TASKS)
    extended["new_benchmark"] = ("new/benchmark", "main", "accuracy")
    monkeypatch.setattr(w4a_quality_regression, "TASKS", extended)
    with pytest.raises(ValueError, match="does not cover"):
        load_calibration_artifact(artifact)


def test_rejects_missing_exclusion_row_even_with_updated_hash(artifact):
    rows = (artifact / "exclusion_questions.jsonl").read_text().splitlines()
    (artifact / "exclusion_questions.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    _edit_manifest(artifact, lambda value: value["files"].update({
        "exclusion_questions.jsonl": file_digest(artifact / "exclusion_questions.jsonl")}))
    with pytest.raises(ValueError, match="exclusion questions"):
        load_calibration_artifact(artifact)


@pytest.mark.parametrize("partition", ["fit", "selection"])
def test_rechecks_overlap_even_with_updated_hashes(artifact, partition):
    def edit(rows):
        row = next(row for row in rows if row["partition"] == partition)
        row["text"] += " How many oranges remain after the farmer sells fifteen?"
        row["text_sha256"] = text_digest(row["text"])
    _edit_samples(artifact, edit)
    with pytest.raises(ValueError, match="overlaps evaluation"):
        load_calibration_artifact(artifact)


def test_rejects_article_shared_across_partitions(artifact):
    def edit(rows):
        fit = next(row for row in rows if row["partition"] == "fit")
        selected = next(row for row in rows if row["partition"] == "selection")
        selected["article_id"] = fit["article_id"]
    _edit_samples(artifact, edit)
    with pytest.raises(ValueError, match="Duplicate|different partition"):
        load_calibration_artifact(artifact)


def test_rejects_duplicate_content_under_different_article_id(artifact):
    def edit(rows):
        rows[1]["text"], rows[1]["text_sha256"] = rows[0]["text"], rows[0]["text_sha256"]
    _edit_samples(artifact, edit)
    with pytest.raises(ValueError, match="Duplicate"):
        load_calibration_artifact(artifact)


def test_selection_filters_contamination_before_returning():
    index = QuestionExclusionIndex([{"id": "test/0", "question": "Forbidden question?"}])
    candidates = [{"article_id": "bad", "source_rows": [9999], "text": "Forbidden question?"}, *_candidates()]
    records, audit = select_records(candidates, index, fit_rows=4, selection_rows=2, seed=787)
    assert all(row["article_id"] != "bad" for row in records)
    assert audit["overlap_rejections"] == [{"article_id": "bad", "evaluation_question_id": "test/0"}]


def test_selection_does_not_silently_return_too_few_rows():
    index = QuestionExclusionIndex([{"id": "test/0", "question": "a question"}])
    with pytest.raises(ValueError, match="Insufficient"):
        select_records([], index, fit_rows=4, selection_rows=2, seed=787)


def test_article_grouping_retains_source_ids_and_caps_words():
    rows = [{"text": text} for text in ["= First Article =", "alpha " * 300,
            "= = Section = =", "beta " * 300, "= Second Article =", "gamma " * 300]]
    articles = list(_wikitext_articles(rows, max_words=256))
    assert len(articles) == 2
    assert articles[0]["source_rows"] == [0, 1]
    assert articles[1]["source_rows"] == [4, 5]
    assert all(len(row["text"].split()) == 256 for row in articles)


def test_sample_tokenization_uses_requested_partition(artifact):
    from tests.models.w4a_nvfp4_norm_qat import _calibration_ids
    observed = []
    def tokenizer(text, **kwargs):
        observed.append(text)
        assert kwargs == {"add_special_tokens": True, "truncation": True, "max_length": 8}
        return {"input_ids": list(range(8))}
    partitions, _ = load_calibration_artifact(artifact)
    samples = _calibration_ids(tokenizer, artifact, 2, 8, partition="selection")
    assert observed == [row["text"] for row in partitions["selection"]]
    assert len(samples) == 2
    with pytest.raises(ValueError, match="artifact has"):
        _calibration_ids(tokenizer, artifact, 3, 8, partition="selection")


def test_checkpoint_provenance_does_not_modify_source_manifest(artifact, tmp_path):
    from tests.models.w4a_nvfp4_norm_qat import _copy_checkpoint_shell
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "w4a_calibration_manifest.json").write_text("old")
    _copy_checkpoint_shell(source, output, calibration=artifact)
    assert (source / "w4a_calibration_manifest.json").read_text() == "old"
    provenance = json.loads((output / "w4a_calibration_manifest.json").read_text())
    assert len(provenance["selected_samples"]) == 6
    assert all("text" not in row for row in provenance["selected_samples"])
    assert provenance["manifest_sha256"] == file_digest(artifact / "manifest.json")


@pytest.mark.parametrize("report", ["w4a_weight_qad_report.json", "w4a_scale_qad_report.json",
                                    "w4a_norm_qat_report.json", "w4a_scale_reconstruction_report.json",
                                    "w4a_producer_calibration_report.json", "w4a_producer_reconstruction_report.json"])
def test_derived_checkpoint_reports_cannot_overwrite_source_evidence(tmp_path, report):
    from tests.models.w4a_nvfp4_norm_qat import _copy_checkpoint_shell

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / report).write_text("original-run-evidence")
    (source / "config.json").write_text("{}")
    _copy_checkpoint_shell(source, output)
    assert not (output / report).exists() and not (output / report).is_symlink()
    (output / report).write_text("new-run-evidence")
    assert (source / report).read_text() == "original-run-evidence"
    assert (output / "config.json").is_symlink()


def test_1b_quantization_loader_requires_audited_artifact(monkeypatch):
    from test_llama3_2_w4a_nvfp4 import TestLlama3_2_W4ANVFP4 as W4ATest

    monkeypatch.delenv("GPTQMODEL_W4A_CALIBRATION_ARTIFACT", raising=False)
    with pytest.raises(ValueError, match="audited fit data"):
        W4ATest.load_dataset(rows=2)


@pytest.mark.parametrize("token_limit", [512, 2048])
def test_1b_quantization_loader_uses_only_fit_articles(artifact, monkeypatch, token_limit):
    from test_llama3_2_w4a_nvfp4 import TestLlama3_2_W4ANVFP4 as W4ATest

    monkeypatch.setenv("GPTQMODEL_W4A_CALIBRATION_ARTIFACT", str(artifact))
    monkeypatch.setenv("GPTQMODEL_W4A_CALIBRATION_MAX_TOKENS", str(token_limit))
    observed = []
    def tokenizer(text, **kwargs):
        observed.append(text)
        assert kwargs == {"add_special_tokens": True, "truncation": True, "max_length": token_limit}
        return {"input_ids": [1, 2, 3]}
    partitions, manifest = load_calibration_artifact(artifact)
    samples = W4ATest.load_dataset(tokenizer, rows=2)
    assert observed == [row["text"] for row in partitions["fit"][:2]]
    assert samples == [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}] * 2
    provenance = W4ATest._w4a_fit_provenance
    assert provenance["partition"] == "fit" and provenance["tokens_before_concatenation"] == 6
    assert provenance["manifest_sha256"] == manifest["manifest_sha256"]
    assert provenance["max_tokens_per_article"] == token_limit
    assert provenance["article_ids"] == [row["article_id"] for row in partitions["fit"][:2]]


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_1b_quantization_loader_rejects_invalid_token_limit(monkeypatch, value):
    from test_llama3_2_w4a_nvfp4 import TestLlama3_2_W4ANVFP4 as W4ATest

    monkeypatch.setenv("GPTQMODEL_W4A_CALIBRATION_MAX_TOKENS", value)
    with pytest.raises(ValueError):
        W4ATest.calibration_token_limit()


@pytest.fixture
def weight_qad_paths(tmp_path):
    source, teacher = tmp_path / "source", tmp_path / "teacher"
    source.mkdir()
    teacher.mkdir()
    (source / "model.safetensors").write_bytes(b"native-weight-fixture")
    (teacher / "model.safetensors").symlink_to(source / "model.safetensors")
    config = {"bits": 4, "group_size": 128, "sym": True, "desc_act": False,
              "pack_dtype": "int32", "quant_method": "gptq", "rotation": "hadamard"}
    for path in (source, teacher):
        (path / "quantize_config.json").write_text(json.dumps(config))
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (path / name).write_text("same-tokenizer-fixture")
    return teacher, source, tmp_path / "output", tmp_path / "cache"


def test_weight_qad_preflight_records_exact_disjoint_article_ids(artifact, weight_qad_paths):
    from tests.models.w4a_nvfp4_weight_qad import weight_qad_preflight

    teacher, source, output, cache = weight_qad_paths
    partitions, manifest = load_calibration_artifact(artifact)
    result = weight_qad_preflight(teacher, source, artifact, output, rows=3, validation_rows=1,
                                  teacher_cache_dir=cache)
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["fit_article_ids"] == [r["article_id"] for r in partitions["fit"][:3]]
    assert result["selection_article_ids"] == [r["article_id"] for r in partitions["selection"][:1]]
    assert not set(result["fit_article_ids"]) & set(result["selection_article_ids"])
    assert not output.exists() and not cache.exists()


def test_weight_qad_tokenizes_the_same_verified_snapshot(artifact, weight_qad_paths, monkeypatch):
    from tests.models import w4a_calibration_data as data
    from tests.models.w4a_nvfp4_norm_qat import _calibration_ids_from_verified_records
    from tests.models.w4a_nvfp4_weight_qad import weight_qad_preflight

    teacher, source, output, cache = weight_qad_paths
    calls = []
    real_loader = data.load_calibration_artifact
    def loader(path):
        calls.append(path)
        return real_loader(path)
    monkeypatch.setattr(data, "load_calibration_artifact", loader)
    report, records = weight_qad_preflight(teacher, source, artifact, output, rows=3,
                                          validation_rows=1, teacher_cache_dir=cache, include_records=True)
    seen = []
    def tokenizer(text, **kwargs):
        seen.append(text)
        return {"input_ids": [1, 2, 3, 4]}
    fit = _calibration_ids_from_verified_records(tokenizer, records, 3, 3)
    selection = _calibration_ids_from_verified_records(tokenizer, records, 1, 3, partition="selection")
    assert calls == [artifact]
    assert seen == [r["text"] for r in records["fit"][:3]] + [records["selection"][0]["text"]]
    assert report["fit_article_ids"] == [r["article_id"] for r in records["fit"][:3]]
    assert [x.tolist() for x in fit + selection] == [[1, 2, 3]] * 4


def test_chat_calibration_preserves_roles_and_avoids_duplicate_special_tokens():
    from tests.models.w4a_nvfp4_norm_qat import _calibration_ids_from_verified_records

    records = {"fit": [{"article_id": "example", "text": "unused plain text", "problems": [
        {"question": "What is 2 plus 3?", "reasoning": "2 + 3 = 5.", "answer": 5},
        {"question": "What is 7 minus 1?", "reasoning": "7 - 1 = 6.", "answer": 6},
    ]}]}
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            assert messages == [
                {"role": "user", "content": "What is 2 plus 3?"},
                {"role": "assistant", "content": "2 + 3 = 5. The result is 5."},
                {"role": "user", "content": "What is 7 minus 1?"},
                {"role": "assistant", "content": "7 - 1 = 6. The result is 6."},
            ]
            return "rendered-with-special-tokens"

        def __call__(self, text, **kwargs):
            assert text == "rendered-with-special-tokens"
            assert kwargs == {"add_special_tokens": False, "truncation": True, "max_length": 3}
            return {"input_ids": [1, 2, 3]}

    ids = _calibration_ids_from_verified_records(Tokenizer(), records, 1, 3,
                                                text_format="chat_worked_examples")
    assert ids[0].tolist() == [1, 2, 3]


@pytest.mark.parametrize("text_format", ["unknown", "chat_worked_examples"])
def test_calibration_format_rejects_unsupported_inputs(text_format):
    from tests.models.w4a_nvfp4_norm_qat import _calibration_ids_from_verified_records

    with pytest.raises(ValueError):
        _calibration_ids_from_verified_records(None, {"fit": [{"text": "plain"}]}, 1, 3,
                                               text_format=text_format)


@pytest.mark.parametrize("rendered", [None, "", [1, 2]])
def test_chat_calibration_rejects_invalid_template_output(rendered):
    from types import SimpleNamespace

    from tests.models.w4a_nvfp4_norm_qat import _calibration_ids_from_verified_records

    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: rendered)
    records = {"fit": [{"text": "plain", "problems": [
        {"question": "question", "reasoning": "reason", "answer": 1}]}]}
    with pytest.raises(ValueError, match="template returned"):
        _calibration_ids_from_verified_records(tokenizer, records, 1, 3,
                                               text_format="chat_worked_examples")


def test_chat_preflight_requires_audited_arithmetic_and_pins_template(artifact, weight_qad_paths, tmp_path):
    from tests.models.w4a_arithmetic_calibration import prepare
    from tests.models.w4a_nvfp4_weight_qad import weight_qad_preflight

    teacher, source, output, cache = weight_qad_paths
    options = {"rows": 3, "validation_rows": 1, "teacher_cache_dir": cache,
                   "calibration_format": "chat_worked_examples"}
    with pytest.raises(ValueError, match="audited arithmetic"):
        weight_qad_preflight(teacher, source, artifact, output, **options)
    arithmetic = tmp_path / "chat_arithmetic"
    prepare(arithmetic, artifact, fit_rows=4, selection_rows=2, seed=9431, examples=12)
    report = weight_qad_preflight(teacher, source, arithmetic, output, **options)
    assert report["calibration_format"] == "chat_worked_examples"
    assert report["chat_template_sha256"] == file_digest(teacher / "chat_template.jinja")
    assert not set(report["fit_article_ids"]) & set(report["selection_article_ids"])


@pytest.mark.parametrize("mismatch", ["weights", "tokenizer", "config", "output", "cache", "rows", "artifact"])
def test_weight_qad_preflight_rejects_invalid_run_before_loading(artifact, weight_qad_paths, mismatch):
    from tests.models.w4a_nvfp4_weight_qad import weight_qad_preflight

    teacher, source, output, cache = weight_qad_paths
    rows = 3
    if mismatch == "weights":
        (teacher / "model.safetensors").unlink()
        (teacher / "model.safetensors").write_bytes(b"different-weights")
    elif mismatch == "tokenizer":
        (teacher / "chat_template.jinja").write_text("different-template")
    elif mismatch == "config":
        config = json.loads((teacher / "quantize_config.json").read_text())
        config["group_size"] = 64
        (teacher / "quantize_config.json").write_text(json.dumps(config))
    elif mismatch == "output":
        output.mkdir()
    elif mismatch == "cache":
        cache.mkdir()
    elif mismatch == "rows":
        rows = 5
    elif mismatch == "artifact":
        artifact = artifact / "unchecked.parquet"
    with pytest.raises((ValueError, FileExistsError)):
        weight_qad_preflight(teacher, source, artifact, output, rows=rows, validation_rows=1,
                             teacher_cache_dir=cache)
