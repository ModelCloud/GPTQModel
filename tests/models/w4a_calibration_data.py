# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Prepare and verify benchmark-disjoint text for W4A calibration.

No model or CUDA is needed. Evaluation questions are used only for exclusion;
answers, model predictions, and evaluation scores never enter sample selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import re
import unicodedata


CORPUS = {
    "dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
    "split": "train", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
}
ARITHMETIC_CORPUS = {
    "dataset": "gptqmodel/procedural-arithmetic", "config": "worked-examples-v2",
    "split": "generated", "revision": "2",
}
# Include few-shot/development splits as well as scored rows. New evaluation
# tasks must extend this registry before using the artifact for calibration.
EXCLUSIONS = (
    ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", ("train", "test")),
    ("madrylab/gsm8k-platinum", "main", "e762492455a1cf7967de89f05b6bef72fc713b66", ("test",)),
    ("TIGER-Lab/MMLU-Pro", "default", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ("test", "validation")),
    ("cais/mmlu", "all", "c30699e8356da336a370243923dbaf21066bb9fe", ("test", "validation", "dev")),
    ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", ("test", "train", "validation")),
    ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453", ("test", "train", "validation")),
)
POLICY = {"normalization": "unicode_nfkc_casefold_alphanumeric_v1", "ngram_words": 13,
          "short_questions": "full_normalized_question", "partition": "article_sha256_seed_mod8_v1"}


def excluded_dataset_configs() -> set[tuple[str, str]]:
    """Dataset/config pairs the artifact builder refuses to use as calibration text."""
    return {(dataset, config) for dataset, config, _, _ in EXCLUSIONS}


def evaluated_dataset_configs() -> set[tuple[str, str]]:
    """Dataset/config pairs the W4A quality harness can score.

    Derived from the harness registry so a new evaluation task cannot be added
    without either covering it in EXCLUSIONS or failing the disjointness test.
    """
    from tests.models.w4a_quality_regression import TASKS

    return {(dataset, config) for dataset, config, _ in TASKS.values()}


def require_evaluation_exclusions(required: Iterable[tuple[str, str]], *, source: str) -> None:
    """Fail closed when a declared evaluation set is not excluded from calibration."""
    missing = sorted(set(required) - excluded_dataset_configs())
    if missing:
        raise ValueError(
            f"{source} evaluates {missing}, which the calibration exclusion registry does "
            "not cover; add the dataset and every scored/development split to EXCLUSIONS "
            "and rebuild the calibration artifact")


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def text_digest(text: str) -> str:
    return hashlib.sha256(" ".join(words(text)).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(tokens: list[str]) -> bytes:
    return hashlib.sha256(" ".join(tokens).encode()).digest()


class QuestionExclusionIndex:
    """Match complete short questions or any 13-word question excerpt.

    This detects normalized lexical overlap, not arbitrary paraphrases. False
    positives are conservatively excluded. Hash collisions also only exclude.
    """

    def __init__(self, questions: list[dict]):
        if not questions:
            raise ValueError("Evaluation exclusion questions are required")
        self.by_length: dict[int, dict[bytes, str]] = {}
        self.question_ids = set()
        for record in questions:
            identity, tokens = record["id"], words(record["question"])
            if not tokens or identity in self.question_ids:
                raise ValueError(f"Empty or duplicate evaluation question: {identity}")
            self.question_ids.add(identity)
            length = min(POLICY["ngram_words"], len(tokens))
            index = self.by_length.setdefault(length, {})
            for start in range(len(tokens) - length + 1):
                index.setdefault(_key(tokens[start:start + length]), identity)

    def match(self, text: str) -> str | None:
        tokens = words(text)
        for length, index in self.by_length.items():
            for start in range(len(tokens) - length + 1):
                match = index.get(_key(tokens[start:start + length]))
                if match is not None:
                    return match
        return None


def article_partition(article_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{article_id}".encode()).hexdigest(), 16)
    return "selection" if value % 8 == 0 else "fit"


def select_records(candidates, index: QuestionExclusionIndex, *, fit_rows: int,
                   selection_rows: int, seed: int) -> tuple[list[dict], dict]:
    if fit_rows <= 0 or selection_rows <= 0:
        raise ValueError("Both fit and selection sets must be nonempty")
    targets, counts = {"fit": fit_rows, "selection": selection_rows}, Counter()
    selected, rejected, seen_articles, seen_text = [], [], set(), set()
    for candidate in candidates:
        article_id, text = candidate["article_id"], candidate["text"]
        partition = article_partition(article_id, seed)
        if counts[partition] >= targets[partition]:
            continue
        digest = text_digest(text)
        if article_id in seen_articles or digest in seen_text:
            continue
        seen_articles.add(article_id)
        seen_text.add(digest)
        overlap = index.match(text)
        if overlap is not None:
            rejected.append({"article_id": article_id, "evaluation_question_id": overlap})
            continue
        selected.append({**candidate, "partition": partition, "text_sha256": digest})
        counts[partition] += 1
        if counts == targets:
            break
    if counts != targets:
        raise ValueError(f"Insufficient disjoint calibration articles: {dict(counts)} / {targets}")
    return selected, {"accepted": dict(counts), "overlap_rejections": rejected,
                      "accepted_overlap_count": 0}


def _write_jsonl(path: Path, records) -> None:
    with path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def save_artifact(output: Path, records: list[dict], questions: list[dict],
                  coverage: list[dict], audit: dict, *, seed: int, max_words: int,
                  corpus: dict | None = None, generator_source: Path | None = None) -> dict:
    corpus = CORPUS if corpus is None else corpus
    if corpus not in (CORPUS, ARITHMETIC_CORPUS):
        raise ValueError("Unknown calibration corpus")
    generated = corpus == ARITHMETIC_CORPUS
    if generated != (generator_source is not None):
        raise ValueError("Generated arithmetic requires its generator source snapshot")
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "samples.jsonl", records)
    # Exclusion-only reference data, never returned as calibration samples.
    _write_jsonl(output / "exclusion_questions.jsonl", questions)
    filenames = ["samples.jsonl", "exclusion_questions.jsonl"]
    if generator_source is not None:
        (output / "generator.py").write_bytes(generator_source.read_bytes())
        filenames.append("generator.py")
    manifest = {
        "version": 1, "corpus": corpus, "policy": POLICY, "seed": seed,
        "max_words_per_article": max_words, "evaluation_coverage": coverage,
        "audit": audit,
        "files": {name: file_digest(output / name) for name in filenames},
    }
    # Write the completion marker last. Interrupted preparation cannot be read.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_calibration_artifact(
    directory: Path,
    *,
    required_evaluations: Iterable[tuple[str, str]] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Fail closed, including on legacy unchecked parquet input.

    Recheck file hashes, reference coverage, article/content disjointness and
    question overlap before returning any fitting or selection text. When
    required_evaluations is given, the artifact must have been built to exclude
    every one of those dataset/config pairs; otherwise loading fails closed.

    The default is the quality harness registry, so a caller that forgets to
    declare its evaluation set still cannot fit on a benchmark the harness can
    score. A new harness task fails closed here until it is added to EXCLUSIONS.
    """
    directory = Path(directory)
    if not directory.is_dir() or not (directory / "manifest.json").is_file():
        raise ValueError("W4A calibration requires a verified dataset directory with manifest.json; "
                         "prepare it with tests.models.w4a_calibration_data")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("version") != 1 or manifest.get("corpus") not in (CORPUS, ARITHMETIC_CORPUS)
            or manifest.get("policy") != POLICY):
        raise ValueError("Unknown or stale calibration dataset policy/corpus")
    expected = {(dataset, config, revision, split)
                for dataset, config, revision, splits in EXCLUSIONS for split in splits}
    coverage = manifest["evaluation_coverage"]
    actual = {(row["dataset"], row["config"], row["revision"], row["split"]) for row in coverage}
    if actual != expected or len(actual) != len(coverage) or any(row["rows"] <= 0 for row in coverage):
        raise ValueError("Incomplete evaluation exclusion coverage")
    required = evaluated_dataset_configs() if required_evaluations is None else required_evaluations
    require_evaluation_exclusions(required, source="the requested evaluation set")
    filenames = ["samples.jsonl", "exclusion_questions.jsonl"]
    if manifest["corpus"] == ARITHMETIC_CORPUS:
        filenames.append("generator.py")
    for name in filenames:
        if file_digest(directory / name) != manifest["files"].get(name):
            raise ValueError(f"Calibration artifact hash mismatch: {name}")
    questions = _read_jsonl(directory / "exclusion_questions.jsonl")
    expected_ids = {f'{row["dataset"]}/{row["config"]}/{row["split"]}/{i}'
                    for row in coverage for i in range(row["rows"])}
    if {row["id"] for row in questions} != expected_ids:
        raise ValueError("Incomplete evaluation exclusion questions")
    index = QuestionExclusionIndex(questions)
    partitions: dict[str, list[dict]] = {"fit": [], "selection": []}
    seen_articles, seen_text = set(), set()
    for sample in _read_jsonl(directory / "samples.jsonl"):
        article_id, partition = sample["article_id"], sample["partition"]
        digest = text_digest(sample["text"])
        if (not words(sample["text"]) or digest != sample["text_sha256"]
                or digest in seen_text or article_id in seen_articles):
            raise ValueError("Duplicate or corrupted calibration sample")
        if partition != article_partition(article_id, manifest["seed"]):
            raise ValueError("Calibration article belongs to a different partition")
        if manifest["corpus"] == ARITHMETIC_CORPUS:
            problems = sample.get("problems", [])
            if not 8 <= len(problems) <= 32 or any(
                    article_partition(text_digest(p["question"]), manifest["seed"]) != partition
                    for p in problems):
                raise ValueError("Arithmetic question belongs to a different partition")
            rendered = "\n\n".join(f'Question: {p["question"]}\nAnswer: {p["reasoning"]} '
                                   f'The result is {p["answer"]}.' for p in problems)
            if sample["text"] != rendered:
                raise ValueError("Arithmetic text differs from its audited questions")
        overlap = index.match(sample["text"])
        if overlap is not None:
            raise ValueError(f"Calibration overlaps evaluation: {article_id} -> {overlap}")
        seen_articles.add(article_id)
        seen_text.add(digest)
        partitions[partition].append(sample)
    counts = {key: len(value) for key, value in partitions.items()}
    if (any(count <= 0 for count in counts.values()) or counts != manifest["audit"]["accepted"]
            or manifest["audit"]["accepted_overlap_count"] != 0):
        raise ValueError("Calibration sample counts do not match the completed audit")
    manifest = {**manifest, "manifest_sha256": file_digest(directory / "manifest.json")}
    return partitions, manifest


def _wikitext_articles(rows, *, max_words: int):
    """One bounded text sample per article; never split an article across roles."""
    article_id, chunks, source_rows, count = None, [], [], 0
    for row_id, row in enumerate(rows):
        text = row["text"].strip()
        if re.fullmatch(r"= [^=]+ =", text):
            if article_id is not None and count >= 256:
                yield {"article_id": article_id, "source_rows": source_rows, "text": "\n".join(chunks)}
            # Use normalized title so duplicated articles cannot enter both sets.
            article_id = text_digest(text)
            chunks, source_rows, count = [], [], 0
        if article_id is None or not text or count >= max_words:
            continue
        tokens = text.split()
        chunk = tokens[:max_words - count]
        chunks.append(" ".join(chunk))
        source_rows.append(row_id)
        count += len(chunk)
    if article_id is not None and count >= 256:
        yield {"article_id": article_id, "source_rows": source_rows, "text": "\n".join(chunks)}


def prepare(output: Path, *, fit_rows=512, selection_rows=64, seed=787, max_words=4096) -> dict:
    from datasets import load_dataset

    if output.exists():
        raise FileExistsError(output)
    if max_words < 256:
        raise ValueError("max_words must be at least 256")
    questions, coverage = [], []
    for dataset, config, revision, splits in EXCLUSIONS:
        for split in splits:
            rows = load_dataset(dataset, config, revision=revision, split=split)
            coverage.append({"dataset": dataset, "config": config, "revision": revision,
                             "split": split, "rows": len(rows)})
            for row_id, row in enumerate(rows):
                questions.append({"id": f"{dataset}/{config}/{split}/{row_id}", "question": row["question"]})
            print(f"Exclusion coverage: {dataset}/{config}/{split}: {len(rows)}", flush=True)
    index = QuestionExclusionIndex(questions)
    rows = load_dataset(CORPUS["dataset"], CORPUS["config"], revision=CORPUS["revision"],
                        split=CORPUS["split"], streaming=True)
    records, audit = select_records(_wikitext_articles(rows, max_words=max_words), index,
                                    fit_rows=fit_rows, selection_rows=selection_rows, seed=seed)
    return save_artifact(output, records, questions, coverage, audit, seed=seed, max_words=max_words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--fit-rows", type=int, default=512)
    build.add_argument("--selection-rows", type=int, default=64)
    build.add_argument("--seed", type=int, default=787)
    build.add_argument("--max-words", type=int, default=4096)
    verify = sub.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        report = prepare(args.output, fit_rows=args.fit_rows, selection_rows=args.selection_rows,
                         seed=args.seed, max_words=args.max_words)
    else:
        _, report = load_calibration_artifact(args.directory)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
