# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Build reproducible arithmetic text independently of benchmark content.

Questions and worked answers come from fixed arithmetic rules and seeded
integers. Evaluation questions are consulted only by the exclusion filter.
This supplies plain text to the existing tokenizer path; it does not alter
chat templates or introduce benchmark prompts into fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .w4a_calibration_data import (
    ARITHMETIC_CORPUS, QuestionExclusionIndex, article_partition, file_digest, load_calibration_artifact,
    save_artifact, select_records, text_digest,
)

KINDS = ("inventory", "bundles", "shares", "production", "discount", "ratio", "change", "distance")
ITEMS = ("tiles", "beads", "markers", "bolts", "stickers", "tickets")


def problem(kind: str, rng: random.Random) -> dict:
    item = rng.choice(ITEMS)
    if kind == "inventory":
        a, b = rng.randint(60, 400), rng.randint(10, 200)
        c = rng.randint(1, a + b - 1)
        answer = a + b - c
        question = f"A storeroom holds {a} {item}. A delivery adds {b}, and a shipment removes {c}. What count remains?"
        reasoning = f"After delivery there are {a} + {b} = {a+b}. Removing the shipment leaves {a+b} - {c} = {answer}."
        operands = [a, b, c]
    elif kind == "bundles":
        a, b = rng.randint(2, 20), rng.randint(3, 40)
        c = rng.randint(0, a * b - 1)
        answer = a * b - c
        question = f"There are {a} sealed packs with {b} {item} per pack. Inspection discards {c} individual {item}. Find the usable count."
        reasoning = f"Opening the packs gives {a} * {b} = {a*b}. Discarding {c} leaves {a*b} - {c} = {answer}."
        operands = [a, b, c]
    elif kind == "shares":
        groups, share = rng.randint(2, 12), rng.randint(5, 50)
        spare = rng.randint(0, groups - 1)
        total = groups * share + spare
        answer = share
        question = f"A box contains {total} {item}. Set aside {spare}, then distribute the rest equally among {groups} tables. What does each table receive?"
        reasoning = f"The distributable count is {total} - {spare} = {total-spare}. Division by {groups} gives {total-spare} / {groups} = {answer}."
        operands = [total, spare, groups]
    elif kind == "production":
        rate, first, second = rng.randint(5, 50), rng.randint(2, 12), rng.randint(2, 12)
        answer = rate * (first + second)
        question = f"A machine makes {rate} {item} each minute while operating. It operates for {first} minutes, pauses, then operates for {second} more minutes. Find its total production."
        reasoning = f"Operating time is {first} + {second} = {first+second} minutes. Production is {rate} * {first+second} = {answer}."
        operands = [rate, first, second]
    elif kind == "discount":
        price, percent = 20 * rng.randint(5, 200), 5 * rng.randint(1, 9)
        saving = price * percent // 100
        answer = price - saving
        question = f"A tool is priced at {price} dollars before a {percent} percent discount. There are no other fees. Find the discounted price in dollars."
        reasoning = f"The saving is {price} * {percent} / 100 = {saving} dollars. The price becomes {price} - {saving} = {answer} dollars."
        operands = [price, percent]
    elif kind == "ratio":
        first, second, unit = rng.randint(1, 8), rng.randint(1, 8), rng.randint(3, 40)
        total = (first + second) * unit
        answer = second * unit
        question = f"Red and blue {item} are counted in the ratio {first} to {second}. Their combined count is {total}. Find the blue count."
        reasoning = f"There are {first} + {second} = {first+second} ratio parts. Each part contains {total} / {first+second} = {unit}. Blue accounts for {second} * {unit} = {answer}."
        operands = [first, second, total]
    elif kind == "change":
        price, count, change = rng.randint(1, 40), rng.randint(2, 12), rng.randint(1, 80)
        cash = price * count + change
        answer = change
        question = f"Each entry ticket costs {price} dollars. A visitor buys {count} tickets and pays {cash} dollars. Find the change due in dollars."
        reasoning = f"The cost is {price} * {count} = {price*count} dollars. Change is {cash} - {price*count} = {answer} dollars."
        operands = [price, count, cash]
    elif kind == "distance":
        first, second, remaining = rng.randint(5, 90), rng.randint(5, 90), rng.randint(1, 100)
        total = first + second + remaining
        answer = remaining
        question = f"A route covers {total} kilometers. A traveler completes a {first} kilometer section followed by a {second} kilometer section. Find the distance still to cover."
        reasoning = f"Completed distance is {first} + {second} = {first+second} kilometers. The remainder is {total} - {first+second} = {answer} kilometers."
        operands = [total, first, second]
    else:
        raise ValueError(f"Unknown arithmetic family: {kind}")
    return {"kind": kind, "operands": operands, "question": question,
            "reasoning": reasoning, "answer": answer}


def article(seed: int, index: int, examples: int = 12) -> dict:
    if isinstance(examples, bool) or not 8 <= examples <= 32 or index < 0:
        raise ValueError("Arithmetic articles require 8–32 examples and a nonnegative index")
    identity = text_digest(f"procedural-arithmetic-v2 seed {seed} article {index}")
    partition = article_partition(identity, seed)
    rng = random.Random(f"gptqmodel-arithmetic-v2:{seed}:{index}")
    families = list(KINDS)
    rng.shuffle(families)
    families += [rng.choice(KINDS) for _ in range(examples - len(KINDS))]
    problems = []
    for kind in families:
        # A normalized question always belongs to the same role, even when
        # independently generated again inside another article. This prevents
        # accidental fitting/selection overlap in finite arithmetic domains.
        for attempt in range(1000):
            candidate = problem(kind, rng)
            if article_partition(text_digest(candidate["question"]), seed) == partition:
                problems.append(candidate)
                break
        else:
            raise ValueError("Could not generate an arithmetic question in the requested partition")
    text = "\n\n".join(f'Question: {row["question"]}\nAnswer: {row["reasoning"]} '
                         f'The result is {row["answer"]}.' for row in problems)
    return {"article_id": identity,
            "source_rows": [index], "generator_seed": seed, "problems": problems, "text": text}


def prepare(output: Path, exclusions: Path, *, fit_rows=4096, selection_rows=512,
            seed=9431, examples=12) -> dict:
    if output.exists():
        raise FileExistsError(output)
    # Validate the existing reference corpus before reusing its exclusion-only
    # questions and pinned split coverage. None of its calibration text is used.
    verified_records, reference = load_calibration_artifact(exclusions)
    del verified_records
    question_bytes = (exclusions / "exclusion_questions.jsonl").read_bytes()
    if hashlib.sha256(question_bytes).hexdigest() != reference["files"]["exclusion_questions.jsonl"]:
        raise ValueError("Exclusion questions changed after verification")
    questions = [json.loads(line) for line in question_bytes.splitlines()]
    index = QuestionExclusionIndex(questions)
    print(json.dumps({"stage": "arithmetic_generation", "exclusion_questions": len(questions),
                      "reference_manifest_sha256": reference["manifest_sha256"]}), flush=True)
    candidates = (article(seed, i, examples) for i in range(100 * (fit_rows + selection_rows)))
    records, audit = select_records(candidates, index, fit_rows=fit_rows, selection_rows=selection_rows, seed=seed)
    if file_digest(exclusions / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("Exclusion reference changed during arithmetic preparation")
    audit["reference_manifest_sha256"] = reference["manifest_sha256"]
    audit["problem_families"] = dict(Counter(p["kind"] for row in records for p in row["problems"]))
    identities = {role: {text_digest(p["question"]) for row in records if row["partition"] == role
                         for p in row["problems"]} for role in ("fit", "selection")}
    if identities["fit"] & identities["selection"]:
        raise AssertionError("Generated questions overlap fitting and selection")
    audit["question_partition"] = "normalized_question_sha256_seed_mod8_v1"
    audit["distinct_questions"] = {role: len(values) for role, values in identities.items()}
    audit["cross_partition_questions"] = 0
    return save_artifact(output, records, questions, reference["evaluation_coverage"], audit,
                         seed=seed, max_words=max(len(row["text"].split()) for row in records),
                         corpus=ARITHMETIC_CORPUS, generator_source=Path(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--fit-rows", type=int, default=4096)
    parser.add_argument("--selection-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9431)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    report = prepare(args.output, args.exclusions, fit_rows=args.fit_rows,
                     selection_rows=args.selection_rows, seed=args.seed, examples=args.examples)
    print(json.dumps({"output": str(args.output), "accepted": report["audit"]["accepted"],
                      "overlap_rejections": len(report["audit"]["overlap_rejections"]),
                      "accepted_overlap_count": report["audit"]["accepted_overlap_count"]}), flush=True)


if __name__ == "__main__":
    main()
