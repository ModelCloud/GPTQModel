# SPDX-License-Identifier: Apache-2.0

import json
import sys

from scripts.check_calibration_disjointness import main


def test_disjointness_audit_preserves_multiple_collision_groups(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.jsonl"
    d300 = tmp_path / "d300.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": text}]})
            for text in ("Alpha prompt", "Beta prompt")
        )
        + "\n",
        encoding="utf-8",
    )
    d300.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": text}]})
            for text in ("alpha prompt", "beta prompt")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_calibration_disjointness.py",
            "--calibration",
            str(calibration),
            "--d300",
            str(d300),
            "--output",
            str(output),
        ],
    )

    assert main() == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    keys = [key for key in payload["overlap_groups"] if key.startswith("d300:normalized:")]
    assert len(keys) == 2
    assert payload["status"] == "fail"
