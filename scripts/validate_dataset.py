#!/usr/bin/env python3
"""Validate the public 900-case ClinConsensus JSONL release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "case_id", "difficulty", "task", "task_labels", "subject",
    "subject_labels", "user_role", "clinical_context", "user_request", "rubrics",
}
FORBIDDEN_FIELDS = {"reference_answer"}
FORBIDDEN_FIELD_FRAGMENTS = (
    "source_file", "prompt_id", "row", "provenance", "physician", "doctor",
    "qc", "repair", "internal",
)
CONTACT_PATTERNS = {
    "email_like": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"),
    "phone_like": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card_like": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            rows.append(item)
    return rows


def validate(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    ids = [str(row.get("case_id", "")) for row in rows]
    difficulty = Counter(str(row.get("difficulty", "")) for row in rows)
    missing_fields = sorted(
        field for field in REQUIRED_FIELDS if any(field not in row for row in rows)
    )
    forbidden_fields = sorted({
        key for row in rows for key in row
        if key in FORBIDDEN_FIELDS
        or any(fragment in key.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS)
    })
    rubric_lengths = Counter(
        len(row.get("rubrics", [])) if isinstance(row.get("rubrics"), list) else -1
        for row in rows
    )
    contact_hits: dict[str, list[str]] = {}
    for kind, pattern in CONTACT_PATTERNS.items():
        values = sorted({
            match.group(0)
            for row in rows
            for match in pattern.finditer(json.dumps(row, ensure_ascii=False))
        })
        if values:
            contact_hits[kind] = values[:100]

    task_labels = Counter(
        str(label) for row in rows for label in row.get("task_labels", [])
    )
    subject_labels = Counter(
        str(label) for row in rows for label in row.get("subject_labels", [])
    )
    return {
        "file": path.name,
        "sha256": sha256(path),
        "records": len(rows),
        "unique_case_ids": len(set(ids)),
        "duplicate_case_ids": len(ids) - len(set(ids)),
        "difficulty_counts": dict(sorted(difficulty.items())),
        "total_rubrics": sum(len(row.get("rubrics", [])) for row in rows),
        "rubric_length_counts": dict(sorted((str(k), v) for k, v in rubric_lengths.items())),
        "distinct_task_labels": len(task_labels),
        "distinct_subject_labels": len(subject_labels),
        "missing_required_fields": missing_fields,
        "forbidden_field_keys": forbidden_fields,
        "contact_like_matches_for_manual_review": contact_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--strict-contact", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    errors: list[str] = []
    if report["records"] != 900 or report["unique_case_ids"] != 900:
        errors.append("release must contain exactly 900 unique cases")
    if report["difficulty_counts"] != {"low": 900}:
        errors.append("release must contain only 900 low-tier cases")
    if report["rubric_length_counts"] != {"30": 900} or report["total_rubrics"] != 27000:
        errors.append("every case must contain exactly 30 rubrics")
    if report["duplicate_case_ids"]:
        errors.append("duplicate case IDs")
    if report["missing_required_fields"]:
        errors.append("missing required fields")
    if report["forbidden_field_keys"]:
        errors.append("reference-answer or internal-looking fields present")
    if args.strict_contact and report["contact_like_matches_for_manual_review"]:
        errors.append("contact-like strings present")
    if errors:
        print("VALIDATION FAILED: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("VALIDATION PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
