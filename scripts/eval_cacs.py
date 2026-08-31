#!/usr/bin/env python3
"""Compute rubric accuracy, pass@k, and CACS@k from rubric-level judgments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "met"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def judgement_list(item: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("criteria_results", "rubric_results", "rubric_judgments"):
        value = item.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


def case_id_for_judgment(item: dict[str, Any], index: int) -> str:
    for key in ("case_id", "id"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    prompt_id = str(item.get("prompt_id", "")).strip()
    if prompt_id.isdigit():
        return f"CC-{int(prompt_id):04d}"
    return f"CC-{index + 1:04d}"


def cacs(scores: Iterable[int], rubric_count: int, threshold: int) -> float:
    values = list(scores)
    if not values or threshold > rubric_count:
        return 0.0
    return 100.0 * sum(
        sum(score >= level for score in values) / len(values)
        for level in range(threshold, rubric_count + 1)
    ) / (rubric_count - threshold + 1)


def aggregate(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "rubrics": 0, "positive": 0, "rubric_accuracy_pct": 0.0, "pass_at_k_pct": 0.0, "cacs_at_k_pct": 0.0}
    rubric_count = sum(int(row["rubric_count"]) for row in rows)
    positive = sum(int(row["positive_count"]) for row in rows)
    scores = [int(row["positive_count"]) for row in rows]
    # CACS is calibrated against the public benchmark definition (30
    # case-specific rubrics), not against accidental extra/missing rows in a
    # provider-generated judgement file.
    per_case_rubrics = max(int(row["expected_rubric_count"]) for row in rows)
    return {
        "cases": len(rows),
        "rubrics": rubric_count,
        "positive": positive,
        "rubric_accuracy_pct": 100.0 * positive / rubric_count if rubric_count else 0.0,
        "pass_at_k_pct": 100.0 * sum(score >= threshold for score in scores) / len(scores),
        "cacs_at_k_pct": cacs(scores, per_case_rubrics, threshold),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--judgements-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()

    cases = load_jsonl(args.data)
    case_map = {str(case["case_id"]): case for case in cases}
    if len(case_map) != len(cases):
        raise SystemExit("duplicate case_id values in public data")
    judge_files = sorted(args.judgements_dir.glob("*.json"))
    if not judge_files:
        raise SystemExit(f"no JSON judgement files found in {args.judgements_dir}")

    case_rows: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for judge_file in judge_files:
        model = re.sub(r"_judge\.json$", "", judge_file.name, flags=re.IGNORECASE)
        data = json.loads(judge_file.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != len(cases):
            raise SystemExit(f"row count mismatch: {judge_file} has {len(data) if isinstance(data, list) else 'non-list'} rows; expected {len(cases)}")
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise SystemExit(f"non-object judgement row: {judge_file}:{index + 1}")
            case_id = case_id_for_judgment(item, index)
            if case_id not in case_map:
                raise SystemExit(f"cannot resolve {judge_file}:{index + 1} to public case: {case_id}")
            judgments = judgement_list(item)
            positive = sum(truthy(entry.get("criteria_met")) for entry in judgments)
            case = case_map[case_id]
            row = {
                "model": model,
                "case_id": case_id,
                "difficulty": case.get("difficulty", ""),
                "subject_labels": "|".join(map(str, case.get("subject_labels", []))),
                "task_labels": "|".join(map(str, case.get("task_labels", []))),
                "rubric_count": len(judgments),
                "expected_rubric_count": len(case.get("rubrics", [])),
                "positive_count": positive,
                "rubric_accuracy_pct": 100.0 * positive / len(judgments) if judgments else 0.0,
                "pass_at_k": int(positive >= args.threshold),
                "cacs_at_k_pct": cacs([positive], len(case.get("rubrics", [])), args.threshold),
            }
            case_rows.append(row)
            summary_groups[(model, "overall")].append(row)
            summary_groups[(model, str(case.get("difficulty", "")))].append(row)
            for label in case.get("subject_labels", []):
                summary_groups[(model, f"subject:{label}")].append(row)

    summaries = []
    for (model, slice_name), rows in sorted(summary_groups.items()):
        summaries.append({"model": model, "slice": slice_name, "threshold": args.threshold, **aggregate(rows, args.threshold)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "case_scores.csv", case_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    manifest = {
        "data": {"path": str(args.data), "sha256": sha256(args.data), "cases": len(cases)},
        "judgements": [{"path": str(path), "sha256": sha256(path), "model": re.sub(r"_judge\.json$", "", path.name, flags=re.IGNORECASE)} for path in judge_files],
        "threshold": args.threshold,
        "files": ["case_scores.csv", "summary.csv", "run_manifest.json"],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "models": len(judge_files), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
