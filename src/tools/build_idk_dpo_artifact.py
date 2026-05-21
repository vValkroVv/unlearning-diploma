#!/usr/bin/env python3
"""Build IdkDPO JSONL artifacts from an existing QA/DualCF-style JSONL file.

The output keeps the original question and answer, rewrites the preferred
``alternate`` answer to an IDK template, and preserves extra metadata. This lets
repo-native DPO train with preferred="I don't know." and rejected=original.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--template", default="I don't know.")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if args.fail_on_empty and not rows:
        raise ValueError(f"No rows read from {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(rows):
            missing = [key for key in (args.question_key, args.answer_key) if key not in row]
            if missing:
                raise KeyError(f"row {row_idx}: missing required keys {missing}")
            output_row = dict(row)
            output_row[args.alternate_key] = args.template
            output_row["idk_template"] = args.template
            output_row["alternate_source"] = "idk_template"
            output_row.setdefault("index", row_idx)
            # Keep DualCF dataset metadata requirements satisfied for plain QA inputs.
            output_row.setdefault("difficulty_score", 0.0)
            output_row.setdefault("attribution_score", 0.0)
            output_row.setdefault("rarity_score", 0.0)
            out.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"[build_idk_dpo_artifact] wrote rows={len(rows)} to {args.output_path}")


if __name__ == "__main__":
    main()
