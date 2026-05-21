#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


_PREFIX_RE = re.compile(
    "^\\s*(alternate\\s+answer|alternative\\s+answer|answer|alt)\\s*[:\\uFF1A-]\\s*",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, obj


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = _PREFIX_RE.sub("", text).strip()
    return text.strip(" \t\n\r\"'`")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip())


def token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.casefold()))
    b_tokens = set(re.findall(r"\w+", b.casefold()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def iter_alternates(value: Any, flatten: bool) -> list[str]:
    if flatten and isinstance(value, list):
        return [clean_text(item) for item in value]
    return [clean_text(value)]


def parse_index(value: Any, fallback: int, path: Path, line_no: int, key: str) -> int:
    raw_index = value if value is not None else fallback
    try:
        return int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_no}: index key {key!r} must be integer-like, got {raw_index!r}"
        ) from exc


def select_alternate_key(row: dict[str, Any], requested_key: str | None) -> str | None:
    if requested_key is not None:
        return requested_key
    if "sub_answer" in row:
        return "sub_answer"
    if "alternate" in row:
        return "alternate"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build AltPO-compatible JSONL artifacts from AltPO sub_answer files "
            "or existing counterfactual artifacts."
        )
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--index-key", default="index")
    parser.add_argument(
        "--alternate-key",
        default=None,
        help="Key containing the generated alternate. If omitted, tries sub_answer then alternate.",
    )
    parser.add_argument("--flatten-list-alternates", action="store_true")
    parser.add_argument("--max-alternates-per-row", type=int, default=5)
    parser.add_argument("--preserve-extra", action="store_true")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    parser.add_argument("--max-overlap-ratio", type=float, default=None)
    args = parser.parse_args()

    if args.max_alternates_per_row <= 0:
        raise ValueError("--max-alternates-per-row must be positive")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    dropped = 0
    seen_indices: set[int] = set()

    with args.output_path.open("w", encoding="utf-8") as output_handle:
        for line_no, row in read_jsonl(args.input_path):
            if args.question_key not in row:
                raise KeyError(
                    f"{args.input_path}:{line_no}: missing question key {args.question_key!r}"
                )
            if args.answer_key not in row:
                raise KeyError(
                    f"{args.input_path}:{line_no}: missing answer key {args.answer_key!r}"
                )

            alt_key = select_alternate_key(row, args.alternate_key)
            if alt_key is None:
                raise KeyError(
                    f"{args.input_path}:{line_no}: missing alternate source; pass --alternate-key"
                )
            if alt_key not in row:
                raise KeyError(f"{args.input_path}:{line_no}: missing alternate key {alt_key!r}")

            question = clean_text(row[args.question_key])
            answer = clean_text(row[args.answer_key])
            base_index = parse_index(
                row.get(args.index_key),
                fallback=line_no - 1,
                path=args.input_path,
                line_no=line_no,
                key=args.index_key,
            )
            alternates = iter_alternates(row[alt_key], flatten=args.flatten_list_alternates)
            alternates = alternates[: args.max_alternates_per_row]

            for alt_pos, alternate in enumerate(alternates):
                if not question or not answer or not alternate:
                    dropped += 1
                    continue

                norm_answer = normalize_for_match(answer)
                norm_alt = normalize_for_match(alternate)
                if norm_answer == norm_alt:
                    dropped += 1
                    continue

                if args.reject_gold_substring and (
                    norm_answer in norm_alt or norm_alt in norm_answer
                ):
                    dropped += 1
                    continue

                if args.require_short_answer and len(alternate) > args.max_alt_length_chars:
                    dropped += 1
                    continue

                if args.max_overlap_ratio is not None:
                    if token_overlap_ratio(answer, alternate) > args.max_overlap_ratio:
                        dropped += 1
                        continue

                output_index = (
                    base_index * 100 + alt_pos if args.flatten_list_alternates else base_index
                )
                if output_index in seen_indices:
                    dropped += 1
                    continue
                seen_indices.add(output_index)

                out_row = dict(row) if args.preserve_extra else {}
                out_row[args.index_key] = output_index
                out_row[args.question_key] = question
                out_row[args.answer_key] = answer
                out_row["alternate"] = alternate

                output_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

    print(
        json.dumps(
            {
                "input_path": str(args.input_path),
                "output_path": str(args.output_path),
                "written": written,
                "dropped": dropped,
            },
            indent=2,
        )
    )

    if written == 0:
        raise SystemExit("No rows written; check alternate key and filters.")


if __name__ == "__main__":
    main()
