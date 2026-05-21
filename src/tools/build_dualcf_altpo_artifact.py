#!/usr/bin/env python3
"""Compose a DualCF artifact that uses AltPO-generated alternates.

The output preserves the DualCF row contract and routing metadata from
``--dualcf-path`` while replacing only ``alternate`` from an AltPO generation
file matched by original/source index. This is intended for ablations such as
"DualCF routing/weights with AltPO alternates".
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_PREFIX_RE = re.compile(
    r"^\s*(alternate\s+answer|alternative\s+answer|answer|alt)\s*[:\uFF1A-]\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AltCandidate:
    source_index: int
    repeat: int
    alternate: str
    line_no: int
    seed: Any


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, row


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def parse_int(value: Any, *, path: Path, line_no: int, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_no}: key {key!r} must be integer-like, got {value!r}"
        ) from exc


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = _PREFIX_RE.sub("", text).strip()
    return text.strip(" \t\r\n\"'`")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip())


def token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.casefold()))
    b_tokens = set(re.findall(r"\w+", b.casefold()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def choose_alt_key(row: dict[str, Any], requested_key: str | None) -> str | None:
    if requested_key:
        return requested_key
    if "sub_answer" in row:
        return "sub_answer"
    if "alternate" in row:
        return "alternate"
    return None


def infer_source_index(
    row: dict[str, Any],
    *,
    path: Path,
    line_no: int,
    source_index_key: str,
    altpo_index_key: str,
    altpo_repeat_key: str,
    repeats: int,
) -> int:
    if source_index_key in row and row[source_index_key] is not None:
        return parse_int(row[source_index_key], path=path, line_no=line_no, key=source_index_key)

    if altpo_index_key not in row:
        return line_no - 1

    altpo_index = parse_int(row[altpo_index_key], path=path, line_no=line_no, key=altpo_index_key)
    repeat = None
    if altpo_repeat_key in row and row[altpo_repeat_key] is not None:
        repeat = parse_int(row[altpo_repeat_key], path=path, line_no=line_no, key=altpo_repeat_key)

    if repeat is not None and repeats > 0:
        return (altpo_index - repeat) // repeats
    if repeats > 0:
        return altpo_index // repeats
    return altpo_index


def infer_repeat(
    row: dict[str, Any],
    *,
    path: Path,
    line_no: int,
    altpo_index_key: str,
    altpo_repeat_key: str,
    repeats: int,
) -> int:
    if altpo_repeat_key in row and row[altpo_repeat_key] is not None:
        return parse_int(row[altpo_repeat_key], path=path, line_no=line_no, key=altpo_repeat_key)
    if altpo_index_key in row and repeats > 0:
        altpo_index = parse_int(row[altpo_index_key], path=path, line_no=line_no, key=altpo_index_key)
        return altpo_index % repeats
    return 0


def collect_altpo_candidates(args: argparse.Namespace) -> dict[int, list[AltCandidate]]:
    candidates: dict[int, list[AltCandidate]] = defaultdict(list)
    missing_alt_key = 0
    empty_alternates = 0

    for line_no, row in read_jsonl(args.altpo_path):
        alt_key = choose_alt_key(row, args.altpo_alternate_key)
        if alt_key is None or alt_key not in row:
            missing_alt_key += 1
            continue
        alternate = clean_text(row[alt_key])
        if not alternate:
            empty_alternates += 1
            continue

        source_index = infer_source_index(
            row,
            path=args.altpo_path,
            line_no=line_no,
            source_index_key=args.source_index_key,
            altpo_index_key=args.altpo_index_key,
            altpo_repeat_key=args.altpo_repeat_key,
            repeats=args.repeats,
        )
        repeat = infer_repeat(
            row,
            path=args.altpo_path,
            line_no=line_no,
            altpo_index_key=args.altpo_index_key,
            altpo_repeat_key=args.altpo_repeat_key,
            repeats=args.repeats,
        )
        candidates[source_index].append(
            AltCandidate(
                source_index=source_index,
                repeat=repeat,
                alternate=alternate,
                line_no=line_no,
                seed=row.get("altpo_seed"),
            )
        )

    if missing_alt_key:
        print(f"[build_dualcf_altpo] skipped rows missing alternate key: {missing_alt_key}")
    if empty_alternates:
        print(f"[build_dualcf_altpo] skipped rows with empty alternates: {empty_alternates}")

    for source_index in candidates:
        candidates[source_index].sort(key=lambda cand: (cand.repeat, cand.line_no))
    return candidates


def candidate_is_valid(
    candidate: AltCandidate,
    answer: str,
    *,
    reject_gold_substring: bool,
    max_overlap_ratio: float | None,
) -> tuple[bool, str | None]:
    alternate_norm = normalize_for_match(candidate.alternate)
    answer_norm = normalize_for_match(answer)
    if not alternate_norm:
        return False, "empty"
    if alternate_norm == answer_norm:
        return False, "matches_gold"
    if reject_gold_substring and answer_norm and answer_norm in alternate_norm:
        return False, "contains_gold"
    if max_overlap_ratio is not None:
        overlap = token_overlap_ratio(answer, candidate.alternate)
        if overlap > max_overlap_ratio:
            return False, f"overlap>{max_overlap_ratio}"
    return True, None


def select_candidate(
    candidates: list[AltCandidate],
    answer: str,
    args: argparse.Namespace,
    invalid_counts: Counter[str],
) -> tuple[AltCandidate | None, bool]:
    if args.altpo_repeat >= 0:
        exact = [cand for cand in candidates if cand.repeat == args.altpo_repeat]
        ordered = exact + [cand for cand in candidates if cand.repeat != args.altpo_repeat]
    else:
        ordered = candidates

    for cand in ordered:
        valid, reason = candidate_is_valid(
            cand,
            answer,
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
        )
        if not valid:
            invalid_counts[reason or "invalid"] += 1
            continue
        used_fallback = args.altpo_repeat >= 0 and cand.repeat != args.altpo_repeat
        return cand, used_fallback

    return None, False


def update_range(ranges: dict[str, list[float]], key: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return
    ranges[key][0] = min(ranges[key][0], float(value))
    ranges[key][1] = max(ranges[key][1], float(value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace DualCF alternates with matched AltPO alternates."
    )
    parser.add_argument("--dualcf-path", required=True, type=Path)
    parser.add_argument("--altpo-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--dualcf-index-key", default="index")
    parser.add_argument("--source-index-key", default="source_index")
    parser.add_argument("--altpo-index-key", default="index")
    parser.add_argument("--altpo-repeat-key", default="altpo_repeat")
    parser.add_argument("--altpo-alternate-key", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--altpo-repeat",
        type=int,
        default=0,
        help="Preferred AltPO repeat to select. Use -1 for first valid repeat.",
    )
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--keep-original-alternate-key",
        default="dualcf_original_alternate",
        help="Store the original DualCF alternate under this key. Set empty to disable.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not args.dualcf_path.exists():
        raise FileNotFoundError(f"DualCF artifact not found: {args.dualcf_path}")
    if not args.altpo_path.exists():
        raise FileNotFoundError(f"AltPO artifact not found: {args.altpo_path}")

    altpo_by_source = collect_altpo_candidates(args)
    output_rows: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    fallback_count = 0
    invalid_counts: Counter[str] = Counter()
    repeat_counts: Counter[int] = Counter()
    ranges = {
        "difficulty_score": [float("inf"), float("-inf")],
        "attribution_score": [float("inf"), float("-inf")],
        "rarity_score": [float("inf"), float("-inf")],
    }

    dualcf_rows = 0
    for line_no, row in read_jsonl(args.dualcf_path):
        dualcf_rows += 1
        for key in (
            args.dualcf_index_key,
            args.question_key,
            args.answer_key,
            "alternate",
            "difficulty_score",
            "attribution_score",
        ):
            if key not in row:
                raise KeyError(f"{args.dualcf_path}:{line_no}: missing required key {key!r}")

        source_index = parse_int(
            row[args.dualcf_index_key],
            path=args.dualcf_path,
            line_no=line_no,
            key=args.dualcf_index_key,
        )
        candidates = altpo_by_source.get(source_index, [])
        candidate, used_fallback = select_candidate(
            candidates,
            clean_text(row[args.answer_key]),
            args,
            invalid_counts,
        )
        if candidate is None:
            missing_indices.append(source_index)
            if not args.allow_missing:
                continue
            output_rows.append(dict(row))
            continue

        updated = dict(row)
        if args.keep_original_alternate_key:
            updated[args.keep_original_alternate_key] = row["alternate"]
        updated["alternate"] = candidate.alternate
        updated["dualcf_altpo_source_index"] = candidate.source_index
        updated["dualcf_altpo_repeat"] = candidate.repeat
        updated["dualcf_altpo_source_line"] = candidate.line_no
        if candidate.seed is not None:
            updated["dualcf_altpo_seed"] = candidate.seed

        for score_key in ranges:
            update_range(ranges, score_key, updated.get(score_key))

        repeat_counts[candidate.repeat] += 1
        fallback_count += int(used_fallback)
        output_rows.append(updated)

    if missing_indices and not args.allow_missing:
        sample = missing_indices[:20]
        raise RuntimeError(
            "Missing valid AltPO alternate for "
            f"{len(missing_indices)} DualCF rows; sample indices={sample}. "
            "Check seed/repeats/source_index alignment or use --allow-missing."
        )

    write_jsonl(output_rows, args.output_path)

    summary = {
        "dualcf_path": str(args.dualcf_path),
        "altpo_path": str(args.altpo_path),
        "output_path": str(args.output_path),
        "dualcf_rows": dualcf_rows,
        "rows_written": len(output_rows),
        "altpo_source_indices": len(altpo_by_source),
        "missing_indices_count": len(missing_indices),
        "missing_indices_sample": missing_indices[:20],
        "selected_repeat_counts": dict(sorted(repeat_counts.items())),
        "fallback_selected_count": fallback_count,
        "invalid_candidate_counts": dict(sorted(invalid_counts.items())),
        "score_ranges": {
            key: tuple(value)
            for key, value in ranges.items()
            if value[0] != float("inf")
        },
    }
    summary_path = args.output_path.with_suffix(args.output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
