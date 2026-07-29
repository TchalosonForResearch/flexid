#!/usr/bin/env python3
"""Split FLEXID instances into deterministic, article-spread annotation lots."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANNOTATOR_FIELDS = ("id", "premise", "hypothesis_facts", "law_ref")
DEFAULT_INPUT = Path("data/flexid.jsonl")
DEFAULT_OUT_DIR = Path("shuffled_and_batches")
PRIVATE_DROP_FIELDS = {
    "label",
    "rationale_start",
    "rationale_end",
    "rationale_text",
    "rationale",
    "span_start",
    "span_end",
    "span_text",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects.")
        records = data
    else:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Every instance must be a JSON object.")
    return records


def normalise_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    law_ref = raw.get("law_ref") or meta.get("law_ref") or ""
    hypothesis = raw.get("hypothesis_facts", raw.get("hypothesis", ""))

    record = dict(raw)
    record["id"] = str(raw.get("id", "")).strip()
    record["premise"] = str(raw.get("premise", "")).strip()
    record["hypothesis_facts"] = str(hypothesis).strip()
    record["law_ref"] = str(law_ref).strip()
    record["_source_index"] = index

    missing = [
        field
        for field in ("id", "premise", "hypothesis_facts", "law_ref")
        if not record[field]
    ]
    if missing:
        raise ValueError(
            f"Instance #{index + 1} is missing required field(s): {', '.join(missing)}"
        )
    return record


def article_key(record: dict[str, Any]) -> str:
    return record["law_ref"]


def choose_batch_combo(
    eligible: list[int],
    count: int,
    rng: random.Random,
    batch_loads: list[int],
    batch_size: int,
) -> tuple[int, ...] | None:
    if len(eligible) < count:
        return None

    best_combo: tuple[int, ...] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for combo in itertools.combinations(eligible, count):
        if count == 1:
            min_gap = len(batch_loads)
            gap_sum = len(batch_loads)
        else:
            gaps = [abs(a - b) for a, b in itertools.combinations(combo, 2)]
            min_gap = min(gaps)
            gap_sum = sum(gaps)

        loads_after = [batch_loads[idx] + 1 for idx in combo]
        max_load_after = max(loads_after)
        sum_sq_load_after = sum(load * load for load in loads_after)
        remaining_after = sum(batch_size - load for load in loads_after)
        jitter = rng.random()
        score = (
            -float(max_load_after),
            -float(sum_sq_load_after),
            float(min_gap),
            float(gap_sum),
            float(remaining_after),
            jitter,
        )

        if best_score is None or score > best_score:
            best_score = score
            best_combo = combo

    return best_combo


def assign_article_spread(
    records: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    max_attempts: int,
) -> list[list[dict[str, Any]]]:
    if len(records) % batch_size != 0:
        raise ValueError(
            f"{len(records)} instances cannot be split into full lots of {batch_size}. "
            "Use another batch size or add/remove instances."
        )

    batch_count = len(records) // batch_size
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[article_key(record)].append(record)

    too_large = {key: len(items) for key, items in grouped.items() if len(items) > batch_count}
    if too_large:
        examples = ", ".join(f"{key}={count}" for key, count in list(too_large.items())[:5])
        raise ValueError(
            "Some articles have more instances than there are lots, so one lot would "
            f"necessarily repeat an article: {examples}"
        )

    base_groups = list(grouped.items())
    last_error = "unknown assignment failure"

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        groups = [(key, list(items)) for key, items in base_groups]
        for _, items in groups:
            rng.shuffle(items)
        rng.shuffle(groups)
        groups.sort(key=lambda item: (-len(item[1]), rng.random()))

        batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
        batch_articles: list[set[str]] = [set() for _ in range(batch_count)]
        batch_loads = [0 for _ in range(batch_count)]

        try:
            for key, items in groups:
                eligible = [
                    idx
                    for idx in range(batch_count)
                    if batch_loads[idx] < batch_size and key not in batch_articles[idx]
                ]
                combo = choose_batch_combo(eligible, len(items), rng, batch_loads, batch_size)
                if combo is None:
                    raise RuntimeError(
                        f"no eligible batch combo for article {key!r} ({len(items)} item(s))"
                    )

                shuffled_combo = list(combo)
                rng.shuffle(shuffled_combo)
                for batch_idx, record in zip(shuffled_combo, items):
                    batches[batch_idx].append(record)
                    batch_articles[batch_idx].add(key)
                    batch_loads[batch_idx] += 1

            if all(load == batch_size for load in batch_loads):
                return batches

            last_error = f"unbalanced loads: {batch_loads}"
        except RuntimeError as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"Could not create article-spread batches after {max_attempts} attempts: {last_error}"
    )


def shuffle_batches_for_reading_order(
    batches: list[list[dict[str, Any]]], seed: int, attempts_per_batch: int = 2000
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled_batches: list[list[dict[str, Any]]] = []
    previous_last_key: str | None = None

    for batch in batches:
        best = list(batch)
        best_score = -1

        for _ in range(attempts_per_batch):
            candidate = list(batch)
            rng.shuffle(candidate)

            boundary_ok = previous_last_key is None or article_key(candidate[0]) != previous_last_key
            position_jitter = sum((idx + 1) * (rng.random() / 1000) for idx, _ in enumerate(candidate))
            score = (1000 if boundary_ok else 0) + position_jitter

            if score > best_score:
                best_score = score
                best = candidate
            if boundary_ok:
                break

        shuffled_batches.append(best)
        previous_last_key = article_key(best[-1])

    return shuffled_batches


def annotator_view(record: dict[str, Any], hide_law_ref: bool) -> dict[str, Any]:
    fields = ("id", "premise", "hypothesis_facts") if hide_law_ref else ANNOTATOR_FIELDS
    return {field: record[field] for field in fields}


def master_view(record: dict[str, Any], lot: int, position_in_lot: int, global_position: int) -> dict[str, Any]:
    out = {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }
    out["lot"] = lot
    out["position_in_lot"] = position_in_lot
    out["global_position"] = global_position
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_batches(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    duplicate_lots: list[dict[str, Any]] = []
    adjacent_pairs = 0
    positions_by_article: dict[str, list[int]] = defaultdict(list)
    global_position = 0
    previous_key: str | None = None

    for lot_idx, batch in enumerate(batches, start=1):
        counts = Counter(article_key(record) for record in batch)
        repeats = {key: count for key, count in counts.items() if count > 1}
        if repeats:
            duplicate_lots.append({"lot": lot_idx, "repeats": repeats})

        for record in batch:
            key = article_key(record)
            if previous_key == key:
                adjacent_pairs += 1
            positions_by_article[key].append(global_position)
            previous_key = key
            global_position += 1

    gaps = []
    for positions in positions_by_article.values():
        if len(positions) > 1:
            gaps.extend(b - a for a, b in zip(positions, positions[1:]))

    return {
        "duplicate_articles_inside_lots": duplicate_lots,
        "adjacent_same_article_pairs": adjacent_pairs,
        "min_gap_between_same_article": min(gaps) if gaps else None,
        "mean_gap_between_same_article": round(sum(gaps) / len(gaps), 2) if gaps else None,
    }


def label_distribution(batch: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("label", "")) for record in batch).items()))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def dataset_output_dir(base_out_dir: Path, input_path: Path) -> Path:
    return base_out_dir / input_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic FLEXID annotation lots from one JSONL/JSON file. "
            "Annotator files keep only id, premise, hypothesis_facts, and law_ref."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input JSONL/JSON file, relative to the project root by default. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Base output directory, relative to the project root by default. "
            f"Default: {DEFAULT_OUT_DIR}"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--expected-count", type=int, default=340)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument(
        "--hide-law-ref",
        action="store_true",
        help="Do not include law_ref in annotator lots.",
    )
    parser.add_argument(
        "--write-master",
        action="store_true",
        help="Also write a private master file with original fields plus lot positions.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write a manifest JSON file with seed and validation details.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    input_path = project_path(args.input, root)
    base_out_dir = project_path(args.out_dir, root)
    out_dir = dataset_output_dir(base_out_dir, input_path)

    raw_records = load_records(input_path)
    records = [normalise_record(record, idx) for idx, record in enumerate(raw_records)]

    if args.expected_count and len(records) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} instances, found {len(records)}.")

    ids = [record["id"] for record in records]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate ids found: {duplicate_ids[:10]}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output directory is not empty: {display_path(out_dir, root)}. Use --force to write anyway."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = assign_article_spread(records, args.batch_size, args.seed, args.max_attempts)
    batches = shuffle_batches_for_reading_order(batches, seed=args.seed + 10_000)

    total = len(records)
    annotator_files = []

    for lot_idx, batch in enumerate(batches, start=1):
        lot_name = f"lot{lot_idx}.jsonl"
        lot_path = out_dir / lot_name
        public_records = [annotator_view(record, args.hide_law_ref) for record in batch]
        write_jsonl(lot_path, public_records)
        annotator_files.append(lot_name)

    master_name = None
    if args.write_master:
        master_records = []
        global_position = 1
        for lot_idx, batch in enumerate(batches, start=1):
            for position_in_lot, record in enumerate(batch, start=1):
                master_records.append(master_view(record, lot_idx, position_in_lot, global_position))
                global_position += 1
        master_name = f"flexid_master_{total}.jsonl"
        write_jsonl(out_dir / master_name, master_records)

    validation = validate_batches(batches)
    manifest_path = None
    if args.write_manifest:
        manifest = {
            "input": display_path(input_path, root),
            "output_dir": display_path(out_dir, root),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "total_instances": total,
            "batch_count": len(batches),
            "logic": (
                "Deterministic article-aware spreading by law_ref, then deterministic "
                "shuffle inside each lot. Annotator files exclude labels and rationale spans."
            ),
            "annotator_fields": list(
                ("id", "premise", "hypothesis_facts") if args.hide_law_ref else ANNOTATOR_FIELDS
            ),
            "private_fields_removed_from_lots": sorted(PRIVATE_DROP_FIELDS),
            "annotator_files": annotator_files,
            "validation": validation,
            "label_distribution_private": {
                f"lot{idx}": label_distribution(batch) for idx, batch in enumerate(batches, start=1)
            },
        }
        if master_name:
            manifest["master_file"] = master_name
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {len(batches)} lots of {args.batch_size} instances to {display_path(out_dir, root)}")
    print(f"Seed: {args.seed}")
    print(f"Annotator files: lot1.jsonl ... lot{len(batches)}.jsonl")
    if master_name:
        print(f"Master file: {master_name}")
    if manifest_path:
        print(f"Manifest: {manifest_path.name}")
    print("Validation:")
    print(f"  duplicate articles inside lots: {len(validation['duplicate_articles_inside_lots'])}")
    print(f"  adjacent same-article pairs: {validation['adjacent_same_article_pairs']}")
    print(f"  min gap between same article: {validation['min_gap_between_same_article']}")
    print(f"  mean gap between same article: {validation['mean_gap_between_same_article']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
