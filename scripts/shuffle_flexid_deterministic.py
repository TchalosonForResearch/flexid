#!/usr/bin/env python3


from __future__ import annotations

import csv
import hashlib
import heapq
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "flexid.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "flexid_shuffled.jsonl"
POSITIONS_CSV = PROJECT_ROOT / "data" / "flexid_shuffle_positions.csv"
AUDIT_JSON = PROJECT_ROOT / "data" / "flexid_shuffle_audit.json"

SEED = 42

ORDER_ATTEMPTS = 200


FEASIBILITY_ATTEMPTS = 12


TARGET_MIN_DISPLACEMENT = 20

ID_FIELD = "id"
LAW_REF_FIELD = "law_ref"


# ============================================================================
# LECTURE ET ÉCRITURE
# ============================================================================

def load_records(path: Path) -> list[dict[str, Any]]:
    """Charge un fichier JSONL ou, par compatibilité, un tableau JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Le fichier d'entrée est vide : {path}")

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Le fichier JSON doit contenir une liste.")
        records = data
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide à la ligne {line_number} : {exc}"
                ) from exc

    if not records:
        raise ValueError("Aucune instance n'a été chargée.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Chaque instance doit être un objet JSON.")

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Écrit une instance JSON par ligne."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================================
# IDENTIFIANTS ET RÉFÉRENCES
# ============================================================================

def record_id(record: dict[str, Any], index: int) -> str:
    value = str(record.get(ID_FIELD, "")).strip()
    if not value:
        raise ValueError(f"Identifiant absent à la position {index + 1}.")
    return value


def law_reference(record: dict[str, Any]) -> str:
    """Lit law_ref au niveau racine ou dans meta.law_ref."""
    direct = record.get(LAW_REF_FIELD)
    if direct is not None and str(direct).strip():
        return str(direct).strip()

    meta = record.get("meta")
    if isinstance(meta, dict):
        nested = meta.get(LAW_REF_FIELD)
        if nested is not None and str(nested).strip():
            return str(nested).strip()

    return ""


def article_keys(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Retourne :
    - les clés internes utilisées pour l'espacement ;
    - les law_ref visibles utilisées dans les rapports.

    Une instance sans law_ref reçoit une clé unique afin que toutes les valeurs
    vides ne soient pas artificiellement considérées comme le même article.
    """
    keys: list[str] = []
    visible_refs: list[str] = []

    for index, record in enumerate(records):
        ref = law_reference(record)
        visible_refs.append(ref)
        keys.append(ref if ref else f"__NO_LAW_REF__{index}")

    return keys, visible_refs


def canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def dataset_hash(records: list[dict[str, Any]]) -> str:
    """Empreinte indépendante de l'ordre."""
    canonical = sorted(canonical_record(record) for record in records)
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


# ============================================================================
# CONSTRUCTION D'UN ORDRE AVEC DISTANCE MINIMALE
# ============================================================================

def build_article_sequence(
    counts: Counter[str],
    minimum_gap: int,
    rng: random.Random,
) -> list[str] | None:
    """
    Construit une séquence dans laquelle deux occurrences d'une même clé sont
    séparées d'au moins `minimum_gap` positions.

    Exemple : minimum_gap=2 interdit les occurrences adjacentes.
    """
    available: list[tuple[int, float, str]] = []
    cooldown: list[tuple[int, int, float, str]] = []

    for key, count in counts.items():
        heapq.heappush(available, (-count, rng.random(), key))

    sequence: list[str] = []
    total = sum(counts.values())

    for position in range(total):
        while cooldown and cooldown[0][0] <= position:
            _, negative_count, tie, key = heapq.heappop(cooldown)
            heapq.heappush(available, (negative_count, tie, key))

        if not available:
            return None

        negative_count, _, key = heapq.heappop(available)
        sequence.append(key)

        remaining = -negative_count - 1
        if remaining > 0:
            release_position = position + minimum_gap
            heapq.heappush(
                cooldown,
                (release_position, -remaining, rng.random(), key),
            )

    return sequence


def maximum_theoretical_gap(counts: Counter[str], total: int) -> int:
    repeated_counts = [count for count in counts.values() if count > 1]
    if not repeated_counts:
        return total

    maximum_count = max(repeated_counts)
    return (total - 1) // (maximum_count - 1)


def sequence_is_possible(
    counts: Counter[str],
    minimum_gap: int,
    seed: int,
    attempts: int,
) -> bool:
    for attempt in range(attempts):
        rng = random.Random(seed + minimum_gap * 1_000_003 + attempt)
        if build_article_sequence(counts, minimum_gap, rng) is not None:
            return True
    return False


def find_maximum_feasible_gap(
    counts: Counter[str],
    total: int,
    seed: int,
    attempts: int,
) -> int:
    """Recherche binaire de la plus grande distance minimale réalisable."""
    if all(count == 1 for count in counts.values()):
        return total

    low = 1
    high = maximum_theoretical_gap(counts, total)
    best = 1

    while low <= high:
        middle = (low + high) // 2
        if sequence_is_possible(counts, middle, seed, attempts):
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    return best


# ============================================================================
# AFFECTATION DES INSTANCES AUX POSITIONS DE LEUR ARTICLE
# ============================================================================

def local_assignment_score(
    old_positions: list[int],
    new_positions: list[int],
) -> tuple[float, ...]:
    shifts = [
        abs(new_position - old_position)
        for old_position, new_position in zip(old_positions, new_positions)
    ]
    return (
        -float(sum(shift == 0 for shift in shifts)),
        -float(sum(shift < TARGET_MIN_DISPLACEMENT for shift in shifts)),
        float(min(shifts)),
        float(statistics.mean(shifts)),
    )


def assign_group_positions(
    old_positions: list[int],
    target_positions: list[int],
    rng: random.Random,
) -> dict[int, int]:
    """
    Associe les anciennes positions aux nouvelles positions du même article.
    Plusieurs affectations déterministes sont testées pour maximiser le
    déplacement sans modifier l'espacement entre articles.
    """
    old_sorted = sorted(old_positions)
    targets_sorted = sorted(target_positions)
    size = len(old_sorted)

    if size == 1:
        return {targets_sorted[0]: old_sorted[0]}

    candidates: list[list[int]] = []
    candidates.append(list(reversed(targets_sorted)))
    candidates.append(list(targets_sorted))

    # Décalages cycliques de l'ordre direct et inversé.
    if size <= 50:
        for shift in range(size):
            candidates.append(
                targets_sorted[shift:] + targets_sorted[:shift]
            )
            reversed_targets = list(reversed(targets_sorted))
            candidates.append(
                reversed_targets[shift:] + reversed_targets[:shift]
            )

    # Quelques permutations supplémentaires uniquement pour les groupes
    # exceptionnellement grands. Les petits groupes sont déjà bien couverts
    # par les ordres direct, inversé et les décalages cycliques.
    if size > 50:
        for _ in range(20):
            candidate = list(targets_sorted)
            rng.shuffle(candidate)
            candidates.append(candidate)

    best_targets = candidates[0]
    best_score = local_assignment_score(old_sorted, best_targets)

    for candidate in candidates[1:]:
        score = local_assignment_score(old_sorted, candidate)
        if score > best_score:
            best_score = score
            best_targets = candidate

    return {
        new_position: old_position
        for old_position, new_position in zip(old_sorted, best_targets)
    }


def sequence_to_permutation(
    sequence: list[str],
    keys: list[str],
    rng: random.Random,
) -> list[int]:
    old_positions_by_key: dict[str, list[int]] = defaultdict(list)
    target_positions_by_key: dict[str, list[int]] = defaultdict(list)

    for old_position, key in enumerate(keys):
        old_positions_by_key[key].append(old_position)

    for new_position, key in enumerate(sequence):
        target_positions_by_key[key].append(new_position)

    permutation = [-1] * len(sequence)

    for key, old_positions in old_positions_by_key.items():
        mapping = assign_group_positions(
            old_positions,
            target_positions_by_key[key],
            rng,
        )
        for new_position, old_position in mapping.items():
            permutation[new_position] = old_position

    if any(position < 0 for position in permutation):
        raise RuntimeError("Affectation incomplète de la permutation.")

    return permutation


# ============================================================================
# MÉTRIQUES ET CHOIX DU MEILLEUR ORDRE
# ============================================================================

def percentile(values: list[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return float(ordered[index])


def permutation_statistics(
    permutation: list[int],
    keys: list[str],
    visible_refs: list[str],
) -> dict[str, Any]:
    total = len(permutation)

    old_to_new = [0] * total
    for new_position, old_position in enumerate(permutation):
        old_to_new[old_position] = new_position

    absolute_shifts = [
        abs(new_position - old_position)
        for old_position, new_position in enumerate(old_to_new)
    ]

    positions_by_key: dict[str, list[int]] = defaultdict(list)
    previous_key: str | None = None
    adjacent_same_article_pairs = 0

    for new_position, old_position in enumerate(permutation):
        key = keys[old_position]
        positions_by_key[key].append(new_position)
        if previous_key == key and visible_refs[old_position]:
            adjacent_same_article_pairs += 1
        previous_key = key

    same_article_gaps: list[int] = []
    per_article_gap: dict[str, dict[str, Any]] = {}

    for key, positions in positions_by_key.items():
        # Les clés artificielles sans law_ref sont uniques et ne comptent pas.
        if key.startswith("__NO_LAW_REF__") or len(positions) < 2:
            continue

        gaps = [
            right - left
            for left, right in zip(positions, positions[1:])
        ]
        same_article_gaps.extend(gaps)
        per_article_gap[key] = {
            "instances": len(positions),
            "positions": [position + 1 for position in positions],
            "minimum_gap": min(gaps),
            "mean_gap": statistics.mean(gaps),
            "maximum_gap": max(gaps),
        }

    return {
        "old_to_new": old_to_new,
        "absolute_shifts": absolute_shifts,
        "fixed_positions": sum(shift == 0 for shift in absolute_shifts),
        "below_target": sum(
            shift < TARGET_MIN_DISPLACEMENT for shift in absolute_shifts
        ),
        "minimum_absolute_shift": min(absolute_shifts),
        "mean_absolute_shift": statistics.mean(absolute_shifts),
        "median_absolute_shift": statistics.median(absolute_shifts),
        "maximum_absolute_shift": max(absolute_shifts),
        "adjacent_same_article_pairs": adjacent_same_article_pairs,
        "same_article_gaps": same_article_gaps,
        "minimum_same_article_gap": (
            min(same_article_gaps) if same_article_gaps else None
        ),
        "p10_same_article_gap": percentile(same_article_gaps, 0.10),
        "median_same_article_gap": (
            statistics.median(same_article_gaps)
            if same_article_gaps
            else None
        ),
        "mean_same_article_gap": (
            statistics.mean(same_article_gaps)
            if same_article_gaps
            else None
        ),
        "maximum_same_article_gap": (
            max(same_article_gaps) if same_article_gaps else None
        ),
        "per_article_gap": per_article_gap,
    }


def candidate_score(stats: dict[str, Any]) -> tuple[float, ...]:
    """Priorité absolue à l'espacement des instances du même article."""
    minimum_gap = stats["minimum_same_article_gap"]
    p10_gap = stats["p10_same_article_gap"]
    mean_gap = stats["mean_same_article_gap"]

    return (
        float(minimum_gap if minimum_gap is not None else 10**9),
        float(p10_gap if p10_gap is not None else 10**9),
        float(mean_gap if mean_gap is not None else 10**9),
        -float(stats["adjacent_same_article_pairs"]),
        -float(stats["fixed_positions"]),
        -float(stats["below_target"]),
        float(stats["minimum_absolute_shift"]),
        float(stats["mean_absolute_shift"]),
    )


def choose_best_permutation(
    counts: Counter[str],
    maximum_gap: int,
    keys: list[str],
    visible_refs: list[str],
    seed: int,
    attempts: int,
) -> tuple[list[int], dict[str, Any]]:
    best_permutation: list[int] | None = None
    best_stats: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None

    successful_attempts = 0

    for attempt in range(attempts):
        rng = random.Random(seed + 50_000_003 + attempt)
        sequence = build_article_sequence(counts, maximum_gap, rng)
        if sequence is None:
            continue

        successful_attempts += 1
        permutation = sequence_to_permutation(sequence, keys, rng)
        stats = permutation_statistics(permutation, keys, visible_refs)
        score = candidate_score(stats)

        if best_score is None or score > best_score:
            best_permutation = permutation
            best_stats = stats
            best_score = score

    if best_permutation is None or best_stats is None:
        raise RuntimeError(
            "Aucun ordre n'a pu être construit avec la distance maximale."
        )

    best_stats["successful_order_attempts"] = successful_attempts
    return best_permutation, best_stats


# ============================================================================
# RAPPORT POSITION PAR POSITION
# ============================================================================

def nearest_same_article_gap_by_old_position(
    permutation: list[int],
    keys: list[str],
) -> dict[int, int | None]:
    positions_by_key: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for new_position, old_position in enumerate(permutation):
        positions_by_key[keys[old_position]].append(
            (new_position, old_position)
        )

    result: dict[int, int | None] = {
        old_position: None for old_position in permutation
    }

    for key, items in positions_by_key.items():
        if key.startswith("__NO_LAW_REF__") or len(items) < 2:
            continue

        for index, (new_position, old_position) in enumerate(items):
            candidate_gaps: list[int] = []
            if index > 0:
                candidate_gaps.append(new_position - items[index - 1][0])
            if index + 1 < len(items):
                candidate_gaps.append(items[index + 1][0] - new_position)
            result[old_position] = min(candidate_gaps)

    return result


def write_positions_csv(
    path: Path,
    records: list[dict[str, Any]],
    permutation: list[int],
    stats: dict[str, Any],
    visible_refs: list[str],
    keys: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nearest_gaps = nearest_same_article_gap_by_old_position(
        permutation,
        keys,
    )

    fields = [
        "id",
        "law_ref",
        "old_position",
        "new_position",
        "signed_shift",
        "absolute_shift",
        "nearest_same_article_gap_in_shuffled_order",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for old_position, record in enumerate(records):
            new_position = stats["old_to_new"][old_position]
            signed_shift = new_position - old_position
            writer.writerow(
                {
                    "id": record_id(record, old_position),
                    "law_ref": visible_refs[old_position],
                    "old_position": old_position + 1,
                    "new_position": new_position + 1,
                    "signed_shift": signed_shift,
                    "absolute_shift": abs(signed_shift),
                    "nearest_same_article_gap_in_shuffled_order": (
                        nearest_gaps[old_position]
                        if nearest_gaps[old_position] is not None
                        else ""
                    ),
                }
            )


def threshold_summary(
    values: list[int],
    thresholds: list[int],
) -> dict[str, Any]:
    total = len(values)
    return {
        f"moved_at_least_{threshold}": {
            "count": sum(value >= threshold for value in values),
            "share": sum(value >= threshold for value in values) / total,
        }
        for threshold in thresholds
    }


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main() -> int:
    records = load_records(INPUT_FILE)
    total = len(records)

    ids = [record_id(record, index) for index, record in enumerate(records)]
    duplicate_ids = [
        value for value, count in Counter(ids).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Identifiants dupliqués : {duplicate_ids[:10]}")

    keys, visible_refs = article_keys(records)
    counts = Counter(keys)

    theoretical_gap_upper_bound = maximum_theoretical_gap(counts, total)

    maximum_gap = find_maximum_feasible_gap(
        counts=counts,
        total=total,
        seed=SEED,
        attempts=FEASIBILITY_ATTEMPTS,
    )

    permutation, stats = choose_best_permutation(
        counts=counts,
        maximum_gap=maximum_gap,
        keys=keys,
        visible_refs=visible_refs,
        seed=SEED,
        attempts=ORDER_ATTEMPTS,
    )

    shuffled_records = [records[old_position] for old_position in permutation]

    original_hash = dataset_hash(records)
    shuffled_hash = dataset_hash(shuffled_records)
    if original_hash != shuffled_hash:
        raise RuntimeError("Le contenu a changé pendant le mélange.")

    shuffled_ids = [
        record_id(record, index)
        for index, record in enumerate(shuffled_records)
    ]
    if len(shuffled_ids) != len(set(shuffled_ids)):
        raise RuntimeError("Le mélange a créé des identifiants dupliqués.")
    if set(shuffled_ids) != set(ids):
        raise RuntimeError("Le mélange a perdu ou ajouté des instances.")

    write_jsonl(OUTPUT_FILE, shuffled_records)
    write_positions_csv(
        POSITIONS_CSV,
        records,
        permutation,
        stats,
        visible_refs,
        keys,
    )

    shifts = stats["absolute_shifts"]
    gaps = stats["same_article_gaps"]

    audit = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "positions_file": str(POSITIONS_CSV),
        "seed": SEED,
        "feasibility_attempts_per_gap": FEASIBILITY_ATTEMPTS,
        "order_attempts": ORDER_ATTEMPTS,
        "successful_order_attempts": stats["successful_order_attempts"],
        "instances": total,
        "article_frequency": {
            "distinct_law_refs": len({ref for ref in visible_refs if ref}),
            "instances_without_law_ref": sum(not ref for ref in visible_refs),
            "maximum_instances_for_one_law_ref": max(
                Counter(ref for ref in visible_refs if ref).values(),
                default=1,
            ),
        },
        "validation": {
            "same_instance_count": len(shuffled_records) == len(records),
            "same_ids": set(shuffled_ids) == set(ids),
            "no_duplicate_ids": len(shuffled_ids) == len(set(shuffled_ids)),
            "records_unchanged": original_hash == shuffled_hash,
            "order_changed": shuffled_ids != ids,
            "content_sha256_order_independent": original_hash,
        },
        "same_law_ref_separation": {
            "theoretical_upper_bound_for_minimum_gap": theoretical_gap_upper_bound,
            "best_minimum_gap_found": maximum_gap,
            "achieved_minimum_gap": stats["minimum_same_article_gap"],
            "reached_theoretical_upper_bound": maximum_gap == theoretical_gap_upper_bound,
            "p10_gap": stats["p10_same_article_gap"],
            "median_gap": stats["median_same_article_gap"],
            "mean_gap": stats["mean_same_article_gap"],
            "maximum_gap": stats["maximum_same_article_gap"],
            "adjacent_same_article_pairs": (
                stats["adjacent_same_article_pairs"]
            ),
            "gap_count": len(gaps),
            "per_article": stats["per_article_gap"],
        },
        "position_displacement": {
            "unchanged_positions": stats["fixed_positions"],
            "positions_below_target": stats["below_target"],
            "target_min_displacement": TARGET_MIN_DISPLACEMENT,
            "minimum_absolute_shift": stats["minimum_absolute_shift"],
            "mean_absolute_shift": stats["mean_absolute_shift"],
            "median_absolute_shift": stats["median_absolute_shift"],
            "maximum_absolute_shift": stats["maximum_absolute_shift"],
            **threshold_summary(shifts, [10, 20, 50, 100, 200]),
        },
    }

    write_json(AUDIT_JSON, audit)

    separation = audit["same_law_ref_separation"]
    displacement = audit["position_displacement"]

    print("Mélange déterministe avec espacement maximal terminé.")
    print(f"Entrée       : {INPUT_FILE}")
    print(f"Sortie       : {OUTPUT_FILE}")
    print(f"Positions    : {POSITIONS_CSV}")
    print(f"Audit        : {AUDIT_JSON}")
    print(f"Instances    : {total}")
    print(f"Graine       : {SEED}")
    print()
    print("Validation du contenu :")
    print(f"  mêmes instances            : {audit['validation']['same_ids']}")
    print(f"  aucun doublon              : {audit['validation']['no_duplicate_ids']}")
    print(f"  contenu inchangé           : {audit['validation']['records_unchanged']}")
    print(f"  ordre effectivement changé : {audit['validation']['order_changed']}")
    print()
    print("Éloignement des instances du même article :")
    print(
        "  borne supérieure théorique            : "
        f"{separation['theoretical_upper_bound_for_minimum_gap']}"
    )
    print(
        "  meilleure distance minimale trouvée   : "
        f"{separation['best_minimum_gap_found']}"
    )
    print(
        "  distance minimale effectivement obtenue: "
        f"{separation['achieved_minimum_gap']}"
    )
    print(
        "  borne théorique atteinte               : "
        f"{separation['reached_theoretical_upper_bound']}"
    )
    print(
        "  paires adjacentes de même article     : "
        f"{separation['adjacent_same_article_pairs']}"
    )
    print(
        "  distance médiane entre mêmes articles : "
        f"{separation['median_gap']:.2f}"
        if separation["median_gap"] is not None
        else "  distance médiane entre mêmes articles : N/A"
    )
    print(
        "  distance moyenne entre mêmes articles : "
        f"{separation['mean_gap']:.2f}"
        if separation["mean_gap"] is not None
        else "  distance moyenne entre mêmes articles : N/A"
    )
    print()
    print("Déplacement par rapport à l'ordre initial :")
    print(f"  positions inchangées       : {displacement['unchanged_positions']}")
    print(f"  déplacement minimal        : {displacement['minimum_absolute_shift']}")
    print(f"  déplacement moyen          : {displacement['mean_absolute_shift']:.2f}")
    print(f"  déplacement médian         : {displacement['median_absolute_shift']:.2f}")
    print(f"  déplacement maximal        : {displacement['maximum_absolute_shift']}")
    print(
        "  déplacées d'au moins 20    : "
        f"{displacement['moved_at_least_20']['count']}/{total} "
        f"({displacement['moved_at_least_20']['share']:.1%})"
    )
    print(
        "  déplacées d'au moins 100   : "
        f"{displacement['moved_at_least_100']['count']}/{total} "
        f"({displacement['moved_at_least_100']['share']:.1%})"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
