#!/usr/bin/env python3
"""
Calcule séparément les métriques FLEXID du second annotateur humain et de DeepSeek-V4-Flash contre le même fichier gold.



Mesures :
- accord brut ;
- Cohen's kappa non pondéré + IC bootstrap 95 % ;
- matrice de confusion 3x3 ;
- S_span = même label non neutre ;
- Macro-F1 token + IC bootstrap 95 % ;
- Macro-IoU ;
- Exact Match strict ;
- Neutral Rationale Absence Agreement (NRA) ;
- Joint IoU@0.50.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


LABELS = ("entailment", "contradiction", "neutral")
NON_NEUTRAL = {"entailment", "contradiction"}


# ---------------------------------------------------------------------------
# Lecture et validation
# ---------------------------------------------------------------------------

def normalise_label(value: Any) -> str:
    label = str(value).strip().casefold()
    aliases = {
        "entailment": "entailment",
        "entails": "entailment",
        "implication": "entailment",
        "contradiction": "contradiction",
        "contradict": "contradiction",
        "neutral": "neutral",
        "neutre": "neutral",
    }
    return aliases.get(label, label)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Fichier vide : {path}")

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: le JSON doit contenir une liste.")
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
                    f"{path}, ligne {line_number}: JSONL invalide : {exc}"
                ) from exc

    if not records:
        raise ValueError(f"Aucune instance trouvée dans {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: chaque ligne doit être un objet JSON.")

    return records


def parse_boundary(value: Any, field: str, record_id: str) -> int | None:
    """
    Valide une borne token.

    Valeurs acceptées :
    - null ;
    - entier JSON strictement positif ;
    - chaîne composée uniquement de chiffres, par compatibilité.

    Les flottants non entiers, par exemple 8.5, sont rejetés au lieu
    d'être silencieusement tronqués.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{record_id}: {field} ne peut pas être booléen.")

    if isinstance(value, int):
        parsed = value

    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(
                f"{record_id}: {field} doit être un entier exact ou null, "
                f"pas {value!r}."
            )
        parsed = int(value)

    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            raise ValueError(
                f"{record_id}: {field} doit être un entier positif ou null."
            )
        parsed = int(stripped)

    else:
        raise ValueError(
            f"{record_id}: {field} doit être un entier positif ou null."
        )

    if parsed < 1:
        raise ValueError(f"{record_id}: {field} doit être >= 1 ou null.")

    return parsed


def prepare_file(
    path: Path,
    role: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    prepared: dict[str, dict[str, Any]] = {}

    for position, record in enumerate(load_jsonl(path), start=1):
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError(f"{role}, ligne {position}: id absent.")
        if record_id in prepared:
            raise ValueError(f"{role}: id dupliqué : {record_id}")

        label = normalise_label(record.get("label", ""))
        if label not in LABELS:
            raise ValueError(
                f"{role}, {record_id}: label invalide {label!r}."
            )

        start = parse_boundary(
            record.get("rationale_start_token"),
            "rationale_start_token",
            record_id,
        )
        end = parse_boundary(
            record.get("rationale_end_token"),
            "rationale_end_token",
            record_id,
        )

        if label == "neutral":
            if start is not None or end is not None:
                raise ValueError(
                    f"{role}, {record_id}: neutral exige deux bornes null."
                )
        else:
            if start is None or end is None:
                raise ValueError(
                    f"{role}, {record_id}: {label} exige deux bornes."
                )
            if start > end:
                raise ValueError(
                    f"{role}, {record_id}: start_token > end_token."
                )

        order.append(record_id)
        prepared[record_id] = {
            "id": record_id,
            "label": label,
            "start": start,
            "end": end,
        }

    return order, prepared


def align_files(
    gold_path: Path,
    candidate_path: Path,
    candidate_role: str,
) -> list[dict[str, Any]]:
    gold_order, gold = prepare_file(gold_path, "gold")
    _, candidate = prepare_file(candidate_path, candidate_role)

    missing = sorted(set(gold) - set(candidate))
    unknown = sorted(set(candidate) - set(gold))

    if missing or unknown:
        parts = []
        if missing:
            parts.append("absents dans le fichier évalué : " + ", ".join(missing[:20]))
        if unknown:
            parts.append("inconnus dans le fichier évalué : " + ", ".join(unknown[:20]))
        raise ValueError("Identifiants non alignés — " + " ; ".join(parts))

    return [
        {
            "id": record_id,
            "gold_label": gold[record_id]["label"],
            "candidate_label": candidate[record_id]["label"],
            "gold_start": gold[record_id]["start"],
            "gold_end": gold[record_id]["end"],
            "candidate_start": candidate[record_id]["start"],
            "candidate_end": candidate[record_id]["end"],
        }
        for record_id in gold_order
    ]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def confusion_matrix(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for record in records:
        matrix[index[record["gold_label"]]][index[record["candidate_label"]]] += 1
    return matrix


def cohen_kappa(
    records: Sequence[dict[str, Any]],
) -> tuple[float, float, float]:
    if not records:
        raise ValueError("Kappa impossible sur zéro instance.")

    n = len(records)
    agreements = sum(
        r["gold_label"] == r["candidate_label"] for r in records
    )
    p_o = agreements / n

    gold_counts = Counter(r["gold_label"] for r in records)
    candidate_counts = Counter(r["candidate_label"] for r in records)
    p_e = sum(
        (gold_counts[label] / n) * (candidate_counts[label] / n)
        for label in LABELS
    )

    if math.isclose(1.0 - p_e, 0.0):
        if math.isclose(p_o, 1.0):
            return 1.0, p_o, p_e
        raise ValueError("Kappa indéfini car p_e = 1.")

    return (p_o - p_e) / (1.0 - p_e), p_o, p_e


# ---------------------------------------------------------------------------
# Rationales
# ---------------------------------------------------------------------------

def span_values(record: dict[str, Any]) -> tuple[float, float, bool]:
    gold = set(range(record["gold_start"], record["gold_end"] + 1))
    candidate = set(
        range(record["candidate_start"], record["candidate_end"] + 1)
    )
    intersection = len(gold & candidate)
    f1 = 2.0 * intersection / (len(gold) + len(candidate))
    iou = intersection / len(gold | candidate)
    exact = (
        record["gold_start"] == record["candidate_start"]
        and record["gold_end"] == record["candidate_end"]
    )
    return f1, iou, exact


def build_s_span(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    subset = []
    for record in records:
        if (
            record["gold_label"] == record["candidate_label"]
            and record["gold_label"] in NON_NEUTRAL
        ):
            f1, iou, exact = span_values(record)
            enriched = dict(record)
            enriched.update(
                {"token_f1": f1, "token_iou": iou, "exact_match": exact}
            )
            subset.append(enriched)
    return subset


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Moyenne impossible sur une liste vide.")
    return sum(values) / len(values)


def rationale_metrics(s_span: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not s_span:
        raise ValueError("S_span est vide.")
    f1_values = [float(r["token_f1"]) for r in s_span]
    iou_values = [float(r["token_iou"]) for r in s_span]
    em_count = sum(bool(r["exact_match"]) for r in s_span)
    return {
        "s_span_size": len(s_span),
        "macro_token_f1": mean(f1_values),
        "macro_token_iou": mean(iou_values),
        "exact_match_count": em_count,
        "exact_match_rate": em_count / len(s_span),
    }


def neutral_rationale_absence_agreement(
    records: Sequence[dict[str, Any]],
) -> tuple[int, int, float]:
    gold_neutral = [r for r in records if r["gold_label"] == "neutral"]
    if not gold_neutral:
        raise ValueError("Aucune instance gold neutral.")
    count = sum(
        r["candidate_label"] == "neutral"
        and r["candidate_start"] is None
        and r["candidate_end"] is None
        for r in gold_neutral
    )
    return count, len(gold_neutral), count / len(gold_neutral)


def joint_iou_at_050(
    records: Sequence[dict[str, Any]],
) -> tuple[int, int, float]:
    success = 0
    for record in records:
        if (
            record["gold_label"] == "neutral"
            and record["candidate_label"] == "neutral"
        ):
            success += 1
        elif (
            record["gold_label"] == record["candidate_label"]
            and record["gold_label"] in NON_NEUTRAL
        ):
            _, iou, _ = span_values(record)
            if iou >= 0.50:
                success += 1
    return success, len(records), success / len(records)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Percentile impossible sur une liste vide.")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_kappa_ci(
    records: Sequence[dict[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[float, float, int]:
    strata = {
        label: [r for r in records if r["gold_label"] == label]
        for label in LABELS
    }
    empty = [label for label, items in strata.items() if not items]
    if empty:
        raise ValueError(
            "Bootstrap stratifié impossible, strates vides: "
            + ", ".join(empty)
        )

    rng = random.Random(seed)
    estimates = []

    for _ in range(iterations):
        sample = []
        for label in LABELS:
            items = strata[label]
            sample.extend(rng.choice(items) for _ in range(len(items)))
        try:
            estimate, _, _ = cohen_kappa(sample)
        except ValueError:
            continue
        if math.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        raise ValueError("Aucune réplication bootstrap valide pour kappa.")

    return (
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
        len(estimates),
    )


def bootstrap_macro_f1_ci(
    s_span: Sequence[dict[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[float, float, int]:
    if not s_span:
        raise ValueError("Bootstrap Macro-F1 impossible: S_span vide.")
    rng = random.Random(seed)
    n = len(s_span)
    estimates = []
    for _ in range(iterations):
        sample = [rng.choice(s_span) for _ in range(n)]
        estimates.append(mean([float(r["token_f1"]) for r in sample]))
    return (
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
        len(estimates),
    )


# ---------------------------------------------------------------------------
# Évaluation globale
# ---------------------------------------------------------------------------

def evaluate(
    records: Sequence[dict[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    matrix = confusion_matrix(records)
    kappa, p_o, p_e = cohen_kappa(records)
    k_low, k_high, k_valid = bootstrap_kappa_ci(
        records, bootstrap_iterations, seed
    )

    s_span = build_s_span(records)
    span_metrics = rationale_metrics(s_span)
    f_low, f_high, f_valid = bootstrap_macro_f1_ci(
        s_span, bootstrap_iterations, seed + 1
    )

    nra_count, nra_total, nra_rate = (
        neutral_rationale_absence_agreement(records)
    )
    joint_count, joint_total, joint_rate = joint_iou_at_050(records)

    agreement_count = sum(
        r["gold_label"] == r["candidate_label"] for r in records
    )
    gold_counts = Counter(r["gold_label"] for r in records)
    candidate_counts = Counter(r["candidate_label"] for r in records)

    return {
        "validation": {
            "aligned_instances": len(records),
            "unique_ids": len({r["id"] for r in records}),
            "all_ids_aligned": True,
            "all_spans_valid": True,
        },
        "label_distributions": {
            "gold": {label: gold_counts[label] for label in LABELS},
            "evaluated_system": {
                label: candidate_counts[label] for label in LABELS
            },
        },
        "label_agreement": {
            "agreement_count": agreement_count,
            "disagreement_count": len(records) - agreement_count,
            "raw_agreement": p_o,
            "expected_agreement": p_e,
            "cohen_kappa": kappa,
            "kappa_bootstrap_ci_95": [k_low, k_high],
            "bootstrap_replicates_valid": k_valid,
            "confusion_matrix": {
                "row_labels_gold": list(LABELS),
                "column_labels_evaluated_system": list(LABELS),
                "values": matrix,
            },
        },
        "rationale_agreement": {
            **span_metrics,
            "definition_s_span": (
                "gold_label == candidate_label and label != neutral"
            ),
            "macro_f1_bootstrap_ci_95": [f_low, f_high],
            "bootstrap_replicates_valid": f_valid,
        },
        "neutral_rationale_absence_agreement": {
            "agreement_count": nra_count,
            "gold_neutral_count": nra_total,
            "rate": nra_rate,
        },
        "joint_iou_at_0_50": {
            "success_count": joint_count,
            "total_instances": joint_total,
            "rate": joint_rate,
            "threshold": 0.50,
        },
        "bootstrap": {
            "iterations_requested": bootstrap_iterations,
            "seed": seed,
            "kappa_method": "stratified percentile bootstrap by gold label",
            "macro_f1_method": "percentile bootstrap on S_span",
            "confidence_level": 0.95,
        },
    }


# ---------------------------------------------------------------------------
# Écriture des résultats
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_confusion_csv(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\système", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


def write_details_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    fields = [
        "id",
        "gold_label",
        "candidate_label",
        "label_agreement",
        "in_s_span",
        "gold_start",
        "gold_end",
        "candidate_start",
        "candidate_end",
        "token_f1",
        "token_iou",
        "exact_match",
        "joint_iou_at_0_50_success",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for record in records:
            same_label = record["gold_label"] == record["candidate_label"]
            in_s_span = same_label and record["gold_label"] in NON_NEUTRAL
            token_f1: float | str = ""
            token_iou: float | str = ""
            exact: bool | str = ""
            joint = False

            if in_s_span:
                token_f1, token_iou, exact = span_values(record)
                joint = token_iou >= 0.50
            elif (
                record["gold_label"] == "neutral"
                and record["candidate_label"] == "neutral"
            ):
                joint = True

            writer.writerow(
                {
                    "id": record["id"],
                    "gold_label": record["gold_label"],
                    "candidate_label": record["candidate_label"],
                    "label_agreement": same_label,
                    "in_s_span": in_s_span,
                    "gold_start": record["gold_start"],
                    "gold_end": record["gold_end"],
                    "candidate_start": record["candidate_start"],
                    "candidate_end": record["candidate_end"],
                    "token_f1": token_f1,
                    "token_iou": token_iou,
                    "exact_match": exact,
                    "joint_iou_at_0_50_success": joint,
                }
            )


def write_summary(
    path: Path,
    gold_path: Path,
    candidate_path: Path,
    results: dict[str, Any],
    display_name: str,
    evaluation_kind: str,
) -> None:
    labels = results["label_agreement"]
    spans = results["rationale_agreement"]
    nra = results["neutral_rationale_absence_agreement"]
    joint = results["joint_iou_at_0_50"]
    n = results["validation"]["aligned_instances"]

    if evaluation_kind == "human":
        title = "FLEXID — Accord inter-annotateurs"
        candidate_line = f"Second annotateur humain : {candidate_path}"
    else:
        title = "FLEXID — Évaluation du modèle contre le gold"
        candidate_line = f"Modèle évalué ({display_name}) : {candidate_path}"

    lines = [
        title,
        "=" * 60,
        f"Système évalué : {display_name}",
        f"Gold : {gold_path}",
        candidate_line,
        "",
        "A. LABELS",
        f"Instances alignées : {n}",
        (
            f"Accord brut : {labels['raw_agreement']:.4f} "
            f"({labels['agreement_count']}/{n})"
        ),
        f"Cohen's kappa : {labels['cohen_kappa']:.4f}",
        (
            "IC bootstrap 95 % de kappa : "
            f"[{labels['kappa_bootstrap_ci_95'][0]:.4f}, "
            f"{labels['kappa_bootstrap_ci_95'][1]:.4f}]"
        ),
        f"Accord attendu p_e : {labels['expected_agreement']:.4f}",
        "",
        "Matrice de confusion — lignes=gold, colonnes=système évalué",
        "                         entailment  contradiction  neutral",
    ]

    for label, row in zip(
        LABELS, labels["confusion_matrix"]["values"]
    ):
        lines.append(
            f"{label:<24} {row[0]:>10} {row[1]:>14} {row[2]:>8}"
        )

    lines.extend(
        [
            "",
            "B. RATIONALE SPANS",
            f"S_span : N={spans['s_span_size']}",
            f"Macro-F1 token : {spans['macro_token_f1']:.4f}",
            (
                "IC bootstrap 95 % du Macro-F1 : "
                f"[{spans['macro_f1_bootstrap_ci_95'][0]:.4f}, "
                f"{spans['macro_f1_bootstrap_ci_95'][1]:.4f}]"
            ),
            f"Macro-IoU : {spans['macro_token_iou']:.4f}",
            (
                f"Exact Match strict : {spans['exact_match_rate']:.4f} "
                f"({spans['exact_match_count']}/{spans['s_span_size']})"
            ),
            "",
            "C. NEUTRAL",
            (
                f"NRA : {nra['rate']:.4f} "
                f"({nra['agreement_count']}/{nra['gold_neutral_count']})"
            ),
            "",
            "D. BOUT-EN-BOUT",
            (
                f"Joint IoU@0.50 : {joint['rate']:.4f} "
                f"({joint['success_count']}/{joint['total_instances']})"
            ),
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration directe — aucun argument de terminal
# ---------------------------------------------------------------------------

GOLD_FILENAME = (
    "flexid_kappa_rationale_180_unique_law_refs_60_each_gold.jsonl"
)

EVALUATION_TARGETS = (
    {
        "key": "second_annotator",
        "display_name": "Second annotateur humain",
        "evaluation_kind": "human",
        "filename": "flexid_kappa_rational_second_annotator.jsonl",
        "output_prefix": "flexid_second_annotator",
    },
    {
        "key": "deepseek_v4_flash",
        "display_name": "DeepSeek-V4-Flash",
        "evaluation_kind": "model",
        "filename": "flexid_deepseek_v4_flash_predictions.jsonl",
        "output_prefix": "flexid_deepseek_v4_flash",
    },
)

OUTPUT_ROOT_DIRECTORY_NAME = "results_evaluations"

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 2026


def find_project_root() -> Path:
    """
    Détecte la racine du projet sans chemin absolu.

    Le script peut être placé dans scripts/ ou directement à la racine.
    La racine doit contenir le gold et les deux fichiers à évaluer dans data/.
    """
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()

    candidates: list[Path] = []
    for starting_point in (script_dir, current_dir):
        candidates.append(starting_point)
        candidates.extend(starting_point.parents)

    seen: set[Path] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        data_dir = candidate / "data"
        gold_file = data_dir / GOLD_FILENAME

        target_files_exist = all(
            (data_dir / target["filename"]).is_file()
            for target in EVALUATION_TARGETS
        )

        if gold_file.is_file() and target_files_exist:
            return candidate

    expected = [f"data/{GOLD_FILENAME}"]
    expected.extend(
        f"data/{target['filename']}"
        for target in EVALUATION_TARGETS
    )

    raise FileNotFoundError(
        "Impossible de détecter la racine du projet.\n"
        "Les fichiers suivants doivent exister :\n  - "
        + "\n  - ".join(expected)
    )


def add_evaluation_metadata(
    results: dict[str, Any],
    *,
    target: dict[str, str],
    gold_file: Path,
    candidate_file: Path,
) -> dict[str, Any]:
    """Ajoute l'identité du système sans modifier les métriques calculées."""
    return {
        "evaluation": {
            "key": target["key"],
            "display_name": target["display_name"],
            "evaluation_kind": target["evaluation_kind"],
            "gold_file": str(gold_file),
            "evaluated_file": str(candidate_file),
        },
        **results,
    }


def print_result_block(
    *,
    display_name: str,
    candidate_file: Path,
    output_dir: Path,
    results: dict[str, Any],
) -> None:
    labels = results["label_agreement"]
    spans = results["rationale_agreement"]
    nra = results["neutral_rationale_absence_agreement"]
    joint = results["joint_iou_at_0_50"]

    print("")
    print("=" * 72)
    print(display_name)
    print("=" * 72)
    print(f"Fichier évalué           : {candidate_file}")
    print(
        "Instances alignées      : "
        f"{results['validation']['aligned_instances']}"
    )
    print(f"Accord brut              : {labels['raw_agreement']:.4f}")
    print(f"Cohen's kappa            : {labels['cohen_kappa']:.4f}")
    print(
        "IC 95 % kappa           : "
        f"[{labels['kappa_bootstrap_ci_95'][0]:.4f}, "
        f"{labels['kappa_bootstrap_ci_95'][1]:.4f}]"
    )
    print(f"Taille de S_span         : {spans['s_span_size']}")
    print(f"Macro-F1 token           : {spans['macro_token_f1']:.4f}")
    print(
        "IC 95 % Macro-F1        : "
        f"[{spans['macro_f1_bootstrap_ci_95'][0]:.4f}, "
        f"{spans['macro_f1_bootstrap_ci_95'][1]:.4f}]"
    )
    print(f"Macro-IoU                : {spans['macro_token_iou']:.4f}")
    print(f"Exact Match              : {spans['exact_match_rate']:.4f}")
    print(f"NRA                      : {nra['rate']:.4f}")
    print(f"Joint IoU@0.50           : {joint['rate']:.4f}")
    print(f"Dossier de sortie        : {output_dir}")


def run_evaluation(
    *,
    gold_file: Path,
    data_dir: Path,
    output_root: Path,
    target: dict[str, str],
) -> dict[str, Any]:
    candidate_file = data_dir / target["filename"]
    output_dir = output_root / target["key"]
    output_dir.mkdir(parents=True, exist_ok=True)

    records = align_files(
        gold_file,
        candidate_file,
        target["display_name"],
    )

    metric_results = evaluate(
        records,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_SEED,
    )
    results = add_evaluation_metadata(
        metric_results,
        target=target,
        gold_file=gold_file,
        candidate_file=candidate_file,
    )

    prefix = target["output_prefix"]
    results_path = output_dir / f"{prefix}_results.json"
    summary_path = output_dir / f"{prefix}_summary.txt"
    confusion_path = output_dir / f"{prefix}_confusion_matrix.csv"
    details_path = output_dir / f"{prefix}_instance_details.csv"

    write_json(results_path, results)
    write_summary(
        summary_path,
        gold_file,
        candidate_file,
        results,
        target["display_name"],
        target["evaluation_kind"],
    )
    write_confusion_csv(
        confusion_path,
        results["label_agreement"]["confusion_matrix"]["values"],
    )
    write_details_csv(details_path, records)

    print_result_block(
        display_name=target["display_name"],
        candidate_file=candidate_file,
        output_dir=output_dir,
        results=results,
    )

    return results


def comparison_row(results: dict[str, Any]) -> dict[str, Any]:
    evaluation = results["evaluation"]
    labels = results["label_agreement"]
    spans = results["rationale_agreement"]
    nra = results["neutral_rationale_absence_agreement"]
    joint = results["joint_iou_at_0_50"]

    return {
        "system_key": evaluation["key"],
        "system_name": evaluation["display_name"],
        "evaluation_kind": evaluation["evaluation_kind"],
        "instances": results["validation"]["aligned_instances"],
        "raw_agreement": labels["raw_agreement"],
        "cohen_kappa": labels["cohen_kappa"],
        "kappa_ci_low": labels["kappa_bootstrap_ci_95"][0],
        "kappa_ci_high": labels["kappa_bootstrap_ci_95"][1],
        "s_span_size": spans["s_span_size"],
        "macro_token_f1": spans["macro_token_f1"],
        "macro_f1_ci_low": spans["macro_f1_bootstrap_ci_95"][0],
        "macro_f1_ci_high": spans["macro_f1_bootstrap_ci_95"][1],
        "macro_token_iou": spans["macro_token_iou"],
        "exact_match_rate": spans["exact_match_rate"],
        "nra": nra["rate"],
        "joint_iou_at_0_50": joint["rate"],
    }


def write_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("Aucun résultat à écrire dans le comparatif.")

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if BOOTSTRAP_ITERATIONS < 1:
        raise ValueError(
            "BOOTSTRAP_ITERATIONS doit être supérieur ou égal à 1."
        )

    project_root = find_project_root()
    data_dir = project_root / "data"
    gold_file = data_dir / GOLD_FILENAME
    output_root = data_dir / OUTPUT_ROOT_DIRECTORY_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    print("Évaluation FLEXID — humain et DeepSeek")
    print(f"Racine du projet         : {project_root}")
    print(f"Fichier gold             : {gold_file}")
    print(f"Itérations bootstrap     : {BOOTSTRAP_ITERATIONS}")
    print(f"Graine bootstrap         : {BOOTSTRAP_SEED}")

    all_results: list[dict[str, Any]] = []

    for target in EVALUATION_TARGETS:
        all_results.append(
            run_evaluation(
                gold_file=gold_file,
                data_dir=data_dir,
                output_root=output_root,
                target=target,
            )
        )

    comparison_rows = [
        comparison_row(results)
        for results in all_results
    ]

    comparison_json = output_root / "flexid_evaluations_comparison.json"
    comparison_csv = output_root / "flexid_evaluations_comparison.csv"

    write_json(
        comparison_json,
        {
            "gold_file": str(gold_file),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "systems": comparison_rows,
        },
    )
    write_comparison_csv(comparison_csv, comparison_rows)

    print("")
    print("=" * 72)
    print("ÉVALUATIONS TERMINÉES")
    print("=" * 72)
    print(f"Résultats séparés        : {output_root}")
    print(f"Comparatif JSON          : {comparison_json}")
    print(f"Comparatif CSV           : {comparison_csv}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
