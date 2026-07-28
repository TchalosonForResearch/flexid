#!/usr/bin/env python3


from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_NAME = "flexid_shuffled_tokenized.jsonl"

OUTPUT_NAME = "flexid_kappa_rationale_180_unique_law_refs_60_each.jsonl"
GOLD_NAME = "flexid_kappa_rationale_180_unique_law_refs_60_each_gold.jsonl"
AUDIT_NAME = "flexid_kappa_rationale_180_selection_audit.json"

SELECTION_SEED = 2026

LABEL_QUOTAS = {
    "entailment": 60,
    "contradiction": 60,
    "neutral": 60,
}


def find_project_root() -> Path:
    """
    Repère automatiquement le dossier du projet sans chemin absolu.

    Le script peut être placé :
    - dans le dossier scripts/ ;
    - ou directement à la racine du projet.

    Le dossier retenu doit contenir :
        data/flexid_shuffled_tokenized.jsonl
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

        expected_input = candidate / "data" / INPUT_NAME
        if expected_input.is_file():
            return candidate

    searched = "\n".join(
        f"  - {candidate / 'data' / INPUT_NAME}"
        for candidate in seen
    )

    raise FileNotFoundError(
        "Impossible de trouver le fichier d'entrée.\n"
        f"Fichier attendu : data/{INPUT_NAME}\n"
        "Emplacements examinés :\n"
        f"{searched}"
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / INPUT_NAME
OUTPUT_FILE = DATA_DIR / OUTPUT_NAME
GOLD_FILE = DATA_DIR / GOLD_NAME
AUDIT_FILE = DATA_DIR / AUDIT_NAME


# ============================================================================
# LECTURE ET NORMALISATION
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
            raise ValueError(
                "Le fichier JSON doit contenir une liste d'objets."
            )

        records = data

    else:
        records: list[dict[str, Any]] = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide à la ligne {line_number} : {exc}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError("Aucune instance n'a été chargée.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Chaque instance doit être un objet JSON.")

    return records


def normalise_label(value: Any) -> str:
    """Normalise les variantes usuelles des trois labels FLEXID."""
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


def extract_law_ref(record: dict[str, Any]) -> str:
    """Lit law_ref au niveau racine ou dans meta.law_ref."""
    direct = record.get("law_ref")

    if direct is not None and str(direct).strip():
        return str(direct).strip()

    meta = record.get("meta")

    if isinstance(meta, dict):
        nested = meta.get("law_ref")

        if nested is not None and str(nested).strip():
            return str(nested).strip()

    return ""


def normalise_law_ref(value: str) -> str:
    """
    Produit une clé stable pour l'unicité des références juridiques.

    La casse et les espaces multiples ne créent pas de fausses références
    distinctes.
    """
    return " ".join(value.split()).casefold()


def parse_token_boundary(
    value: Any,
    *,
    field_name: str,
    record_id: str,
) -> int | None:
    """Valide une borne token : entier positif ou null."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"Instance {record_id} : {field_name} ne peut pas être booléen."
        )

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Instance {record_id} : {field_name} doit être un entier "
            "positif ou null."
        ) from exc

    if parsed < 1:
        raise ValueError(
            f"Instance {record_id} : {field_name} doit être supérieur "
            "ou égal à 1 pour un span non vide."
        )

    return parsed


def prepare_record(
    record: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Valide et normalise une instance du fichier tokenisé."""
    record_id = str(record.get("id", "")).strip()
    premise = str(record.get("premise", "")).strip()
    tokenized_premise = str(record.get("tokenized_premise", "")).strip()

    hypothesis_facts = str(
        record.get("hypothesis_facts", record.get("hypothesis", ""))
    ).strip()

    label = normalise_label(record.get("label", ""))
    law_ref = extract_law_ref(record)
    law_key = normalise_law_ref(law_ref)

    missing = [
        field_name
        for field_name, value in (
            ("id", record_id),
            ("premise", premise),
            ("tokenized_premise", tokenized_premise),
            ("hypothesis_facts", hypothesis_facts),
            ("label", label),
            ("law_ref/meta.law_ref", law_ref),
        )
        if not value
    ]

    if missing:
        raise ValueError(
            f"Instance #{index + 1} : champ(s) manquant(s) : "
            + ", ".join(missing)
        )

    if label not in LABEL_QUOTAS:
        raise ValueError(
            f"Instance {record_id} : label inconnu {label!r}. "
            f"Labels attendus : {', '.join(LABEL_QUOTAS)}."
        )

    raw_start = record.get("rationale_start_token")
    raw_end = record.get("rationale_end_token")

    if label == "neutral":
        if raw_start is not None or raw_end is not None:
            raise ValueError(
                f"Instance {record_id} : une instance neutral doit avoir "
                "rationale_start_token=null et rationale_end_token=null."
            )

        rationale_start_token = None
        rationale_end_token = None

    else:
        rationale_start_token = parse_token_boundary(
            raw_start,
            field_name="rationale_start_token",
            record_id=record_id,
        )
        rationale_end_token = parse_token_boundary(
            raw_end,
            field_name="rationale_end_token",
            record_id=record_id,
        )

        if (
            rationale_start_token is None
            or rationale_end_token is None
        ):
            raise ValueError(
                f"Instance {record_id} : une instance {label} doit avoir "
                "deux bornes token non nulles."
            )

        if rationale_start_token > rationale_end_token:
            raise ValueError(
                f"Instance {record_id} : rationale_start_token "
                f"({rationale_start_token}) dépasse rationale_end_token "
                f"({rationale_end_token})."
            )

    return {
        "id": record_id,
        "tokenized_premise": tokenized_premise,
        "hypothesis_facts": hypothesis_facts,
        "premise": premise,
        "label": label,
        "rationale_start_token": rationale_start_token,
        "rationale_end_token": rationale_end_token,
        "law_ref": law_ref,
        "law_key": law_key,
        "source_index": index,
    }


# ============================================================================
# FLUX MAXIMAL : UNE RÉFÉRENCE NE PEUT ÊTRE UTILISÉE QU'UNE FOIS
# ============================================================================

@dataclass
class Edge:
    to: int
    reverse_index: int
    capacity: int
    original_capacity: int


class Dinic:
    """Implémentation minimale de l'algorithme de flot maximal de Dinic."""

    def __init__(self, node_count: int) -> None:
        self.graph: list[list[Edge]] = [[] for _ in range(node_count)]
        self.level = [-1] * node_count
        self.progress = [0] * node_count

    def add_edge(
        self,
        source: int,
        target: int,
        capacity: int,
    ) -> None:
        forward = Edge(
            to=target,
            reverse_index=len(self.graph[target]),
            capacity=capacity,
            original_capacity=capacity,
        )

        backward = Edge(
            to=source,
            reverse_index=len(self.graph[source]),
            capacity=0,
            original_capacity=0,
        )

        self.graph[source].append(forward)
        self.graph[target].append(backward)

    def build_levels(self, source: int, sink: int) -> bool:
        self.level = [-1] * len(self.graph)
        self.level[source] = 0

        queue: deque[int] = deque([source])

        while queue:
            node = queue.popleft()

            for edge in self.graph[node]:
                if edge.capacity > 0 and self.level[edge.to] < 0:
                    self.level[edge.to] = self.level[node] + 1
                    queue.append(edge.to)

        return self.level[sink] >= 0

    def send_flow(
        self,
        node: int,
        sink: int,
        available: int,
    ) -> int:
        if node == sink:
            return available

        while self.progress[node] < len(self.graph[node]):
            edge_index = self.progress[node]
            edge = self.graph[node][edge_index]

            if (
                edge.capacity > 0
                and self.level[edge.to] == self.level[node] + 1
            ):
                sent = self.send_flow(
                    edge.to,
                    sink,
                    min(available, edge.capacity),
                )

                if sent > 0:
                    edge.capacity -= sent
                    reverse_edge = self.graph[
                        edge.to
                    ][edge.reverse_index]
                    reverse_edge.capacity += sent
                    return sent

            self.progress[node] += 1

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        total_flow = 0

        while self.build_levels(source, sink):
            self.progress = [0] * len(self.graph)

            while True:
                sent = self.send_flow(source, sink, 10**9)

                if sent == 0:
                    break

                total_flow += sent

        return total_flow


# ============================================================================
# SÉLECTION ÉQUILIBRÉE
# ============================================================================

def group_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Groupe les instances par law_ref normalisée, puis par label."""
    grouped: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))

    for record in records:
        grouped[record["law_key"]][record["label"]].append(record)

    return grouped


def select_records(
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Résout simultanément les contraintes suivantes :

    - 180 instances au total ;
    - 60 instances pour chacun des trois labels ;
    - une seule instance par law_ref ;
    - sélection et ordre final reproductibles.
    """
    rng = random.Random(seed)
    grouped = group_records(records)

    law_keys = list(grouped)
    rng.shuffle(law_keys)

    labels = list(LABEL_QUOTAS)
    rng.shuffle(labels)

    source = 0

    law_node = {
        law_key: index + 1
        for index, law_key in enumerate(law_keys)
    }

    first_label_node = 1 + len(law_keys)

    label_node = {
        label: first_label_node + index
        for index, label in enumerate(labels)
    }

    sink = first_label_node + len(labels)
    flow = Dinic(sink + 1)

    # Chaque référence juridique a une capacité maximale de 1.
    for law_key in law_keys:
        flow.add_edge(source, law_node[law_key], 1)

        available_labels = list(grouped[law_key])
        rng.shuffle(available_labels)

        for label in available_labels:
            flow.add_edge(
                law_node[law_key],
                label_node[label],
                1,
            )

    # Quotas exacts : 60 + 60 + 60.
    for label, quota in LABEL_QUOTAS.items():
        flow.add_edge(label_node[label], sink, quota)

    expected_total = sum(LABEL_QUOTAS.values())
    achieved_total = flow.max_flow(source, sink)

    if achieved_total != expected_total:
        unique_refs_by_label = {
            label: sum(
                label in grouped[law_key]
                for law_key in law_keys
            )
            for label in LABEL_QUOTAS
        }

        raise RuntimeError(
            "Sélection impossible avec les contraintes demandées. "
            f"Flot obtenu : {achieved_total}/{expected_total}. "
            "Références distinctes disponibles par label : "
            f"{unique_refs_by_label}."
        )

    selected: list[dict[str, Any]] = []

    label_by_node = {
        node: label
        for label, node in label_node.items()
    }

    for law_key in law_keys:
        node = law_node[law_key]
        chosen_label: str | None = None

        for edge in flow.graph[node]:
            if (
                edge.to in label_by_node
                and edge.original_capacity == 1
                and edge.capacity == 0
            ):
                chosen_label = label_by_node[edge.to]
                break

        if chosen_label is None:
            continue

        candidates = list(grouped[law_key][chosen_label])
        rng.shuffle(candidates)

        # Une seule instance concrète pour la law_ref retenue.
        selected.append(candidates[0])

    if len(selected) != expected_total:
        raise RuntimeError(
            f"Erreur interne : {len(selected)} instances sélectionnées "
            f"au lieu de {expected_total}."
        )

    # Mélange final pour éviter de regrouper les labels.
    final_rng = random.Random(seed + 100_003)
    final_rng.shuffle(selected)

    return selected


# ============================================================================
# VUES DE SORTIE
# ============================================================================

def annotator_view(record: dict[str, Any]) -> dict[str, str]:
    """
    Champs visibles par le second annotateur.

    Aucun label, aucune référence juridique et aucune borne gold
    ne sont exposés.
    """
    return {
        "id": record["id"],
        "tokenized_premise": record["tokenized_premise"],
        "hypothesis_facts": record["hypothesis_facts"],
        "premise": record["premise"],
    }


def gold_view(record: dict[str, Any]) -> dict[str, Any]:
    """Gold aligné avec le fichier annotateur."""
    return {
        "id": record["id"],
        "label": record["label"],
        "rationale_start_token": record["rationale_start_token"],
        "rationale_end_token": record["rationale_end_token"],
    }


# ============================================================================
# VALIDATION ET ÉCRITURE
# ============================================================================

def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Écrit exactement un objet JSON par ligne."""
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


def validate_selection(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Vérifie toutes les contraintes avant l'écriture."""
    ids = [record["id"] for record in selected]
    law_keys = [record["law_key"] for record in selected]
    labels = Counter(record["label"] for record in selected)

    expected_total = sum(LABEL_QUOTAS.values())

    non_neutral = [
        record
        for record in selected
        if record["label"] != "neutral"
    ]

    neutral = [
        record
        for record in selected
        if record["label"] == "neutral"
    ]

    validation = {
        "input_file": str(INPUT_FILE),
        "selection_seed": SELECTION_SEED,
        "selected_instances": len(selected),
        "expected_instances": expected_total,
        "unique_ids": len(set(ids)),
        "unique_law_refs": len(set(law_keys)),
        "label_distribution": dict(sorted(labels.items())),
        "non_neutral_instances": len(non_neutral),
        "neutral_instances": len(neutral),
        "correct_total": len(selected) == expected_total,
        "all_ids_unique": len(ids) == len(set(ids)),
        "all_law_refs_unique": len(law_keys) == len(set(law_keys)),
        "correct_label_quotas": all(
            labels[label] == quota
            for label, quota in LABEL_QUOTAS.items()
        ),
        "all_non_neutral_spans_present": all(
            record["rationale_start_token"] is not None
            and record["rationale_end_token"] is not None
            for record in non_neutral
        ),
        "all_neutral_spans_empty": all(
            record["rationale_start_token"] is None
            and record["rationale_end_token"] is None
            for record in neutral
        ),
    }

    required_checks = (
        "correct_total",
        "all_ids_unique",
        "all_law_refs_unique",
        "correct_label_quotas",
        "all_non_neutral_spans_present",
        "all_neutral_spans_empty",
    )

    if not all(validation[check] for check in required_checks):
        raise RuntimeError(
            "La validation finale de la sélection a échoué : "
            f"{validation}"
        )

    return validation


def write_audit(
    path: Path,
    validation: dict[str, Any],
    selected: list[dict[str, Any]],
) -> None:
    """Enregistre un rapport reproductible sans exposer les données textuelles."""
    audit = {
        **validation,
        "output_annotator_file": str(OUTPUT_FILE),
        "output_gold_file": str(GOLD_FILE),
        "selected_ids_in_output_order": [
            record["id"]
            for record in selected
        ],
    }

    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    raw_records = load_records(INPUT_FILE)

    records = [
        prepare_record(record, index)
        for index, record in enumerate(raw_records)
    ]

    input_ids = [record["id"] for record in records]

    duplicate_ids = [
        record_id
        for record_id, count in Counter(input_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            "Identifiants dupliqués dans le corpus : "
            f"{duplicate_ids[:10]}"
        )

    selected = select_records(records, SELECTION_SEED)
    validation = validate_selection(selected)

    annotator_records = [
        annotator_view(record)
        for record in selected
    ]

    gold_records = [
        gold_view(record)
        for record in selected
    ]

    # Les deux fichiers sont écrits dans exactement le même ordre.
    write_jsonl(OUTPUT_FILE, annotator_records)
    write_jsonl(GOLD_FILE, gold_records)
    write_audit(AUDIT_FILE, validation, selected)

    print("Sous-ensemble Kappa + rationales créé.")
    print(f"Racine du projet        : {PROJECT_ROOT}")
    print(f"Entrée                  : {INPUT_FILE}")
    print(f"Fichier annotateur      : {OUTPUT_FILE}")
    print(f"Fichier gold            : {GOLD_FILE}")
    print(f"Rapport d'audit         : {AUDIT_FILE}")
    print(f"Graine                  : {SELECTION_SEED}")
    print(f"Instances sélectionnées : {validation['selected_instances']}")
    print(f"Identifiants uniques    : {validation['unique_ids']}")
    print(f"law_ref distinctes      : {validation['unique_law_refs']}")
    print("Répartition des labels  :")

    for label in ("entailment", "contradiction", "neutral"):
        print(
            f"  {label:<13}: "
            f"{validation['label_distribution'].get(label, 0)}"
        )

    print(
        "Champs annotateur       : "
        "id, tokenized_premise, hypothesis_facts, premise"
    )
    print(
        "Champs gold             : "
        "id, label, rationale_start_token, rationale_end_token"
    )
    print("Labels dans annotateur  : non")
    print("Rationales dans annotateur : non")
    print("law_ref dans annotateur : non")
    print("Ordre annotateur/gold   : identique")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
