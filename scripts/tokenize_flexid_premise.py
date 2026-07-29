#!/usr/bin/env python3


from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION — CHEMINS RELATIFS AU DOSSIER DU SCRIPT
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "flexid_shuffled.jsonl"
OUTPUT_FILE = DATA_DIR / "flexid_shuffled_tokenized.jsonl"
WARNINGS_FILE = DATA_DIR / "flexid_shuffled_tokenization_warnings.txt"

ALLOWED_LABELS = {"entailment", "contradiction", "neutral"}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def nfc(value: Any) -> str:
    """Normalise une valeur textuelle en Unicode NFC."""
    return unicodedata.normalize("NFC", str(value or ""))


def tokenize_with_offsets(text: str) -> list[dict[str, Any]]:
    """
    Tokenisation simple et déterministe : chaque séquence non blanche est
    un token. La ponctuation et les apostrophes restent attachées aux mots.
    """
    return [
        {
            "token_id": token_id,
            "text": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
        }
        for token_id, match in enumerate(re.finditer(r"\S+", text), start=1)
    ]


def make_tokenized_premise(tokens: list[dict[str, Any]]) -> str:
    """Crée une prémisse numérotée, adaptée aux prompts ou à l'annotation."""
    return "\n".join(
        f"[{token['token_id']}] {token['text']}"
        for token in tokens
    )


def char_span_to_token_span(
    tokens: list[dict[str, Any]],
    start_char: int,
    end_char: int,
) -> tuple[int | None, int | None, list[int]]:
    """
    Convertit [start_char, end_char) en un intervalle de tokens inclusif.
    Un token est sélectionné dès qu'il chevauche le span caractère.
    """
    selected_ids = [
        token["token_id"]
        for token in tokens
        if token["char_end"] > start_char
        and token["char_start"] < end_char
    ]

    if not selected_ids:
        return None, None, []

    return min(selected_ids), max(selected_ids), selected_ids


def token_span_text(
    tokens: list[dict[str, Any]],
    start_token: int | None,
    end_token: int | None,
) -> str:
    """Reconstruit une vue lisible du span tokenisé à des fins de contrôle."""
    if start_token is None or end_token is None:
        return ""

    return " ".join(
        token["text"]
        for token in tokens
        if start_token <= token["token_id"] <= end_token
    )


def validate_character_span(
    instance_id: str,
    premise: str,
    start_char: int,
    end_char: int,
    rationale_text: str,
) -> list[str]:
    """Vérifie les bornes et l'égalité exacte avec rationale_text."""
    messages: list[str] = []

    if start_char < 0:
        messages.append(
            f"ERREUR {instance_id}: rationale_start négatif ({start_char})."
        )

    if end_char < start_char:
        messages.append(
            f"ERREUR {instance_id}: rationale_end ({end_char}) est inférieur "
            f"à rationale_start ({start_char})."
        )

    if end_char > len(premise):
        messages.append(
            f"ERREUR {instance_id}: rationale_end ({end_char}) dépasse "
            f"la longueur de la prémisse ({len(premise)})."
        )

    if messages:
        return messages

    extracted = premise[start_char:end_char]
    if extracted != rationale_text:
        messages.append(
            f"ERREUR {instance_id}: désaccord offsets/rationale_text.\n"
            f"  offsets   : [{start_char}, {end_char})\n"
            f"  extrait   : {extracted!r}\n"
            f"  rationale : {rationale_text!r}"
        )

    return messages


# ============================================================================
# TRAITEMENT D'UNE INSTANCE
# ============================================================================

def process_instance(
    item: dict[str, Any],
    line_number: int,
) -> tuple[dict[str, Any], list[str]]:
    messages: list[str] = []

    instance_id = str(item.get("id", "")).strip()
    if not instance_id:
        raise ValueError(f"Ligne {line_number}: champ id absent.")

    raw_premise = str(item.get("premise", ""))
    premise = nfc(raw_premise)
    hypothesis_facts = nfc(
        item.get("hypothesis_facts", item.get("hypothesis", ""))
    )
    label = str(item.get("label", "")).strip().casefold()
    rationale_text = nfc(item.get("rationale_text", ""))

    if not premise:
        raise ValueError(f"{instance_id}: prémisse vide.")

    if not hypothesis_facts:
        raise ValueError(f"{instance_id}: hypothèse vide.")

    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"{instance_id}: label invalide {label!r}. "
            f"Labels autorisés : {sorted(ALLOWED_LABELS)}."
        )

    try:
        rationale_start_char = int(item.get("rationale_start", 0))
        rationale_end_char = int(item.get("rationale_end", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{instance_id}: offsets caractères non entiers."
        ) from exc

    # Les offsets gold sont supposés avoir été établis sur un texte NFC.
    if raw_premise != premise:
        messages.append(
            f"ERREUR {instance_id}: la prémisse source n'était pas déjà en NFC; "
            "les offsets caractères doivent être revérifiés."
        )

    tokens = tokenize_with_offsets(premise)
    tokenized_premise = make_tokenized_premise(tokens)

    is_empty_span = (
        rationale_start_char == 0
        and rationale_end_char == 0
        and rationale_text == ""
    )

    if label == "neutral":
        if not is_empty_span:
            messages.append(
                f"ERREUR {instance_id}: neutral exige rationale_start=0, "
                'rationale_end=0 et rationale_text="".'
            )

        rationale_start_token = None
        rationale_end_token = None
        rationale_token_text = ""

    else:
        if is_empty_span:
            messages.append(
                f"ERREUR {instance_id}: {label} exige un rationale non vide."
            )

        messages.extend(
            validate_character_span(
                instance_id=instance_id,
                premise=premise,
                start_char=rationale_start_char,
                end_char=rationale_end_char,
                rationale_text=rationale_text,
            )
        )

        (
            rationale_start_token,
            rationale_end_token,
            _selected_token_ids,
        ) = char_span_to_token_span(
            tokens=tokens,
            start_char=rationale_start_char,
            end_char=rationale_end_char,
        )

        if rationale_start_token is None:
            messages.append(
                f"ERREUR {instance_id}: aucun token ne chevauche le rationale."
            )

        rationale_token_text = token_span_text(
            tokens=tokens,
            start_token=rationale_start_token,
            end_token=rationale_end_token,
        )

    processed = {
        "id": instance_id,
        "premise": premise,
        "tokenized_premise": tokenized_premise,
        "hypothesis_facts": hypothesis_facts,
        "label": label,
        "rationale_start_char": rationale_start_char,
        "rationale_end_char": rationale_end_char,
        "rationale_text": rationale_text,
        "rationale_start_token": rationale_start_token,
        "rationale_end_token": rationale_end_token,
        "rationale_token_text": rationale_token_text,
        "premise_tokens": tokens,
        "meta": item.get("meta", {}),
    }

    return processed, messages


# ============================================================================
# TRAITEMENT DU DATASET JSONL
# ============================================================================

def write_report(path: Path, messages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if messages:
        path.write_text("\n\n".join(messages) + "\n", encoding="utf-8")
    else:
        path.write_text(
            "Aucun warning. Toutes les instances ont été traitées "
            "correctement.\n",
            encoding="utf-8",
        )


def process_dataset(
    input_file: Path,
    output_file: Path,
    warnings_file: Path,
) -> int:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_file}\n"
            "Le script doit être placé dans scripts/ et le corpus dans "
            "data/flexid_shuffled.jsonl."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    total = 0
    written = 0

    with (
        input_file.open("r", encoding="utf-8-sig") as input_handle,
        output_file.open("w", encoding="utf-8", newline="\n") as output_handle,
    ):
        for line_number, raw_line in enumerate(input_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                messages.append(
                    f"ERREUR ligne {line_number}: JSONL invalide : {exc}"
                )
                continue

            if not isinstance(item, dict):
                messages.append(
                    f"ERREUR ligne {line_number}: objet JSON attendu."
                )
                continue

            try:
                processed, instance_messages = process_instance(
                    item=item,
                    line_number=line_number,
                )

                instance_id = processed["id"]
                if instance_id in seen_ids:
                    messages.append(
                        f"ERREUR {instance_id}: identifiant dupliqué."
                    )
                    continue

                seen_ids.add(instance_id)
                label_counts[processed["label"]] += 1

                output_handle.write(
                    json.dumps(
                        processed,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                written += 1
                messages.extend(instance_messages)

            except Exception as exc:
                instance_id = item.get("id", "ID inconnu")
                messages.append(
                    f"ERREUR ligne {line_number} ({instance_id}): {exc}"
                )

    write_report(warnings_file, messages)

    error_count = sum(message.startswith("ERREUR") for message in messages)

    print("Tokenisation FLEXID terminée.")
    print(f"Projet détecté          : {PROJECT_ROOT}")
    print(f"Entrée                  : {input_file}")
    print(f"Sortie                  : {output_file}")
    print(f"Rapport                 : {warnings_file}")
    print(f"Instances lues          : {total}")
    print(f"Instances écrites       : {written}")
    print(f"Identifiants uniques    : {len(seen_ids)}")
    print(
        "Répartition des labels  : "
        f"entailment={label_counts['entailment']}, "
        f"contradiction={label_counts['contradiction']}, "
        f"neutral={label_counts['neutral']}"
    )
    print(f"Anomalies détectées     : {error_count}")

    if written != total or error_count:
        print(
            "ATTENTION : consulte le rapport avant d'utiliser la sortie.",
            file=sys.stderr,
        )
        return 1

    print("Validation              : réussie")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            process_dataset(
                input_file=INPUT_FILE,
                output_file=OUTPUT_FILE,
                warnings_file=WARNINGS_FILE,
            )
        )
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
