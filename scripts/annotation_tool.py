
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import unicodedata
import re
import hashlib
from collections import Counter
from pathlib import Path


path_abs = Path("data/flexid.jsonl")

MAX_PER_PREMISE = 6


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def hash_premise(premise: str) -> str:
    prem_norm = nfc(premise).strip()
    prem_norm = re.sub(r"\s+", " ", prem_norm)
    return hashlib.sha256(prem_norm.encode("utf-8")).hexdigest()


def validate_entry(entry, existing_premise_stats: dict, existing_pair_hashes: set):
    errors = []

    required_fields = [
        "id",
        "premise",
        "hypothesis_facts",
        "label",
        "rationale_start",
        "rationale_end",
        "rationale_text",
        "meta"
    ]

    for k in required_fields:
        if k not in entry:
            errors.append(f"Champ manquant: {k}")

    if errors:
        return False, errors

    entry_id = str(entry["id"]).strip()
    premise = nfc(entry["premise"])
    hypothesis_facts = nfc(entry["hypothesis_facts"])
    rationale_text = nfc(entry.get("rationale_text", ""))

    if not entry_id:
        errors.append("ID vide.")

    if not re.fullmatch(r"FLEXID-\d{4}", entry_id):
        errors.append("ID invalide. Format attendu : FLEXID-0001")

    if len(premise) == 0:
        errors.append("Prémisse vide.")

    if len(hypothesis_facts) == 0:
        errors.append("Faits / hypothesis_facts vide.")

    if entry["label"] not in {"entailment", "contradiction", "neutral"}:
        errors.append(f"Label invalide: {entry['label']}")

    try:
        start = int(entry["rationale_start"])
        end = int(entry["rationale_end"])
    except Exception:
        errors.append("Offsets non numériques.")
        start = end = -1

    is_neutral_void = entry["label"] == "neutral" and start == 0 and end == 0

    if not is_neutral_void:
        if not (0 <= start < end <= len(premise)):
            errors.append(f"Offsets hors bornes ou inversés: [{start}, {end})")
        else:
            slice_txt = premise[start:end]

            if rationale_text and rationale_text != slice_txt:
                errors.append("rationale_text != premise[start:end].")

            if not slice_txt.strip():
                errors.append("Rationale vide.")

    if rationale_text and rationale_text in hypothesis_facts:
        errors.append("Rationale recopiée dans hypothesis_facts.")

    meta = entry.get("meta", {}) or {}
    law_ref = meta.get("law_ref", "").strip()

    if not law_ref:
        errors.append("meta.law_ref manquant.")

    h = hash_premise(premise)
    stats = existing_premise_stats.get(h, 0)

    if stats >= MAX_PER_PREMISE:
        errors.append("Limite par prémisse atteinte.")

    return len(errors) == 0, errors


def quality_warnings(all_entries: list):
    warns = []

    if not all_entries:
        return warns

    total = len(all_entries)
    labels = Counter(e.get("label", "NA") for e in all_entries)

    if total >= 20:
        ent = labels.get("entailment", 0)
        con = labels.get("contradiction", 0)
        neu = labels.get("neutral", 0)

        counts = [ent, con, neu]

        if max(counts) > 0 and min(counts) / max(counts) < 0.67:
            warns.append(f"Déséquilibre labels: E={ent}, C={con}, N={neu}.")

    return warns


class FLEXIDManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Gestionnaire d'Entrées FLEXID")
        self.root.geometry("1020x850")

        os.makedirs("data", exist_ok=True)

        if path_abs.exists():
            self.filename = str(path_abs)
        else:
            self.filename = os.path.join("data", "flexid.jsonl")

        self.entries = {}
        self.mode = None
        self.selection_mode = None
        self.rationale_start = 0
        self.rationale_end = 0

        self.load_entries()
        self.create_widgets()

    def load_entries(self):
        self.entries = {}

        if not os.path.exists(self.filename):
            return

        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)

                        if "hypothesis" in e and "hypothesis_facts" not in e:
                            e["hypothesis_facts"] = e.pop("hypothesis")

                        e.pop("implicit_conclusion", None)
                        e.pop("split", None)

                        meta = e.get("meta", {}) or {}
                        meta.pop("source", None)

                        e["meta"] = {
                            "law_ref": meta.get("law_ref", ""),
                            "notes": meta.get("notes", "")
                        }

                        if "id" in e:
                            self.entries[e["id"]] = e

            messagebox.showinfo(
                "Chargement",
                f"{len(self.entries)} entrées chargées."
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de lecture: {e}")

    def save_entries(self):
        try:
            parent = os.path.dirname(self.filename)

            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(self.filename, "w", encoding="utf-8") as f:
                for entry in self.entries.values():
                    clean_entry = self.clean_entry_for_save(entry)
                    f.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")

            return True

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return False

    def clean_entry_for_save(self, entry):
        meta = entry.get("meta", {}) or {}

        return {
            "id": entry.get("id", ""),
            "premise": nfc(entry.get("premise", "")),
            "hypothesis_facts": nfc(entry.get("hypothesis_facts", "")),
            "label": entry.get("label", ""),
            "rationale_start": int(entry.get("rationale_start", 0)),
            "rationale_end": int(entry.get("rationale_end", 0)),
            "rationale_text": nfc(entry.get("rationale_text", "")),
            "meta": {
                "law_ref": meta.get("law_ref", ""),
                "notes": meta.get("notes", "")
            }
        }

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Gestionnaire FLEXID",
            font=("Arial", 16, "bold")
        ).pack(pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=5)

        ttk.Button(
            button_frame,
            text="Nouvelle Entrée",
            command=self.new_entry_mode
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Modifier",
            command=self.edit_mode
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Sauvegarder",
            command=self.save_current_entry
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Liste",
            command=self.show_entries_list
        ).pack(side=tk.LEFT, padx=5)

        fields_frame = ttk.Frame(main_frame)
        fields_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        fields_frame.columnconfigure(1, weight=1)

        ttk.Label(fields_frame, text="ID:").grid(row=0, column=0, sticky=tk.W)

        self.id_var = tk.StringVar()
        self.id_entry = ttk.Entry(fields_frame, textvariable=self.id_var, width=30)
        self.id_entry.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(fields_frame, text="Prémisse:").grid(row=1, column=0, sticky=tk.NW)

        self.premise_text = scrolledtext.ScrolledText(fields_frame, height=5, width=90)
        self.premise_text.grid(row=1, column=1, pady=2, sticky=tk.W)
        self.premise_text.bind("<Button-1>", self.on_premise_click)

        ttk.Label(fields_frame, text="Faits:").grid(row=2, column=0, sticky=tk.NW)

        self.hypothesis_facts_text = scrolledtext.ScrolledText(fields_frame, height=4, width=90)
        self.hypothesis_facts_text.grid(row=2, column=1, pady=2, sticky=tk.W)

        ttk.Label(fields_frame, text="Label:").grid(row=3, column=0, sticky=tk.W)

        self.label_var = tk.StringVar(value="entailment")
        label_frame = ttk.Frame(fields_frame)
        label_frame.grid(row=3, column=1, sticky=tk.W)

        for lab in ["entailment", "contradiction", "neutral"]:
            ttk.Radiobutton(
                label_frame,
                text=lab,
                variable=self.label_var,
                value=lab
            ).pack(side=tk.LEFT, padx=5)

        ttk.Label(fields_frame, text="Rationale:").grid(row=4, column=0, sticky=tk.NW)

        self.rationale_text = scrolledtext.ScrolledText(fields_frame, height=4, width=90)
        self.rationale_text.grid(row=4, column=1, pady=2, sticky=tk.W)

        rat_ctrl = ttk.Frame(fields_frame)
        rat_ctrl.grid(row=5, column=1, sticky=tk.W)

        ttk.Button(
            rat_ctrl,
            text="📍 Début",
            command=self.activate_start_selection
        ).pack(side=tk.LEFT)

        ttk.Button(
            rat_ctrl,
            text="📍 Fin",
            command=self.activate_end_selection
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            rat_ctrl,
            text="❌ Vider Neutral",
            command=self.clear_rationale_neutral
        ).pack(side=tk.LEFT, padx=10)

        self.selection_info = tk.StringVar(value="Offsets: [0, 0]")
        ttk.Label(
            rat_ctrl,
            textvariable=self.selection_info,
            foreground="blue"
        ).pack(side=tk.LEFT)

        ttk.Label(fields_frame, text="Réf. Légale:").grid(row=6, column=0, sticky=tk.W)

        self.law_ref_var = tk.StringVar()
        ttk.Entry(
            fields_frame,
            textvariable=self.law_ref_var,
            width=90
        ).grid(row=6, column=1, sticky=tk.W, pady=2)

        ttk.Label(fields_frame, text="Notes:").grid(row=7, column=0, sticky=tk.W)

        self.notes_var = tk.StringVar()
        ttk.Entry(
            fields_frame,
            textvariable=self.notes_var,
            width=90
        ).grid(row=7, column=1, sticky=tk.W, pady=2)

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def activate_start_selection(self):
        self.selection_mode = "start"
        self.status_var.set("Cliquez dans la prémisse pour choisir le début de la rationale.")

    def activate_end_selection(self):
        self.selection_mode = "end"
        self.status_var.set("Cliquez dans la prémisse pour choisir la fin de la rationale.")

    def clear_rationale_neutral(self):
        self.rationale_start = 0
        self.rationale_end = 0
        self.rationale_text.delete("1.0", tk.END)
        self.selection_info.set("Offsets: [0, 0] Neutral autorisé")

    def on_premise_click(self, event):
        if not self.selection_mode:
            return

        index = self.premise_text.index(f"@{event.x},{event.y}")
        pos = len(nfc(self.premise_text.get("1.0", index)))

        if self.selection_mode == "start":
            self.rationale_start = pos
            self.selection_info.set(f"Début: {pos}")

        elif self.selection_mode == "end":
            self.rationale_end = pos
            self.extract_rationale()

        self.selection_mode = None
        self.status_var.set("Prêt")

    def extract_rationale(self):
        p = nfc(self.premise_text.get("1.0", "end-1c"))

        if self.rationale_start > self.rationale_end:
            messagebox.showwarning(
                "Offsets invalides",
                "Le début est après la fin. Veuillez recommencer la sélection."
            )
            return

        txt = p[self.rationale_start:self.rationale_end]

        self.rationale_text.delete("1.0", tk.END)
        self.rationale_text.insert("1.0", txt)
        self.selection_info.set(f"Offsets: [{self.rationale_start}, {self.rationale_end}]")

    def get_next_id(self):
        max_id = 0

        for eid in self.entries:
            try:
                if eid.startswith("FLEXID-"):
                    number_part = eid.split("-")[1]
                    max_id = max(max_id, int(number_part))
            except Exception:
                continue

        return f"FLEXID-{str(max_id + 1).zfill(4)}"

    def normalize_entry_id(self, value: str):
        value = str(value or "").strip()

        if not value:
            return None

        if value in self.entries:
            return value

        if value.upper().startswith("FLEXID-"):
            candidate = value.upper()

            if candidate in self.entries:
                return candidate

            return None

        if value.isdigit():
            candidate = f"FLEXID-{int(value):04d}"

            if candidate in self.entries:
                return candidate

            candidate_raw = f"FLEXID-{value.zfill(4)}"

            if candidate_raw in self.entries:
                return candidate_raw

        return None

    def new_entry_mode(self):
        self.mode = "new"
        self.clear_fields()
        self.id_var.set(self.get_next_id())
        self.status_var.set("Mode nouvelle entrée.")

    def edit_mode(self):
        user_value = simpledialog.askstring(
            "Modifier une entrée",
            "Entrez le numéro ou l'ID de l'instance à modifier.\n\nExemples: 7, 0007 ou FLEXID-0007"
        )

        if user_value is None:
            return

        eid = self.normalize_entry_id(user_value)

        if not eid:
            messagebox.showerror(
                "Introuvable",
                f"Aucune entrée trouvée pour: {user_value}"
            )
            return

        self.load_entry_into_form(eid)
        self.mode = "edit"
        self.status_var.set(f"Mode modification: {eid}")

    def load_entry_into_form(self, eid):
        if eid not in self.entries:
            messagebox.showerror("Erreur", f"ID introuvable: {eid}")
            return

        e = self.entries[eid]

        self.clear_fields()

        self.id_var.set(e.get("id", eid))

        self.premise_text.insert("1.0", e.get("premise", ""))
        self.hypothesis_facts_text.insert("1.0", e.get("hypothesis_facts", ""))

        self.label_var.set(e.get("label", "entailment"))

        self.rationale_start = int(e.get("rationale_start", 0))
        self.rationale_end = int(e.get("rationale_end", 0))

        self.rationale_text.delete("1.0", tk.END)
        self.rationale_text.insert("1.0", e.get("rationale_text", ""))

        self.selection_info.set(
            f"Offsets: [{self.rationale_start}, {self.rationale_end}]"
        )

        meta = e.get("meta", {}) or {}
        self.law_ref_var.set(meta.get("law_ref", ""))
        self.notes_var.set(meta.get("notes", ""))

    def clear_fields(self):
        self.id_var.set("")
        self.law_ref_var.set("")
        self.notes_var.set("")

        self.label_var.set("entailment")

        for text_widget in [
            self.premise_text,
            self.hypothesis_facts_text,
            self.rationale_text
        ]:
            text_widget.delete("1.0", tk.END)

        self.rationale_start = 0
        self.rationale_end = 0
        self.selection_mode = None
        self.selection_info.set("Offsets: [0, 0]")

    def build_current_entry(self):
        return {
            "id": self.id_var.get().strip(),
            "premise": nfc(self.premise_text.get("1.0", "end-1c")),
            "hypothesis_facts": nfc(self.hypothesis_facts_text.get("1.0", "end-1c")),
            "label": self.label_var.get(),
            "rationale_start": self.rationale_start,
            "rationale_end": self.rationale_end,
            "rationale_text": nfc(self.rationale_text.get("1.0", "end-1c")),
            "meta": {
                "law_ref": self.law_ref_var.get().strip(),
                "notes": self.notes_var.get().strip()
            }
        }

    def save_current_entry(self):
        entry = self.build_current_entry()

        ok, errs = validate_entry(entry, {}, set())

        if not ok:
            messagebox.showerror("Erreur de validation", "\n".join(errs))
            return

        self.entries[entry["id"]] = entry

        if self.save_entries():
            saved_id = entry["id"]

            messagebox.showinfo(
                "Succès",
                f"Entrée {saved_id} sauvegardée."
            )

            self.clear_fields()
            self.mode = None
            self.status_var.set(f"Entrée {saved_id} sauvegardée. Champs vidés.")

    def show_entries_list(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Liste des entrées FLEXID")
        dialog.geometry("850x650")

        txt = scrolledtext.ScrolledText(dialog, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True)

        labels = Counter(e.get("label", "NA") for e in self.entries.values())

        txt.insert(tk.END, "STATISTIQUES\n")
        txt.insert(tk.END, "=" * 40 + "\n\n")
        txt.insert(tk.END, f"entailment: {labels.get('entailment', 0)}\n")
        txt.insert(tk.END, f"contradiction: {labels.get('contradiction', 0)}\n")
        txt.insert(tk.END, f"neutral: {labels.get('neutral', 0)}\n\n")
        txt.insert(tk.END, f"TOTAL: {len(self.entries)} entrées\n\n")

        warns = quality_warnings(list(self.entries.values()))

        if warns:
            txt.insert(tk.END, "AVERTISSEMENTS QUALITÉ\n")
            txt.insert(tk.END, "=" * 40 + "\n")

            for w in warns:
                txt.insert(tk.END, f"- {w}\n")

            txt.insert(tk.END, "\n")

        txt.insert(tk.END, "LISTE DES ENTRÉES\n")
        txt.insert(tk.END, "=" * 40 + "\n\n")

        for eid, e in sorted(self.entries.items()):
            txt.insert(
                tk.END,
                f"{eid} | {e.get('label', '')}\n"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = FLEXIDManager(root)
    root.mainloop()
