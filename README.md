# FLEXID reproducibility scripts

This directory contains the scripts aligned with the experimental protocol
reported in the FLEXID manuscript.


## Pipeline

0. `annotator_tool.py`
   Is used to fill the  fields of each instance with data
1. `shuffle_flexid_deterministic.py`
   creates the mixed-order corpus used by the downstream preparation scripts.
2. `tokenize_flexid_premise.py`
   maps character rationales to inclusive whitespace-token spans.
3. `select_flexid_kappa_rationales.py`
   creates the blind 180-instance evaluation sample and aligned gold file.
4. `annotate_flexid_deepseekapi.py`
   produces the zero-shot model predictions. Set `DEEPSEEK_API_KEY` in the
   environment;.
5. `metrics_calculator_human_and_deepseek.py`
   evaluates the external human annotation series and DeepSeek predictions.
6. `premise_hypothesis_classifier.py`
   runs the post-correction partial-input and relational audit, including the
   paired group-bootstrap intervals for the relational accuracy gains.
7. `create_flexid_group_split.py`
   creates the fixed exact-string group-disjoint split.
8. `train_camembert_judibert.py`
   trains CamemBERT with seeds 2026/2027/2028 and the exploratory JuriBERT run
   with seed 2026.


## Important protocol distinctions

- The relational audit uses 339 groups formed from normalised legal-reference,
  premise and hypothesis strings.
- The released supervised split uses 340 connected components based on exact
  decoded JSON equality of the legal-reference and premise strings. It applies
  no canonicalisation.
- Rationale tokens are maximal non-whitespace sequences; punctuation remains
  attached.


## Environment

Python 3.10 or later is required. Install the packages listed in
`requirements.in`, then record the fully resolved environment used for the
published run (for example with `python -m pip freeze`) alongside the released
outputs. Model repository revisions must also be recorded before the final
experiments are rerun.


