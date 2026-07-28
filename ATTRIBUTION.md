# How to Attribute FLEXID

## Short attribution

> FLEXID — French Legal Explainable Inference Dataset, © 2026 Aboudourazakou
> Tetereou, Tarik Boudaa, and El Wardani Dadi, CC BY 4.0. Official French legal
> information: DILA/Légifrance and Cour de cassation/Judilibre, Licence
> Ouverte 2.0.

## Attribution for an academic publication

Please:

1. cite the FLEXID paper;
2. identify the dataset release tag, version, or archived record used;
3. link to the public dataset repository or archive;
4. state whether the dataset was modified; and
5. retain the official legal-source references distributed with each instance.

Suggested prose:

> We use FLEXID (French Legal Explainable Inference Dataset), © 2026
> Aboudourazakou Tetereou, Tarik Boudaa, and El Wardani Dadi, under CC BY 4.0.
> We used the release identified in our reproducibility materials. Official
> French legal information incorporated in FLEXID originates from
> DILA/Légifrance and, where applicable, the Cour de cassation/Judilibre, and
> remains subject to Licence Ouverte 2.0.

## BibTeX

Update this entry with the journal, DOI, volume, pages, and final publication
year once the article is published.

```bibtex
@misc{tetereou2026flexid,
  title        = {{FLEXID}: A Benchmark for Explainable Legal Inference
                  in French Civil Law, with Rationale Spans},
  author       = {Tetereou, Aboudourazakou and Boudaa, Tarik and
                  Dadi, El Wardani},
  year         = {2026},
  howpublished = {Dataset and accompanying manuscript},
  url          = {https://github.com/TchalosonForResearch/flexid},
  note         = {Dataset licensed under CC BY 4.0; incorporated official
                  French legal information remains subject to Licence
                  Ouverte 2.0}
}
```

## Attribution for a derivative dataset

Suggested wording:

> This dataset is adapted from FLEXID, © 2026 Aboudourazakou Tetereou, Tarik
> Boudaa, and El Wardani Dadi, used under CC BY 4.0. The authors of FLEXID do
> not endorse this derivative. A description of all changes is provided in the
> derivative dataset's change log. Official French legal information remains
> attributed to its original public producer and subject to Licence Ouverte
> 2.0 and the applicable source terms.

## Attribution inside a model or dataset card

```yaml
license: cc-by-4.0
license_name: CC BY 4.0 for original FLEXID contributions
license_link: https://creativecommons.org/licenses/by/4.0/legalcode
```

Add the following text directly below the metadata:

> The license field above applies to original FLEXID contributions. Official
> legal texts and related source metadata remain subject to Licence Ouverte
> 2.0 and applicable Légifrance or Cour de cassation/Judilibre terms. See
> `THIRD_PARTY_NOTICES.md`.
