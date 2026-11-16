# Final report source and verification

`main.tex` preserves the original `acmart` options `sigconf,nonacm`, title, authors,
affiliations, top-matter settings and page style. Its subtitle and research content
are updated. References remain inline in `thebibliography`; no BibTeX installation
or external bibliography file is required. The original report folder was removed
with explicit user approval after its four images were verified in this project
and a byte-verified backup of the complete folder was saved outside the repository.

## Build and outstanding acceptance

No LaTeX compiler was available on PATH or in the standard checked macOS TeX
locations during validation. No checkpoint PDF was found in the supplied report
directory. Consequently **no compiled paper preview is delivered**, and page
layout, bibliography rendering, overfull boxes, float placement and page-level
comparison remain unverified. The confusion-matrix PDF is a figure, not the paper.

On a machine with the original ACM class/fonts and a compatible LaTeX engine
already installed, compile from this portable folder. For pdfLaTeX:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Inspect the resulting PDF and compilation log before submission. Static checks
verified citation keys, figure paths, balanced environments/braces and original
format directives; those checks cannot replace compilation and visual review.

To regenerate the new scientific figure with the existing project environment:

```sh
python -B generate_figures.py
```

The local `figures/grouped_confusion_source.json` contains the reconciled 3x3
counts, cohort definition and hashes of the saved experimental prediction inputs.
It preserves the small figure input without copying trained models or datasets.

## Asset accounting

All four original PNGs are copied unchanged into `figures/`:

| Original filename | Use in final source |
| --- | --- |
| `architecture.png` | Included; caption distinguishes historical acquisition from cached audit |
| `cluster_pca.png` | Preserved; superseded by `cluster_pca_verified.png` from `outputs/06_clusters_pca.png` |
| `feature_importance.png` | Preserved but omitted: it does not explain the corrected grouped experimental model |
| `skill_network.png` | Preserved; superseded by `skill_network_verified.png` from `outputs/04_skill_network.png` |

The updated PCA/network are existing post-extraction historical figures, not new
fits. `grouped_confusion.pdf` is newly plotted from saved unadjusted fold-zero
predictions. Its count total is 6,836 and its labels/color scale were visually
checked. The original source was used as the formatting reference before its folder was
removed with explicit user approval; the verified external backup retains it.

## Citation verification and changes

All original citation keys are retained. Corrected metadata: the 2023 textbook's
authors are Han, Pei and Tong; `senger2024` now cites its published NLP4HR version;
`samwi2024` is a retained key for the verified **2026** paper; `debao2021` has three
authors, not four; the Kablaoui DOI is `10.1109/ICECTA57148.2022.9990316`; the
Khaouja DOI is `10.1109/ACCESS.2021.3106120`.

New peer-reviewed references and supported claims:

| Key | Supported methodological claim | Verified primary source |
| --- | --- | --- |
| `skillspan2022` | Skill extraction requires span annotations and explicit guidelines; it does not validate this prototype | https://aclanthology.org/2022.naacl-main.366/ |
| `hamalainen2019` | Frequency/lift filters alone do not control false discoveries across pattern searches | https://doi.org/10.1007/s10618-018-0590-x |
| `kapoor2023` | Leakage and evaluation design can invalidate ML-based scientific claims | https://doi.org/10.1016/j.patter.2023.100804 |
| `sandve2013` | Recording inputs and computational steps supports computational reproducibility | https://doi.org/10.1371/journal.pcbi.1003285 |

Existing references were reconciled with these primary records:

- https://hanj.cs.illinois.edu/bk4/
- https://aclanthology.org/2024.nlp4hr-1.1/
- https://www.journal-isi.org/index.php/isi/article/view/1671
- https://pure.ku.edu.kw/en/publications/machine-learning-models-for-salary-prediction-dataset-using-pytho/
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255419
- https://ensias.um5.ac.ma/content/survey-skill-identification-online-job-ads
- https://www.mdpi.com/2078-2489/13/10/495

## Final validation scope

- Backend: **154 passed**, six undefined-precision warnings from synthetic fixtures;
  installed requirements and `pip check` pass under Python 3.14.2, not a clean
  install or actual execution of the declared Python 3.12 CI target.
- Frontend: **110 passed**, TypeScript check and production build pass. Build
  output was isolated outside the repository; a large-main-bundle warning remains.
- Production User/Admin routes and separate lazy playground routing were inspected
  and tested. jsdom chart/navigation tests do not establish real-browser acceptance.
  Browser host access was unavailable; a native Safari probe timed out.
- Read-only SQL integrity/count checks, cached prediction/rule re-evaluation,
  source-linked salary export and original-file hash checks were performed without
  model fitting, job APIs, dependency installation, staging or committing.

Detailed machine-readable validation and the internal numerical evidence ledger
remain in `/private/tmp/skillmap-final-validation-v1/`. Phase 2 evidence remains in
`/private/tmp/skillmap-evidence-v1-20261115/`; temporary directories are not a durable
research archive. The report explicitly distinguishes these limits from its
portable figure input and preserved source files.
