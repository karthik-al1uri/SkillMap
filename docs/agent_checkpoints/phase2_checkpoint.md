# Phase 2 final checkpoint

Status: **Computations and reviewed integration complete; stopped before Phase 3.** One permission-blocked operation: early wake-process termination. No interactive approval requested.

UTC: 2026-11-11T06:47:30.392690+00:00
Git commit: `e47cdf1cc1a7c4d1c24a03f9cff55253bb63e41d` (unchanged).

## Preserved work and changed files

All prior uncommitted files match their starting SHA-256 values in `/private/tmp/skillmap-phase2/lead/starting_state.json`. Added this phase:

- `scripts/evaluate_grouped_classification.py`
- `scripts/check_grouped_feasibility.py`
- `scripts/validate_cached_live.py`
- `tests/test_grouped_classification.py`
- `tests/test_grouped_feasibility.py`
- `tests/test_cached_live.py`
- `docs/agent_checkpoints/phase2_checkpoint.md`

Full current modified/untracked inventory:

```text
M CLAUDE.md
 M README.md
 M notebooks/05_classification.ipynb
 M notebooks/08_muse_pipeline.ipynb
 M requirements.txt
 M scripts/build_dashboard.py
 M src/classification.py
 M src/muse_pipeline.py
?? .github/workflows/tests.yml
?? docs/agent_checkpoints/phase2_checkpoint.md
?? docs/phase6b_classification_contract.md
?? scripts/build_mining_facts.py
?? scripts/build_rule_views.py
?? scripts/check_cluster_robustness.py
?? scripts/check_data_quality.py
?? scripts/check_extraction_phase6b.py
?? scripts/check_grouped_feasibility.py
?? scripts/evaluate_grouped_classification.py
?? scripts/validate_cached_live.py
?? tests/test_association.py
?? tests/test_cached_live.py
?? tests/test_classification.py
?? tests/test_cluster_robustness.py
?? tests/test_dashboard.py
?? tests/test_data_quality.py
?? tests/test_extraction_phase6b.py
?? tests/test_grouped_classification.py
?? tests/test_grouped_feasibility.py
?? tests/test_muse_pipeline.py
?? tests/test_preprocessing.py
?? tests/test_rule_views.py
?? tests/test_smoke.py
```

## Completed classification

Frozen normalized-description SGKF outer 5 folds, seed2026, test fold0; inner3 folds seed2027. Canonical class encoding Low0/Mid1/High2. 27,343 train / 6,836 test; all three classes retained, zero normalized-description overlap in outer/inner splits. Exactly9 sequential inner fits and1 selected refit, single threads, hard20-minute cap; actual elapsed 50.31s.

Inner CV macroF1: nominal logistic0.669220; ordinal LogisticIT0.642931; fixed XGBoost0.683701. Same fold-contained PostingFeatures+TFIDF for all; scaled nominal control and ordinal baseline. CV-only selection chose fixed XGBoost (200trees,depth4,learning_rate0.1,subsample1). No parameter search or selection from outer test.

Selected holdout: macroF1 0.695879 (200 description-group bootstrap95% interval0.685092–0.707173), accuracy0.700848, macro OvR ROC-AUC0.869863 (interval0.864127–0.876426). F1 Low/Mid/High0.779258/0.569943/0.738435.

Confusion rows actual, columns predicted, order Low/Mid/High:

```text
[[1901, 328, 78], [524, 1206, 526], [147, 442, 1684]]
```

Full per-fold/per-class metrics, probabilities and uncertainty:
`/private/tmp/skillmap-phase2/classification/run1/{frozen_design.json,split_manifest.csv,fit_milestones.json,cv_selection.csv,outer_predictions.csv,selected_model.pkl,report.json,execution.json}`.
The external model was reloaded and three representative predictions/probabilities verified. Canonical model was not replaced.

Historical saved F1 0.7273/AUC0.8921 remain historical and test-selected. The new grouped resplit and three fixed models differ; the difference does not isolate a grouping effect or prove regression/improvement. Both reuse previously inspected data; historical global salary thresholds remain fixed.

## Split feasibility and reconciliation

Use `/private/tmp/skillmap-phase2/feasibility/run2_numeric_labels/`, not run1, for the evaluated classifier's exact assignments. Run1 used strings (internal alphabetical High/Low/Mid), which changes SGKF tie resolution. Numeric coding was independently reconciled after freezing: zero missing/extra jobIDs, zero outer and inner assignment mismatches. **No model or classifier assignments were rerun or changed.**

Frozen sorted test-ID hash: `34980ca481a3a8739dc4c631c5c1a1b7db3057bc9c2fc57833bee45758011a3e`.

Employer holdout feasibility is separate: 27,342 train /6,837 test, all classes retained, no employer overlap, but14 normalized-description groups/15 test rows still cross. Description holdout shares2,152 employer groups. Actual grouped test shares1,346 IDs with historical test;5,490 former historical test IDs are now training. These are exploratory reused-data splits. Employer holdout model performance was not evaluated, respecting the single predefined training comparison.

## Cached Adzuna validation

1,149 cached predictions /660 employers. Accuracy0.579634 (employer-bootstrap95% interval0.540444–0.612650), macroF1 0.464431 (0.424557–0.495394). F1 Low/Mid/High0.311594/0.350820/0.730878; recall0.558442/0.298050/0.723703.
Observed counts77/359/713; predicted199/251/699 (Low/Mid/High). Confusion:

```text
[[43, 18, 16], [85, 107, 167], [71, 126, 516]]
```

Historical-training-majority Low baseline: accuracy0.067015, macroF1 0.041871. Evaluation-label High-majority **oracle**, explicitly not a training baseline: accuracy0.620540, macroF1 0.255281. Classifier accuracy is below that oracle while macroF1 is higher. 1,092 targets are estimated salaries,57 advertised. ROC-AUC is unavailable because cached class probabilities/decision scores are absent; no inference or training was fabricated to fill this gap.

## Cached Muse validation

Fixed historical top20 rules; primary support>=.05, confidence>.5, lift>1.5, applied unrounded. Eligible denominator>=2 skills within the selected dated snapshot; cap25 applied before eligibility, latest publication then ID. Cached date2026-10-14 only.

Uncapped1/20 (`powerpoint -> excel`), 1,372 eligible/1,378 total. Cap25: different1/20 (`marketing -> sales`),602 eligible/606 total. Separately predeclared .02-support sensitivity gives3/20 and4/20; threshold was not selected from these counts. Historical primary confirmation flags agree exactly; lift differences are four-decimal rounding.

Employer-cluster bootstrap200 retains all employer rows with multiplicity and varying row totals. For uncapped confirmed rule: support95% interval0.026408–0.091945, lift4.703948–10.743218, passing fraction0.54. Capped confirmed rule: support0.031858–0.093432, lift2.583295–6.092182, passing fraction0.755. Undefined metric replicates are counted, excluded from percentile intervals and treated as nonpassing for pass fractions. No temporal-market interpretation is supported.

Artifacts: `/private/tmp/skillmap-phase2/live/validation/{report.json,muse_uncapped_fixed_rules.csv,muse_company_capped_fixed_rules.csv,historical_saved_comparison.csv,execution.json}`.

## Tests, verification, warnings

Final integrated suite: **77 passed,0 failures,0 errors**, 13.67s in JUnit. Six pytest UndefinedMetricWarnings arise from synthetic dummy predictions. Three ordinal training SciPy solver deprecation warnings were captured; no failed fit or convergence failure. `git diff --check` passed.

Independent verification recomputed confusion/F1/accuracy/AUC from saved predictions, checked split reconciliation, and reloaded the external model. Evidence: `/private/tmp/skillmap-phase2/lead/independent_verification.json`.

Canonical preservation at feasibility, classification/live, and final milestones: **188/188 SHA-256+size matches**, zero failures. No prior uncommitted file changed. Evidence: `/private/tmp/skillmap-phase2/lead/preservation_{feasibility,classification_live,final}.json`.

## Permissions and managed wake process

`/usr/bin/caffeinate -i -t 28800` was started and observed running at startup; recorded PID59303. It never changed system power settings. Sandbox denied `/bin/ps` process-list inspection and signal-0 permission. Early termination therefore requires unavailable permission and was skipped, as instructed, without interactive escalation or bypass. **Termination is not claimed.** The original eight-hour automatic expiry remains the safety bound. Current process state cannot be verified within these permissions. Records: `/private/tmp/skillmap-phase2/lead/{caffeinate.json,permissions.json}`.

No API/network calls, installs, Git history changes, canonical overwrites or Phase3 work. No task was completely blocked; the wake cleanup limit did not prevent computations or safe checkpointing.

## Executed commands and next steps

Exact training/test command, working directory, software versions and captured warnings: classification `execution.json`. Live and feasibility exact commands: their `execution.json` files. Combined tests:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/skillmap-phase2/lead/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -B -m pytest -p no:cacheprovider --basetemp=/private/tmp/skillmap-phase2/lead/final-tests --junitxml=/private/tmp/skillmap-phase2/lead/final-tests.xml tests -q
```

All generated output directories are now occupied and intentionally protected against overwrite. Do not rerun training commands into them.

Phase3 readiness: **ready for review/planning**, with reused-data/conditional-uncertainty limitations and employer-holdout performance still unmeasured. Independent generalization/deployment claims are not established. Stop before Phase3.

Exact resume prompt:

> Read docs/agent_checkpoints/phase2_checkpoint.md and /private/tmp/skillmap-phase2/lead/checkpoint.json. Phase2 is complete: do not retrain, change splits, or repeat cached validation. Preserve all canonical artifacts and existing uncommitted work. Recheck only stale/missing evidence. Handle caffeinate cleanup only if permitted; otherwise retain recorded automatic expiry. Begin Phase3 only on explicit user instruction.

Full machine-readable checkpoint, metrics and inventory: `/private/tmp/skillmap-phase2/lead/checkpoint.json`.
