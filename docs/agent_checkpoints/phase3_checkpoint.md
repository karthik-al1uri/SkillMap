# Phase 3 checkpoint — dashboard and offline engineering

Status: **Six-view dashboard implemented and offline checks passed; stopped before Phase 4.** Real-browser acceptance, clean dependency installation, and hosted Python 3.12 CI remain unverified.

Commit: `e47cdf1cc1a7c4d1c24a03f9cff55253bb63e41d`; UTC `2026-11-11T16:47:48.576180+00:00`. No Git-history changes. Existing work was reused, not retrained or re-audited.

## Changed files

- `scripts/build_dashboard.py`
- `scripts/build_dashboard_facts.py`
- `scripts/verify_dashboard.py`
- `scripts/check_offline_environment.py`
- `tests/test_dashboard.py`
- `tests/test_dashboard_facts.py`
- `tests/test_dashboard_verification.py`
- `tests/test_offline_environment.py`
- `requirements.txt`
- `requirements-test.txt`
- `.github/workflows/tests.yml`
- `README.md`
- `docs/offline_reproducibility.md`
- `docs/agent_checkpoints/phase3_checkpoint.md`

## Dashboard behavior and facts contract

Six analyst-first views: **Overview, Skills, Salary, Clusters, Live, Model**. Retained existing Plotly styling, theme logic, useful rule/cluster/model/live plots and legacy tab links; added keyboard navigation and explicit tab/panel ARIA attributes.

- Overview: saved-job/vocabulary/skill/salary facts, grouped F1, cached dates and source hash registry.
- Skills: prevalence, top10 lift, strict technical-only view including explicit zero-rule state.
- Salary: job-level distribution, tier counts, experience medians and top5 High-tier associations. Salary is never aggregated after a job-skill fact join. Canonical job-grain median is $82,500 across34,179 jobs.
- Clusters: saved tier/salary/distinctive-skill plots, saved k sweep and bounded common-space robustness comparison. Exploratory limitations explicit.
- Live: cached observed/predicted tier distribution, confusion, employer uncertainty and baselines, fixed-rule validation, cross-source skill shares and date2026-10-14. No single-snapshot temporal claims.
- Model: grouped training-CV comparison, selected-model F1/AUC/per-class/confusion/uncertainty, separately labeled historical plots. Grouped winner is fixed_xgboost; macroF1 0.6959. No test-F1 model selection.

`skillmap.dashboard.v1` JSON is the **sole rendering data source**. Every metric/table/evaluation has availability, evidence status, source IDs and data.28 actually consumed sources have path/byte/SHA provenance and were verified before/after facts generation. Optional missing evaluations stay unavailable; historical metrics do not replace grouped results. Policies such as target scores and frozen salary boundaries have hashed code/contract sources. The renderer does not open a database or re-score a model.

## Verified results

- Repository guarded offline suite: **114 passed,0 failed**, six existing synthetic-prediction warnings,14.64s.
- Data-free source export: **114 passed,0 failed**, same warnings,14.04s, using the existing Python3.14.2 interpreter/system site packages. Export contains53 allowlisted files and no `.git`, canonical `data/` or `outputs/`, credentials or virtualenv.
- Installed constraints: **18/18 compatible**, guarded `pip check`: no broken requirements. Python socket/DNS guard verified. It is not an OS network sandbox.
- Static HTML verifier: **six tabs/panels,22 chart containers, unique IDs, source metadata consistent, embedded Plotly and no external required assets**. Static verification does not execute browser JavaScript.
- Controlled Node unit test simulates tab clicks, keyboard arrows/Home/End and legacy links; not a real browser interaction.
- Exported source rendered the same supplied facts with Python networking blocked and no canonical data directories; resulting HTML is **byte-identical** to the repository render.
- `git diff --check` passed.
- Canonical preservation: **188/188 size+SHA matches**; no unexpected modifications to prior uncommitted files. Intended existing-file edits: README, requirements, CI workflow, dashboard source and its tests. Other prior work remains unchanged.

## Artifacts

- dashboard: `/private/tmp/skillmap-phase3/lead/dashboard-v2.html`
- dashboard_sha256: `eddc4ca9399d2a09921d3bde8bc84205db2227d119085de0dfea8f00fb7b0b8b`
- facts: `/private/tmp/skillmap-phase3/facts/dashboard-facts-final.json`
- static_verification: `/private/tmp/skillmap-phase3/verification/dashboard-static-v2.json`
- installed_suite: `/private/tmp/skillmap-phase3/lead/installed-suite/environment.json`
- exported_suite: `/private/tmp/skillmap-phase3/lead/exported-suite/environment.json`
- source_export: `/private/tmp/skillmap-phase3/source-export`
- exported_render: `/private/tmp/skillmap-phase3/lead/export-render/dashboard.html`
- exported_render_verification: `/private/tmp/skillmap-phase3/lead/export-render/verification.json`
- preservation: `/private/tmp/skillmap-phase3/lead/preservation.json`

Dashboard HTML SHA-256: `eddc4ca9399d2a09921d3bde8bc84205db2227d119085de0dfea8f00fb7b0b8b`.

The facts bundle is about70MB and self-contained HTML about5.9MB. HTML can be opened without the original database, model, reports or network. Absolute source paths shown in provenance are evidence references, not rendering dependencies.

## Engineering and dependency specification

Plotly lower bound raised5.18→5.19 to support rounded bars. Existing sklearn>=1.5 retained. Direct threadpoolctl>=3.1 declared; requirements-test includes pytest>=8 and packaging>=23. CI declaratively targets Python3.12 and invokes guarded offline checks after installing declared test requirements. **No installs or hosted CI were run here.** Lower bounds are not a lockfile or a tested minimum-version matrix.

An initial environment-guard issue shadowed Homebrew's existing sitecustomize hook; the worker fixed hook chaining and added a regression test before the final checks. An initial dashboard build failed for an empty strict-technical table; fixed and regression-tested. Failed/empty initial artifact and failed verification report were retained rather than overwritten. Only dashboard-v2.html is the verified repository delivery artifact.

## Permissions and remaining limitations

Browser automation host was unavailable. Native Safari preview returned **Computer Use permissions are not granted**. No permission was requested and no alternate browser-control technology was used to bypass it. Real visual/browser-runtime/responsive acceptance is therefore **not claimed**.

No API fetches, external data requests, model retraining, canonical overwrite, dependency installation, push or history modification. No Phase4 work.

Readiness: self-contained artifact is ready for **offline analyst review**. Delivery sign-off is conditional on real-browser acceptance and any required clean-install/hosted-CI checks. Passing the installed-environment/data-free-export suite does not establish clean-clone dependency reproducibility or reproduction of historical training results.

## Commands and safe resume

Worker exact facts/engineering execution records: `/private/tmp/skillmap-phase3/{facts,engineering}/execution.json`. Source export file hashes: `/private/tmp/skillmap-phase3/lead/source-export.json`.

Executed renderer:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/skillmap-phase3/lead/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -B scripts/build_dashboard.py --facts /private/tmp/skillmap-phase3/facts/dashboard-facts-final.json --out /private/tmp/skillmap-phase3/lead/dashboard-v2.html
```

Executed verifier:

```bash
.venv/bin/python -B scripts/verify_dashboard.py --html /private/tmp/skillmap-phase3/lead/dashboard-v2.html --out /private/tmp/skillmap-phase3/verification/dashboard-static-v2.json
```

Executed installed suite (Agg, threads1, Python socket guard, no pytest cache):

```bash
.venv/bin/python -B -m scripts.check_offline_environment --root '/Users/genjishimada49/Documents/CU-Boulder/Fall 2026/Data Mining/SkillMap' --out-dir /private/tmp/skillmap-phase3/lead/installed-suite --pytest
```

Executed exported suite, working directory `/private/tmp/skillmap-phase3/source-export`:

```bash
/Users/genjishimada49/Documents/CU-Boulder/Fall 2026/Data Mining/SkillMap/.venv/bin/python -B -m scripts.check_offline_environment --root /private/tmp/skillmap-phase3/source-export --out-dir /private/tmp/skillmap-phase3/lead/exported-suite --pytest
```

Executed data-free facts-only render check:

```bash
.venv/bin/python -B /private/tmp/skillmap-phase3/lead/verify_export_render.py
```

Example future facts generation using the completed saved evidence (not executed again):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m scripts.build_dashboard_facts --root . --out <fresh-external>/dashboard-facts.json --rule-views /private/tmp/skillmap-phase6b/association/views --grouped /private/tmp/skillmap-phase2/classification/run1 --live-validation /private/tmp/skillmap-phase2/live/validation --cluster-robustness /private/tmp/skillmap-phase6b/clustering/run --salary-quality /private/tmp/skillmap-phase6b/salary/run2
```

All listed artifact paths already exist and are protected against overwrite. Use fresh isolated destinations for an authorized future build. Do not rerun training or earlier audits.

Exact resume prompt:

> Read docs/agent_checkpoints/phase3_checkpoint.md and /private/tmp/skillmap-phase3/lead/checkpoint.json. Preserve all existing uncommitted work and canonical artifacts. Do not repeat training or completed audits. Phase3 offline tests and artifact checks are complete. If permission becomes available, verify the existing self-contained dashboard in a real browser (six tabs, charts, keyboard, mobile/desktop); do not bypass permissions. Do not claim clean-clone reproducibility without an authorized actual clean-environment install and test run. Start Phase4 only on explicit instruction.

Full machine-readable checkpoint includes current Git modified-file inventory, test output, artifact hashes, commands, skipped permissions and blockers: `/private/tmp/skillmap-phase3/lead/checkpoint.json`.
