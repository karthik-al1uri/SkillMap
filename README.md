# SkillMap — Mining Job Market Data for Career Path Intelligence and Salary Prediction

**Team:** Karthik Alluri, Shashidhar Reddy Kanaparthi
**Course:** CSCI 5502 Data Mining, Fall 2026, University of Colorado Boulder

SkillMap is a data mining pipeline over large-scale job posting data. It answers three questions:

1. Which skills consistently appear together and predict higher compensation?
2. Do distinct job role archetypes exist beyond traditional job titles?
3. Can salary tier be reliably predicted from job posting features alone?

The full write-up is in [`outputs/final_report.md`](outputs/final_report.md).

---

## Bounded validation status

The saved November 1 results are historical exploratory results, not a newly reproduced evaluation. The original classifier test split influenced model-family selection; its scores must not be described as an untouched final evaluation. Current saved values are 62 general rules, 436,441 warehouse fact rows, 11.1 extracted skills per training job, classifier macro F1 0.7273 / ROC-AUC 0.8921, and 1/20 Muse confirmations. A company cap confirms a different rule, also 1/20.

The current implementation pass preserves existing data, snapshots, models, reports, and saved outputs. Do not run the historical end-to-end commands below against canonical artifacts during this pass: several overwrite outputs, and the warehouse builder replaces the database. Any new evaluation must select models using training-only cross-validation and fit learned preprocessing within each fold. Reusing these data does not create an independent external test.

The live snapshots were fetched on October 14, 2026. Cross-source skill-share differences do not establish temporal market trends. A lift ceiling above the 1.5 acceptance threshold does not make passing mathematically impossible. Weak clustering scores describe the tested methods and representations, not an impossibility theorem.

## Datasets

The raw data is **not included in this repository** because the files are up to 4.8 GB. Download each Kaggle dataset and unzip it into the folder shown.

| Dataset | Source | Local folder | Role |
|---|---|---|---|
| LinkedIn Job Postings 2023-2024 | [Kaggle — arshkon](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | `data/raw/linkedin_postings/` | Main table: 123,849 postings with salary, experience, company, industry |
| 1.3M LinkedIn Jobs & Skills 2024 | [Kaggle — asaniczka](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | `data/raw/linkedin_jobs/` | Source of the 245-skill vocabulary |
| Data Science Salaries 2020-2025 | [Kaggle — arnabchaki](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2025) | `data/raw/ds_salaries/` | External salary benchmark |
| Adzuna API (live) | [api.adzuna.com](https://api.adzuna.com/v1/api/jobs/us/search) | `data/raw/adzuna_live/` | Stage 7: 1,150 current US IT postings (needs a free API key) |
| The Muse API (live) | [themuse.com/api/public/jobs](https://www.themuse.com/api/public/jobs) | `data/raw/muse_live/` | Stage 8: 1,378 current tech postings with full descriptions (no key) |

Expected layout of the Kaggle data:

```
data/raw/
├── linkedin_jobs/       linkedin_job_postings.csv, job_skills.csv, job_summary.csv
├── linkedin_postings/   postings.csv, jobs/, companies/, mappings/
└── ds_salaries/         DataScience_salaries_2025.csv
```

`CLAUDE.md` lists the full column schema for every file.

---

## Pipeline and results

| Stage | What it does | Code | Key result |
|---|---|---|---|
| 1 | Ingestion: profile all 15 raw tables | `notebooks/01_data_ingestion.ipynb` | `outputs/01_summary.txt` |
| 2 | Preprocessing: join, annualize and filter salaries, drop reposts, tier at the 33rd/66th percentiles, regex-match 245 skills in descriptions | `src/preprocessing.py` | 34,179 jobs; Low ≤ $62,400 < Mid ≤ $109,200 < High; 11.1 skills per job |
| 3 | Warehousing: SQLite star schema, (job, skill) fact table, `v_jobs` view, OLAP queries | `src/warehouse.py` | `data/processed/skillmap.db`, 436,441 fact rows |
| 4 | Association mining: Apriori (support 0.05, confidence > 0.5, lift > 1.5), High-tier rules, co-occurrence network, 0.02-support technical pass | `src/association.py` | 62 rules; {python, engineering} jobs are 81% High tier (2.4× baseline); aws + python 88% High |
| 5 | Classification: Logistic Regression, Random Forest, XGBoost, tuned XGBoost + title/company TF-IDF, plus two ordinal models | `src/classification.py` | Best macro F1 **0.7273** (target 0.75 ✗), ROC-AUC **0.8921** (target 0.80 ✓); ordinal models don't help |
| 6 | Clustering: K-Means, DBSCAN, t-SNE vs PCA, Isolation Forest outliers | `src/clustering.py` | Silhouette ≤ 0.192 (target 0.50 ✗): the tested methods and representations showed weak cluster separation. A tech cluster earns $50k more than service roles; outliers are skill-dense postings |
| 7 | Live data (Adzuna): fetch, preprocess, warehouse, re-test rules and model | `src/live_pipeline.py` | 1,150 IT postings, but ~500-character snippets give only 2.0 skills each, too few to test rules |
| 8 | Live skill validation (The Muse): full descriptions, rule validation, skill-demand comparison | `src/muse_pipeline.py` | 15.6 skills per posting; 1 of 20 training rules confirmed; Python appears in 17.3% of the training tech cluster and 31.8% of Muse postings (different populations) |

Each stage has a notebook (`notebooks/0N_*.ipynb`) with executed outputs and a module that runs it end to end with `python -m src.<module>`. Summaries are written to `outputs/0N_summary.txt`.

---

## Setup

```bash
git clone https://github.com/karthik-al1uri/SkillMap.git
cd SkillMap

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: XGBoost needs OpenMP -> brew install libomp
```

Download the three Kaggle datasets into `data/raw/` (see above), then run the stages in order:

```bash
python -m src.preprocessing    # Stage 2, ~8 min
python -m src.warehouse        # Stage 3
python -m src.association      # Stage 4
python -m src.classification --output-dir /tmp/new-skillmap-evaluation  # Stage 5; training requires separate approval
python -m src.clustering       # Stage 6, ~1.5 min
```

Or open the notebooks in order with `jupyter notebook notebooks/`. `job_summary.csv` is 4.8 GB, so the ingestion notebook loads only its first 200,000 rows by default (`LARGE_FILE_MB` / `LARGE_FILE_NROWS` in Step 2).

### Live data (Stages 7-8)

Stage 7 needs Adzuna credentials. Copy the template and fill in your own values from [developer.adzuna.com](https://developer.adzuna.com):

```bash
cp .env.example .env
# edit .env:
# ADZUNA_APP_ID=<your application id>
# ADZUNA_API_KEY=<your application key>
```

`.env` is gitignored: never commit it. The Adzuna free tier allows only 25 calls, so `src/live_pipeline.py` logs every call in `data/raw/adzuna_live/api_calls.json` before sending it and refuses to go past 25. Pages are cached, so reruns and the notebook make no further calls.

Stage 8 needs no key. Each fetch is a snapshot keyed by `fetch_date`: pages are cached in `data/raw/muse_live/pages/<fetch_date>/` (the first fetch, 2026-10-14, sits directly in `pages/`), and `muse_cleaned_jobs.csv` and the `muse_job_postings` table keep one set of rows per snapshot. Rerunning a snapshot replaces only its own rows. Analysis runs on one snapshot, and the company-capped repeat keeps at most `--max-per-company` postings per employer (most recent first) through a SQL query; no rows are deleted.

```bash
python -m src.live_pipeline                    # Stage 7 (uses the cache once fetched)
python -m src.muse_pipeline --output-dir /tmp/new-muse-review  # Stage 8, cache only
python -m src.muse_pipeline --list-snapshots   # cached snapshots
python -m src.muse_pipeline --fetch-date 2026-10-14 --max-per-company 50 --output-dir /tmp/new-muse-capped-review
```

`python -m src.warehouse` rebuilds `skillmap.db` from scratch and drops the live tables. Re-run the two commands above to restore them from the cache without any API calls (Stage 8 reloads every snapshot in `muse_cleaned_jobs.csv`).

### Dashboard

```bash
# Generate artifact-derived facts into a fresh external file (no training).
python -m scripts.build_dashboard_facts --root . --out /tmp/new-skillmap-facts.json
python scripts/build_dashboard.py --facts /tmp/new-skillmap-facts.json --out /tmp/new-skillmap-dashboard.html
python scripts/verify_dashboard.py --html /tmp/new-skillmap-dashboard.html --out /tmp/new-skillmap-verification.json
```

One self-contained HTML file with six views: Overview, Skills, Salary, Clusters, Live, and Model. The renderer reads only the generated facts JSON; no database connection, API call, or training occurs. Every value has artifact provenance. Missing optional evaluations appear as unavailable rather than being replaced with historical results. Salary summaries use one row per job.

To include the completed isolated evaluations, pass their directories to the facts generator using `--rule-views`, `--grouped`, `--live-validation`, `--cluster-robustness`, and `--salary-quality`. The grouped model is selected using training CV metadata. Historical exploratory results and dated cached live validation remain separately labeled; one snapshot does not establish temporal trends. See [offline verification instructions](docs/offline_reproducibility.md) for installed-environment checks and a data-free source export. Python 3.12 CI and clean dependency installation remain unverified until actually run.

---

## Project structure

```
SkillMap/
├── .env.example       credential template for Stage 7 (copy to .env)
├── data/raw/          raw Kaggle data and live API caches (gitignored)
├── data/processed/    cleaned tables and skillmap.db (gitignored)
├── notebooks/         01-08, one per stage, with executed outputs
├── src/               one module per stage
└── outputs/           summaries, plots, models (gitignored), final_report.md, progress_report.md
```
