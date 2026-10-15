# SkillMap — Mining Job Market Data for Career Path Intelligence and Salary Prediction

**Team:** Karthik Alluri, Shashidhar Reddy Kanaparthi
**Course:** CSCI 5502 Data Mining, Fall 2026, University of Colorado Boulder

SkillMap is a data mining pipeline over large-scale job posting data. It answers three questions:

1. Which skills consistently appear together and predict higher compensation?
2. Do distinct job role archetypes exist beyond traditional job titles?
3. Can salary tier be reliably predicted from job posting features alone?

The full write-up is in [`outputs/final_report.md`](outputs/final_report.md).

---

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
| 2 | Preprocessing: join, annualize and filter salaries, drop reposts, tier at the 33rd/66th percentiles, regex-match 245 skills in descriptions | `src/preprocessing.py` | 34,179 jobs; Low ≤ $62,400 < Mid ≤ $109,200 < High; 11.5 skills per job |
| 3 | Warehousing: SQLite star schema, (job, skill) fact table, `v_jobs` view, OLAP queries | `src/warehouse.py` | `data/processed/skillmap.db`, 449,665 fact rows |
| 4 | Association mining: Apriori (support 0.05, confidence > 0.5, lift > 1.5), High-tier rules, co-occurrence network, 0.02-support technical pass | `src/association.py` | 91 rules; {python, engineering} jobs are 81% High tier (2.4× baseline); aws + python 88% High |
| 5 | Classification: Logistic Regression, Random Forest, XGBoost, tuned XGBoost + title/company TF-IDF, plus two ordinal models | `src/classification.py` | Best macro F1 **0.728** (target 0.75 ✗), ROC-AUC **0.891** (target 0.80 ✓); ordinal models don't help |
| 6 | Clustering: K-Means, DBSCAN, t-SNE vs PCA, Isolation Forest outliers | `src/clustering.py` | Silhouette ≤ 0.21 (target 0.50 ✗): skills form a continuum. A tech cluster earns $50k more than service roles; outliers are skill-dense postings |
| 7 | Live data (Adzuna): fetch, preprocess, warehouse, re-test rules and model | `src/live_pipeline.py` | 1,150 IT postings, but ~500-character snippets give only 2.0 skills each, too few to test rules |
| 8 | Live skill validation (The Muse): full descriptions, rule validation, skill-demand comparison | `src/muse_pipeline.py` | 16.2 skills per posting; 2 of 20 training rules confirmed, most others hit a lift ceiling with *higher* confidence; Python demand in tech postings 18% → 32% |

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
python -m src.classification   # Stage 5, ~2.5 min
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

Stage 8 needs no key. It caches every page in `data/raw/muse_live/pages/` and retries network errors and HTTP 429/5xx with backoff.

```bash
python -m src.live_pipeline    # Stage 7 (uses the cache once fetched)
python -m src.muse_pipeline    # Stage 8
```

`python -m src.warehouse` rebuilds `skillmap.db` from scratch and drops the live tables. Re-run the two commands above to restore them from the cache without any API calls.

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
