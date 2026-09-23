# CLAUDE.md — SkillMap Project

## Overview
You are helping build **SkillMap** — a data mining pipeline that analyzes large-scale job posting data to answer three core questions:
1. Which skills consistently appear together and predict higher compensation?
2. Do distinct job role archetypes exist beyond traditional job titles?
3. Can salary tier be reliably predicted from job posting features alone?

---

## Project Structure
```
SkillMap/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              ← one folder per dataset (see Datasets); gitignored
│   │   ├── linkedin_jobs/
│   │   ├── linkedin_postings/
│   │   └── ds_salaries/
│   └── processed/        ← pipeline outputs go here; gitignored
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_data_warehouse.ipynb
│   ├── 04_association_mining.ipynb
│   ├── 05_classification.ipynb
│   └── 06_clustering.ipynb
├── src/
│   ├── preprocessing.py
│   ├── warehouse.py
│   ├── association.py
│   ├── classification.py
│   └── clustering.py
└── outputs/
```

---

## Datasets
Each dataset lives in its own folder under `data/raw/`. Do not modify raw files. Raw data is gitignored (files are up to 4.8 GB); see README for download links. The multi-file datasets are **not merged yet** — joins happen in Stage 2.

### `data/raw/linkedin_jobs/` — 1.3M LinkedIn Jobs & Skills 2024
Join key across all files: `job_link`. No salary columns.

| File | Size | Columns |
|---|---|---|
| linkedin_job_postings.csv | 396 MB | job_link, last_processed_time, got_summary, got_ner, is_being_worked, job_title, company, job_location, first_seen, search_city, search_country, search_position, job_level, job_type |
| job_skills.csv | 642 MB | job_link, job_skills (comma-separated skill string) |
| job_summary.csv | 4.8 GB | job_link, job_summary (full description text) |

### `data/raw/linkedin_postings/` — LinkedIn Job Postings 2023-2024
Join key: `job_id` (jobs/*), `company_id` (companies/*); `skill_abr` / `industry_id` map via `mappings/`.

| File | Columns |
|---|---|
| postings.csv (493 MB) | job_id, company_name, title, description, max_salary, pay_period, location, company_id, views, med_salary, min_salary, formatted_work_type, applies, original_listed_time, remote_allowed, job_posting_url, application_url, application_type, expiry, closed_time, formatted_experience_level, skills_desc, listed_time, posting_domain, sponsored, work_type, currency, compensation_type, normalized_salary, zip_code, fips |
| jobs/salaries.csv | salary_id, job_id, max_salary, med_salary, min_salary, pay_period, currency, compensation_type |
| jobs/job_skills.csv | job_id, skill_abr |
| jobs/job_industries.csv | job_id, industry_id |
| jobs/benefits.csv | job_id, inferred, type |
| companies/companies.csv | company_id, name, description, company_size, state, country, city, zip_code, address, url |
| companies/company_industries.csv | company_id, industry |
| companies/company_specialities.csv | company_id, speciality |
| companies/employee_counts.csv | company_id, employee_count, follower_count, time_recorded |
| mappings/skills.csv | skill_abr, skill_name |
| mappings/industries.csv | industry_id, industry_name |

Note: salary columns are `min_salary`/`max_salary`/`med_salary` (plus `normalized_salary`, annualized); `pay_period` mixes hourly/monthly/yearly. Experience level is `formatted_experience_level`.

### `data/raw/ds_salaries/` — Data Science Salaries 2020-2025
| File | Columns |
|---|---|
| DataScience_salaries_2025.csv (93,597 rows) | work_year, experience_level, employment_type, job_title, salary, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location, company_size |
| DataScience_salaries_2025.json | same data in JSON |

Note: there is no `work_type` column — use `remote_ratio` (0/50/100) and `employment_type`.

---

## Tech Stack
- **pandas, NumPy** — data processing
- **spaCy, NLTK** — NLP and skill extraction
- **mlxtend** — Apriori and FP-Growth association mining
- **scikit-learn** — classification and clustering
- **XGBoost** — gradient boosted salary classifier
- **NetworkX** — skill co-occurrence graph
- **Matplotlib, Plotly** — visualization

Install all with:
```bash
pip install -r requirements.txt
```

Local environment: Homebrew Python blocks `pip install` (PEP 668), so the project uses `.venv/` (gitignored), created with `--system-site-packages`. It reuses the system pandas/numpy/matplotlib/networkx/sklearn and adds mlxtend, ipython, and xgboost. xgboost needs `brew install libomp`. Run pipelines with `.venv/bin/python -m src.<module>`.

---

## Pipeline Instructions

### Stage 1 — Data Ingestion (`notebooks/01_data_ingestion.ipynb`)
- Load all three CSV files from `data/raw/`
- Print shape, column names, and sample rows for each
- Check for nulls and data types
- Save a merged exploration summary to `outputs/01_summary.txt`

### Stage 2 — Data Preprocessing (`notebooks/02_preprocessing.ipynb`, `src/preprocessing.py`)
**Status: complete.** Run with `python -m src.preprocessing` (~3 min).
- Main table: `linkedin_postings/postings.csv`, joined with `jobs/salaries.csv`, `jobs/job_skills.csv` + `mappings/skills.csv`, and `jobs/job_industries.csv` + `mappings/industries.csv`
- Normalize job titles to lowercase and strip whitespace
- Salary: `normalized_salary` is primary (pay_period annualized: hourly x2080, weekly x52, biweekly x26, monthly x12). Drop rows with no salary, non-USD rows, and salaries outside $10k-$1M. Impute remaining gaps by `experience_level` median
- Drop reposted duplicate ads (same company_name, title, description, normalized_salary; 1,425 rows). Without this, copies land on both sides of Stage 5's split and inflate scores
- Discretize salary into three tiers at the 33rd/66th percentiles: Low (<= $62,400), Mid (<= $109,200), High
- `skills_list` = 35 coarse job-function categories. `matched_skills` = concrete skills: the top 100 skills from `linkedin_jobs/job_skills.csv` (spelling variants and `SKILL_SYNONYMS` merged, `EXCLUDED_TERMS` removed), regex-matched in each posting's description. Phrases containing another skill ("project management" ⊃ "management") are matched first and blanked out of the text, so there are no tautological rules. **Stages 4-6 should use `matched_skills`**
- Outputs:
  - `data/processed/cleaned_jobs.csv` (34,179 rows): job_id, job_title, company_name, location, experience_level, industry, skills_list, matched_skills, normalized_salary, salary_tier. Read it with `src.preprocessing.load_cleaned_jobs()` so the list columns are parsed
  - `data/processed/linkedin_jobs_skills.csv` (1.29M job_link → skills_list)
  - `data/processed/skill_vocabulary.csv`
  - `outputs/02_summary.txt`

### Stage 3 — Data Warehousing (`notebooks/03_data_warehouse.ipynb`, `src/warehouse.py`)
**Status: complete.** Run with `python -m src.warehouse` (~5 s). Output: `data/processed/skillmap.db` (SQLite, ~50 MB) and `outputs/03_summary.txt`.
- Fact table `job_postings` at (job, skill) grain: job_id, skill_id, location_id, company_id, time_id, salary_tier, experience_level, normalized_salary (337,421 rows). Jobs with no skills get one row with skill_id NULL
- Dimensions:
  - `dim_skills`: skill_type 'extracted' = 100 matched_skills, 'category' = 35 LinkedIn categories
  - `dim_location`: city/state/location_type parsed from the raw string
  - `dim_company`: company_size 0-7, latest employee_count; company_id 0 = Unknown
  - `dim_time`: time_id YYYYMMDD, from original_listed_time
  - `dim_industry` + `bridge_job_industry` (jobs have up to 3 industries)
- **`v_jobs` view = one row per job. Use it for any salary aggregate**, since the fact table repeats salary per skill
- OLAP queries (in `src/warehouse.OLAP_QUERIES`): top 10 skills, average salary by experience level, job count by industry × state

### Stage 4 — Association Rule Mining (`notebooks/04_association_mining.ipynb`, `src/association.py`)
- Load skill lists per job posting as transactions
- Apply Apriori algorithm (min_support=0.05) using mlxtend
- Generate association rules filtered by:
  - confidence > 0.5
  - lift > 1.5
- Build a skill co-occurrence graph using NetworkX
- Output top 20 rules sorted by lift to `outputs/04_association_rules.csv`
- Plot top skill co-occurrence network and save to `outputs/04_skill_network.png`

**Status: complete.** Run with `python -m src.association` (~2 s). Transactions = `matched_skills`.
- All jobs: 241 frequent itemsets, 35 rules pass the filters, graph has 33 skills / 138 edges. The plot draws edges with lift >= 1.2 (`PLOT_MIN_LIFT`)
- High-tier jobs only: 121 rules → `outputs/04_high_salary_rules.csv`. Each rule has `jobs_with_itemset`, `high_tier_rate`, and `high_tier_lift` (High share among ALL jobs with the itemset ÷ 33.3% baseline), which shows whether a combination actually predicts High pay
- Summary: `outputs/04_summary.txt`

### Stage 5 — Classification (`notebooks/05_classification.ipynb`, `src/classification.py`)
- Features: extracted skills (TF-IDF), experience_level, company_size, location, industry
- Target: salary_tier (Low / Mid / High)
- Train three models:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
- Use 80/20 train-test split with stratification
- Evaluate all three using:
  - F1-score (macro) — target > 0.75
  - ROC-AUC — target > 0.80
  - Confusion matrix
- Save best model to `outputs/05_best_model.pkl`
- Save results comparison to `outputs/05_classification_results.csv`

**Status: complete.** Run with `python -m src.classification` (~5 s). Features as specified by the user (multi-hot skills rather than TF-IDF):
- `matched_skills` multi-hot (100 columns)
- experience_level ordinal: Internship 0, Entry 1, Associate 2, Mid-Senior 3, Director 4, Executive 5, Unknown −1
- company_size ordinal 0-8 binned from warehouse `employee_count`
- top 20 states one-hot + Other
- top 15 industries multi-hot + Other

Top states and industries are chosen on the train split only.
- Results (test): Random Forest F1 0.690 / AUC 0.862 (best, saved); XGBoost 0.667 / 0.847; LR 0.612 / 0.801. **F1 target not met, AUC target met**
- Adding job_title TF-IDF (1-2 grams, 2,000 terms) to XGBoost gave F1 0.740 / AUC 0.895 in a quick experiment. This is the most promising way to close the F1 gap
- Outputs: `05_classification_results.csv`, `05_best_model.pkl` (dict with model + feature metadata), `05_confusion_matrix.png`, `05_feature_importance.png`, `05_summary.txt`

### Stage 6 — Clustering (`notebooks/06_clustering.ipynb`, `src/clustering.py`)
- Vectorize job postings using TF-IDF on skill columns
- Apply K-Means (k=5 to 10, use elbow method to pick k)
- Apply DBSCAN (tune eps and min_samples)
- Evaluate both using:
  - Silhouette Score — target > 0.50
  - Davies-Bouldin Index — target < 1.0
- Label each cluster with top 5 skills
- Save cluster assignments to `outputs/06_cluster_labels.csv`
- Plot cluster visualization (PCA to 2D) and save to `outputs/06_clusters.png`

---

## Progress Reporting
After every stage or major task is completed, regenerate `outputs/progress_report.md` in this format (fill in the brackets; keep it concise):

```markdown
## SkillMap Progress Report
**Last Updated:** [date and time]
**Current Stage:** [stage number and name]

### ✅ Completed Stages
- Stage X — [name]: [one line summary of what was built and key output file]

### 🔄 Current Stage Summary
**What was built:** [2-3 sentences]
**Key outputs:** [list of files created]
**Rows/records processed:** [numbers]
**Issues found:** [any problems or decisions needed]

### ⏭ Next Stage
**Stage X — [name]:** [one line on what comes next]

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete / ⏳ Pending | outputs/01_summary.txt |
| ... one row per stage 1-6 ... | | |
```

---

## Coding Standards
- Every function must have a docstring
- Use `logging` instead of print statements in `src/` files
- Notebooks should have a markdown cell explaining each step before the code
- All file paths should use `pathlib.Path` not hardcoded strings
- Random seed: always set `random_state=42`

---

## Evaluation Targets Summary

| Metric | Task | Target |
|---|---|---|
| F1-Score | Classification | > 0.75 |
| ROC-AUC | Classification | > 0.80 |
| Silhouette Score | Clustering | > 0.50 |
| Davies-Bouldin Index | Clustering | < 1.0 |
| Lift | Association Mining | > 1.5 |

---

## Milestones

| Phase | Task | Deadline |
|---|---|---|
| Phase 1 | Data Acquisition | Sep 21 |
| Phase 2 | Preprocessing and Cleaning | Sep 28 |
| Phase 3 | Data Warehousing and EDA | Oct 5 |
| Phase 4 | Association Mining and Classification | Oct 19 |
| Phase 5 | Clustering and Outlier Detection | Nov 1 |
| Phase 6 | Evaluation and Visualization | Nov 14 |

---

## How to Use This File with Claude Code

When starting a new session, Claude will automatically read this file and understand the full project context. You can then give direct commands like:

```
implement Stage 2 preprocessing pipeline
```
```
build the association rule mining notebook
```
```
run the classification pipeline and compare all three models
```
```
fix the clustering silhouette score it is below 0.50
```

Claude will follow the pipeline stages, coding standards, and evaluation targets defined above without needing re-explanation each session.