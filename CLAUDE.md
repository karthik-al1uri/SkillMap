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

---

## Pipeline Instructions

### Stage 1 — Data Ingestion (`notebooks/01_data_ingestion.ipynb`)
- Load all three CSV files from `data/raw/`
- Print shape, column names, and sample rows for each
- Check for nulls and data types
- Save a merged exploration summary to `outputs/01_summary.txt`

### Stage 2 — Data Preprocessing (`notebooks/02_preprocessing.ipynb`, `src/preprocessing.py`)
- Drop duplicate rows and irrelevant columns
- Normalize job titles to lowercase and strip whitespace
- Extract skills from job descriptions using spaCy and a keyword list
- Impute missing salary values using median grouped by job_title and experience_level
- Discretize salary into three tiers: Low (bottom 33%), Mid (33-66%), High (top 33%)
- Save cleaned dataset to `data/processed/cleaned_jobs.csv`

### Stage 3 — Data Warehousing (`notebooks/03_data_warehouse.ipynb`)
- Design a star schema:
  - Fact table: job_postings (job_id, skill_id, location_id, company_id, salary_tier, experience_level)
  - Dimension tables: dim_skills, dim_location, dim_company, dim_time
- Load into SQLite using pandas and sqlite3
- Run 3 sample OLAP queries:
  1. Top 10 most demanded skills
  2. Average salary by experience level
  3. Job count by industry and location
- Save database to `data/processed/skillmap.db`

### Stage 4 — Association Rule Mining (`notebooks/04_association_mining.ipynb`, `src/association.py`)
- Load skill lists per job posting as transactions
- Apply Apriori algorithm (min_support=0.05) using mlxtend
- Generate association rules filtered by:
  - confidence > 0.5
  - lift > 1.5
- Build a skill co-occurrence graph using NetworkX
- Output top 20 rules sorted by lift to `outputs/04_association_rules.csv`
- Plot top skill co-occurrence network and save to `outputs/04_skill_network.png`

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