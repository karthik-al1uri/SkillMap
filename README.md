# SkillMap — Mining Job Market Data for Career Path Intelligence and Salary Prediction

**Team:** Karthik Alluri, Shashidhar Reddy Kanaparthi
**Course:** CSCI 5502 Data Mining, Fall 2025, University of Colorado Boulder

SkillMap is a data mining pipeline over large-scale job posting data. It answers three questions:

1. Which skills consistently appear together and predict higher compensation?
2. Do distinct job role archetypes exist beyond traditional job titles?
3. Can salary tier be reliably predicted from job posting features alone?

---

## Datasets

The raw data is **not included in this repository** because the files are up to 4.8 GB. Download each dataset from Kaggle and unzip it into the folder shown.

| Dataset | Source | Local folder | Main files |
|---|---|---|---|
| 1.3M LinkedIn Jobs & Skills 2024 | [Kaggle — asaniczka](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | `data/raw/linkedin_jobs/` | `linkedin_job_postings.csv`, `job_skills.csv`, `job_summary.csv` |
| LinkedIn Job Postings 2023-2024 | [Kaggle — arshkon](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | `data/raw/linkedin_postings/` | `postings.csv`, `jobs/`, `companies/`, `mappings/` |
| Data Science Salaries 2020-2025 | [Kaggle — arnabchaki](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2025) | `data/raw/ds_salaries/` | `DataScience_salaries_2025.csv` |

Expected layout after download:

```
data/raw/
├── linkedin_jobs/
│   ├── linkedin_job_postings.csv
│   ├── job_skills.csv
│   └── job_summary.csv
├── linkedin_postings/
│   ├── postings.csv
│   ├── jobs/        (salaries, job_skills, job_industries, benefits)
│   ├── companies/   (companies, company_industries, company_specialities, employee_counts)
│   └── mappings/    (skills, industries)
└── ds_salaries/
    └── DataScience_salaries_2025.csv
```

`CLAUDE.md` lists the full column schema for every file.

---

## Pipeline Overview

| Phase | Stage | Notebook | Status |
|---|---|---|---|
| 1 | Data acquisition and ingestion | `notebooks/01_data_ingestion.ipynb` | ✅ Done |
| 2 | Preprocessing and cleaning | `notebooks/02_preprocessing.ipynb`, `src/preprocessing.py` | ✅ Done |
| 3 | Data warehousing (SQLite star schema) | `notebooks/03_data_warehouse.ipynb`, `src/warehouse.py` | ✅ Done |
| 4 | Association rule mining and classification | `notebooks/04_association_mining.ipynb`, `notebooks/05_classification.ipynb`, `src/association.py`, `src/classification.py` | ✅ Done |
| 5 | Clustering and outlier detection | `notebooks/06_clustering.ipynb` | ⏳ Planned |
| 6 | Evaluation and visualization | `outputs/` | ⏳ Planned |

**Phase 1 — Ingestion.** Load every raw table and record its shape, dtypes, null rates, duplicates, and key columns in `outputs/01_summary.txt`.

**Phase 2 — Preprocessing.** Join the postings tables (salaries, skills, industries), normalize titles, annualize salaries by `pay_period`, drop rows with no salary or implausible salary and reposted duplicate ads, impute by experience-level median, and bin salary into Low / Mid / High tiers at the 33rd and 66th percentiles. The top 100 skills from the 1.3M LinkedIn Jobs dataset are matched in each posting's description to produce `matched_skills`. Outputs `data/processed/cleaned_jobs.csv` (34,179 jobs) and `data/processed/linkedin_jobs_skills.csv` (1.29M per-job skill lists). Run with `python -m src.preprocessing`.

**Phase 3 — Warehousing.** Build a SQLite star schema in `data/processed/skillmap.db`: a `job_postings` fact table at (job, skill) grain; skill, location, company, time, and industry dimensions; and a one-row-per-job `v_jobs` view. Then run OLAP queries for top skills, salary by experience level, and jobs by industry × state. Run with `python -m src.warehouse`.

**Phase 4 — Association mining and classification.** Mine skill association rules with Apriori (support ≥ 0.05, confidence > 0.5, lift > 1.5) and build a skill co-occurrence network. Mining is repeated on High-tier jobs, and each rule is scored by how much more likely jobs with that skill set are to be High-paying (`python -m src.association`). Headline result: 81% of jobs listing both python and engineering are High tier (2.4× the baseline). Predict salary tier with Logistic Regression, Random Forest, and XGBoost (targets: macro F1 > 0.75, ROC-AUC > 0.80; `python -m src.classification`). Best: Random Forest with macro F1 0.690 and ROC-AUC 0.862, so the AUC target is met but the F1 target is not. Experience level and company size are the strongest predictors.

**Phase 5 — Clustering.** Cluster TF-IDF skill vectors with K-Means and DBSCAN to find role archetypes (targets: silhouette > 0.50, Davies-Bouldin < 1.0).

**Phase 6 — Evaluation and visualization.** Compare the models and produce the final figures and report.

---

## Setup

```bash
git clone https://github.com/karthik-al1uri/SkillMap.git
cd SkillMap

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: XGBoost needs OpenMP -> brew install libomp
python -m spacy download en_core_web_sm

# Download the three datasets into data/raw/ (see above), then:
jupyter notebook notebooks/01_data_ingestion.ipynb
```

`job_summary.csv` is 4.8 GB. By default, the ingestion notebook loads only its first 200,000 rows. To change this, adjust `LARGE_FILE_MB` / `LARGE_FILE_NROWS` in Step 2.

---

## Project Structure

```
SkillMap/
├── data/raw/          raw Kaggle datasets (gitignored)
├── data/processed/    pipeline outputs (gitignored)
├── notebooks/         one notebook per pipeline stage
├── src/               reusable pipeline modules
└── outputs/           results, models, and figures (gitignored)
```
