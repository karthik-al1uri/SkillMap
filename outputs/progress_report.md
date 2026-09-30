## SkillMap Progress Report
**Last Updated:** 2026-09-30 17:10 MDT
**Current Stage:** Expanded technical skill vocabulary; reran Stages 2-6

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with description matching over a 245-skill vocabulary (top 200 + 50 forced technical skills) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema with 3 OLAP queries, rebuilt on the new vocabulary → `data/processed/skillmap.db`
- Stage 4 — Association Mining: Apriori rules on all jobs and on High-tier jobs, plus a skill co-occurrence network, rerun on the new vocabulary → `outputs/04_association_rules.csv`, `outputs/04_high_salary_rules.csv`
- Stage 5 — Classification: 4 models. Best is XGBoost + title TF-IDF (F1 0.731, AUC 0.891) → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means (k=3) and DBSCAN on multi-hot skills; a Data, Tech & Engineering cluster now appears → `outputs/06_cluster_labels.csv`

### 🔄 Current Stage Summary
**What was built:**
- `TOP_N_SKILLS` went from 100 to 200. A new `FORCED_SKILLS` list of 50 technical skills (languages, ML/AI, data, cloud, databases, git/linux/rest api/agile/scrum) is always added; 45 of them were outside the top 200.
- Ambiguous terms (r, go, rust, swift, spark, airflow, dbt, transformers) only count when listed next to, or near, other technical terms. This keeps out "ready to go", HVAC airflow, DBT therapy and electrical transformers. Extra spellings are searched too (golang, postgres, k8s, natural language processing, …).
- 19 benefits/boilerplate terms that entered the top 200 were added to `EXCLUDED_TERMS` (paid holidays, 401(k) plan, dental benefits, background check, …), along with "word" and "outlook", which mostly match everyday English.

**Key outputs:** `data/processed/cleaned_jobs.csv`, `skill_vocabulary.csv`, `skillmap.db`; `outputs/02-06_*` summaries, rules, model, plots, and cluster labels

**Rows/records processed:**
- 34,179 jobs (unchanged); fact table 449,665 rows (was 337,421)
- Skill coverage: mean 11.5 matched skills per job (was 8.2); 99.3% of jobs have ≥ 1 skill (33,924, was 98.5% / 33,675); 5,057 jobs (14.8%) have ≥ 1 forced technical skill
- Top technical skills: agile 1,641 · sql 1,430 · python 1,338 · aws 879 · azure 750 · java 631 · machine learning 589

| Stage 5 model | F1 macro (was) | ROC-AUC (was) |
|---|---|---|
| XGBoost + title TF-IDF (best) | **0.731** (0.719) | **0.891** (0.884) |
| Random Forest | 0.707 (0.690) | 0.872 (0.862) |
| XGBoost | 0.679 (0.667) | 0.857 (0.847) |
| Logistic Regression | 0.632 (0.612) | 0.815 (0.801) |

| Stage 6 method | Silhouette > 0.50 (was) | Davies-Bouldin < 1.0 (was) |
|---|---|---|
| K-Means k = 3 | 0.056 ❌ (0.031) | 4.72 ❌ (4.19) |
| DBSCAN eps = 2.83, min_samples = 20 | 0.214 ❌ (0.167) | **0.93 ✅** (1.04) |

| K-Means cluster | Jobs | Mean salary | Defining skills |
|---|---|---|---|
| Data, Tech & Engineering (new) | 5,978 (17%) | $131,946 (64% High) | python, computer science, engineering, design, sql |
| Management & Leadership | 10,497 (31%) | $100,722 | finance, leadership, reporting, planning, interpersonal skills |
| Sales, Retail & Customer Service | 17,449 (51%) | $81,952 (22% High) | customer service |

**Issues found:**
- **F1 > 0.75 is still not met.** Every model gained 0.01-0.02 F1, and the best reaches 0.731. Only 14.8% of postings mention any technical skill, which caps what technical skills can add. python, aws, machine learning and sql are now among the top 20 features.
- **Clustering improved but the silhouette target is still not met.**
  - K-Means silhouette rose from 0.03 to 0.06, and it now finds a distinct, high-paying tech cluster.
  - DBSCAN now meets the Davies-Bouldin target (0.93), but its 2 clusters hold 99.8% of jobs in one and 31.8% of jobs are noise.
  - The "continuum" conclusion stands.
- Stage 2 now takes ~8 min (was ~3), because of the larger vocabulary and the context regexes.
- Stages 3 and 4 were also rerun so the warehouse and rules match the new `cleaned_jobs.csv`. Stage 4 now has 91 rules for all jobs (was 35) and 277 for High-tier jobs (was 121).
- The notebooks were not re-executed, so their saved cell outputs show the old 100-skill numbers.

### ⏭ Next Stage
**Phase 6 — Evaluation & Visualization:** pull results from Stages 4-6 into the final report and figures, answering the three research questions.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete (rerun, 245 skills) | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete (rerun) | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete (rerun) | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (F1 target not met: 0.731) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (silhouette not met; DBSCAN DB met) | outputs/06_cluster_labels.csv |
