## SkillMap Progress Report
**Last Updated:** 2026-10-14 12:45 MDT
**Current Stage:** Stage 7 — Live Data Pipeline (Adzuna API)

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with a 245-skill description vocabulary → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema (449,665 fact rows) with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: 91 rules on all jobs, 277 on High-tier jobs, a co-occurrence network, and a 0.02-support technical pass → `outputs/04_association_rules.csv`
- Stage 5 — Classification: 6 models. Best is tuned XGBoost + title & company TF-IDF (F1 0.728, AUC 0.891) → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means, DBSCAN, t-SNE vs PCA, and Isolation Forest outliers → `outputs/06_cluster_labels.csv`
- Stage 7 — Live Data: 1,150 current Adzuna IT postings, preprocessed and warehoused, and tested against the Stage 4 rules and the Stage 5 model → `outputs/07_live_summary.txt`
- Final report → `outputs/final_report.md` (does not yet cover Stage 7)

### 🔄 Current Stage Summary
**What was built:**
- **`src/live_pipeline.py`** loads credentials from `.env`, then:
  - runs a one-call health check, then fetches IT postings under a hard 25-call budget (every call is logged before it is sent)
  - caches every response, so reruns make no calls
  - extracts skills with the Stage 2 matcher and tiers salaries with the exact training thresholds
  - loads `live_job_postings` into the warehouse, mines rules, and compares them with training
  - scores the saved Stage 5 model on live salaries
- **`notebooks/07_live_pipeline.ipynb`** runs Steps 1-7 from the cache.

**Key outputs:**
- `data/raw/adzuna_live/` (raw_jobs.json, live_jobs.csv, pages/, api_calls.json)
- `data/processed/live_cleaned_jobs.csv`
- `skillmap.db` table `live_job_postings`
- `outputs/07_live_association_rules.csv`, `07_rules_comparison.csv`, `07_live_predictions.csv`, `07_live_summary.txt`

**Rows/records processed:** 1,150 unique postings (25 of 25 API calls: 2 health checks + 23 pages of 50); 2,570 warehouse rows; 1,149 postings with salary

**Issues found:**
- **Authentication.** Adzuna needs `app_id` + `app_key`. The specified `api_key`-only call returned HTTP 400, which cost 1 call, so 23 pages were fetched instead of 24.
- **Salaries are mostly estimates.** 1,092 of 1,149 (95%) are Adzuna's own estimates (`salary_is_predicted=1`); only 57 are advertised.
- **Descriptions are snippets** of about 500 characters: 2.0 skills per posting vs 11.5 in training.
- **Rules don't transfer.** There are no rules at 0.05 support and one at 0.02 (compliance → security). 0 of the 20 training rules reappear.
- **The model falls below baseline.** Accuracy is 0.594 and macro F1 0.478, below the 0.621 "always High" baseline. Experience level, company size, and industry are unavailable.
- **Pay holds up.** Live IT postings are 62% High tier (median $123,689 vs $82,500), which agrees with the training finding that technical roles pay more.
- **Credential exposure, now fixed.** Adzuna embeds the app ID in each `redirect_url`. Cached pages and outputs are now scrubbed, and a scan finds no credential in any saved file or notebook.

### ⏭ Next Stage
**No API budget remains.** Optional: add a Stage 7 section to `final_report.md`. A meaningful live validation would need a source with full descriptions and advertised salaries.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (AUC met; F1 0.728 < 0.75) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (silhouette not met) | outputs/06_cluster_labels.csv |
| Stage 7 — Live Data | ✅ Complete (API budget used: 25/25) | outputs/07_live_summary.txt |
| Final report | ⚠️ Stage 7 not yet included | outputs/final_report.md |
