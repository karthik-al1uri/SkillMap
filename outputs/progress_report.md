## SkillMap Progress Report
**Last Updated:** 2026-10-13 00:10 MDT
**Current Stage:** Improvements — Stage 6 outlier detection + Stage 5 ordinal classification experiment

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with a 245-skill description vocabulary (top 200 + 50 forced technical skills) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema (449,665 fact rows) with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: 91 rules on all jobs (top lift 5.54) and 277 on High-tier jobs, plus a co-occurrence network → `outputs/04_association_rules.csv`
- Stage 5 — Classification: 6 models. Best is still tuned XGBoost + title & company TF-IDF (F1 0.728, AUC 0.891); the ordinal models (5-6) don't beat it → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means (k=3) and DBSCAN, plus Isolation Forest outliers (342 jobs) → `outputs/06_cluster_labels.csv`, `outputs/06_outliers.csv`
- Final report: research-paper write-up answering Q1-Q3 → `outputs/final_report.md` (does not yet cover outliers or the ordinal experiment)

### 🔄 Current Stage Summary
**What was built:**
- **Stage 6 outlier detection.** An Isolation Forest (contamination 0.01, 300 trees) runs on 245 skill columns + experience ordinal + salary. A salary-mismatch test z-scores log salary within each job's (K-Means cluster, experience) group. A sensitivity run compresses skills to SVD(10) so salary carries more weight.
- **Stage 5 ordinal experiment.** Model 5 is Ordinal Ridge and Model 6 is Logistic IT (both `mord`), on the same features as Model 4. A same-feature multinomial Logistic Regression control isolates the effect of the ordering. The results table now has per-tier F1.

**Key outputs:**
- New: `outputs/06_outliers.csv`
- Updated: `05_classification_results.csv`, `05_summary.txt`, `05_confusion_matrix.png`, `06_summary.txt`
- Updated and re-executed: notebooks 05 and 06 (their stale 100-skill markdown is also fixed)
- `requirements.txt` adds `mord`

**Rows/records processed:** 34,179 jobs. Stage 5 uses 27,343 train / 6,836 test rows; 342 jobs (1%) are flagged as outliers.

| Model | Macro F1 (> 0.75) | Mid F1 | ROC-AUC (> 0.80) |
|---|---|---|---|
| XGBoost + title & company TF-IDF (tuned) | **0.728** ❌ | 0.615 | **0.891** ✅ |
| Random Forest | 0.703 ❌ | 0.588 | 0.873 ✅ |
| XGBoost | 0.683 ❌ | 0.565 | 0.858 ✅ |
| Logistic IT (mord) — Model 6 | 0.642 ❌ | 0.485 | 0.820 ✅ |
| Logistic Regression | 0.632 ❌ | 0.496 | 0.815 ✅ |
| Ordinal Ridge (mord) — Model 5 | 0.607 ❌ | 0.559 | 0.817 ✅ |
| *LR control, same features as Models 4-6* | *0.667* | *0.535* | *0.849* |

**Issues found:**
- **The ordinal approach doesn't help.** Logistic IT (0.642) is below both the control (0.667) and the best model (0.728). Ordinal Ridge has a higher Mid F1 than the control (0.559 vs 0.535) only because it predicts Mid for 57% of test jobs (the actual share is 33%). Its macro F1 is 0.060 lower than the control's. F1 > 0.75 is still not met.
- **The outliers are skill-dense postings, not salary mismatches.**
  - The anomaly score correlates 0.95 (Spearman) with skill count and 0.20 with salary.
  - Flagged jobs average 30.8 skills vs 11.5 overall.
  - Only 5.6% of flagged jobs are salary mismatches, vs 4.5% of all jobs.
  - In the salary-weighted sensitivity run, 9.9% are mismatches and 38% of flagged jobs overlap with the main run.
- **Silhouette > 0.50 is still not met** (best 0.214); DBSCAN meets Davies-Bouldin < 1.0 (0.93).
- `final_report.md` doesn't yet mention the outlier or ordinal results.

### ⏭ Next Stage
**Remaining improvements:** association mining at min_support 0.02 and a t-SNE visualization. Then add the outlier and ordinal results to `final_report.md`.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (AUC met; F1 0.728 < 0.75; ordinal experiment done) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (DB met by DBSCAN; silhouette not met; outliers done) | outputs/06_cluster_labels.csv, outputs/06_outliers.csv |
| Final report | ⚠️ Needs update for new results | outputs/final_report.md |
