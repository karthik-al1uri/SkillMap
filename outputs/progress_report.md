## SkillMap Progress Report
**Last Updated:** 2026-09-30 17:35 MDT
**Current Stage:** All stages complete — final report written

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with a 245-skill description vocabulary (top 200 + 50 forced technical skills) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema (449,665 fact rows) with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: 91 rules on all jobs (top lift 5.54) and 277 on High-tier jobs, plus a co-occurrence network → `outputs/04_association_rules.csv`
- Stage 5 — Classification: 4 models. Best is tuned XGBoost + title & company TF-IDF (F1 0.728, AUC 0.891) → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means (k=3) and DBSCAN; the tech cluster earns $50k more than the service cluster → `outputs/06_cluster_labels.csv`
- Final report: research-paper write-up answering Q1-Q3 → `outputs/final_report.md`

### 🔄 Current Stage Summary
**What was built:**
- Stage 5 Model 4 now adds TF-IDF of the top 30 company-name words (legal suffixes such as "inc" and "llc" removed) and a skill-count feature.
- Model 4 is tuned by a 3-fold CV grid on the training data: max_depth {4, 6, 8} × n_estimators {200, 400} × learning_rate {0.05, 0.1} × subsample {0.8, 1.0}.
- The requested `salary_per_skill` feature (salary ÷ number of skills) was **not** added: it is computed from the target and would leak the answer.
- `outputs/final_report.md` presents the whole project as a data mining paper.

**Key outputs:**
- `outputs/final_report.md`
- Updated `outputs/05_classification_results.csv`, `05_best_model.pkl`, `05_summary.txt`, and the Stage 5 plots
- `notebooks/05_classification.ipynb` updated to the new API

**Rows/records processed:** 34,179 jobs (27,343 train / 6,836 test), 285 base features + 80 TF-IDF features

| Model | F1 macro (> 0.75) | ROC-AUC (> 0.80) |
|---|---|---|
| XGBoost + title & company TF-IDF (tuned) | **0.728** ❌ | **0.891** ✅ |
| Random Forest | 0.703 ❌ | 0.873 ✅ |
| XGBoost | 0.683 ❌ | 0.858 ✅ |
| Logistic Regression | 0.632 ❌ | 0.815 ✅ |

**Issues found:**
- **F1 > 0.75 is not met.** The final push scored 0.728, slightly below the 0.731 from the previous title-only setup with 600 trees. The requested grid stops at 400 trees, and the company-name words are generic ("america", "bank", "health"). Almost all errors involve the Mid tier; only 2% of test jobs are confused between Low and High.
- **Silhouette > 0.50 is not met** (best 0.214). Skill profiles form a continuum. DBSCAN meets Davies-Bouldin < 1.0 (0.93).
- **References.** The proposal was not in the repo, so the report cites 7 standard method papers (Apriori, Random Forest, XGBoost, k-means, DBSCAN, silhouette, Davies-Bouldin). Replace them if the proposal's list differs.

### ⏭ Next Stage
**Project complete.** Optional: re-execute notebooks 02-06 so their saved outputs match the final numbers, and add the outlier-detection piece from Phase 5.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (AUC met; F1 0.728 < 0.75) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (DB met by DBSCAN; silhouette not met) | outputs/06_cluster_labels.csv |
| Final report | ✅ Complete | outputs/final_report.md |
