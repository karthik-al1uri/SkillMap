## SkillMap Progress Report
**Last Updated:** 2026-09-23 02:32 MDT
**Current Stage:** Stage 5 — Classification (complete)

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings with 100-skill description matching (`matched_skills`) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: Apriori rules on all jobs and on High-tier jobs, plus a skill co-occurrence network → `outputs/04_association_rules.csv`, `outputs/04_high_salary_rules.csv`
- Stage 5 — Classification: Logistic Regression, Random Forest, and XGBoost salary tier models; Random Forest is best → `outputs/05_best_model.pkl`

### 🔄 Current Stage Summary
**What was built:** `src/classification.py` and `notebooks/05_classification.ipynb` build 139 features (100 skill flags, experience ordinal, company size ordinal from warehouse employee_count, top-20 states, and top-15 industries). They train three models on an 80/20 stratified split with random_state=42. The best model (Random Forest) is saved together with its feature metadata, along with confusion-matrix and feature-importance plots.
**Key outputs:**
- `outputs/05_classification_results.csv`
- `outputs/05_best_model.pkl`
- `outputs/05_confusion_matrix.png`
- `outputs/05_feature_importance.png`
- `outputs/05_summary.txt`

**Rows/records processed:** 34,179 jobs (27,343 train / 6,836 test), 139 features

| Model | F1 macro | ROC-AUC (OvR) |
|---|---|---|
| Random Forest (best) | 0.690 | 0.862 |
| XGBoost | 0.667 | 0.847 |
| Logistic Regression | 0.612 | 0.801 |

**Issues found:**
- **F1 target (> 0.75) not met; ROC-AUC target (> 0.80) met by all three models.** Almost all errors are at the Mid tier boundary. Only about 5% of Low jobs are predicted High, and vice versa.
- **Duplicate leakage found and fixed in Stage 2:** 15% of test rows had an identical feature vector in train, and Random Forest scored F1 0.935 on those vs 0.670 on the rest. Stage 2 now drops 1,425 reposted ads (same company, title, description, salary). Stages 2-4 were re-run, which lowered Random Forest from 0.709 to 0.690 F1 (the honest number).
- Random Forest's train F1 is 0.99, so it memorizes heavily, though it still generalizes best of the three.
- **Recommendation:** add job_title TF-IDF as a feature. A quick XGBoost experiment reached F1 0.740 / AUC 0.895.
- Encoding notes:
  - The data's "Associate" / "Mid-Senior level" are mapped to Mid = 2 / Senior = 3.
  - LinkedIn's company_size is a 1-7 code, so the 0-8 size buckets are binned from employee_count.
- Setup: xgboost needed `brew install libomp`, and is installed in the project `.venv`.
- Stage 4 numbers moved slightly after deduplication:
  - 35 rules; python + engineering is still 2.4× High.
  - retail + sales dropped below 5% support, because it had been inflated by reposted retail ads.

### ⏭ Next Stage
**Stage 6 — Clustering:** vectorize jobs by TF-IDF of skills, then run K-Means (k = 5-10, elbow method) and DBSCAN. Evaluate with silhouette (> 0.50) and Davies-Bouldin (< 1.0), and label clusters by their top 5 skills.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (F1 target not met) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ⏳ Pending | |
