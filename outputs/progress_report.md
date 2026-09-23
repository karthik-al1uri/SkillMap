## SkillMap Progress Report
**Last Updated:** 2026-09-23 02:51 MDT
**Current Stage:** Stage 6 — Clustering (complete)

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings with 100-skill description matching (`matched_skills`) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: Apriori rules on all jobs and on High-tier jobs, plus a skill co-occurrence network → `outputs/04_association_rules.csv`, `outputs/04_high_salary_rules.csv`
- Stage 5 — Classification: 4 models, including the new XGBoost + title TF-IDF, which is best (F1 0.719, AUC 0.884) → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means (k=3) and DBSCAN on multi-hot skills, with cluster profiles and names → `outputs/06_cluster_labels.csv`

### 🔄 Current Stage Summary
**What was built:**
- `src/clustering.py` and `notebooks/06_clustering.ipynb` cluster jobs by their 100-skill multi-hot vectors.
- K-Means is swept over k = 3-10 (elbow + silhouette, k = 3 chosen), and DBSCAN's eps is tuned from the k-distance staircase.
- Each K-Means cluster is profiled by defining skills (lift), common titles, and salary, and named automatically from a skill-theme map.
- Stage 5 also gained Model 4: XGBoost + TF-IDF of the top 50 job-title words, with hyperparameters tuned by 3-fold CV on train only.

**Key outputs:**
- `outputs/06_cluster_labels.csv`
- `outputs/06_elbow_plot.png`
- `outputs/06_kdistance_plot.png`
- `outputs/06_clusters_pca.png`
- `outputs/06_cluster_profiles.txt`
- `outputs/06_summary.txt`
- Updated `outputs/05_classification_results.csv`, `05_summary.txt`, `05_best_model.pkl`, and the Stage 5 plots

**Rows/records processed:**
- Clustering: 34,179 jobs; 33,675 clustered (504 with no matched skills labelled −1); 27,058 distinct skill sets
- Stage 5: 27,343 train / 6,836 test, 189 features for Model 4

| Stage 6 method | Clusters | Silhouette (> 0.50) | Davies-Bouldin (< 1.0) |
|---|---|---|---|
| K-Means k = 3 | 3 | 0.031 ❌ | 4.19 ❌ |
| DBSCAN eps = 2.45, min_samples = 20 | 2 (99.9% in one) + 14.7% noise | 0.167 ❌ | 1.04 ❌ |

| Cluster | Jobs | Mean salary | Defining skills |
|---|---|---|---|
| Management & Leadership | 9,524 (28%) | $113,785 (45% High) | project management, leadership, planning, collaboration, reporting |
| Education & Training | 9,186 (27%) | $80,343 (47% Low) | training, high school diploma, safety, customer service, education |
| Generalist (few listed skills) | 14,965 (44%) | $95,119 (even tier mix) | none over-represented; 5.4 skills per job on average |

**Issues found:**
- **Clustering targets are not met, and this is a property of the data, not of the tuning.**
  - Silhouette is < 0.04 for every k, and the inertia curve has no elbow.
  - Every DBSCAN eps gives one dominant cluster.
  - A denser TF-IDF → SVD(10) representation only reaches 0.14.
  - Conclusion: skill profiles form a continuum rather than discrete archetypes. That answers research question 2, and the K-Means partition still shows a $34k pay gap between the leadership and training profiles.
- **Classification F1 target is still not met.**
  - Model 4 lifts macro F1 from 0.690 to 0.719 and AUC from 0.862 to 0.884. It's saved as the best model because it beats the others on every metric, even though it misses 0.75.
  - A 2,000-term title vocabulary reaches about 0.74 in exploration.
- The small vocabulary (only python and sql are technical) limits both stages. A larger technical vocabulary is the most promising improvement.
- Fixed during Stage 6:
  - The DBSCAN eps candidates were being rounded below the √n steps.
  - A cluster whose "defining" skills were actually under-represented had been named "Sales" by mistake.

### ⏭ Next Stage
**Phase 6 — Evaluation & Visualization:**
- Pull results from Stages 4-6 into the final report and figures, answering the three research questions.
- Optional improvements: a larger technical skill vocabulary and outlier detection.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (F1 target not met) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (targets not met) | outputs/06_cluster_labels.csv |
