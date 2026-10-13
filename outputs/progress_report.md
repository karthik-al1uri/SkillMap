## SkillMap Progress Report
**Last Updated:** 2026-10-13 00:45 MDT
**Current Stage:** All improvements complete — low-support mining, t-SNE, final report updated

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with a 245-skill description vocabulary (top 200 + 50 forced technical skills) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema (449,665 fact rows) with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: 91 rules on all jobs (top lift 5.54), 277 on High-tier jobs, a co-occurrence network, and a 0.02-support pass for technical rules → `outputs/04_association_rules.csv`, `outputs/04_technical_rules_low_support.csv`
- Stage 5 — Classification: 6 models. Best is tuned XGBoost + title & company TF-IDF (F1 0.728, AUC 0.891); the ordinal models don't help → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means (k=3), DBSCAN, t-SNE vs PCA, and Isolation Forest outliers → `outputs/06_cluster_labels.csv`, `outputs/06_clusters_tsne.png`, `outputs/06_outliers.csv`
- Final report: covers every result, including the low-support rules, t-SNE, outliers, and the ordinal negative result → `outputs/final_report.md`

### 🔄 Current Stage Summary
**What was built:**
- **Stage 4 low-support pass.** All jobs are re-mined at min_support 0.02, keeping rules with a technical skill. Each rule gets High-tier scores and flags for whether it is new (support < 0.05) and whether it already appears in the High-tier rules. The most common technical–technical pairs are listed separately.
- **Stage 6 t-SNE.** t-SNE runs on the same 8,000-job sample as PCA and is plotted 2×2 against it. The views are compared by kNN overlap and purity, and High-tier hotspots are found with DBSCAN on the t-SNE coordinates.
- **Final report.**
  - Methodology adds Isolation Forest, t-SNE, the ordinal models, and the 0.02 pass.
  - Q1 adds the technical rules, Q2 adds t-SNE and outliers, and Q3 adds the ordinal negative result.
  - The Discussion has a new outlier section. The abstract and conclusion are updated, and three references are added.
  - The old future-work item 6 (outlier detection) is removed, and item 1 now records that ordinal was tried.

**Key outputs:**
- New: `outputs/04_technical_rules_low_support.csv`, `outputs/06_clusters_tsne.png`
- Updated: `04_summary.txt`, `06_summary.txt`, `final_report.md`
- Re-executed: notebooks 04 and 06 (notebook 04's stale Findings numbers also fixed)

**Rows/records processed:** 34,179 transactions (2,444 frequent itemsets at 0.02 support, 1,421 rules, 10 technical); 8,000-job t-SNE sample

**Issues found:**
- **0.02 support is still too high for most technical skills.**
  - Only agile, sql, python, aws, and azure reach 2%.
  - All 10 technical rules are new, but each pairs one of those skills with engineering, design, or development.
  - Three are new high-pay combinations not already in the High-tier rules: python+development (78% High), sql+design (71%), and sql+development (66%).
  - Technical–technical pairs stay below 2% (python+sql 1.5%) but are the best paid: aws+python 88% High, ML+python 85%.
- **t-SNE finds no hidden clusters, but it does find local structure PCA misses.**
  - It keeps 18.1% of each job's true neighbours vs 3.0% for PCA.
  - It shows High-tier pockets, notably acute-care nursing (79% High), which K-Means puts in the low-paid Service cluster.
- Targets unchanged: F1 0.728 < 0.75 and silhouette 0.214 < 0.50 are still not met.

### ⏭ Next Stage
**Project complete.** Optional extensions: salary regression with residual-based anomaly detection, and association mining restricted to technical postings (to get tech–tech rules).

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete (incl. 0.02 technical pass) | outputs/04_association_rules.csv, outputs/04_technical_rules_low_support.csv |
| Stage 5 — Classification | ✅ Complete (AUC met; F1 0.728 < 0.75; ordinal didn't help) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (DB met by DBSCAN; silhouette not met; t-SNE + outliers done) | outputs/06_cluster_labels.csv, outputs/06_clusters_tsne.png, outputs/06_outliers.csv |
| Final report | ✅ Complete (updated) | outputs/final_report.md |
