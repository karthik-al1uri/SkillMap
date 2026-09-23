## SkillMap Progress Report
**Last Updated:** 2026-09-23 02:16 MDT
**Current Stage:** Stage 4 — Association Mining (complete)

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, salary-tiered postings with 100-skill description matching (`matched_skills`) → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: Apriori rules on all jobs and on High-tier jobs, plus a skill co-occurrence network → `outputs/04_association_rules.csv`, `outputs/04_high_salary_rules.csv`

### 🔄 Current Stage Summary
**What was built:** `src/association.py` and `notebooks/04_association_mining.ipynb` run Apriori (min_support 0.05) on each job's `matched_skills` and keep rules with confidence > 0.5 and lift > 1.5. They also build a NetworkX co-occurrence graph (node size = skill frequency, edge weight = lift). Mining is repeated on High-tier jobs, and each rule is scored against all jobs (`high_tier_lift`) to show which skill combinations actually predict High pay.
**Key outputs:**
- `outputs/04_association_rules.csv` (top 20 by lift)
- `outputs/04_high_salary_rules.csv` (123 rules)
- `outputs/04_skill_network.png`
- `outputs/04_summary.txt`

**Rows/records processed:**
- All jobs: 35,604 transactions → 242 frequent itemsets → 38 rules
- High tier: 11,868 transactions → 377 itemsets → 123 rules, of which 100 have high_tier_lift > 1.2
- Graph: 34 skills, 139 edges (91 plotted with lift ≥ 1.2)

**Issues found:**
- Fixed in Stage 2: nested phrases ("project management" ⊃ "management") created tautological rules with confidence 1.0. Longer phrases are now matched first and removed from the text. Stages 2 and 3 were re-run; the fact table is now 351,540 rows.
- Key findings:
  - 81% of jobs listing both python and engineering are High tier (2.4× baseline).
  - Leadership + collaboration + management + organization reaches about 2×.
  - training + education is common but *below* baseline (0.87×).
  - Strongest pairs: retail ↔ sales (lift 2.7), marketing ↔ sales (2.5), maintenance ↔ safety (2.1).
- min_support 0.05 excludes most technical skills (sql appears in only 4.1% of jobs). A lower-support run (e.g. 0.02) would surface technical combinations.
- `mlxtend` and `ipython` are installed in a project `.venv` (Homebrew Python blocks pip). Run pipelines with `.venv/bin/python`.

### ⏭ Next Stage
**Stage 5 — Classification:** predict salary_tier from TF-IDF of skills plus experience_level, company_size (from `dim_company`), location, and industry, comparing Logistic Regression, Random Forest, and XGBoost (targets: macro F1 > 0.75, ROC-AUC > 0.80).

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ⏳ Pending | |
| Stage 6 — Clustering | ⏳ Pending | |
