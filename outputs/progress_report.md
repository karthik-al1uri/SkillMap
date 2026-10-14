## SkillMap Progress Report
**Last Updated:** 2026-10-14 13:05 MDT
**Current Stage:** Stage 8 — The Muse Live Skill Validation

### ✅ Completed Stages
- Stage 1 — Ingestion: notebook loads and profiles all 15 raw tables across the 3 datasets → `outputs/01_summary.txt`
- Stage 2 — Preprocessing: joined, cleaned, deduplicated, salary-tiered postings, with a 245-skill description vocabulary → `data/processed/cleaned_jobs.csv`
- Stage 3 — Warehousing: SQLite star schema (449,665 fact rows) with 3 OLAP queries → `data/processed/skillmap.db`
- Stage 4 — Association Mining: 91 rules on all jobs, 277 on High-tier jobs, a co-occurrence network, and a 0.02-support technical pass → `outputs/04_association_rules.csv`
- Stage 5 — Classification: 6 models. Best is tuned XGBoost + title & company TF-IDF (F1 0.728, AUC 0.891) → `outputs/05_best_model.pkl`
- Stage 6 — Clustering: K-Means, DBSCAN, t-SNE vs PCA, and Isolation Forest outliers → `outputs/06_cluster_labels.csv`
- Stage 7 — Live Data (Adzuna): 1,150 IT postings, but the snippets were too short to test rules → `outputs/07_live_summary.txt`
- Stage 8 — Live Skill Validation (The Muse): 1,378 full-length tech postings; training rules re-tested and skill demand compared → `outputs/08_muse_summary.txt`
- Final report: now includes the live validation (Section 4.4) → `outputs/final_report.md`

### 🔄 Current Stage Summary
**What was built:**
- **`src/muse_pipeline.py`** fetches The Muse public API (no key) for four tech categories, then:
  - caches every page, so reruns make no calls
  - removes duplicates by id and reposts, and strips the HTML with BeautifulSoup
  - extracts skills with the Stage 2 matcher, maps experience levels, and parses states
  - loads `muse_job_postings` into the warehouse
  - validates the Stage 4 rules (plus lift-ceiling and company-capped checks) and compares skill demand with the training tech cluster
- **`notebooks/08_muse_pipeline.ipynb`** runs these steps from the cache.
- **Updated:** `final_report.md` (datasets table, methodology, Section 4.4, Discussion, Conclusion, Future Work) and the Stage 7 summary note.

**Key outputs:**
- `data/raw/muse_live/` (raw_jobs.json, live_jobs.csv, pages/, fetch_log.json)
- `data/processed/muse_cleaned_jobs.csv`
- `skillmap.db` table `muse_job_postings`
- `outputs/08_muse_association_rules.csv`, `08_rules_validation.csv`, `08_skill_comparison.csv`, `08_muse_summary.txt`

**Rows/records processed:** 1,502 postings fetched (76 page calls) → 1,378 unique after dropping 124 reposts; 1,372 with ≥ 2 skills; 20,599 Muse rules

**Issues found:**
- **API access.** CloudFront blocks generic User-Agents (HTTP 403), so requests use a descriptive project User-Agent. Three of the four requested categories don't exist under those names (mapped to Software Engineering, Computer and IT, and Data and Analytics). "Data Science" has only 2 jobs, so the total is 1,502, not 2,000.
- **Full descriptions work.** Postings average 813 words and 16.2 skills (Adzuna 2.0, training 11.5).
- **Only 2 of 20 training rules are confirmed** (1 of 20 when capped at 25 postings per company). But 14 of the 16 failures hit a lift ceiling: their consequents are in 55-62% of tech postings. All 14 have higher confidence than in training, so the groupings persist and lift doesn't transfer to a tech-only sample.
- **Concentration.** SpaceX is 34% of postings and Bank of America 12%. The top-lift Muse rules are employer template text.
- **Demand shift (indicative).** Python rose from 18% to 32% of tech postings; agile and aws fell.

### ⏭ Next Stage
**Project complete.** Optional: validate on a live source with an all-industry mix, and track demand across repeated fetches.

### 📊 Pipeline Health
| Stage | Status | Output File |
|---|---|---|
| Stage 1 — Ingestion | ✅ Complete | outputs/01_summary.txt |
| Stage 2 — Preprocessing | ✅ Complete | data/processed/cleaned_jobs.csv |
| Stage 3 — Warehousing | ✅ Complete | data/processed/skillmap.db |
| Stage 4 — Association Mining | ✅ Complete | outputs/04_association_rules.csv |
| Stage 5 — Classification | ✅ Complete (AUC met; F1 0.728 < 0.75) | outputs/05_best_model.pkl |
| Stage 6 — Clustering | ✅ Complete (silhouette not met) | outputs/06_cluster_labels.csv |
| Stage 7 — Live Data (Adzuna) | ✅ Complete (API budget used: 25/25) | outputs/07_live_summary.txt |
| Stage 8 — Live Skill Validation (The Muse) | ✅ Complete (2/20 rules confirmed; lift-ceiling effect) | outputs/08_muse_summary.txt |
| Final report | ✅ Complete (includes Section 4.4) | outputs/final_report.md |
