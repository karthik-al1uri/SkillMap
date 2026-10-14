# SkillMap: Mining Skill Co-occurrence, Role Archetypes, and Salary Tiers from LinkedIn Job Postings

**Author:** Karthik Alluri · CU Boulder, Data Mining, Fall 2026
**Code:** https://github.com/karthik-al1uri/SkillMap

## Abstract

We mine 34,179 salaried U.S. LinkedIn postings, using a 245-skill vocabulary built from 1.3 million skill lists, to ask which skills co-occur and predict pay, whether role archetypes exist beyond titles, and whether salary tier is predictable. Apriori finds 91 rules (lift up to 5.54). Technical combinations such as {computer science, design, engineering} are 83% High-tier, 2.5× the baseline. Clustering finds only a weak continuum (silhouette ≤ 0.21), yet it separates a technical profile earning $50k more than service roles. t-SNE finds small well-paid pockets that the partition misses, such as acute-care nursing. Isolation Forest outliers turn out to be skill-dense postings, not salary mismatches. Tuned XGBoost reaches ROC-AUC 0.891 but macro F1 0.728, short of the 0.75 target. Ordinal models do not close the gap. On 1,378 current (2026) tech postings from The Muse, only 2 of the 20 top rules are confirmed. Most of the rest fail because of a base-rate ceiling on lift, while their confidence rises.

## 1. Introduction

### 1.1 Problem statement and motivation

Job seekers, educators, and workforce analysts all ask which skills are worth learning and what they pay. Job titles answer this poorly. The same title ("analyst", "associate", "engineer") covers very different work and pay, and postings list skills inconsistently. Large public LinkedIn datasets now make it possible to study skills, roles, and compensation together, using standard data mining techniques [1]–[7].

SkillMap is an end-to-end pipeline that goes from raw job postings to a data warehouse. Three analyses then run on top of it: association rule mining, clustering, and classification.

### 1.2 Research questions

1. **Q1.** Which skills consistently appear together, and which combinations predict higher compensation?
2. **Q2.** Do distinct job role archetypes exist beyond traditional job titles?
3. **Q3.** Can salary tier (Low / Mid / High) be reliably predicted from job posting features alone?

## 2. Dataset

Three public Kaggle datasets are used for training and analysis. Two live APIs are used to validate the results on current postings:

| Dataset | Size | Key fields | Role in SkillMap |
|---|---|---|---|
| **LinkedIn Job Postings 2023-2024** (arshkon) | 123,849 postings, 11 linked tables | title, description, `normalized_salary`, `pay_period`, `formatted_experience_level`, location, company, industries, 35 job-function skill categories | Main analysis table |
| **1.3M LinkedIn Jobs & Skills 2024** (asaniczka) | 1.29M postings with skill lists (396 MB postings, 642 MB skills, 4.8 GB summaries) | `job_link`, `job_skills` (comma-separated), title, level | Source of the skill vocabulary (no salaries) |
| **Data Science Salaries 2020-2025** (arnabchaki) | 93,597 rows (83,997 U.S. full-time) | `salary_in_usd`, `experience_level`, `job_title`, `remote_ratio`, `company_size` | External salary benchmark |
| **Adzuna API** (live, fetched 2026-10-14) | 1,150 US IT postings (25-call free tier) | title, company, location, `salary_min/max` (95% Adzuna-estimated), ~500-character description snippet | Live check (Stage 7); snippets too short for skill rules |
| **The Muse API** (live, fetched 2026-10-14) | 1,378 unique postings in Software Engineering, Computer and IT, Data and Analytics, and Data Science (no API key) | title, company, locations, levels, categories, **full HTML description** (~810 words), publication date; no salary | **Validation source** for skill co-occurrence and demand (Stage 8) |

**Preprocessing decisions (Stage 2):**

- **Salary**
  - `normalized_salary` is the primary salary. Salaries given per hour, week, two weeks, or month are annualized (×2080, ×52, ×26, ×12).
  - Rows are dropped if they have no salary (87,776), are not in USD (15), or fall outside $10k–$1M (454).
- **Duplicates**
  - 1,425 reposted ads are removed (same company, title, description, and salary).
  - Without this step, 15% of test rows had an identical copy in the training set, which inflated classification scores.
- **Target**
  - Salary is cut into three equal tiers at the 33rd and 66th percentiles: **Low ≤ $62,400**, **Mid ≤ $109,200**, **High > $109,200**. That gives 11,532 / 11,278 / 11,369 jobs.
  - The Data Science Salaries benchmark confirms that data science roles sit high in this distribution: 78% of its U.S. rows would be High tier.
- **Skills**
  - The postings only carry 35 coarse job-function categories, so concrete skills are extracted from the description text instead.
  - **Vocabulary.** The vocabulary is the top 200 skills from the 1.3M-job dataset, plus 50 forced technical skills (languages, ML/AI, data, cloud, databases), for 245 skills in total. Spelling variants and synonyms are merged. Benefits and boilerplate terms such as "401(k)" and "paid holidays" are excluded.
  - **Matching.** Skills are matched with regular expressions.
    - Phrases are matched before any shorter skill they contain ("project management" before "management"). This prevents rules that are true by definition.
    - Ambiguous terms (*r*, *go*, *swift*, *spark*, *dbt*, *transformers*, *airflow*) only count when they appear near other technical terms.
  - **Coverage.** 11.5 skills per job on average; 99.3% of jobs have at least one skill; 14.8% have at least one technical skill.

**Final dataset:** 34,179 jobs, each with `matched_skills`, experience level, industry, location, company, `normalized_salary` (mean $96,586, median $82,500), and `salary_tier`.

## 3. Methodology

| Stage | What was done | Why this technique |
|---|---|---|
| 1. Ingestion | Profiled all 15 raw tables: shapes, types, nulls, samples | Find join keys, salary quirks (mixed pay periods), and missing data before modeling |
| 2. Preprocessing | Joins, salary annualization and filtering, de-duplication, tiering, regex skill extraction | Gives one clean row per job with comparable annual salaries and concrete skills |
| 3. Warehousing | SQLite star schema: fact `job_postings` at the (job, skill) grain (449,665 rows), dimensions for skills, location, company, time, and industry (with a bridge table), and a one-row-per-job view `v_jobs`. OLAP queries: top skills, salary by experience, industry × state | Supports slicing by any dimension. The job-level view prevents a salary being counted once per skill |
| 4. Association mining | Apriori [1] (min support 0.05; confidence > 0.5; lift > 1.5) on all jobs and on High-tier jobs. NetworkX co-occurrence graph. A second pass at min support 0.02 keeps rules that involve one of the 50 technical skills | Apriori is the standard, interpretable way to mine frequent itemsets. Lift controls for skills that are common everywhere. For each High-tier rule, *high-tier lift* = the High share among **all** jobs with that skill combination ÷ 33.3%. It tests whether a combination actually predicts pay |
| 5. Classification | Features: 245 multi-hot skills, number of matched skills, experience level (ordinal), company size (ordinal, 9 buckets), top-20 states, and top-15 industries. Four models: Logistic Regression (baseline), Random Forest [2], XGBoost [3], and XGBoost + TF-IDF of the top 50 title words and top 30 company-name words. Model 4 is tuned by a 3-fold CV grid search on the training data only (max depth {4, 6, 8} × trees {200, 400} × learning rate {0.05, 0.1} × subsample {0.8, 1.0}). Ordinal experiment: Models 5-6 are Ordinal Ridge and Logistic IT [10] (`mord`) on Model 4's features, standardized, with a same-feature multinomial Logistic Regression as a control. 80/20 stratified split, `random_state=42` | Linear baseline → bagged trees → boosted trees shows how much non-linear interaction helps. Ordinal models use the Low < Mid < High order, which could reduce Mid-tier errors. Macro F1 and one-vs-rest ROC-AUC treat the three balanced classes equally |
| 6. Clustering | K-Means [4] (k = 3–10, chosen by silhouette, with an elbow check) and DBSCAN [5] (min_samples 20, eps chosen from the k-distance graph). Scored with silhouette [6] and Davies-Bouldin [7]. Clusters named from skills over-represented in them (lift). t-SNE [9] (perplexity 30, 1,000 iterations) on an 8,000-job sample, compared with PCA. **Outlier detection:** Isolation Forest [8] (300 trees, contamination 0.01) on the 245 skill columns + experience level (ordinal) + `normalized_salary`. To test whether outliers are salary mismatches, each job's log salary is z-scored within its (K-Means cluster, experience level) group; \|z\| > 2 counts as a mismatch | K-Means tests whether the data splits into a few compact archetypes. DBSCAN tests for dense groups of arbitrary shape and lets outliers be labelled as noise. t-SNE keeps local neighbourhoods that a linear projection loses. Isolation Forest isolates unusual postings without assuming a distribution, and needs no feature scaling |

| 7-8. Live validation | Adzuna (Stage 7) and The Muse (Stage 8) postings run through the same skill matcher and tier thresholds and loaded into warehouse tables (`live_job_postings`, `muse_job_postings`). Muse rules are mined with the same Apriori settings (postings with ≥ 2 skills), each training rule is checked against them, and the check is repeated with at most 25 postings per company | Tests whether patterns learned from 2023-2024 data hold on current postings from an independent source. The company cap stops a few large employers' template text from driving the result |

**Leakage controls.**
- Everything learned from data (top states and industries, TF-IDF vocabularies, hyperparameters) is fit on the training split only.
- A proposed `salary_per_skill` feature (salary ÷ number of skills) was rejected because it is computed from the target.

## 4. Results

### Q1 — Skill co-occurrence and compensation (association mining)

**All jobs.** Apriori found 343 frequent itemsets, and **91 rules** pass confidence > 0.5 and lift > 1.5. The co-occurrence graph has 40 skills and 191 edges. The most connected skills are management (37 edges), communication (32), and organization, development, and training (28 each).

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| accounting → finance | 0.055 | 0.56 | **5.54** |
| powerpoint → excel | 0.058 | 0.92 | **5.42** |
| {development, engineering} → design | 0.062 | 0.63 | 3.02 |
| engineering → design | 0.094 | 0.55 | 2.64 |
| marketing → sales | 0.070 | 0.54 | 2.52 |
| {development, management, organization} → leadership | 0.052 | 0.53 | 2.17 |
| compliance + leadership → management | 0.056 | 0.76 | 1.69 |

**Which combinations predict higher pay.** Mining the 11,369 High-tier jobs alone gives 277 rules. For 249 of them, High-tier jobs make up more than 40% of all jobs with the rule's skills (high-tier lift > 1.2, against a 33.3% baseline). The strongest are technical:

| Skill combination | Jobs | Share in High tier | High-tier lift |
|---|---|---|---|
| {computer science, design, engineering} | 836 | **82.8%** | **2.49** |
| {computer science, development, engineering} | 883 | 81.8% | 2.46 |
| {python, engineering} | 810 | 81.5% | 2.45 |
| {design, development, engineering, leadership} | 750 | 80.9% | 2.43 |
| {computer science, engineering} | 1,255 | 78.8% | 2.37 |

Skills that appear together often don't necessarily pay well. {powerpoint, excel} has the second-highest co-occurrence lift (5.42), but only 29.0% of jobs listing both are High tier (high-tier lift 0.87). {training, education} is similar at 29.2% High (0.88).

**Technical rules at lower support (0.02).** Only 14.8% of postings mention a technical skill, so the 5% pass cannot produce technical rules. At 2% support, only agile (4.8%), sql (4.2%), python (3.9%), aws (2.6%), and azure (2.2%) are frequent on their own. **10 technical rules** pass the filters, and **all 10 are new** (support 2.2-3.6%).
- Each pairs python, sql, or agile with a generic skill (engineering, design, development). No rule links two technical skills.
- All 10 predict High pay, at 66-81% High tier. Seven of the itemsets were already in the High-tier rules. The three new high-pay combinations are {python, development} (78% High, 864 jobs), {sql, design} (71%, 773), and {sql, development} (66%, 888).
- Pairs of *two* technical skills stay below 2% (the most common, python + sql, is 1.5%), but they are the best-paid combinations in the data:

| Technical pair | Jobs | Share in High tier | High-tier lift |
|---|---|---|---|
| aws + python | 353 | **88.1%** | **2.65** |
| agile + aws | 264 | 87.9% | 2.64 |
| machine learning + python | 303 | 84.8% | 2.55 |
| java + python | 311 | 83.6% | 2.51 |
| aws + azure | 355 | 82.5% | 2.48 |
| python + sql | 524 | 72.5% | 2.18 |

At the level of single skills, the spread in median salary is large:
- **Highest median salaries:** machine learning **$163,250**, Kubernetes $150,800, C++ $148,225, GCP $147,424, AWS $145,000, Python $140,000.
- **Lowest median salaries:** cash handling **$42,640**, cleaning $44,720, bending $44,720, merchandising $45,760, high school diploma $46,800.

### Q2 — Role archetypes (clustering)

| Method | Clusters | Silhouette (target > 0.50) | Davies-Bouldin (target < 1.0) |
|---|---|---|---|
| K-Means, k = 3 | 3 | 0.056 ❌ | 4.72 ❌ |
| DBSCAN, eps = 2.83, min_samples = 20 | 2 (+31.8% noise; 99.8% of clustered jobs in one) | 0.214 ❌ | **0.93 ✅** |
| Check: TF-IDF → SVD(10), k = 3 | 3 | 0.138 | 2.29 |

- Silhouette falls steadily from k = 3 (0.056) to k = 10 (−0.019), and the inertia curve has no elbow.
- DBSCAN meets the Davies-Bouldin target only by labelling a third of the jobs as noise and putting almost all of the rest into one cluster.

The K-Means partition is weak geometrically, but it is meaningful economically:

| Cluster | Jobs | Mean / median salary | High tier | Defining skills (lift) | Typical titles |
|---|---|---|---|---|---|
| **Data, Tech & Engineering** | 5,978 (17.5%) | **$131,946** / $125,000 | **64%** | python (4.49), computer science (4.47), engineering (4.01), design (3.81), sql (3.59) | software engineer, electrical engineer, manufacturing engineer |
| **Management & Leadership** | 10,497 (30.7%) | $100,722 / $87,500 | 34% | finance (1.97), leadership (1.97), reporting (1.89), planning (1.84) | senior accountant, controller, project manager, general manager |
| **Sales, Retail & Customer Service** | 17,449 (51.1%) | $81,952 / $65,000 | 22% | customer service (1.13); 7.6 skills per job | customer service rep, administrative assistant, package handler |
| No matched skills | 255 (0.7%) | $98,697 / $74,880 | 34% | — | — |

- **Salary gaps.** The technical cluster earns **$50.0k more on average** (median gap $60k) than the service and retail cluster. It also earns $31.2k more than the management cluster.
- **The third cluster is not a clean archetype.** It is mostly a leftover group of postings that list few skills, with only one weakly over-represented skill (customer service, lift 1.13).

**t-SNE vs PCA.** t-SNE also shows one connected mass, with no hidden islands, which agrees with the low silhouette.

| 2D view (8,000 jobs, k = 15) | kNN overlap with 245-D space | K-Means purity | Salary-tier purity |
|---|---|---|---|
| PCA | 3.0% | 0.85 | 0.44 |
| t-SNE | **18.1%** | 0.70 | 0.47 |
| Chance | 0.2% | 0.39 | 0.33 |

- t-SNE keeps six times more of each job's true neighbours than PCA.
- PCA draws the K-Means partition more cleanly, because K-Means splits along the top principal directions.
- That local fidelity reveals **High-tier pockets** (regions where ≥ 70% of a job's 30 neighbours are High tier):
  - a software-engineering core (java, software development, python; 321 sampled jobs, 83% High)
  - product and account roles (marketing, sales; 79 jobs, 75% High)
  - innovation and project-management roles (34 jobs, 79% High)
  - **acute-care nursing** (ACLS, BLS, registered nurse; 38 jobs, 79% High)
- K-Means puts 97% of the acute-care nursing pocket in *Sales, Retail & Customer Service*, its lowest-paid cluster. This is a well-paid niche that a hard 3-way partition cannot represent.

**Outlier detection.** Isolation Forest flags 342 postings (1%).

| Group | Jobs | Mean skills | Median salary | High tier | Salary mismatch (\|z\| > 2) |
|---|---|---|---|---|---|
| All jobs | 34,179 | 11.5 | $82,500 | 33.3% | 4.5% |
| Flagged (main run) | 342 | **30.8** | $117,720 | 58.2% | **5.6%** |
| Flagged (sensitivity: skills → SVD(10)) | 342 | 26.9 | $163,515 | 75.4% | 9.9% |

- **Postings are anomalous because they list many skills.** The anomaly score has Spearman correlation **0.95** with the number of matched skills, against 0.20 with salary and 0.09 with experience level.
- Flagged postings list 2.7× more skills than average. They are long, kitchen-sink descriptions: cleared government roles ("logistics management specialist with security clearance"), lead and director tech roles, and senior clinical roles. They are over-represented at Mid-Senior, Director, and Executive levels.
- **They are not salary mismatches.** Only 5.6% are paid more than two standard deviations above or below their cluster-and-experience group, barely above the 4.5% rate across all jobs.
- **Sensitivity check.** Compressing the skills to 10 SVD components gives salary more weight. This run overlaps the main run by only 38%. It flags more expensive jobs (75% High), but still only 9.9% are salary mismatches.

### Q3 — Predicting salary tier (classification)

Test set: 6,836 jobs (20%, stratified), with 285 base features.

| Model | Features | Macro F1 (target > 0.75) | ROC-AUC OvR (target > 0.80) | Accuracy |
|---|---|---|---|---|
| **XGBoost + title & company TF-IDF** (tuned) | base + 50 title + 30 company TF-IDF | **0.728** ❌ | **0.891** ✅ | 0.729 |
| Random Forest | base | 0.703 ❌ | 0.873 ✅ | 0.705 |
| XGBoost | base | 0.683 ❌ | 0.858 ✅ | 0.685 |
| Logistic Regression | base | 0.632 ❌ | 0.815 ✅ | 0.637 |

- **Tuned hyperparameters for Model 4:** 400 trees, depth 8, learning rate 0.1, subsample 0.8. Its cross-validated macro F1 was 0.725, close to the test score, so the test result is not a lucky split.
- **Errors by class.** Per-class F1 is Low 0.81, Mid 0.62, and High 0.76. Only **2.0%** of test jobs are confused between Low and High. Almost all errors involve the Mid tier.
- **Most important features:**
  - Skill flags: *high school diploma*, *engineering*, *python*, *aws*, *machine learning*, *computer science*, *sql*.
  - Retail industry.
  - Experience level.
  - Title words: *accountant*, *engineer*, *developer*, *shift*.

**How the vocabulary and features affected Model 4:**

| Variant of Model 4 | Macro F1 |
|---|---|
| 100-skill vocabulary, title TF-IDF, 600 trees | 0.719 |
| 245-skill vocabulary, title TF-IDF, 600 trees | 0.731 |
| 245-skill vocabulary, title + company TF-IDF, tuned grid capped at 400 trees (final) | 0.728 |

The company-name words turned out to be generic ("america", "bank", "health", "group") and added no signal.

**Negative result: ordinal classification.** Treating the tiers as ordered does not help.

| Model (base + title & company TF-IDF features) | Macro F1 | Mid-tier F1 | ROC-AUC | Test jobs predicted Mid |
|---|---|---|---|---|
| XGBoost + TF-IDF (Model 4, best) | **0.728** | **0.615** | **0.891** | 32% |
| Logistic Regression, multinomial (control) | 0.667 | 0.535 | 0.849 | 29% |
| Logistic IT, ordinal (Model 6) | 0.642 | 0.485 | 0.820 | 32% |
| Ordinal Ridge (Model 5) | 0.607 | 0.559 | 0.817 | 57% |

- **Both ordinal models score below the unordered control on the same features**, and far below Model 4.
- Ordinal Ridge's higher Mid F1 (0.559 vs 0.535) is an artefact. Rounding a regression output pulls predictions toward the middle, so it calls 57% of test jobs Mid when the true share is 33%. That lowers Low and High F1, and its macro F1 is 0.060 below the control.
- A single linear score with two cut points cannot place Mid jobs better. Mid postings differ from Low and High in *which* skills and title words they contain, not in more or less of one ordered signal.
- Ordinal Ridge has no class probabilities. Its ROC-AUC uses the continuous output *s* as tier scores −*s*, −\|*s* − 1\|, and *s*.

### 4.4 Live Skill Validation

**Why two live sources.** Adzuna (Stage 7) returned 1,150 current IT postings, but its descriptions are ~500-character snippets (71 words). That gave 2.0 skills per posting, too few to mine rules: none of the 20 training rules reappeared, and the salary model scored 59%, below a 62% majority baseline, on mostly Adzuna-estimated salaries. The Muse returns full descriptions.

| Source | Postings | Description length | Skills per posting |
|---|---|---|---|
| Training (LinkedIn 2023-2024, all industries) | 34,179 | full | 11.5 |
| Adzuna (Oct 2026, IT) | 1,150 | 71 words | 2.0 |
| **The Muse (Oct 2026, tech)** | **1,378** | **813 words** | **16.2** |

**The Muse sample.**
- 1,502 postings were fetched (76 page calls); 124 reposted duplicates were removed.
- 73% were published in the 90 days before the fetch.
- Levels: Senior 57%, Mid-Senior 41%.
- The sample is **concentrated**: SpaceX supplies 34% of postings and Bank of America 12%, out of 115 companies.
- The requested "Data Science" category holds only 2 jobs on The Muse, so the tech categories that exist were used.

**Rule validation.** Apriori on the 1,372 postings with ≥ 2 skills (min support 0.05) gives 20,599 rules. The highest-lift ones are long itemsets from a single employer's template text. **2 of the 20 training rules are confirmed** as Muse rules: *powerpoint → excel* (lift 5.42 → 5.83) and *{development, management, organization} → leadership* (2.17 → 1.58). With at most 25 postings per company, only *marketing → sales* passes.

Most failures are a **base-rate ceiling**, not a broken pattern. If a consequent appears in a share *p* of postings, lift can be at most 1/*p*. In tech postings, development (61%), design (62%), and management (55%) are so common that lift is capped at about 1.6-1.8:

| Training rule | Lift (train → Muse) | Max possible Muse lift | Confidence (train → Muse) |
|---|---|---|---|
| engineering → design | 2.64 → 1.14 | 1.60 | 0.55 → **0.71** |
| {development, engineering} → design | 3.02 → 1.28 | 1.60 | 0.63 → **0.80** |
| {education, leadership} → development | 1.69 → 1.37 | 1.63 | 0.62 → **0.84** |
| {compliance, leadership} → management | 1.69 → 1.49 | 1.82 | 0.76 → **0.81** |
| accounting → finance | 5.54 → 6.53 | 8.47 | 0.56 → 0.77 (support 3.4% < 5%) |

14 of the 16 rules that fail the lift test hit this ceiling, and **all 14 have higher confidence on Muse than in training**. *accounting ↔ finance* and *marketing → sales* keep high lift (5.8-6.5), but fall below 5% support in a tech sample.

**Skill demand vs training tech jobs** (the Stage 6 *Data, Tech & Engineering* cluster):

| Skill | Muse 2026 | Training tech 2023-24 | Change |
|---|---|---|---|
| python | 31.8% | 17.6% | **+14.2 pp** |
| sql | 21.3% | 15.0% | +6.3 pp |
| c++ / ci/cd / kubernetes | 11.6% / 10.9% / 8.2% | 5.8% / 7.2% / 4.9% | +3 to +6 pp |
| agile | 10.5% | 17.2% | **−6.7 pp** |
| aws | 7.4% | 11.7% | −4.3 pp |
| design / development | 62.2% / 61.0% | 79.7% / 74.1% | −17.5 / −13.1 pp |

Manufacturing (+14.8 pp) and healthcare (−9.2 pp) also shift, but they mirror the employer mix (SpaceX, Jabil) rather than the market. All shifts mix real change with source differences, so they are indicative.

## 5. Discussion

**What this means for job seekers.**
- The pay premium comes from technical *combinations*, not from single skills or from general skills that are merely common.
  - Postings that combine computer science, engineering, and design (or Python with engineering) are about 80% High-tier.
  - Office-software pairs (Excel + PowerPoint) and training/education pairs co-occur strongly but are *below* the High-tier baseline.
- Adding cloud and infrastructure skills (AWS, GCP, Kubernetes, Docker, CI/CD) or machine learning to a software profile matches the highest median salaries in the data ($141k–$163k).
- Experience level still matters a great deal. 86% of Director postings are High tier, compared with 11.5% of Entry-level postings, so skills and seniority compound.

**Why the F1 target was not met.**
- **The Mid tier is hard by construction.** The tiers cut a continuous salary distribution at two percentiles. A $108k job and a $111k job land in different tiers even though their postings are almost identical. The model separates Low from High well (ROC-AUC 0.89; only 2% Low↔High confusion), but Mid-tier F1 is 0.62.
- **Important pay factors are not in the features.** Posting text does not capture exact location cost of living (only the state is used), company pay policy, negotiated offers, or whether an hourly job is full-time. The hourly-to-annual conversion also assumes 2,080 hours a year.
- **Diminishing returns.** Tuning, 45 extra technical skills, and company-name terms moved F1 by about 0.01. A 2,000-phrase title vocabulary reached about 0.74 in exploration, which suggests the limit is the information in the postings, not the model.

**Why the silhouette target was not met.**
- **Skill profiles form a continuum, not separate groups.** Most skills are shared across roles (communication, management, training), and the vectors are sparse and binary: 245 dimensions with about 11 skills per job. In that setting, distances between points are all similar, so no partition can reach silhouette 0.5.
- **The result holds across methods.** Neither k nor DBSCAN's eps changes this, and a TF-IDF + SVD representation only reaches 0.14.
- **So the answer to Q2 is "weakly".** The data supports broad, overlapping *profiles* rather than crisp archetypes.

**What outlier detection found.**
- The unusual postings in this data are **unusually long descriptions, not unusually paid jobs**. A posting with 30-50 matched skills is far from almost every other job in the 245-dimensional skill space, so Isolation Forest isolates it in a few splits. Its salary, a single column, rarely decides the isolation.
- **Salary mismatches are rare and diffuse.** About 4.5% of jobs are paid 2 SD away from their skill-and-seniority group, and the outliers contain barely more. Salary given the posting's skills is noisy everywhere, which matches the Mid-tier ceiling in Q3, rather than concentrated in a few anomalous ads.
- **Implications.**
  - Skill counts partly reflect how verbose a description is, not only what the job needs. Normalizing for description length, or capping skills per job, would make clustering and classification less sensitive to boilerplate-heavy ads.
  - Finding salary anomalies directly needs a salary model: flag large residuals from a regression of salary on skills and seniority, rather than running a general-purpose detector on the joint feature space.

**What cross-source validation found.**
- **Snippets can't validate skill patterns.** Adzuna's 71-word snippets gave 2 skills per posting, so neither rules nor the classifier could be tested. Full descriptions (The Muse, 813 words) gave 16 skills per posting, more than the training data.
- **Skill groupings persist, but lift doesn't transfer across samples.** The training rules were mined on all industries, where *design* or *development* appear in about 20-37% of jobs. In a tech-only sample those skills are in about 60% of postings, so the same co-occurrence yields a much lower lift. Confidence, the share of postings with A that also list B, rose for 14 of the 16 failing rules. Lift is a property of the pattern *and* the population, so a validation sample needs a similar industry mix, or base-rate-robust measures such as confidence or leverage.
- **Employer concentration is a hazard for live data.** One employer supplied a third of The Muse sample, and one employer's template produced the top Muse rules. The training data's breadth (thousands of companies) protects against this. A live validation set should be capped per company or weighted by company.
- **Demand moved toward Python and infrastructure.** In tech postings Python nearly doubled, with SQL, C++, CI/CD, and Kubernetes up and Agile and AWS down. This fits the High-tier technical combinations in Q1 (python + engineering, aws + python) staying relevant in 2026. The direction is suggestive given the sample's concentration.

**What the larger technical vocabulary changed.**
- Expanding from 100 to 245 skills raised skills per job from 8.2 to 11.5.
- It improved every classifier by 0.01–0.02 F1 and almost doubled K-Means silhouette (0.031 → 0.056).
- Most importantly, it turned an unclear "Generalist" cluster into a clear **Data, Tech & Engineering** cluster: 64% High tier, mean $132k.
- The effect is limited because only **14.8% of postings mention any technical skill**. The dataset is dominated by healthcare, retail, administration, and trades. A larger *domain-appropriate* vocabulary would help those jobs more than further technical terms: nursing credentials, trade certifications, clinical specialties, and licenses. Beyond that, the next step is embedding-based skill extraction (e.g. spaCy entity recognition) instead of a fixed vocabulary.

## 6. Conclusion

**Q1 — Which skills co-occur and predict pay?**
- Skills cluster into clear pairs: accounting–finance (lift 5.54), Excel–PowerPoint (5.42), engineering–design (2.64), and marketing–sales (2.52).
- High pay is predicted by technical combinations such as {computer science, design, engineering} and {python, engineering}. These are 80–83% High tier, about 2.5× the baseline.
- Common office and training skill sets fall below the baseline.
- At 2% support, the technical rules add {python, development} and {sql, design}. Rarer pairs of two technical skills (aws + python, machine learning + python) are the best paid, at 85-88% High tier.

**Q2 — Do role archetypes exist beyond job titles?**
- Only weakly. Silhouette scores (≤ 0.21) are far below 0.50, and no clustering method finds well-separated groups.
- Even so, a skill-defined technical profile appears across many titles: software, electrical, and manufacturing engineers, and some project managers. It earns $50k more than the service and retail profile.
- t-SNE confirms the continuum but finds small well-paid pockets that the partition hides. The clearest is acute-care nursing, 79% High tier but filed under the service cluster.
- The 1% most anomalous postings are skill-dense descriptions (30.8 vs 11.5 skills), not salary mismatches.

**Q3 — Can salary tier be predicted from posting features alone?**
- Partly. The best model (tuned XGBoost with title and company TF-IDF) reaches ROC-AUC 0.891, well above the 0.80 target, and rarely confuses Low with High.
- Its macro F1 is 0.728, short of the 0.75 target, because the Mid-tier boundary is not identifiable from posting text.
- Ordinal models (Ordinal Ridge 0.607, Logistic IT 0.642) do not help. Both trail an unordered linear control (0.667).

**Live validation (Oct 2026).**
- On full-length Muse tech postings, 2 of 20 training rules are confirmed as rules (*powerpoint → excel*, *{development, management, organization} → leadership*), or 1 of 20 when capped per company.
- Most of the rest still co-occur with higher confidence but hit a lift ceiling, because their consequents are near-universal in tech postings. The skill groupings hold; lift-based rules do not transfer from an all-industry to a tech-only population.
- Python demand in tech postings rose from 18% to 32%.
- Adzuna snippets were too short to test rules, and the salary model was not validated on live data (Adzuna's salaries are mostly its own estimates).

**Future work.**
1. Predict salary with regression, then bin the prediction into tiers. Linear ordinal classifiers were tried and did not help (Section 4, Q3). A non-linear ordinal model, such as gradient boosting with cumulative-logit loss, remains untested. Regression residuals would also give a direct way to detect salary anomalies.
2. Use richer text features from job descriptions, such as sentence embeddings, after removing any stated salary figures.
3. Adjust salaries for metro-level cost of living, and add company-level pay history.
4. Extract skills with domain-specific vocabularies or NER models for healthcare and the trades.
5. Use soft or overlapping clustering (topic models, Gaussian mixtures) to describe the skill continuum better than hard partitions do.
6. Re-run live validation on a source with an industry mix like the training data's, and full descriptions and advertised salaries. Compare base-rate-robust measures (confidence, leverage, conviction) alongside lift, and cap or weight postings per company.
7. Collect live postings over several months to track skill demand over time, instead of a single snapshot.

## References

[1] R. Agrawal and R. Srikant, "Fast algorithms for mining association rules," in *Proc. 20th Int. Conf. Very Large Data Bases (VLDB)*, Santiago, Chile, 1994, pp. 487–499.

[2] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794.

[4] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. 5th Berkeley Symp. Mathematical Statistics and Probability*, vol. 1, 1967, pp. 281–297.

[5] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," in *Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining (KDD)*, Portland, OR, USA, 1996, pp. 226–231.

[6] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *J. Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987.

[7] D. L. Davies and D. W. Bouldin, "A cluster separation measure," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. PAMI-1, no. 2, pp. 224–227, 1979.

[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 413–422.

[9] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.

[10] J. D. M. Rennie and N. Srebro, "Loss functions for preference levels: Regression with discrete ordered labels," in *Proc. IJCAI Multidisciplinary Workshop on Advances in Preference Handling*, Edinburgh, U.K., 2005, pp. 180–186.

**Datasets:**
- arshkon, "LinkedIn Job Postings (2023–2024)," Kaggle. https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- asaniczka, "1.3M LinkedIn Jobs & Skills (2024)," Kaggle. https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024
- arnabchaki, "Data Science Salaries 2025," Kaggle. https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2025
