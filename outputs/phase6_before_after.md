# Phase 6 — Skill Extraction Fix: Before vs After

**Date:** 2026-11-01 · **Branch:** `phase6-fix-extraction` · **Code change:** `src/preprocessing.py` (`SKILL_PATTERNS` for `driving` and `organization`, new `SKILL_EXCLUSIONS`)

Stages 2-8 were rerun in order on the same raw data. Stages 7 and 8 reprocessed the cached Adzuna and Muse pages only: both logs report 0 new API calls, the two call ledgers are unchanged since 2026-10-14, and every run had `HTTP(S)_PROXY` pointed at a dead local port so that any request would fail.

## 1. Audit: false skill matches

Method: for each of the 40 most frequent matched skills, 5 random postings that match it (`random_state=42`, 200 snippets in total), each read with ±70 characters of context. Suspicious skills were then checked on all 34,179 postings by counting the words just before and after each match.

| Skill (jobs before) | False in sample | Example false match | Cause | Action |
|---|---|---|---|---|
| **driving** (4,288) | 1/5 | "interested in **driving** business and product strategy" | "driving sales / revenue / results / innovation / force" | **Fixed:** context pattern |
| **organization** (11,448) | 4/5 | "integrity within the **organization**", "our **organization**'s success" | "the / our / an organization" (~6,800 uses) is the employer, not the soft skill | **Fixed:** context pattern |
| **supervision** (3,074) | 4/5 | "under general **supervision**", "with minimal **supervision**" | Says the job *is* supervised, the opposite of the skill | **Fixed:** exclusions |
| **security** (4,789) | 1/5 | "social **security** number" | Another meaning | **Fixed:** exclusion |
| **reporting** (5,768) | 1/5 | "**reporting** to the team leader" | Org chart, not a reporting duty | **Fixed:** exclusion |
| **excel** (5,802) | 0/5 (seen in another skill's snippet) | "where they can **excel**" | The verb (682 jobs on the full data) | **Fixed:** exclusions |
| healthcare (6,731) | 2/5 | "comprehensive **healthcare** (medical, dental, vision)" | Benefit | Not fixed (see §4) |
| flexibility (3,403) | 3/5 | "remote work **flexibility**" | Perk | Not fixed |
| education (12,553) | 3/5 | "**education**: bachelors" | Requirement label | Not fixed |
| development (12,570) | 1/5 | "career **development** programs" | Benefit | Not fixed |
| planning (6,569) | 1/5 | "retirement **planning** (401(k))" | Benefit | Not fixed |
| training (12,693) | 1/5 | "skills **training** and career opportunities" | Benefit | Not fixed |
| travel (6,830) | 1/5 | "space available **travel**" | Benefit | Not fixed |
| management (15,436) | 1/5 | "American Health Information **Management** Association" | Organization name | Not fixed |

The other 26 of the top 40 skills had no suspicious snippets in their sample.

**Rule used to decide what to fix:** fix a match only when the word has a *different meaning* (an unrelated sense, an inverted meaning, or a different part of speech). Mentions of the *same concept* as a benefit (healthcare plans, career development, paid training) were left alone to keep vocabulary changes minimal (see §4).

## 2. What changed in the vocabulary

No skills were added or removed (still 245). Six skills now match more narrowly:

| Skill | Rule now | Jobs before | Jobs after | Dropped | Added |
|---|---|---|---|---|---|
| driving | Only as a driving requirement: "driving record / history / experience / license", "safe / clean / commercial / truck driving", or within 40 characters of driver's / license / vehicle / truck / CDL / MVR | 4,288 | 1,262 | 3,026 | 0 |
| organization | Only as the soft skill: "organizational (and X) skills / abilities", "strong / excellent / … organization(al)", or listed with communication / interpersonal / analytical / prioritization / multitasking / planning | 11,448 | 4,548 | 7,962 | 1,062 |
| supervision | Blanks "under (the) (X) supervision" and "minimal / limited / little / no / close / general / … supervision" | 3,074 | 1,375 | 1,699 | 0 |
| excel | Blanks "to / you / will / can / who / they / we excel" and "excel at / in / as" | 5,802 | 5,120 | 682 | 0 |
| reporting | Blanks "reporting (directly) to" | 5,768 | 5,020 | 748 | 0 |
| security | Blanks "social security" | 4,789 | 4,619 | 170 | 0 |

`organization` gains 1,062 jobs because "strong organizational and communication skills" previously did not match: the old pattern only covered "organizational skills" (adjacent words) and bare "organization". No other skill changed for any job.

A re-audit after the fix (8 random snippets for each fixed skill): driving 8/8, organization 8/8 and excel 8/8 are real uses. Some false matches remain in `security` ("privacy and security of personal information", "benefits … provide the security"), plus 1 in `supervision` ("supervision is received from management") and 1 in `reporting` ("a culture of reporting abuse").

## 3. Before vs after

| Metric | Target | Before | After | Change |
|---|---|---|---|---|
| Mean matched skills per job | — | 11.5 | 11.1 | −0.4 |
| Jobs with ≥ 1 matched skill | — | 33,924 (99.3%) | 33,911 (99.2%) | −13 |
| Warehouse fact rows (job × skill) | — | 449,665 | 436,441 | −13,224 |
| **Association rules** (support 0.05, conf > 0.5, lift > 1.5) | lift > 1.5 | **91** | **62** | −29 |
| Frequent itemsets | — | 343 | 284 | −59 |
| Co-occurrence graph | — | 40 skills / 191 edges | 39 / 162 | driving no longer in any pair at 5% support |
| High-tier rules (High-tier lift > 1.2) | — | 277 (249) | 162 (142) | −115 |
| Low-support technical rules (0.02) | — | 10 | 10 | unchanged |
| **Macro F1, best model** (XGBoost + title & company TF-IDF) | > 0.75 | **0.728** ❌ | **0.727** ❌ | −0.001 |
| **ROC-AUC, best model** | > 0.80 | **0.891** ✅ | **0.892** ✅ | +0.001 |
| Random Forest F1 / AUC | | 0.703 / 0.873 | 0.705 / 0.872 | ≈ |
| XGBoost F1 / AUC | | 0.683 / 0.858 | 0.679 / 0.858 | ≈ |
| Logistic Regression F1 / AUC | | 0.632 / 0.815 | 0.628 / 0.815 | ≈ |
| Low ↔ High confusions (best model) | — | 2.0% | 1.9% | ≈ |
| **Silhouette, K-Means k = 3** | > 0.50 | **0.056** ❌ (0.0562) | **0.056** ❌ (0.0557) | ≈ |
| Davies-Bouldin, K-Means | < 1.0 | 4.72 ❌ | 4.75 ❌ | ≈ |
| **Silhouette, DBSCAN** (eps 2.83) | > 0.50 | **0.214** ❌ | **0.192** ❌ | **−0.022** |
| Davies-Bouldin, DBSCAN | < 1.0 | 0.93 ✅ | 0.92 ✅ | ≈ |
| DBSCAN noise share | — | 31.8% | 29.1% | −2.7 pp |
| **Isolation Forest outliers** | — | **342** | **342** | fixed at 1% contamination |
| Flagged jobs: mean skills / salary mismatch | — | 30.8 / 5.6% (all jobs 4.5%) | 30.1 / 5.8% (all jobs 4.6%) | ≈ |
| Outlier sensitivity run: overlap / mismatch | — | 38% / 9.9% | 42% / 7.9% | |
| t-SNE vs PCA kNN overlap | — | 18.1% vs 3.0% | 18.4% vs 3.4% | ≈ |
| **Adzuna live accuracy** vs always-High baseline | — | **59.4%** vs 62.1% | **58.0%** vs 62.1% | −1.4 pp, still below baseline |
| Adzuna live macro F1 / advertised-salary accuracy (n = 57) | — | 0.478 / 50.9% | 0.464 / 47.4% | |
| **Muse: training top-20 rules confirmed** | — | **2 of 20** | **1 of 20** | −1 |
| Muse, capped at 25 postings per company | — | 1 of 20 | 1 of 20 | unchanged (marketing → sales) |
| Muse skills per posting / Muse rules | — | 16.2 / 20,599 | 15.6 / 8,246 | |
| Muse failures at the lift ceiling | — | 14 of 16 | 14 of 17 | all 14 still gain confidence |

### Rule-level changes
- **29 of the 30 lost rules contain `organization`.** The 30th is {development, reporting} → management. One rule is new: organization → communication (lift 1.54).
- **Top 20 by lift.** Out: {development, management, organization} → leadership; {design, organization} → development; {leadership, management, organization} → development. In: {communication, leadership, management} → development; {leadership, planning} → management; {education, safety} → training.
- **powerpoint → excel** is now the strongest rule: lift 5.42 → **6.11**, because excel is in fewer jobs. On the High-tier side it is still below baseline (29.0% High, High-tier lift 0.87).
- The best-paid combinations didn't change: {computer science, design, engineering} 82.8% High, {python, engineering} 81.5%, aws + python 88.1%.

## 4. Surprises and caveats

1. **The fixed false match was the top skill of a t-SNE hotspot.** Before the fix, the "product and account roles" High-pay pocket (79 jobs, 75% High) listed **driving** as its top skill ("driving revenue / sales / business"), ahead of marketing and sales. After the fix that pocket is gone. Its closest replacement is a {collaboration, marketing, leadership, design, innovation} pocket (90 jobs, 73% High).
2. **The acute-care nursing hotspot is no longer detected.** The ACLS/BLS postings themselves didn't change (227 jobs with both, 54% High, before and after). The t-SNE layout shifted, and the 38-job pocket no longer clears the hotspot threshold (≥ 70% of neighbours High, DBSCAN min_samples 20). The final report presented this as a key Q2 finding, but it turns out not to be robust to small input changes. The report now says so.
3. **The middle cluster was renamed.** "Management & Leadership" became **"Finance, Compliance & Administration"**. Without `organization` noise, `SKILL_THEMES` picks finance / leadership / reporting / organization / microsoft office as the defining skills. The cluster's size (10,604 vs 10,497) and pay (mean $99.9k vs $100.7k) barely moved.
4. **DBSCAN silhouette fell from 0.214 to 0.192.** The fake `organization` signal made part of the data slightly more cohesive. Neither number is close to the 0.50 target, and no metric target changed.
5. **Muse confirmation fell from 2/20 to 1/20.** One of the two confirmed rules, {development, management, organization} → leadership, was partly an artefact: "organization" mostly meant "our organization". Muse descriptions carry plenty of that boilerplate. It no longer reaches the training top 20.
6. **Classification barely moved.** The six skills are low-signal for salary tier, and the title TF-IDF carries most of the signal.
7. **Known residual false positives (not fixed, by design).** Benefit mentions still count for healthcare, flexibility, development ("professional development"), training ("paid training"), education ("continuing education"), planning ("retirement planning") and travel. `education` also matches requirement headers ("Education: Bachelor's"). These are the same concept used in a benefits or requirements context, unlike the six fixed skills. Excluding them would need much larger exclusion lists, and it would weaken the most common skills in the rules (development, education, training). Remaining `security` false positives: "privacy and security", "financial security".
8. **Notebooks were not re-executed.** `src/` modules were rerun; notebook cell outputs still show the old numbers. `CLAUDE.md` also still lists the old Stage 2-8 numbers.
