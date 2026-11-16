# Frozen classification inputs after the bounded Phase 6b review

This pass stops before Phase 2 classification. Existing uncommitted integration work is preserved. All experiment outputs are external to the repository under `/private/tmp/skillmap-phase6b/`; canonical data, warehouse, models and historical reports are unchanged.

## Extraction and labels

Keep the existing Phase 6 extraction, its nested-phrase handling, and the unique 245-skill vocabulary. No proposed Phase 6b exclusions were adopted. Benefits are a context category, not a new skill. The 48 real-posting target-skill examples contain 42 documented assistant-reviewed judgments and six ambiguous/null labels. They are exploratory annotations, not independent human gold; both nominal partitions were inspected. Sample precision/recall does not estimate corpus performance. Proposed exclusions lose a legitimate development responsibility as well as suppressing false positives. Obtain independent review and new blinded examples, including term-absent positive cases, before versioning any extraction change. Never regenerate canonical matched skills as an incidental classification step.

## Salary task

Preserve vendor normalized annual salaries when present. Fill only missing values with the mean of available minimum/maximum (otherwise median) times hourly 2080, weekly 52, biweekly 26, monthly 12 or yearly 1. Unknown periods do not imply annual pay. No currency conversion or inferred hours adjustment.

Freeze Low <= $62,400; Mid > $62,400 and <= $109,200; High > $109,200. These cutoffs originally came from exact 1/3 and 2/3 quantiles of the globally processed dataset, before the classifier split. They were not learned on an untouched training set. Decimal .33/.66 quantiles are different and must not replace the task boundaries. New deployment labels use these fixed boundaries rather than recomputed percentiles. Do not use salary or salary-derived variables as prediction features.

## Evaluation and overlap

The original seed-42 stratified 80/20 split remains exploratory. Its test rows include 135 exact-description overlaps with training, 154 normalized-description overlaps, 324 masked-template candidates and 805 company/title candidates. Candidate equality is a grouping diagnostic, not proof that jobs are duplicates. Existing repost removal included company, title, description and salary; identical descriptions can survive it.

Before Phase 2, predeclare groups from exact/normalized descriptions (combine connected groups if using multiple keys); do not combine all missing descriptions into one group. Review numeric/URL-masked candidates before treating them as duplicate groups. Assess employer holdout separately from description grouping. Freeze job IDs, group IDs, split assignments and class coverage before fitting. Fit categories, TF-IDF, scaling and any SVD inside each grouped training CV fold. Select models only from training CV; preserve the existing family/search scope and same-feature ordinal control. Do not tune grouping or extraction against held-out classification outcomes. Earlier use of these data cannot be undone by renaming a split.

## Mining and clustering

General support >= .05 uses all jobs, including empty baskets. High-tier mining support >= .05 uses all High jobs; High share uses High matching jobs divided by all matching jobs. Technical support >= .02 uses all jobs. Confidence > .5 and lift > 1.5 are strict, applied before rounding. Technical-only means every item on both sides is in the frozen forced-technical set. Salary associations are descriptive, not causal.

The bounded 6,000-row clustering experiment is exploratory and uses saved Phase 6 features. SVD32/L2 is not adopted as a classification transformation. Its preprocessing was fitted on the exploratory sample; never reuse that fitted transform in supervised evaluation. Seed stability does not establish resampling, employer or external generalization. Normalized-feature KMeans is not spherical KMeans. No classifier training or canonical rebuild was performed.

## Evidence locations

- Extraction: `/private/tmp/skillmap-phase6b/extraction/{sample.json,sampling.json,comparison/report.json,comparison/predictions.csv}`
- Salary and groups: `/private/tmp/skillmap-phase6b/salary/run2/{data_quality.json,group_candidates.csv}`
- Rule views: `/private/tmp/skillmap-phase6b/association/views/{top10_lift.csv,top5_high_tier.csv,technical_only.csv,report.json}`
- Clustering: `/private/tmp/skillmap-phase6b/clustering/run/{report.json,summary.csv,labels.csv}`
