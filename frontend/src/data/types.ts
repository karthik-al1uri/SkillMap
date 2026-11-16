export type Tier = 'Low' | 'Mid' | 'High';
export type JobRow = [salary: number, roleIndex: number, locationIndex: number, skillIndices: number[], tierIndex: number];
export interface RoleSkill { skill: string; index: number; count: number; share: number; overall_share: number | null }
export interface Role { id: number; name: string; count: number; median_salary: number; high_share: number; top_skills: RoleSkill[] }
export interface Skill { index: number; name: string; count: number; share: number | null; is_technical: boolean | null }
export interface NetworkNode { id: number; label: string; count: number; role: number | null; x: number; y: number }
export interface NetworkEdge { source: number; target: number; count: number; jaccard: number }
export interface Network { nodes: NetworkNode[]; edges: NetworkEdge[] }
export interface SkillNeighbor { skillIndex: number; count: number; share_among_selected: number }
export interface ModelTargets { f1_macro: number; roc_auc: number; silhouette: number }
export interface SalaryBoundaries { low_max: number; mid_max: number }
export interface ProductMetrics {
  n_jobs?: number | null;
  n_skills_vocab?: number | null;
  zero_skills?: number | null;
  mean_skills?: number | null;
  median_salary?: number | null;
  tier_counts?: Partial<Record<Tier, number>> | null;
  skill_counts?: Record<string, number> | null;
  salary_summary?: { min: number; max: number; mean: number; median: number } | null;
  model_targets?: ModelTargets | null;
  salary_tier_boundaries?: SalaryBoundaries | null;
  dates?: string[] | null;
  live_snapshot_dates?: string[] | null;
}
export interface CvRow { candidate: string; mean_f1_macro: number; fold_f1_macro?: number[] }
export interface GroupedInterval { 'lower_2.5': number | null; 'upper_97.5': number | null; valid_repetitions: number }
export interface GroupedReport {
  status: string;
  winner: string;
  cv_rows: CvRow[];
  outer_metrics: { f1_macro: number; accuracy: number; roc_auc_ovr_macro: number | null; f1_per_class: Record<Tier, number>; confusion_matrix: number[][] };
  bootstrap: { repetitions: number; seed: number; unit: string; intervals: Record<string, GroupedInterval> };
  limitations: string[];
  elapsed_seconds?: number;
  [key: string]: unknown;
}
export interface FrozenDesign {
  candidate_order: string[];
  evaluation_status?: string;
  outer_train_rows?: number;
  outer_test_rows?: number;
  outer_train_groups?: number;
  outer_test_groups?: number;
  outer_group_overlap?: number;
  group_unit?: string;
  groups?: number;
  rows?: number;
  limitations?: string[];
  source_hashes?: Record<string, string>;
  [key: string]: unknown;
}
export interface GroupedModel { report: GroupedReport; frozen_design: FrozenDesign; cv_selection: { winner: string; rows: CvRow[]; selection?: string }; validated_training_cv_winner: string }
export interface HistoricalModel { model: string; f1_macro: number; roc_auc_ovr: number; accuracy?: number; f1_low?: number; f1_mid?: number; f1_high?: number; [key: string]: unknown }
export interface BootstrapInterval { low: number | null; high: number | null; defined_replicates: number }
export interface PredictionMetrics {
  accuracy: number | null;
  macro_f1: number;
  per_class_f1: Record<Tier, number>;
  per_class_recall: Record<Tier, number>;
  actual_counts: Record<Tier, number>;
  predicted_counts: Record<Tier, number>;
  actual_shares: Partial<Record<Tier, number>>;
  predicted_shares: Partial<Record<Tier, number>>;
  confusion_rows_actual_columns_predicted: number[][];
  labels: Tier[];
}
export interface LiveAdzuna extends PredictionMetrics {
  n: number;
  roc_auc: { value: number | null; reason: string };
  bootstrap: { accuracy: BootstrapInterval; macro_f1: BootstrapInterval; employer_groups: number; replicates: number; seed: number; method: string; limitations: string; zero_division: string };
  comparators: { observed_majority_oracle: PredictionMetrics & { label: Tier; status: string }; historical_training_majority: PredictionMetrics & { label: Tier; provenance: string; training_counts: Record<Tier, number>; training_n: number }; all_cleaned_population_counts_not_training_only: Record<Tier, number> };
  salary_label_status: { estimated_rows: number; advertised_rows: number; limitation: string };
  salary_label_subgroups: Record<string, Partial<PredictionMetrics> & { n: number; metrics?: null }>;
  employer_join: { matched_rows: number; unmatched_rows: number; company_name_disagreements: number; missing_employer_rows_unique_fallback: number; prediction_duplicate_ids: number };
}
export interface MuseSummary {
  all_postings: number; eligible_postings: number; excluded_under_two_skills: number;
  empty_baskets_all_postings: number; empty_baskets_eligible: number;
  employers_all: number; employers_eligible: number; fixed_rules: number;
  primary_confirmed: number; sensitivity_confirmed: number;
  confirmed_primary_rules: string[]; confirmed_sensitivity_rules: string[];
  bootstrap_replicates: number; bootstrap_seed: number; bootstrap_method: string;
  undefined_replicates: Record<string, number>; limitations: string;
}
export interface LiveReport {
  source_status: { fetch_date: string; mode: string; [key: string]: unknown };
  predeclared_contract: { primary_support_inclusive: number; sensitivity_support_inclusive: number; confidence_strictly_greater_than: number; lift_strictly_greater_than: number; [key: string]: unknown };
  adzuna: LiveAdzuna;
  muse_uncapped: MuseSummary;
  muse_company_capped: MuseSummary;
  limitations: string[];
  inputs_unchanged: boolean;
  historical_saved_comparison: Record<string, unknown>;
  input_sha256: Record<string, string>;
}
export interface SkillComparison { skill: string; muse_share: number; train_share: number; train_tech_share: number; muse_share_capped: number; is_technical: boolean; change_vs_tech_pp: number; ratio_vs_tech: number | null; fetch_date: string }
export interface Rule { antecedents: string; consequents: string; support: number; confidence: number; lift: number; high_tier_rate?: number; high_tier_lift?: number; jobs_with_itemset?: number; [key: string]: unknown }
export interface FixedRule { rule: string; support: number | null; confidence: number | null; lift: number | null; passes_primary_005: boolean; passes_sensitivity_002: boolean; denominator: number; [key: string]: unknown }
export interface RobustnessSummary { representation: string; k: number; mean_cosine_silhouette_original: number; mean_cosine_silhouette_representation: number; mean_euclidean_silhouette_representation: number; mean_pairwise_seed_ari: number; min_pairwise_seed_ari: number; min_cluster_share: number; max_cluster_share: number }
export interface RobustnessReport { summary: RobustnessSummary[]; provenance: { total_saved_rows: number; nonzero_rows: number; zero_skill_rows_excluded: number; sample_rows: number; sample_seed: number; sampling: string; [key: string]: unknown }; configuration: { k: number[]; seeds: number[]; n_init: number; max_iter: number; threads: number }; limitations: string[]; [key: string]: unknown }
export interface Source { path: string; bytes: number; sha256: string; status: string }
export interface Availability { available: boolean; status: string; source_ids: string[]; reason?: string }
export interface ProductData {
  schema: 'skillmap.dashboard.product.v1';
  metrics: ProductMetrics;
  roles: Role[];
  locations: (string | null)[];
  skills: Skill[];
  rows: JobRow[];
  tiers: Tier[];
  networks: { all: Network; technical: Network };
  skill_neighbors: Record<string, SkillNeighbor[]>;
  models: { grouped: GroupedModel | null; historical: HistoricalModel[] | null };
  live: { evaluation: LiveReport | null; comparison: SkillComparison[] | null; rules_validation: Record<string, unknown>[] | null; fixed_rules_uncapped: FixedRule[] | null; fixed_rules_capped: FixedRule[] | null };
  robustness: RobustnessReport | null;
  rules: { general: Rule[] | null; high: Rule[] | null; technical: Rule[] | null };
  availability: { tables: Record<string, Availability>; metrics: Record<string, Availability>; evaluations: Record<string, Availability> };
  provenance: { salary_grain: string; row_columns: string[]; role_index: string; network_definition: string; neighbor_share_definition: string; role_names: string; sources: Record<string, Source>; sourcefacts_sha256?: string; facts_source?: { path: string; bytes: number; sha256: string; schema: string }; [key: string]: unknown };
}
export interface CohortFilters { role: number | null; location: number | null; skill: number | null }
export interface SalaryGroup { index: number; count: number; median_salary: number; high_share: number }
export interface SalaryHistogram { counts: number[]; edges: number[]; total: number; underflow: number; overflow: number }
