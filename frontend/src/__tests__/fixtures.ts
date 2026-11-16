/** Explicitly synthetic UI fixtures. None of these values are production results. */
import type { LiveAdzuna, MuseSummary, PredictionMetrics, ProductData } from '../data/types';

const predictionMetrics: PredictionMetrics = {
  accuracy: 1, macro_f1: 1,
  per_class_f1: { Low: 1, Mid: 1, High: 1 }, per_class_recall: { Low: 1, Mid: 1, High: 1 },
  actual_counts: { Low: 2, Mid: 2, High: 2 }, predicted_counts: { Low: 2, Mid: 2, High: 2 },
  actual_shares: { Low: 1 / 3, Mid: 1 / 3, High: 1 / 3 }, predicted_shares: { Low: 1 / 3, Mid: 1 / 3, High: 1 / 3 },
  confusion_rows_actual_columns_predicted: [[2, 0, 0], [0, 2, 0], [0, 0, 2]], labels: ['Low', 'Mid', 'High'],
};
const lowMajority: PredictionMetrics = { ...predictionMetrics, accuracy: 1 / 3, macro_f1: 1 / 6,
  per_class_f1: { Low: .5, Mid: 0, High: 0 }, per_class_recall: { Low: 1, Mid: 0, High: 0 },
  predicted_counts: { Low: 6, Mid: 0, High: 0 }, predicted_shares: { Low: 1, Mid: 0, High: 0 },
  confusion_rows_actual_columns_predicted: [[2, 0, 0], [2, 0, 0], [2, 0, 0]] };
const midMajority: PredictionMetrics = { ...predictionMetrics, accuracy: 1 / 3, macro_f1: 1 / 6,
  per_class_f1: { Low: 0, Mid: .5, High: 0 }, per_class_recall: { Low: 0, Mid: 1, High: 0 },
  predicted_counts: { Low: 0, Mid: 6, High: 0 }, predicted_shares: { Low: 0, Mid: 1, High: 0 },
  confusion_rows_actual_columns_predicted: [[0, 2, 0], [0, 2, 0], [0, 2, 0]] };
const adzuna: LiveAdzuna = {
  ...predictionMetrics, n: 6,
  roc_auc: { value: null, reason: 'Synthetic saved predictions lack probabilities.' },
  bootstrap: { accuracy: { low: 1, high: 1, defined_replicates: 10 }, macro_f1: { low: 1, high: 1, defined_replicates: 10 },
    employer_groups: 3, replicates: 10, seed: 7, method: 'Synthetic fixed-prediction bootstrap',
    limitations: 'Synthetic fixture only.', zero_division: 'zero' },
  comparators: {
    observed_majority_oracle: { ...lowMajority, label: 'Low', status: 'Synthetic oracle comparator only.' },
    historical_training_majority: { ...midMajority, label: 'Mid', provenance: 'Synthetic training counts.',
      training_counts: { Low: 2, Mid: 3, High: 1 }, training_n: 6 },
    all_cleaned_population_counts_not_training_only: { Low: 2, Mid: 2, High: 2 },
  },
  salary_label_status: { estimated_rows: 4, advertised_rows: 2, limitation: 'Synthetic imperfect salary labels.' },
  salary_label_subgroups: { advertised: { n: 2, metrics: null }, estimated: { n: 4, metrics: null } },
  employer_join: { matched_rows: 6, unmatched_rows: 0, company_name_disagreements: 0,
    missing_employer_rows_unique_fallback: 0, prediction_duplicate_ids: 0 },
};
function muse(capped = false): MuseSummary {
  return { all_postings: capped ? 4 : 6, eligible_postings: capped ? 3 : 5, excluded_under_two_skills: 1,
    empty_baskets_all_postings: 0, empty_baskets_eligible: 0, employers_all: 2, employers_eligible: 2,
    fixed_rules: 2, primary_confirmed: capped ? 0 : 1, sensitivity_confirmed: 1,
    confirmed_primary_rules: capped ? [] : ['python -> sql'], confirmed_sensitivity_rules: ['python -> sql'],
    bootstrap_replicates: 10, bootstrap_seed: 7, bootstrap_method: 'Synthetic employer bootstrap',
    undefined_replicates: {}, limitations: 'Synthetic fixture only; cross-source comparisons are not temporal trends.' };
}
const cvRows = [{ candidate: 'fixed_xgboost', mean_f1_macro: 0.64 }, { candidate: 'nominal_logistic', mean_f1_macro: 0.60 }];

export function productFixture(): ProductData {
  return structuredClone({
    schema: 'skillmap.dashboard.product.v1',
    metrics: { n_jobs: 6, n_skills_vocab: 4, zero_skills: 0, mean_skills: 11 / 6, median_salary: 90_000,
      tier_counts: { Low: 2, Mid: 2, High: 2 }, skill_counts: { python: 4, sql: 3, healthcare: 3, security: 1 },
      salary_summary: { min: 40_000, max: 140_000, mean: 90_000, median: 90_000 },
      model_targets: { f1_macro: .75, roc_auc: .8, silhouette: .5 }, salary_tier_boundaries: { low_max: 62_400, mid_max: 109_200 },
      dates: ['2026-01-01'], live_snapshot_dates: ['2026-01-01'] },
    roles: [
      { id: 7, name: 'Synthetic Technical Ops', count: 3, median_salary: 80_000, high_share: 1 / 3,
        top_skills: [{ skill: 'python', index: 0, count: 3, share: 1, overall_share: 4 / 6 },
          { skill: 'sql', index: 1, count: 2, share: 2 / 3, overall_share: .5 },
          { skill: 'security', index: 3, count: 1, share: 1 / 3, overall_share: 1 / 6 }] },
      { id: 12, name: 'Synthetic Care Services', count: 3, median_salary: 100_000, high_share: 1 / 3,
        top_skills: [{ skill: 'healthcare', index: 2, count: 3, share: 1, overall_share: .5 },
          { skill: 'python', index: 0, count: 1, share: 1 / 3, overall_share: 4 / 6 },
          { skill: 'sql', index: 1, count: 1, share: 1 / 3, overall_share: .5 }] },
    ],
    locations: ['Synthetic Boulder, CO', 'Synthetic Denver, CO'],
    skills: [
      { index: 0, name: 'python', count: 4, share: 4 / 6, is_technical: true },
      { index: 1, name: 'sql', count: 3, share: .5, is_technical: true },
      { index: 2, name: 'healthcare', count: 3, share: .5, is_technical: false },
      { index: 3, name: 'security', count: 1, share: 1 / 6, is_technical: false },
    ],
    rows: [[40_000, 0, 0, [0, 1], 0], [60_000, 1, 0, [1, 2], 0], [80_000, 0, 1, [0, 1], 1],
      [100_000, 1, 1, [2], 1], [120_000, 0, 0, [0, 3], 2], [140_000, 1, 1, [0, 2], 2]],
    tiers: ['Low', 'Mid', 'High'],
    networks: {
      all: { nodes: [
        { id: 0, label: 'python', count: 4, role: 0, x: 0, y: 0 },
        { id: 1, label: 'sql', count: 3, role: 0, x: 1, y: 0 },
        { id: 2, label: 'healthcare', count: 3, role: 1, x: 0, y: 1 },
        { id: 3, label: 'security', count: 1, role: 0, x: 1, y: 1 },
      ], edges: [{ source: 0, target: 1, count: 2, jaccard: .4 }, { source: 0, target: 2, count: 1, jaccard: 1 / 6 },
        { source: 0, target: 3, count: 1, jaccard: .25 }, { source: 1, target: 2, count: 1, jaccard: .2 }] },
      technical: { nodes: [{ id: 0, label: 'python', count: 4, role: 0, x: 0, y: 0 }, { id: 1, label: 'sql', count: 3, role: 0, x: 1, y: 0 }],
        edges: [{ source: 0, target: 1, count: 2, jaccard: .4 }] },
    },
    skill_neighbors: { '0': [{ skillIndex: 1, count: 2, share_among_selected: .5 }, { skillIndex: 2, count: 1, share_among_selected: .25 }, { skillIndex: 3, count: 1, share_among_selected: .25 }],
      '1': [{ skillIndex: 0, count: 2, share_among_selected: 2 / 3 }, { skillIndex: 2, count: 1, share_among_selected: 1 / 3 }],
      '2': [{ skillIndex: 0, count: 1, share_among_selected: 1 / 3 }, { skillIndex: 1, count: 1, share_among_selected: 1 / 3 }],
      '3': [{ skillIndex: 0, count: 1, share_among_selected: 1 }] },
    models: { grouped: {
      report: { status: 'complete', winner: 'fixed_xgboost', cv_rows: cvRows,
        outer_metrics: { f1_macro: 25 / 36, accuracy: .7, roc_auc_ovr_macro: .78,
          f1_per_class: { Low: 2 / 3, Mid: .75, High: 2 / 3 }, confusion_matrix: [[2, 1, 0], [0, 3, 1], [1, 0, 2]] },
        bootstrap: { repetitions: 10, seed: 7, unit: 'synthetic description group',
          intervals: { f1_macro: { 'lower_2.5': .6, 'upper_97.5': .8, valid_repetitions: 10 } } },
        limitations: ['Synthetic reused development split, not an untouched external test.'] },
      frozen_design: { candidate_order: ['fixed_xgboost', 'nominal_logistic'], evaluation_status: 'exploratory_grouped_resplit_of_previously_used_development_data',
        outer_train_rows: 40, outer_test_rows: 10, outer_train_groups: 35, outer_test_groups: 9, outer_group_overlap: 0,
        group_unit: 'normalized_description_group', groups: 44, rows: 50 },
      cv_selection: { winner: 'fixed_xgboost', rows: cvRows }, validated_training_cv_winner: 'fixed_xgboost',
    }, historical: [{ model: 'Synthetic historical test-F1 leader', f1_macro: .99, roc_auc_ovr: .99, accuracy: .99 }] },
    live: { evaluation: { source_status: { fetch_date: '2026-01-01', mode: 'cached_only' },
      predeclared_contract: { primary_support_inclusive: .05, sensitivity_support_inclusive: .02,
        confidence_strictly_greater_than: .5, lift_strictly_greater_than: 1.5, muse_eligibility_min_skills: 2 },
      adzuna, muse_uncapped: muse(), muse_company_capped: muse(true),
      limitations: ['Synthetic cached snapshot only; no temporal market inference.'], inputs_unchanged: true,
      historical_saved_comparison: {}, input_sha256: {} },
      comparison: [{ skill: 'python', muse_share: .6, train_share: 4 / 6, train_tech_share: 1, muse_share_capped: .5,
        is_technical: true, change_vs_tech_pp: -40, ratio_vs_tech: .6, fetch_date: '2026-01-01' }],
      rules_validation: [], fixed_rules_uncapped: [{ rule: 'python -> sql', support: .4, confidence: .75, lift: 1.8,
        passes_primary_005: true, passes_sensitivity_002: true, denominator: 5 }],
      fixed_rules_capped: [{ rule: 'python -> sql', support: .2, confidence: .5, lift: 1.2,
        passes_primary_005: false, passes_sensitivity_002: false, denominator: 3 }] },
    robustness: { summary: [{ representation: 'synthetic_svd_cosine', k: 2, mean_cosine_silhouette_original: .1,
      mean_cosine_silhouette_representation: .2, mean_euclidean_silhouette_representation: .15,
      mean_pairwise_seed_ari: .7, min_pairwise_seed_ari: .6, min_cluster_share: .3, max_cluster_share: .7 }],
      provenance: { total_saved_rows: 6, nonzero_rows: 6, zero_skill_rows_excluded: 0, sample_rows: 6, sample_seed: 7, sampling: 'Synthetic fixture' },
      configuration: { k: [2], seeds: [7, 8], n_init: 1, max_iter: 5, threads: 1 }, limitations: ['Synthetic fixture only.'] },
    rules: { general: [], high: [], technical: [] },
    availability: { tables: {}, metrics: {}, evaluations: { grouped: { available: true, status: 'exploratory_grouped', source_ids: ['synthetic'] },
      live: { available: true, status: 'cached_snapshot', source_ids: ['synthetic'] }, clustering: { available: true, status: 'bounded_exploratory', source_ids: ['synthetic'] } } },
    provenance: { salary_grain: 'unique_job_id', row_columns: ['salary', 'roleIndex', 'locationIndex', 'skillIndices', 'tierIndex'],
      role_index: 'Role array index, not saved cluster ID.', network_definition: 'Descriptive shared-posting counts; synthetic fixture, not significance or market trends.',
      neighbor_share_definition: 'Shared jobs / selected skill jobs.', role_names: 'Explicitly synthetic archetype labels.',
      sources: { synthetic: { path: 'synthetic-ui-fixture.csv', bytes: 0, sha256: '0'.repeat(64), status: 'synthetic_test_fixture' } },
      generated_utc: '2026-01-01T00:00:00Z', caveats: ['Synthetic test fixture; no production results.'] },
  } satisfies ProductData);
}

export function emptyProductFixture(): ProductData {
  const data = productFixture();
  data.rows = []; data.roles = []; data.locations = []; data.skills = [];
  data.metrics = { n_jobs: 0, n_skills_vocab: 0, zero_skills: 0, mean_skills: null, median_salary: null,
    skill_counts: {}, tier_counts: { Low: 0, Mid: 0, High: 0 }, salary_summary: null };
  data.networks = { all: { nodes: [], edges: [] }, technical: { nodes: [], edges: [] } };
  data.skill_neighbors = {}; data.models = { grouped: null, historical: null };
  data.live = { evaluation: null, comparison: null, rules_validation: null, fixed_rules_uncapped: null, fixed_rules_capped: null };
  data.robustness = null; data.rules = { general: [], high: [], technical: [] };
  data.availability = { tables: {}, metrics: {}, evaluations: {} };
  return data;
}
