import { describe, expect, it } from 'vitest';
import { filterCohort, groupSalary, median, networkNeighbors, normalizeConfusion, salaryHistogram, validateData } from './logic';
import type { JobRow, Network, ProductData } from './types';

const rows: JobRow[] = [[100, 0, 0, [0, 1], 0], [100000, 0, 1, [0], 2], [900, 1, 0, [], 1]];
function fixture(): ProductData {
  return {
    schema: 'skillmap.dashboard.product.v1',
    metrics: { n_jobs: 3, n_skills_vocab: 2, median_salary: 900 },
    roles: [{ id: 7, name: 'Observed role', count: 2, median_salary: 50050, high_share: .5, top_skills: [] },
      { id: -1, name: 'No matched skills', count: 1, median_salary: 900, high_share: 0, top_skills: [] }],
    skills: [{ index: 0, name: 'python', count: 2, share: 2 / 3, is_technical: true },
      { index: 1, name: 'sql', count: 1, share: 1 / 3, is_technical: null }],
    locations: ['Boulder', 'Denver'], tiers: ['Low', 'Mid', 'High'], rows: structuredClone(rows),
    networks: { all: { nodes: [{ id: 0, label: 'python', count: 2, role: 0, x: 0, y: 0 },
      { id: 1, label: 'sql', count: 1, role: 0, x: 1, y: 1 }], edges: [{ source: 0, target: 1, count: 1, jaccard: .5 }] },
      technical: { nodes: [{ id: 0, label: 'python', count: 2, role: 0, x: 0, y: 0 }], edges: [] } },
    skill_neighbors: {}, models: { grouped: null, historical: null },
    live: { evaluation: null, comparison: null, rules_validation: null, fixed_rules_uncapped: null, fixed_rules_capped: null },
    robustness: null, rules: { general: null, high: null, technical: [] },
    availability: { tables: {}, metrics: {}, evaluations: {} },
    provenance: { salary_grain: 'unique_job_id', row_columns: ['salary', 'roleIndex', 'locationIndex', 'skillIndices', 'tierIndex'], role_index: 'array', network_definition: 'Observed shared-posting counts', neighbor_share_definition: 'conditional on selected skill', role_names: 'saved', sources: {} },
  };
}

describe('cohort salary logic', () => {
  it('uses job-grain medians, including empty and finite validation', () => {
    expect(median(rows.map(row => row[0]))).toBe(900);
    expect(median([100, 100000])).toBe(50050);
    expect(median([])).toBeNull();
    expect(() => median([Number.NaN])).toThrow(/finite/);
  });
  it('combines filters without mutating rows', () => {
    expect(filterCohort(rows, { role: 0, location: 0, skill: 1 })).toEqual([rows[0]]);
    expect(filterCohort(rows, { skill: 999 })).toEqual([]);
    expect(rows).toHaveLength(3);
  });
  it('groups skills once per job and applies minimum cohort sizes', () => {
    const duplicate: JobRow[] = [[100, 0, 0, [0, 0, 1], 0], rows[1], rows[2]];
    expect(groupSalary(duplicate, 'skill')).toEqual([
      { index: 0, count: 2, median_salary: 50050, high_share: .5 },
      { index: 1, count: 1, median_salary: 100, high_share: 0 },
    ]);
    expect(groupSalary(rows, 'role', 2)).toEqual([{ index: 0, count: 2, median_salary: 50050, high_share: .5 }]);
    expect(groupSalary(rows, 'location')[0]).toEqual({ index: 1, count: 1, median_salary: 100000, high_share: 1 });
  });
  it('histogram preserves totals and includes the upper boundary', () => {
    expect(salaryHistogram(rows, 2, [0, 100000])).toEqual({ counts: [2, 1], edges: [0, 50000, 100000], total: 3, underflow: 0, overflow: 0 });
    const narrowed = salaryHistogram(rows, 2, [200, 1000]);
    expect(narrowed.counts.reduce((a, b) => a + b, 0) + narrowed.underflow + narrowed.overflow).toBe(3);
    expect(salaryHistogram([], 2).edges).toEqual([]);
  });
  it('normalizes confusion rows without inventing observed classes', () => {
    expect(normalizeConfusion([[2, 2], [0, 0]])).toEqual([[.5, .5], [0, 0]]);
    expect(() => normalizeConfusion([[1, -1]])).toThrow();
  });
});

describe('runtime dashboard data contract', () => {
  it('accepts factual metadata and counts at unique-job grain', () => {
    const data = fixture();
    expect(validateData(data)).toBe(data);
    expect(data.models.grouped).toBeNull();
  });
  it.each(['schema', 'grain', 'count', 'skill', 'salary', 'network', 'technical'] as const)('rejects corrupted %s metadata', field => {
    const data = fixture();
    if (field === 'schema') (data as { schema: string }).schema = 'wrong';
    if (field === 'grain') data.provenance.salary_grain = 'job_skill';
    if (field === 'count') data.metrics.n_jobs = 4;
    if (field === 'skill') data.rows[0][3] = [99];
    if (field === 'salary') data.roles[0].median_salary = 100;
    if (field === 'network') data.networks.all.edges[0].count = 2;
    if (field === 'technical') data.networks.technical.nodes.push(data.networks.all.nodes[1]);
    expect(() => validateData(data)).toThrow();
  });
  it('returns observed neighboring skills from actual edges', () => {
    const graph: Network = fixture().networks.all;
    expect(networkNeighbors(graph, 0)[0]).toMatchObject({ node: { id: 1 }, count: 1, jaccard: .5 });
    expect(networkNeighbors(graph, 999)).toEqual([]);
  });
  it('rejects CV winners that conflict with training-only scores and frozen tie order', () => {
    const data = fixture();
    const cv = [{ candidate: 'first', mean_f1_macro: .6 }, { candidate: 'second', mean_f1_macro: .6 }];
    data.models.grouped = {
      report: { status: 'complete', winner: 'first', cv_rows: cv, outer_metrics: { f1_macro: .4, accuracy: .4, roc_auc_ovr_macro: .5, f1_per_class: { Low: .4, Mid: .4, High: .4 }, confusion_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }, bootstrap: { repetitions: 200, seed: 42, unit: 'groups', intervals: {} }, limitations: [] },
      frozen_design: { candidate_order: ['first', 'second'] }, cv_selection: { winner: 'first', rows: cv }, validated_training_cv_winner: 'first',
    };
    expect(validateData(data)).toBe(data);
    data.models.grouped.report.winner = 'second';
    expect(() => validateData(data)).toThrow(/training CV/);
  });
});
