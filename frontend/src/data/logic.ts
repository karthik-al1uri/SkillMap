import type { CohortFilters, JobRow, Network, NetworkNode, ProductData, SalaryGroup, SalaryHistogram } from './types';

function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value); }
function integer(value: unknown): value is number { return finite(value) && Number.isInteger(value); }
function object(value: unknown, label: string): Record<string, unknown> {
  ensure(typeof value === 'object' && value !== null && !Array.isArray(value), `${label} must be an object`);
  return value as Record<string, unknown>;
}
function list(value: unknown, label: string): unknown[] {
  ensure(Array.isArray(value), `${label} must be an array`);
  return value;
}

/** Median at the supplied observation grain. Empty cohorts are unavailable, never zero. */
export function median(values: readonly number[]): number | null {
  ensure(values.every(finite), 'Median requires finite numbers');
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
}

export function filterCohort(rows: readonly JobRow[], filters: Partial<CohortFilters> = {}): JobRow[] {
  for (const value of [filters.role, filters.location, filters.skill]) {
    ensure(value == null || (integer(value) && value >= 0), 'Cohort filters must be nonnegative indices or null');
  }
  return rows.filter(row => (filters.role == null || row[1] === filters.role)
    && (filters.location == null || row[2] === filters.location)
    && (filters.skill == null || row[3].includes(filters.skill)));
}

/** Skill comparisons count a job once per distinct skill, never once per repeated token. */
export function groupSalary(rows: readonly JobRow[], dimension: 'role' | 'location' | 'skill' | 1 | 2, minN = 1): SalaryGroup[] {
  ensure(integer(minN) && minN >= 1, 'Minimum cohort size must be a positive integer');
  const groups = new Map<number, { salaries: number[]; high: number }>();
  for (const row of rows) {
    ensure(finite(row[0]), 'Salary rows must be finite');
    const indices = dimension === 'skill' ? [...new Set(row[3])]
      : [row[dimension === 'role' || dimension === 1 ? 1 : 2]];
    for (const index of indices) {
      const group = groups.get(index) ?? { salaries: [], high: 0 };
      group.salaries.push(row[0]);
      if (row[4] === 2) group.high += 1;
      groups.set(index, group);
    }
  }
  return [...groups.entries()].filter(([, value]) => value.salaries.length >= minN)
    .map(([index, value]) => ({ index, count: value.salaries.length,
      median_salary: median(value.salaries) as number, high_share: value.high / value.salaries.length }))
    .sort((a, b) => b.median_salary - a.median_salary || b.count - a.count || a.index - b.index);
}

export function salaryHistogram(rows: readonly JobRow[], bins = 24, domain?: readonly [number, number]): SalaryHistogram {
  ensure(integer(bins) && bins >= 1 && bins <= 1000, 'Histogram bins must be an integer from 1 to 1000');
  const salaries = rows.map(row => row[0]);
  ensure(salaries.every(finite), 'Histogram salaries must be finite');
  if (!salaries.length && !domain) return { counts: [], edges: [], total: 0, underflow: 0, overflow: 0 };
  let low: number;
  let high: number;
  if (domain) {
    [low, high] = domain;
    ensure(finite(low) && finite(high) && low < high, 'Histogram domain must be finite and increasing');
  } else {
    low = Math.min(...salaries);
    high = Math.max(...salaries);
    if (low === high) {
      const padding = Math.max(1, Math.abs(low) * 0.01);
      low -= padding;
      high += padding;
    }
  }
  const edges = Array.from({ length: bins + 1 }, (_, i) => low * (1 - i / bins) + high * (i / bins));
  edges[0] = low;
  edges[bins] = high;
  const counts = Array<number>(bins).fill(0);
  let underflow = 0;
  let overflow = 0;
  for (const salary of salaries) {
    if (salary < low) { underflow += 1; continue; }
    if (salary > high) { overflow += 1; continue; }
    const span = high - low;
    const ratio = Number.isFinite(span) ? (salary - low) / span : (salary / 2 - low / 2) / (high / 2 - low / 2);
    const index = Math.min(bins - 1, Math.max(0, Math.floor(ratio * bins)));
    counts[index] += 1;
  }
  return { counts, edges, total: salaries.length, underflow, overflow };
}

/** Row normalization preserves empty observed classes as zero rows. */
export function normalizeConfusion(matrix: readonly (readonly number[])[]): number[][] {
  const width = matrix[0]?.length ?? 0;
  return matrix.map(row => {
    ensure(row.length === width && row.every(value => finite(value) && value >= 0), 'Confusion matrix must be rectangular finite nonnegative counts');
    const total = row.reduce((sum, value) => sum + value, 0);
    ensure(Number.isFinite(total), 'Confusion row total must be finite');
    return row.map(value => total ? value / total : 0);
  });
}

export function networkNeighbors(network: Network, index: number): { node: NetworkNode; count: number; jaccard: number }[] {
  const nodes = new Map(network.nodes.map(node => [node.id, node]));
  const result: { node: NetworkNode; count: number; jaccard: number }[] = [];
  for (const edge of network.edges) {
    const neighbor = edge.source === index ? edge.target : edge.target === index ? edge.source : null;
    if (neighbor !== null) {
      const node = nodes.get(neighbor);
      if (node) result.push({ node, count: edge.count, jaccard: edge.jaccard });
    }
  }
  return result.sort((a, b) => b.jaccard - a.jaccard || b.count - a.count || a.node.id - b.node.id);
}

function validateGrouped(value: unknown): void {
  if (value == null) return;
  const grouped = object(value, 'Grouped evaluation');
  const report = object(grouped.report, 'Grouped report');
  const frozen = object(grouped.frozen_design, 'Frozen grouped design');
  const selector = object(grouped.cv_selection, 'Grouped CV selector');
  const order = list(frozen.candidate_order, 'Frozen candidate order');
  ensure(order.length > 0 && order.every(name => typeof name === 'string') && new Set(order).size === order.length, 'Frozen candidate order must be unique nonempty names');
  const rows = list(selector.rows, 'CV selector rows');
  const scores = new Map<string, number>();
  for (const entry of rows) {
    const row = object(entry, 'CV row');
    ensure(typeof row.candidate === 'string' && finite(row.mean_f1_macro) && row.mean_f1_macro >= 0 && row.mean_f1_macro <= 1, 'CV rows require candidate and finite macro F1');
    ensure(!scores.has(row.candidate), 'CV candidate rows must be unique');
    scores.set(row.candidate, row.mean_f1_macro);
  }
  ensure(scores.size === order.length && order.every(name => scores.has(name as string)), 'CV candidates must match frozen design');
  const winner = (order as string[]).reduce((best, name) => (scores.get(name) as number) > (scores.get(best) as number) ? name : best);
  ensure(report.status === 'complete' && report.winner === winner && selector.winner === winner
    && grouped.validated_training_cv_winner === winner, 'Grouped winner must use training CV and frozen candidate-order tie break');
  const reportRows = list(report.cv_rows, 'Grouped report CV rows');
  ensure(reportRows.length === rows.length, 'Grouped report candidates differ from selector');
  const seen = new Set<string>();
  for (const entry of reportRows) {
    const row = object(entry, 'Report CV row');
    ensure(typeof row.candidate === 'string' && !seen.has(row.candidate) && finite(row.mean_f1_macro)
      && scores.has(row.candidate) && Math.abs(row.mean_f1_macro - (scores.get(row.candidate) as number)) <= 1e-12,
    'Grouped report scores must match training CV selector');
    seen.add(row.candidate);
  }
}

/** Validate nested evidence used directly by the authored views. */
function validateEvidence(data: Record<string, unknown>): void {
  const score = (value: unknown, label: string, nullable = false) => ensure((nullable && value === null) || (finite(value) && value >= 0 && value <= 1), `${label} must be a score in [0,1]`);
  const numericFields = (value: unknown, fields: string[], label: string) => {
    const row = object(value, label);
    for (const key of fields) ensure(finite(row[key]), `${label}.${key} must be finite`);
    return row;
  };
  const models = object(data.models, 'Models');
  if (models.grouped != null) {
    const report = object(object(models.grouped, 'Grouped').report, 'Grouped report');
    const metrics = object(report.outer_metrics, 'Grouped metrics');
    for (const key of ['f1_macro', 'accuracy', 'roc_auc_ovr_macro']) score(metrics[key], key, key === 'roc_auc_ovr_macro');
    const classes = object(metrics.f1_per_class, 'Grouped class F1');
    for (const tier of ['Low', 'Mid', 'High']) score(classes[tier], tier);
    const matrix = list(metrics.confusion_matrix, 'Confusion matrix');
    ensure(matrix.length === 3 && matrix.every(row => Array.isArray(row) && row.length === 3 && row.every(value => integer(value) && value >= 0)), 'Confusion matrix must contain 3x3 nonnegative integer counts');
    ensure(list(report.limitations, 'Grouped limitations').every(value => typeof value === 'string'), 'Limitations must be strings');
    const bootstrap = object(report.bootstrap, 'Grouped bootstrap');
    ensure(integer(bootstrap.repetitions) && typeof bootstrap.unit === 'string', 'Invalid grouped bootstrap');
    for (const value of Object.values(object(bootstrap.intervals, 'Grouped intervals'))) {
      const interval = object(value, 'Bootstrap interval');
      score(interval['lower_2.5'], 'Lower interval', true); score(interval['upper_97.5'], 'Upper interval', true);
    }
  }
  if (models.historical != null) for (const value of list(models.historical, 'Historical models')) {
    const row = object(value, 'Historical model');
    ensure(typeof row.model === 'string', 'Historical model name missing');
    score(row.f1_macro, 'Historical F1'); score(row.roc_auc_ovr, 'Historical AUC', true);
  }
  const live = object(data.live, 'Live');
  if (live.evaluation != null) {
    const report = object(live.evaluation, 'Live evaluation');
    object(report.source_status, 'Live source status');
    numericFields(report.predeclared_contract, ['primary_support_inclusive', 'confidence_strictly_greater_than', 'lift_strictly_greater_than'], 'Live contract');
    if (report.adzuna != null) {
      const a = object(report.adzuna, 'Adzuna');
      ensure(integer(a.n) && a.n >= 0, 'Adzuna population invalid');
      score(a.accuracy, 'Adzuna accuracy', true); score(a.macro_f1, 'Adzuna F1');
      for (const key of ['actual_counts', 'predicted_counts', 'per_class_recall', 'per_class_f1']) {
        const values = numericFields(a[key], ['Low', 'Mid', 'High'], key);
        for (const value of Object.values(values)) ensure(finite(value) && value >= 0, `${key} must be nonnegative`);
      }
      const matrix = list(a.confusion_rows_actual_columns_predicted, 'Cached confusion matrix');
      ensure(matrix.length === 3 && matrix.every(row => Array.isArray(row) && row.length === 3 && row.every(value => integer(value) && value >= 0)), 'Cached confusion matrix must be 3x3 nonnegative counts');
      const actual = object(a.actual_counts, 'Cached observed counts');
      const predicted = object(a.predicted_counts, 'Cached predicted counts');
      const counts = matrix as number[][];
      for (const [index,tier] of ['Low','Mid','High'].entries()) {
        ensure(counts[index].reduce((sum,value)=>sum+value,0) === actual[tier], 'Cached confusion rows disagree with observed counts');
        ensure(counts.reduce((sum,row)=>sum+row[index],0) === predicted[tier], 'Cached confusion columns disagree with predicted counts');
      }
      ensure(counts.flat().reduce((sum,value)=>sum+value,0) === a.n, 'Cached confusion population differs from evaluation n');
      numericFields(a.salary_label_status, ['estimated_rows', 'advertised_rows'], 'Salary label status');
      const bootstrap = object(a.bootstrap, 'Adzuna bootstrap');
      for (const key of ['accuracy', 'macro_f1']) {
        const interval = object(bootstrap[key], 'Adzuna interval'); score(interval.low, 'Lower bound', true); score(interval.high, 'Upper bound', true);
      }
      const comparators = object(a.comparators, 'Adzuna comparators');
      for (const key of ['observed_majority_oracle', 'historical_training_majority']) {
        const comparator = object(comparators[key], key);
        score(comparator.accuracy, 'Comparator accuracy', true); score(comparator.macro_f1, 'Comparator F1');
        ensure(typeof comparator.label === 'string', 'Comparator label missing');
      }
      ensure(typeof object(a.roc_auc, 'Adzuna AUC').reason === 'string', 'Adzuna AUC reason missing');
    }
    for (const key of ['muse_uncapped', 'muse_company_capped']) if (report[key] != null) {
      const m = numericFields(report[key], ['eligible_postings', 'all_postings', 'primary_confirmed', 'fixed_rules', 'employers_eligible', 'sensitivity_confirmed'], key);
      ensure(list(m.confirmed_primary_rules, 'Confirmed rules').every(value => typeof value === 'string'), 'Confirmed rules must be strings');
    }
  }
  if (live.comparison != null) for (const value of list(live.comparison, 'Skill comparison')) {
    const row = numericFields(value, ['muse_share', 'muse_share_capped', 'train_tech_share'], 'Skill comparison');
    ensure(typeof row.skill === 'string', 'Comparison skill missing');
  }
  for (const key of ['fixed_rules_uncapped', 'fixed_rules_capped']) if (live[key] != null) for (const value of list(live[key], key)) {
    const row = object(value, 'Fixed rule');
    ensure(typeof row.rule === 'string' && typeof row.passes_primary_005 === 'boolean', 'Invalid fixed rule');
    for (const metric of ['support', 'confidence', 'lift']) ensure(row[metric] === null || (finite(row[metric]) && row[metric] >= 0), 'Rule metric must be nonnegative or null');
  }
  const rules = object(data.rules, 'Rules');
  for (const key of ['general', 'technical', 'high']) if (rules[key] != null) for (const value of list(rules[key], key)) {
    const row = numericFields(value, ['support', 'confidence', 'lift'], 'Association rule');
    ensure(typeof row.antecedents === 'string' && typeof row.consequents === 'string', 'Invalid association rule text');
  }
  if (data.robustness != null) {
    const robustness = object(data.robustness, 'Robustness');
    for (const value of list(robustness.summary, 'Robustness summary')) {
      const row = numericFields(value, ['k', 'mean_cosine_silhouette_original', 'mean_pairwise_seed_ari', 'min_cluster_share', 'max_cluster_share'], 'Robustness row');
      ensure(typeof row.representation === 'string', 'Robustness representation missing');
    }
    numericFields(robustness.provenance, ['sample_rows'], 'Robustness provenance');
    list(object(robustness.configuration, 'Robustness configuration').seeds, 'Robustness seeds');
  }
}

/** Validate exported compact data before it can drive charts or salary filters. */
export function validateData(input: unknown): ProductData {
  const data = object(input, 'Dashboard data');
  ensure(data.schema === 'skillmap.dashboard.product.v1', 'Unsupported dashboard schema');
  const provenance = object(data.provenance, 'Provenance');
  ensure(provenance.salary_grain === 'unique_job_id', 'Salary grain must be unique_job_id');
  if (provenance.sourcefacts_sha256 !== undefined) ensure(typeof provenance.sourcefacts_sha256 === 'string'
    && /^[a-f0-9]{64}$/.test(provenance.sourcefacts_sha256), 'Facts source SHA-256 is invalid');
  for (const value of Object.values(object(provenance.sources, 'Source registry'))) {
    const source = object(value, 'Source');
    ensure(typeof source.path === 'string' && typeof source.status === 'string' && integer(source.bytes) && source.bytes >= 0
      && typeof source.sha256 === 'string' && /^[a-f0-9]{64}$/.test(source.sha256), 'Invalid source provenance');
  }
  const metrics = object(data.metrics, 'Metrics');
  const roles = list(data.roles, 'Roles');
  const skills = list(data.skills, 'Skills');
  const locations = list(data.locations, 'Locations');
  const tiers = list(data.tiers, 'Tiers');
  const rows = list(data.rows, 'Rows');
  ensure(integer(metrics.n_jobs) && metrics.n_jobs >= 0 && metrics.n_jobs === rows.length, 'Row count must match unique-job n_jobs');
  ensure(tiers.length === 3 && tiers.join('|') === 'Low|Mid|High', 'Tier order must be Low, Mid, High');
  ensure(locations.every(value => value === null || typeof value === 'string'), 'Locations must preserve raw labels or null');
  const skillCounts = Array<number>(skills.length).fill(0);
  const roleSalaries = Array.from({ length: roles.length }, () => [] as number[]);
  const roleHigh = Array<number>(roles.length).fill(0);
  for (const value of rows) {
    const row = list(value, 'Job row');
    ensure(row.length === 5 && finite(row[0]), 'Compact job rows require a finite salary');
    ensure(integer(row[1]) && row[1] >= 0 && row[1] < roles.length, 'Role index is out of range');
    ensure(integer(row[2]) && row[2] >= 0 && row[2] < locations.length, 'Location index is out of range');
    ensure(integer(row[4]) && row[4] >= 0 && row[4] < 3, 'Tier index is out of range');
    const indices = list(row[3], 'Job skill indices');
    ensure(new Set(indices).size === indices.length, 'A job must contain each skill at most once');
    for (const index of indices) {
      ensure(integer(index) && index >= 0 && index < skills.length, 'Skill index is out of range');
      skillCounts[index] += 1;
    }
    roleSalaries[row[1]].push(row[0]);
    if (row[4] === 2) roleHigh[row[1]] += 1;
  }
  for (let index = 0; index < roles.length; index += 1) {
    const role = object(roles[index], 'Role');
    ensure(integer(role.id) && typeof role.name === 'string' && integer(role.count) && role.count === roleSalaries[index].length,
      'Role counts must use unique-job rows');
    ensure(role.count > 0 && finite(role.median_salary) && Math.abs(role.median_salary - (median(roleSalaries[index]) as number)) <= 1e-6,
      'Role median must use unique-job salaries');
    ensure(finite(role.high_share) && Math.abs(role.high_share - roleHigh[index] / role.count) <= 1e-12, 'Role High share conflicts with rows');
  }
  for (const value of roles) {
    const role = object(value, 'Role');
    for (const item of list(role.top_skills, 'Archetype skill signature')) {
      const skill = object(item, 'Signature skill');
      ensure(typeof skill.skill === 'string' && finite(skill.share) && skill.share >= 0 && skill.share <= 1
        && (skill.overall_share === null || (finite(skill.overall_share) && skill.overall_share >= 0 && skill.overall_share <= 1)), 'Invalid archetype skill shares');
    }
  }
  ensure(new Set(roles.map(value => object(value, 'Role').id)).size === roles.length, 'Saved role IDs must be unique');
  for (let index = 0; index < skills.length; index += 1) {
    const skill = object(skills[index], 'Skill');
    ensure(skill.index === index && typeof skill.name === 'string' && integer(skill.count) && skill.count === skillCounts[index],
      'Skill indices/counts must match distinct job-skill rows');
    ensure(skill.is_technical === null || typeof skill.is_technical === 'boolean', 'Technical eligibility must be recorded true/false/null');
  }
  if (metrics.n_skills_vocab != null) ensure(metrics.n_skills_vocab === skills.length, 'Vocabulary size conflicts with skills');
  if (metrics.median_salary != null) ensure(finite(metrics.median_salary)
    && Math.abs(metrics.median_salary - (median((rows as JobRow[]).map(row => row[0])) as number)) <= 1e-6,
  'Dataset median must use unique-job rows');
  const networks = object(data.networks, 'Networks');
  for (const name of ['all', 'technical']) {
    const graph = object(networks[name], `${name} network`);
    const nodes = list(graph.nodes, 'Network nodes');
    const edges = list(graph.edges, 'Network edges');
    ensure(nodes.length <= 24 && edges.length <= 48, 'Network must retain its bounded factual view');
    const nodeIds = new Set<number>();
    for (const value of nodes) {
      const node = object(value, 'Network node');
      ensure(integer(node.id) && node.id >= 0 && node.id < skills.length && !nodeIds.has(node.id), 'Network node skill index invalid or duplicated');
      nodeIds.add(node.id);
      const skill = object(skills[node.id], 'Network skill');
      ensure(node.label === skill.name && node.count === skill.count && finite(node.x) && finite(node.y), 'Network node must match observed skill facts');
      ensure(node.role === null || (integer(node.role) && node.role >= 0 && node.role < roles.length), 'Network role index invalid');
      if (name === 'technical') ensure(skill.is_technical === true, 'Technical graph contains a skill without true technical eligibility');
    }
    const seen = new Set<string>();
    for (const value of edges) {
      const edge = object(value, 'Network edge');
      ensure(integer(edge.source) && integer(edge.target) && edge.source !== edge.target && nodeIds.has(edge.source) && nodeIds.has(edge.target), 'Network edge endpoints invalid');
      const key = [edge.source, edge.target].sort((a, b) => a - b).join(':');
      ensure(!seen.has(key), 'Network edge duplicated');
      seen.add(key);
      const shared = (rows as JobRow[]).reduce((count, row) => count + (row[3].includes(edge.source as number) && row[3].includes(edge.target as number) ? 1 : 0), 0);
      ensure(integer(edge.count) && edge.count > 0 && edge.count === shared, 'Network edge must count actual shared postings');
      const jaccard = shared / (skillCounts[edge.source] + skillCounts[edge.target] - shared);
      ensure(finite(edge.jaccard) && Math.abs(edge.jaccard - jaccard) <= 1e-12, 'Network Jaccard conflicts with shared postings');
    }
  }
  validateGrouped(object(data.models, 'Models').grouped);
  object(data.live, 'Live data');
  object(data.rules, 'Rules');
  object(data.availability, 'Availability');
  validateEvidence(data);
  return input as ProductData;
}
