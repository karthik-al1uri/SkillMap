/** Posting-level saved snapshots are separate from model-validation populations. */
export type MarketSourceKey = 'adzuna' | 'muse';
export interface MarketPosting {
  id: string;
  title: string;
  company: string | null;
  location: string | null;
  date: string;
  skills: string[];
  salary: number | null;
  salaryStatus: 'estimated' | 'advertised' | 'unavailable';
  tier: 'Low' | 'Mid' | 'High' | null;
}
export interface MarketSource {
  label: string;
  postings: MarketPosting[];
  provenance: { path: string; bytes: number; sha256: string };
  limitations: string[];
}
export interface MarketData {
  schema: 'skillmap.market.v1';
  sources: Record<MarketSourceKey, MarketSource>;
  provenance: {
    approved_html: { path: string; bytes: number; sha256: string };
    source_checks_requested: boolean;
    execution: string;
    posting_grain: string;
    salary_currency: string;
    salary_period: string;
    refresh_status: string;
  };
}
const DAY = 86_400_000;
export const STALE_AFTER_DAYS = 30;
function calendarTime(value: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const time = Date.parse(`${value}T00:00:00Z`);
  return Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === value ? time : null;
}
/** Collection freshness only; never represents whether a vacancy remains open. */
export function snapshotFreshness(date: string, now = new Date()): { ageDays: number | null; stale: boolean; label: string } {
  const time = calendarTime(date);
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  if (time === null || !Number.isFinite(today) || time > today) return { ageDays: null, stale: true, label: 'Collection date unavailable' };
  const ageDays = Math.floor((today - time) / DAY);
  return { ageDays, stale: ageDays > STALE_AFTER_DAYS,
    label: ageDays > STALE_AFTER_DAYS ? `Stale snapshot · ${ageDays} days old` : `Cached snapshot · ${ageDays} ${ageDays === 1 ? 'day' : 'days'} old` };
}
function ensure(condition: unknown, message: string): asserts condition { if (!condition) throw new Error(message); }
function record(value: unknown, label: string): Record<string, unknown> {
  ensure(typeof value === 'object' && value !== null && !Array.isArray(value), `${label} must be an object`);
  return value as Record<string, unknown>;
}
function sourceReference(value: unknown) {
  const source = record(value, 'Snapshot provenance');
  ensure(typeof source.path === 'string' && Number.isInteger(source.bytes) && Number(source.bytes) >= 0
    && typeof source.sha256 === 'string' && /^[a-f0-9]{64}$/.test(source.sha256), 'Invalid snapshot provenance');
}
export function validateMarketData(input: unknown): MarketData {
  const data = record(input, 'Market snapshot');
  ensure(data.schema === 'skillmap.market.v1', 'Unsupported market snapshot schema');
  const sources = record(data.sources, 'Market sources');
  for (const key of ['adzuna', 'muse']) {
    const source = record(sources[key], `${key} snapshot`);
    ensure(typeof source.label === 'string' && Array.isArray(source.postings), 'Snapshot label/postings unavailable');
    sourceReference(source.provenance);
    ensure(Array.isArray(source.limitations) && source.limitations.every(value => typeof value === 'string'), 'Snapshot limitations must be strings');
    const seen = new Set<string>();
    for (const inputPosting of source.postings) {
      const posting = record(inputPosting, 'Saved posting');
      ensure(typeof posting.id === 'string' && posting.id.length > 0 && typeof posting.title === 'string' && posting.title.length > 0, 'Posting identity missing');
      ensure(typeof posting.date === 'string' && calendarTime(posting.date) !== null, 'Invalid posting collection date');
      const identity = `${posting.id}\0${posting.date}`;
      ensure(!seen.has(identity), 'Duplicate saved posting within source and collection date');
      seen.add(identity);
      ensure((posting.company === null || typeof posting.company === 'string') && (posting.location === null || typeof posting.location === 'string'), 'Invalid saved posting labels');
      ensure(Array.isArray(posting.skills) && posting.skills.every(value => typeof value === 'string' && value.length > 0)
        && new Set(posting.skills).size === posting.skills.length, 'Saved skill mentions must be unique strings');
      ensure(['estimated', 'advertised', 'unavailable'].includes(String(posting.salaryStatus)), 'Invalid saved salary status');
      ensure(posting.tier === null || ['Low', 'Mid', 'High'].includes(String(posting.tier)), 'Invalid saved salary tier');
      if (posting.salaryStatus === 'unavailable') ensure(posting.salary === null && posting.tier === null, 'Unavailable salary must remain null');
      else ensure(typeof posting.salary === 'number' && Number.isFinite(posting.salary) && posting.salary >= 0 && posting.tier !== null, 'Available salary requires finite saved value and tier');
      if (key === 'muse') ensure(posting.salaryStatus === 'unavailable', 'Muse salary was not saved in this export contract');
    }
  }
  const provenance = record(data.provenance, 'Market provenance');
  sourceReference(provenance.approved_html);
  ensure(provenance.source_checks_requested === true, 'Market export must verify saved sources');
  ensure(provenance.posting_grain === '(source, id, collection date)', 'Unsupported posting observation grain');
  ensure(provenance.salary_currency === 'USD' && provenance.salary_period === 'annual', 'Unsupported saved salary units');
  ensure(typeof provenance.execution === 'string' && typeof provenance.refresh_status === 'string', 'Market execution provenance missing');
  return input as MarketData;
}
