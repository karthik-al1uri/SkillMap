import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { snapshotFreshness, STALE_AFTER_DAYS, validateMarketData } from './market';
const load = () => JSON.parse(readFileSync('public/data/market.json', 'utf8'));

describe('verified saved market export', () => {
  it('retains posting-level populations and source salary status without fabricating Muse compensation', () => {
    const data = validateMarketData(load());
    const a = data.sources.adzuna.postings;
    const m = data.sources.muse.postings;
    expect(a).toHaveLength(1150);
    expect(m).toHaveLength(1378);
    expect(a.filter(row => row.salaryStatus === 'estimated')).toHaveLength(1092);
    expect(a.filter(row => row.salaryStatus === 'advertised')).toHaveLength(57);
    expect(a.filter(row => row.salaryStatus === 'unavailable')).toHaveLength(1);
    expect(m.every(row => row.salary === null && row.salaryStatus === 'unavailable')).toBe(true);
    expect(a.some(row => row.skills.length === 0)).toBe(true);
    expect(m.some(row => row.skills.length < 2)).toBe(true);
    expect(new Set([...a, ...m].map(row => row.date))).toEqual(new Set(['2026-10-14']));
  });
  it('matches both canonical source CSV hashes and the approved Phase 3 HTML', () => {
    const data = validateMarketData(load());
    for (const source of [...Object.values(data.sources).map(value => value.provenance), data.provenance.approved_html]) {
      const raw = readFileSync(source.path);
      expect(raw.byteLength).toBe(source.bytes);
      expect(createHash('sha256').update(raw).digest('hex')).toBe(source.sha256);
    }
    expect(data.provenance.source_checks_requested).toBe(true);
  });
  it.each(['schema', 'duplicate', 'salary', 'skills', 'date', 'source', 'muse'] as const)('rejects corrupted %s evidence', field => {
    const data = load();
    if (field === 'schema') data.schema = 'wrong';
    if (field === 'duplicate') data.sources.adzuna.postings.push(data.sources.adzuna.postings[0]);
    if (field === 'salary') data.sources.adzuna.postings[0].salary = Number.NaN;
    if (field === 'skills') data.sources.adzuna.postings[0].skills = ['python', 'python'];
    if (field === 'date') data.sources.adzuna.postings[0].date = '2026-02-31';
    if (field === 'source') data.sources.adzuna.provenance.sha256 = 'bad';
    if (field === 'muse') Object.assign(data.sources.muse.postings[0], {salary:100, salaryStatus:'advertised',tier:'Low'});
    expect(() => validateMarketData(data)).toThrow();
  });
  it('accepts explicit empty saved source populations', () => {
    const data = load();
    data.sources.adzuna.postings = []; data.sources.muse.postings = [];
    expect(validateMarketData(data).sources.adzuna.postings).toHaveLength(0);
  });
});

describe('snapshot freshness is collection age, never real-time status', () => {
  const today = new Date('2026-11-14T20:00:00Z');
  it('makes older-than-30-day snapshots explicitly stale', () => {
    expect(STALE_AFTER_DAYS).toBe(30);
    expect(snapshotFreshness('2026-10-14', today)).toMatchObject({ageDays:31,stale:true,label:'Stale snapshot · 31 days old'});
    expect(snapshotFreshness('2026-10-15', today)).toMatchObject({ageDays:30,stale:false});
  });
  it('uses UTC calendar days and handles invalid or future dates without fake freshness', () => {
    expect(snapshotFreshness('2026-11-14', today)).toMatchObject({ageDays:0,stale:false});
    expect(snapshotFreshness('2026-02-31', today)).toMatchObject({ageDays:null,stale:true});
    expect(snapshotFreshness('2026-11-15', today)).toMatchObject({ageDays:null,stale:true});
    expect(snapshotFreshness('', today)).toMatchObject({ageDays:null,stale:true});
  });
});
