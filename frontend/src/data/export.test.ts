/** Production export integrity, separate from explicitly synthetic interaction fixtures. */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { validateData } from './logic';

describe('saved production export', () => {
  it('satisfies the runtime contract and records checked export provenance', () => {
    const data = validateData(JSON.parse(readFileSync('public/data/skillmap.json', 'utf8')));
    expect(data.provenance.sourcefacts_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(data.provenance.export_contract).toMatchObject({ source_checks_requested: true, rows: data.rows.length });
    expect(data.provenance.sources).not.toHaveProperty('synthetic');
    expect(data.models.grouped?.report.winner).toBe(data.models.grouped?.validated_training_cv_winner);
    expect(data.rows.length).toBe(data.metrics.n_jobs);
    for (const source of Object.values(data.provenance.sources)) {
      const raw = readFileSync(source.path);
      expect(raw.byteLength).toBe(source.bytes);
      expect(createHash('sha256').update(raw).digest('hex')).toBe(source.sha256);
    }
  });
});
