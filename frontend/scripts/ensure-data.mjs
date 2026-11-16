import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
if (!existsSync(new URL('../public/data/skillmap.json', import.meta.url))) {
  console.log('Exporting the saved SkillMap analysis data…');
  const result = spawnSync('python3', ['-B', 'scripts/export-data.py'], { stdio: 'inherit', env: { ...process.env, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1' } });
  if (result.status !== 0) {
    console.error('Data export unavailable. Run npm run data:export -- --facts /path/to/dashboard-facts.json. No data or metrics will be fabricated.');
    process.exit(result.status ?? 1);
  }
}
