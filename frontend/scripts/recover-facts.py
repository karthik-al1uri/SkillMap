"""Recover saved evidence from the approved Phase 3 HTML and hash-matching CSVs.

Does not recreate evaluations, mine rules, train models or change source artifacts.
The recovered facts are distinct from the unavailable original Phase 3 facts file.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
APPROVED_HTML_SHA256 = 'eddc4ca9399d2a09921d3bde8bc84205db2227d119085de0dfea8f00fb7b0b8b'


def recover(out):
    import pandas as pd
    html = ROOT / 'dashboard_preview/dashboard-v2.html'
    encoded = html.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != APPROVED_HTML_SHA256:
        raise ValueError('Saved dashboard does not match the approved Phase 3 checkpoint')
    match = re.search(r'<script type="application/json" id="skillmap-provenance">(.*?)</script>', encoded.decode(), re.S)
    if not match:
        raise ValueError('Approved HTML has no embedded evidence registry')
    saved = json.loads(match.group(1))
    sources = {'approved_html': {'path': str(html), 'bytes': len(encoded), 'sha256': APPROVED_HTML_SHA256,
                                 'status': 'approved_phase3_html_recovery'}}
    checked = []
    missing = []
    for key, source in saved['sources'].items():
        path = Path(source['path'])
        if not path.is_file():
            missing.append(key)
            continue
        raw = path.read_bytes()
        if len(raw) != source['bytes'] or hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError(f'Saved source differs from Phase 3 registry: {path}')
        checked.append(key)
    tables = {}
    for key, meta in saved['table_provenance'].items():
        tables[key] = {**meta, 'available': False, 'data': None,
                       'reason': 'Original table payload unavailable; no experiment recreated'}
        if key not in ['jobs', '06_cluster_labels', '05_classification_results', '08_skill_comparison', '08_rules_validation']:
            continue
        source_id = meta['source_ids'][0]
        if source_id not in checked:
            continue
        source = saved['sources'][source_id]
        frame = pd.read_csv(source['path'], dtype={'job_id': str})
        if key == 'jobs':
            frame['matched_skills'] = frame['matched_skills'].map(ast.literal_eval)
        # pandas emits JSON null for missing scalar values, never nonstandard NaN.
        tables[key] = {**meta, 'source_ids': ['approved_html', source_id],
                       'data': json.loads(frame.to_json(orient='records'))}
        sources[source_id] = source
    if not tables['jobs']['available']:
        raise ValueError('Hash-verified unique-job CSV is required for cohort recovery')
    notice = ('Recovered from the SHA-256-verified approved Phase 3 HTML and matching canonical CSVs. '
              'Original facts bundle and temporary evaluation files are unavailable. '
              'Embedded evaluations are verified through the approved HTML, not against missing originals. '
              'Unavailable association-rule tables remain unavailable.')
    facts = {'schema': saved['schema'], 'generated_utc': saved.get('generated_utc'),
             'sources': sources, 'metrics': saved['metrics'], 'evaluations': saved['evaluations'],
             'tables': tables, 'provenance': {**saved['provenance'], 'recovery_notice': notice,
                 'original_facts_sha256': saved['bundle_sha256'], 'original_source_registry': saved['sources'],
                 'verified_original_sources': checked, 'unavailable_original_sources': missing},
             'caveats': [notice]}
    for section in ['metrics', 'evaluations']:
        for wrapper in facts[section].values():
            wrapper['original_source_ids'] = wrapper.get('source_ids', [])
            wrapper['source_ids'] = ['approved_html']
    out = Path(out)
    if out.exists():
        raise ValueError('Recovery output must be a new file')
    out.write_text(json.dumps(facts, allow_nan=False, separators=(',', ':')) + '\n')
    print(json.dumps({'facts': str(out), 'verified_original_sources': len(checked),
                      'unavailable_original_sources': len(missing), 'notice': notice}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    recover(args.out)
