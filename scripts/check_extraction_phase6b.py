"""Compare frozen extraction with experimental exclusions using explicit reviewed labels.

Never generates labels. Labels must reference real descriptions, include reviewer/rationale,
and use null for ambiguity. Results on a targeted sample are exploratory, not population accuracy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src import preprocessing as pp

# Experimental only: these rules are NOT installed in canonical preprocessing.
PROPOSED_EXCLUSIONS = {
    "training": r"(?<![a-z0-9])(?:paid(?:,\s*formal)?|free(?:\s+online)?|ongoing)\s+training(?![a-z0-9])",
    "development": r"(?<![a-z0-9])(?:career|personal|professional|workforce|continual)\s+development(?![a-z0-9])"
                   r"|(?<![a-z0-9])learning\s+and\s+development(?![a-z0-9])"
                   r"|(?<![a-z0-9])development\s+course\s+offerings(?![a-z0-9])",
    "healthcare": r"(?<![a-z0-9])(?:fully\s+paid|affordable|subsidized)\s+health[ -]?care(?:\s+options)?(?![a-z0-9])"
                  r"|(?<![a-z0-9])health[ -]?care\s+decisions\s+for\s+you\s+and\s+your\s+family",
    "security": r"(?<![a-z0-9])financial\s+security(?![a-z0-9])"
                r"|(?<![a-z0-9])security\s+of\s+working(?![a-z0-9])"
                r"|(?<![a-z0-9])for\s+added\s+security(?![a-z0-9])",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def score(labels, predictions):
    """Calculate metrics from reviewed Boolean labels; unknown labels never become negatives."""
    pairs = [(y, p) for y, p in zip(labels, predictions) if y is not None]
    tp = sum(y and p for y, p in pairs)
    fp = sum(not y and p for y, p in pairs)
    fn = sum(y and not p for y, p in pairs)
    tn = sum(not y and not p for y, p in pairs)
    return dict(n=len(pairs), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def compare(rows, vocab):
    """Run the full frozen vocabulary and existing nested-phrase logic for both variants."""
    for row in rows:
        if row['skill'] not in PROPOSED_EXCLUSIONS:
            raise ValueError('Unsupported target skill')
        if row['label'] is not None and (type(row['label']) is not bool or not row.get('labeler')
                                        or not row.get('label_rationale') or not row.get('job_id')):
            raise ValueError('Explicit Boolean label, real job ID, reviewer and rationale required')
    frame = pd.DataFrame(rows)
    baseline = pp.match_description_skills(frame, vocab)
    exclusions = dict(pp.SKILL_EXCLUSIONS)
    for skill, pattern in PROPOSED_EXCLUSIONS.items():
        exclusions[skill] = f"(?:{exclusions[skill]})|(?:{pattern})" if skill in exclusions else pattern
    with patch.dict(pp.SKILL_EXCLUSIONS, exclusions):
        proposed = pp.match_description_skills(frame, vocab)
    predictions = []
    for row, old, new in zip(rows, baseline, proposed):
        predictions.append({**{k: row[k] for k in ('sample_id', 'job_id', 'skill', 'label', 'labeler', 'label_rationale')},
                            'baseline': row['skill'] in old, 'proposed': row['skill'] in new})
    results = {}
    for skill in ['all', *PROPOSED_EXCLUSIONS]:
        subset = predictions if skill == 'all' else [r for r in predictions if r['skill'] == skill]
        results[skill] = {name: score([r['label'] for r in subset], [r[name] for r in subset])
                          for name in ('baseline', 'proposed')}
    return predictions, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    root, out = args.root.resolve(), args.out_dir.resolve()
    if out == root or root in out.parents or out.exists():
        parser.error('Use a fresh output directory outside the repository')
    rows = json.loads(args.sample.read_text())
    vocab_path = root/'data/processed/skill_vocabulary.csv'
    vocab = pd.read_csv(vocab_path)
    vocab['forms'] = vocab['forms'].map(ast.literal_eval)
    sources = [args.sample, vocab_path, root/'src/preprocessing.py']
    before = {str(p): digest(p) for p in sources}
    predictions, metrics = compare(rows, vocab)
    report = {'sample_size': len(rows), 'reviewed': sum(r['label'] is not None for r in rows),
              'unknown': sum(r['label'] is None for r in rows), 'metrics': metrics,
              'label_provenance': sorted({r['labeler'] for r in rows}),
              'limitations': 'Targeted stratified sample reviewed by assistant, not independent human gold. '
                             'Both nominal splits were inspected; no untouched holdout. No prevalence-weighted population estimate. '
                             'Recall covers reviewed term-containing postings, not unobserved synonyms or the whole corpus.',
              'decision': 'Do not adopt: tiny, development-inspected, non-independent sample; production rules remain frozen.',
              'proposed_exclusions': PROPOSED_EXCLUSIONS, 'source_sha256': before}
    if before != {str(p): digest(p) for p in sources}:
        raise RuntimeError('An input changed during comparison')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    pd.DataFrame(predictions).to_csv(out/'predictions.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
