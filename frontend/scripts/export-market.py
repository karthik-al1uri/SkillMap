"""Export hash-verified saved job snapshots. No API, extraction, training or database access."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'frontend/public/data/market.json'
APPROVED_HTML = ROOT / 'dashboard_preview/dashboard-v2.html'
APPROVED_SHA256 = 'eddc4ca9399d2a09921d3bde8bc84205db2227d119085de0dfea8f00fb7b0b8b'
SOURCE_FILES = {'adzuna': 'live_cleaned_jobs.csv', 'muse': 'muse_cleaned_jobs.csv'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def date(value):
    from datetime import datetime
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Snapshot date must be an ISO calendar date')
    datetime.strptime(value, '%Y-%m-%d')
    return value


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Saved salary flags must be explicit True/False')
    return value == 'True'


def postings(raw, source):
    records = []
    seen = set()
    for row in csv.DictReader(io.StringIO(raw.decode('utf-8'))):
        collected = date(row['fetch_date'])
        key = row['id'], collected
        if not row['id'] or not row['job_title'] or key in seen:
            raise ValueError('Posting IDs must be unique within each source and collection date')
        seen.add(key)
        skills = ast.literal_eval(row['matched_skills'])
        if not isinstance(skills, list) or not all(isinstance(skill, str) and skill for skill in skills) or len(skills) != len(set(skills)):
            raise ValueError('Saved extracted skill mentions must be unique nonempty strings')
        salary, status, tier = None, 'unavailable', None
        if source == 'adzuna':
            available = boolean(row['has_salary'])
            predicted = boolean(row['salary_is_predicted'])
            if available:
                salary = float(row['mean_salary'])
                if not math.isfinite(salary) or salary < 0:
                    raise ValueError('Saved salary must be finite and nonnegative')
                status = 'estimated' if predicted else 'advertised'
                tier = row['salary_tier']
                if tier not in ('Low', 'Mid', 'High'):
                    raise ValueError('Available Adzuna salaries require their saved tier label')
            elif row['mean_salary'] or row['salary_tier']:
                raise ValueError('Unavailable salary conflicts with saved salary fields')
        records.append({'id': row['id'], 'title': row['job_title'], 'company': row['company_name'] or None,
                        'location': row['location'] or None, 'date': collected, 'skills': skills,
                        'salary': salary, 'salaryStatus': status, 'tier': tier})
    return records


def export(out, validate_only=False):
    out = Path(out).absolute()
    if out.is_symlink() or (out == DEFAULT_OUT and out.resolve() != out):
        raise ValueError('Export may not follow output symlinks')
    if out != DEFAULT_OUT and (ROOT in out.resolve().parents or out.exists()):
        raise ValueError('Only designated market.json may be replaced; external output must be new')
    html = APPROVED_HTML.read_bytes()
    if sha(html) != APPROVED_SHA256:
        raise ValueError('Saved dashboard does not match the approved Phase 3 SHA-256')
    match = re.search(r'<script type="application/json" id="skillmap-provenance">(.*?)</script>', html.decode(), re.S)
    if not match:
        raise ValueError('Approved dashboard has no embedded source registry')
    registry = json.loads(match.group(1))['sources']
    consumed = {str(APPROVED_HTML): (len(html), sha(html))}
    sources = {}
    for source, filename in SOURCE_FILES.items():
        path = ROOT / 'data/processed' / filename
        entries = [entry for entry in registry.values() if entry['path'] == str(path)]
        if len(entries) != 1:
            raise ValueError(f'Snapshot has no unique approved provenance record: {path}')
        recorded = entries[0]
        raw = path.read_bytes()
        if len(raw) != recorded['bytes'] or sha(raw) != recorded['sha256']:
            raise ValueError(f'Cached snapshot differs from approved provenance: {path}')
        consumed[str(path)] = (len(raw), sha(raw))
        sources[source] = {'label': 'Adzuna' if source == 'adzuna' else 'The Muse',
                           'postings': postings(raw, source),
                           'provenance': {key: recorded[key] for key in ['path', 'bytes', 'sha256']},
                           'limitations': [
                               'Cached collection, not real-time availability. Collection date is not posting publication date.',
                               'All saved cleaned postings are included, including zero-skill postings; no employer cap or minimum-skill eligibility filter.',
                               'Source coverage, geography and extraction differ from historical data; comparisons do not establish temporal demand trends.',
                               'Salary estimates are source estimates, not independently verified advertised compensation.' if source == 'adzuna'
                               else 'Salary was not saved for these Muse postings; compensation remains unavailable.',
                           ]}
    payload = {'schema': 'skillmap.market.v1', 'sources': sources,
               'provenance': {'approved_html': {'path': str(APPROVED_HTML), 'bytes': len(html), 'sha256': APPROVED_SHA256},
                              'source_checks_requested': True,
                              'execution': 'Existing cached CSVs only; no requests, skill extraction, experiments or database access',
                              'posting_grain': '(source, id, collection date)',
                              'salary_currency': 'USD', 'salary_period': 'annual',
                              'refresh_status': 'Manual refresh is not run or authorized by this application'}}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
    for name, expected in consumed.items():
        raw = Path(name).read_bytes()
        if (len(raw), sha(raw)) != expected:
            raise ValueError(f'Source changed during export: {name}')
    if validate_only:
        if not out.is_file() or out.read_bytes() != encoded.encode('utf-8'):
            raise ValueError('Existing market export differs from the hash-verified saved snapshots')
        return {'validated': str(out), 'sources_checked': len(consumed), 'sha256': sha(encoded.encode()),
                'postings': {source: len(value['postings']) for source, value in sources.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=out.parent, prefix='.market-export-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'out': str(out), 'sources_checked': len(consumed), 'sha256': sha(encoded.encode()),
            'postings': {source: len(value['postings']) for source, value in sources.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--validate', action='store_true', help='Verify the existing export without writing files')
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.out, args.validate)))
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        parser.error(str(error))
