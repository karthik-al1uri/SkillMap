"""Project existing verified facts into playground-only title/employer metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'frontend/public/data/playground.json'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def export(validate=False):
    product_raw = (ROOT / 'frontend/public/data/skillmap.json').read_bytes()
    product = json.loads(product_raw)
    source = product['provenance']['facts_source']
    raw = Path(source['path']).read_bytes()
    if sha(raw) != source['sha256'] or len(raw) != source['bytes']:
        raise ValueError('Existing facts bundle does not match verified export')
    facts = json.loads(raw)
    for ref in facts['sources'].values():
        content = Path(ref['path']).read_bytes()
        if sha(content) != ref['sha256'] or len(content) != ref['bytes']:
            raise ValueError('Facts source changed: ' + ref['path'])
    jobs = facts['tables']['jobs']['data']
    if len(jobs) != len(product['rows']) or len({str(row['job_id']) for row in jobs}) != len(jobs):
        raise ValueError('Unique-job metadata does not match compact export')
    metadata = []
    for job, row in zip(jobs, product['rows']):
        if job['normalized_salary'] != row[0] or set(job['matched_skills']) != {product['skills'][index]['name'] for index in row[3]} or job.get('location') != product['locations'][row[2]]:
            raise ValueError('Job order or fields conflict with compact export')
        if not isinstance(job['job_title'], str) or not job['job_title'].strip():
            raise ValueError('Saved normalized job title missing')
        metadata.append({'id':str(job['job_id']), 'title':job['job_title'], 'company':job.get('company_name')})
    payload = {'schema':'skillmap.playground.v1','jobs':metadata,'provenance':{
        'facts_sha256':source['sha256'],'product_sha256':sha(product_raw),
        'source_checks_requested':True,'salary_grain':'unique_job_id',
        'title_definition':'Saved normalized job_title; not a standardized occupation code'}}
    encoded = (json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n').encode()
    if validate:
        if TARGET.read_bytes() != encoded:
            raise ValueError('Playground export does not reproduce')
    else:
        if TARGET.is_symlink():
            raise ValueError('Refusing symlink output')
        with tempfile.NamedTemporaryFile(dir=TARGET.parent,delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary,TARGET)
    if sha(Path(source['path']).read_bytes()) != source['sha256']:
        raise ValueError('Facts changed during projection')
    print(json.dumps({'jobs':len(jobs),'sha256':sha(encoded),'validated':validate}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate',action='store_true')
    export(parser.parse_args().validate)
