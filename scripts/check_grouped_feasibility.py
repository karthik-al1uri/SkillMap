"""Independent split feasibility only: no feature fitting or model training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

LABELS = ('Low', 'Mid', 'High')
TIER_TO_CODE = {'Low': 0, 'Mid': 1, 'High': 2}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_employers(jobs):
    """Names are proxies for employers, not verified legal-entity identities."""
    names = jobs.company_name.fillna('').astype(str).str.casefold().str.split().str.join(' ')
    return pd.Series([('employer:' + name) if name else ('missing-employer:' + str(job_id))
                      for name, job_id in zip(names, jobs.job_id)], index=jobs.index)


def description_groups(jobs, source):
    if jobs.job_id.duplicated().any() or source.job_id.duplicated().any():
        raise ValueError('job_id must be unique in both sources')
    series = jobs.job_id.map(source.set_index('job_id').normalized_description_group).fillna('')
    return pd.Series([str(key) if str(key).strip() else ('singleton-job:' + str(job_id))
                      for key, job_id in zip(series, jobs.job_id)], index=jobs.index)


def counts(y):
    result = pd.Series(y).value_counts()
    return {label:int(result.get(label, 0)) for label in LABELS}


def split_stats(jobs, groups, train, test, description=None):
    train_groups, test_groups = set(groups.iloc[train]), set(groups.iloc[test])
    sizes = groups.value_counts()
    result = {'train_rows':len(train), 'test_rows':len(test),
              'train_class_counts':counts(jobs.salary_tier.iloc[train]),
              'test_class_counts':counts(jobs.salary_tier.iloc[test]),
              'train_groups':len(train_groups), 'test_groups':len(test_groups),
              'shared_groups':len(train_groups & test_groups),
              'all_classes_present':all(v > 0 for v in counts(jobs.salary_tier.iloc[train]).values()) and all(v > 0 for v in counts(jobs.salary_tier.iloc[test]).values()),
              'groups_total':len(sizes), 'largest_group_rows':int(sizes.max()),
              'largest_group_fraction':float(sizes.max()/len(jobs)),
              'group_size_quantiles':{str(q):float(sizes.quantile(q)) for q in (.5,.9,.99,1.)},
              'test_job_ids_sha256_sorted_json':hashlib.sha256(json.dumps(sorted(jobs.job_id.iloc[test].tolist())).encode()).hexdigest()}
    if description is not None:
        shared = set(description.iloc[train]) & set(description.iloc[test])
        # Missing descriptions have unique groups and therefore do not produce false overlap.
        result['shared_normalized_description_groups'] = len(shared)
        result['test_rows_sharing_description_with_train'] = int(description.iloc[test].isin(shared).sum())
    return result


def grouped_design(jobs, groups, description=None):
    y = jobs.salary_tier.map(TIER_TO_CODE)
    if y.isna().any():
        raise ValueError('Unknown salary tier; canonical coding is required')
    y = y.to_numpy(dtype=int)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    outer_splits = list(outer.split(jobs, y, groups))
    train, test = outer_splits[0]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2027)
    training = jobs.iloc[train].reset_index(drop=True)
    training_groups = groups.iloc[train].reset_index(drop=True)
    training_description = description.iloc[train].reset_index(drop=True) if description is not None else None
    inner_splits = list(inner.split(training, y[train], training_groups))
    report = {'outer_n_splits':5, 'outer_seed':2026, 'selected_outer_fold':0,
              'inner_n_splits':3, 'inner_seed':2027,
              'label_encoding': TIER_TO_CODE,
              'group_encoding': 'Unprefixed normalized-description SHA strings; singleton-job:{job_id} fallback for missing descriptions.',
              'outer_selected':split_stats(jobs, groups, train, test, description),
              'outer_all_folds':[split_stats(jobs, groups, a, b, description) for a,b in outer_splits],
              'inner_folds':[split_stats(training, training_groups, a, b, training_description) for a,b in inner_splits]}
    assignments = np.full(len(jobs), 'test', dtype=object)
    assignments[train] = 'train'
    inner_validation_fold = np.full(len(jobs), -1, dtype=int)
    for fold, (_, valid) in enumerate(inner_splits):
        inner_validation_fold[train[valid]] = fold
    return report, assignments, inner_validation_fold


def run(root, groups_path, out, reference_manifest=None):
    root, groups_path, out = Path(root).resolve(), Path(groups_path).resolve(), Path(out).resolve()
    if out == root or root in out.parents or out.exists():
        raise ValueError('Require a fresh external output directory')
    cleaned = root/'data/processed/cleaned_jobs.csv'
    jobs = pd.read_csv(cleaned, usecols=['job_id','company_name','salary_tier'])
    if set(jobs.salary_tier) != set(LABELS):
        raise ValueError('Expected all three existing tier labels')
    source = pd.read_csv(groups_path, usecols=['job_id','normalized_description_group'])
    description = description_groups(jobs, source)
    employers = normalized_employers(jobs)
    grouped, assignment, inner = grouped_design(jobs, description, description)
    employer, employer_assignment, employer_inner = grouped_design(jobs, employers, description)
    old_train, old_test = train_test_split(np.arange(len(jobs)), test_size=.2, stratify=jobs.salary_tier, random_state=42)
    new_test = set(jobs.job_id[assignment=='test'])
    old_test_ids = set(jobs.job_id.iloc[old_test])
    report = {'scope':'Feasibility and split assignments only; no model, features, extraction or raw data loaded.',
              'description_grouped':grouped, 'employer_grouped_separate_estimand':employer,
              'historical_seed42':split_stats(jobs, description, old_train, old_test, description),
              'historical_vs_new_description_design':{
                  'old_test_rows':len(old_test), 'new_test_rows':len(new_test),
                  'shared_test_job_ids':len(old_test_ids & new_test),
                  'old_test_job_ids_now_in_training':len(old_test_ids-new_test),
                  'new_test_job_ids_previously_training':len(new_test-old_test_ids)},
              'limitations':['New folds reuse historically inspected data; split redesign does not create an untouched independent evaluation.',
                             'Employer-name normalization is a proxy; spelling aliases and parent/subsidiary identities are not resolved.',
                             'Description holdout does not isolate employers; employer holdout may retain repeated descriptions. These measure different generalization targets.',
                             'Numeric/URL-masked template candidates are not part of the frozen primary grouping definition.'],
              'sources':{str(p):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in (cleaned, groups_path, Path(__file__).resolve())}}
    report['description_grouped']['outer_selected']['shared_normalized_employer_groups'] = len(set(employers[assignment=='train']) & set(employers[assignment=='test']))
    if reference_manifest is not None:
        reference_path = Path(reference_manifest).resolve()
        reference = pd.read_csv(reference_path, dtype={'job_id': str})
        actual = pd.DataFrame({'job_id': jobs.job_id.astype(str), 'outer_role': assignment,
                               'inner_validation_fold': inner})
        checked = actual.merge(reference[['job_id', 'outer_role', 'inner_validation_fold']],
                               on='job_id', how='outer', suffixes=('_independent', '_classifier'),
                               indicator=True, validate='one_to_one')
        reference_test_ids = sorted(int(x) for x in reference.loc[reference.outer_role.eq('test'), 'job_id'])
        report['frozen_classifier_reconciliation'] = {
            'reference_manifest': str(reference_path), 'reference_sha256': sha256(reference_path),
            'missing_or_extra_job_ids': int(checked['_merge'].ne('both').sum()),
            'outer_assignment_mismatches': int(checked.outer_role_independent.ne(checked.outer_role_classifier).sum()),
            'inner_assignment_mismatches': int(checked.inner_validation_fold_independent.ne(checked.inner_validation_fold_classifier).sum()),
            'reference_test_ids_sha256_sorted_integer_json': hashlib.sha256(json.dumps(reference_test_ids).encode()).hexdigest(),
            'reason_for_run1_difference': 'Run1 supplied salary strings, whose internal alphabetical class order is High,Low,Mid; canonical integer coding Low0,Mid1,High2 changes deterministic SGKF tie resolution. Group-key representation now matches the frozen classifier manifest. No outcomes or fitted models read; no classifier assignments changed.'}
    out.mkdir(parents=True)
    assignments = pd.DataFrame({'job_id':jobs.job_id, 'normalized_description_group':description,
                               'normalized_employer_group':employers,'description_outer_partition':assignment,
                               'description_inner_validation_fold':inner, 'employer_outer_partition':employer_assignment,
                               'employer_inner_validation_fold':employer_inner})
    assignments.to_csv(out/'split_assignments.csv', index=False)
    (out/'feasibility.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path)
    args = parser.parse_args(argv)
    report = run(args.root,args.groups,args.out,args.reference_manifest)
    print(json.dumps({'description':report['description_grouped']['outer_selected'],
                      'employer':report['employer_grouped_separate_estimand']['outer_selected'],
                      'historical':report['historical_seed42'],
                      'overlap':report['historical_vs_new_description_design'],
                      'reconciliation': report.get('frozen_classifier_reconciliation')}))


if __name__ == '__main__':
    main()
