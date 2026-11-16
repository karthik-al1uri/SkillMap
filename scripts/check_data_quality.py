"""Read-only salary provenance and conservative description-overlap diagnostics."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src import preprocessing as pp


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_description(text):
    """Conservative equality key; numeric masking is a separate template diagnostic."""
    text = html.unescape(str(text or ''))
    text = re.sub(r'<[^>]+>', ' ', text)
    return ' '.join(text.lower().split())


def overlap_report(keys, train_indices, test_indices):
    """Count groups/rows shared across a frozen split, without fitting any model."""
    keys = pd.Series(keys).fillna('')
    valid = keys.ne('')
    counts = keys[valid].value_counts()
    train_keys = set(keys.iloc[list(train_indices)]) - {''}
    test = keys.iloc[list(test_indices)]
    crossing = train_keys & (set(test) - {''})
    return {'nonempty_rows': int(valid.sum()), 'repeated_groups': int(counts.gt(1).sum()),
            'rows_in_repeated_groups': int(counts[counts.gt(1)].sum()),
            'excess_rows': int((counts - 1).clip(lower=0).sum()),
            'cross_split_groups': len(crossing),
            'test_rows_with_train_group': int(test.isin(crossing).sum()),
            'largest_group': int(counts.max()) if len(counts) else 0}


def salary_checks(raw, cleaned):
    """Check the implementation and vendor normalization independently."""
    midpoint = raw[['min_salary', 'max_salary']].mean(axis=1).fillna(raw.med_salary)
    factors = raw.pay_period.str.upper().map({'HOURLY': 2080, 'WEEKLY': 52,
                                            'BIWEEKLY': 26, 'MONTHLY': 12, 'YEARLY': 1})
    computed = midpoint * factors
    annualized, fills = pp.annualize_salary(raw)
    expected = raw.normalized_salary.combine_first(computed)
    comparable = raw.normalized_salary.notna() & computed.notna()
    mismatch = comparable & ((raw.normalized_salary - computed).abs() > .011)
    tiers = pd.cut(cleaned.normalized_salary, [-np.inf, 62400, 109200, np.inf],
                   labels=['Low', 'Mid', 'High']).astype(str)
    by_id = raw.drop_duplicates('job_id').set_index('job_id')
    original = cleaned.job_id.map(by_id.normalized_salary)
    return {'raw_rows': len(raw), 'raw_missing_normalized': int(raw.normalized_salary.isna().sum()),
            'annualization_fills': fills,
            'implementation_matches_independent_formula': bool(np.allclose(annualized.normalized_salary,
                                                                           expected, equal_nan=True)),
            'vendor_vs_midpoint_comparable': int(comparable.sum()),
            'vendor_vs_midpoint_mismatches_over_1cent': int(mismatch.sum()),
            'vendor_vs_midpoint_mismatch_by_period': raw.loc[mismatch, 'pay_period'].value_counts().to_dict(),
            'cleaned_rows': len(cleaned),
            'cleaned_missing_raw_normalized': int(original.isna().sum()),
            'cleaned_vs_raw_normalized_mismatches': int(((cleaned.normalized_salary-original).abs() > .011).sum()),
            'saved_tier_mismatches_fixed_boundaries': int((tiers != cleaned.salary_tier).sum()),
            'saved_salary_exact_tercile_thresholds': cleaned.normalized_salary.quantile([1/3, 2/3]).tolist(),
            'decimal_33_66_percent_thresholds_for_comparison': cleaned.normalized_salary.quantile([.33, .66]).tolist(),
            'source_quantile_probabilities': list(pp.TIER_QUANTILES),
            'tier_counts': cleaned.salary_tier.value_counts().to_dict(),
            'raw_pay_period_counts': raw.pay_period.fillna('MISSING').value_counts().to_dict(),
            'tier_provenance': 'Stage 2 computes exact 1/3 and 2/3 quantiles (not decimal 0.33/0.66) after salary filtering, exact repost removal and any experience-median imputation, before the later model split. Existing cutoffs are dataset-derived, not training-fold-derived. Freeze 62400/109200 as task definitions for comparability; do not claim label thresholds were learned independently of held-out data.',
            'annualization_contract': 'Keep existing vendor normalized_salary; fill missing using mean of available min/max else median, multiplied by HOURLY 2080 / WEEKLY 52 / BIWEEKLY 26 / MONTHLY 12 / YEARLY 1. No currency conversion or implied work-hour adjustment.'}


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents or out.exists():
        raise ValueError('Use a fresh external output directory')
    out.mkdir(parents=True)
    cleaned_path = root/'data/processed/cleaned_jobs.csv'
    raw_path = root/'data/raw/linkedin_postings/postings.csv'
    cleaned = pd.read_csv(cleaned_path)
    ids = set(cleaned.job_id)
    numeric, descriptions = [], []
    columns = ['job_id', 'min_salary', 'max_salary', 'med_salary', 'pay_period',
               'normalized_salary', 'description', 'company_name', 'title']
    for chunk in pd.read_csv(raw_path, usecols=columns, chunksize=10000, low_memory=False):
        descriptions.append(chunk.loc[chunk.job_id.isin(ids), ['job_id', 'description', 'company_name', 'title']])
        numeric.append(chunk.drop(columns=['description', 'company_name', 'title']))
    raw = pd.concat(numeric, ignore_index=True)
    text = pd.concat(descriptions, ignore_index=True).drop_duplicates('job_id').set_index('job_id')
    text = text.reindex(cleaned.job_id).reset_index()
    train, test = train_test_split(np.arange(len(cleaned)), test_size=.2, stratify=cleaned.salary_tier, random_state=42)
    exact = text.description.fillna('')
    normalized = exact.map(normalized_description)
    masked = normalized.str.replace(r'https?://\S+', '<url>', regex=True).str.replace(r'\d+(?:[.,]\d+)*', '<num>', regex=True)
    company_title = text.company_name.fillna('').str.strip().str.lower() + '|' + text.title.fillna('').str.strip().str.lower()
    company_title = company_title.where(text.company_name.notna(), '')
    report = {'scope': 'All saved cleaned jobs and all raw postings salary rows; descriptions only retained for saved job_ids. No pipeline, model or canonical writes.',
              'salary': salary_checks(raw, cleaned),
              'descriptions_missing': int(text.description.isna().sum()),
              'overlap_historical_split_80_20_seed42': {
                  'exact_description': overlap_report(exact, train, test),
                  'normalized_description': overlap_report(normalized, train, test),
                  'numeric_url_masked_template_candidate': overlap_report(masked, train, test),
                  'company_title_candidate': overlap_report(company_title, train, test)},
              'limitations': ['Numeric/URL-masked equality and company/title groups are candidate grouping keys, not verified duplicate labels.',
                             'No exhaustive fuzzy pairwise template search; semantic near duplicates can remain.',
                             'Raw salaries.csv fallback and experience-median imputation are described by source, not rerun here; zero saved-vs-raw discrepancies would establish saved rows needed neither.',
                             'Historical split overlap is a leakage diagnostic only, not a new evaluation.'],
              'classification_contract': ['Freeze existing 245-skill vocabulary and salary boundaries unless a separately versioned experiment changes them.',
                                          'Predeclare grouped evaluation using at least exact/normalized description connected groups; assess employer holdout separately.',
                                          'Do not tune grouping, extraction, model families or thresholds on held-out labels.',
                                          'Fit feature preprocessing inside grouped training CV folds; preserve job_id and group assignments in split manifest.'],
              'sources': {str(p.relative_to(root)): {'bytes':p.stat().st_size, 'sha256':digest(p)} for p in
                          (raw_path, cleaned_path, root/'src/preprocessing.py', root/'outputs/02_summary.txt')}}
    (out/'data_quality.json').write_text(json.dumps(report, indent=2)+'\n')
    groups = pd.DataFrame({'job_id':cleaned.job_id, 'exact_description_group':exact.map(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else ''),
                           'normalized_description_group':normalized.map(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else ''),
                           'template_candidate_group':masked.map(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else ''),
                           'company_title_candidate_group':company_title.map(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else '')})
    groups.to_csv(out/'group_candidates.csv', index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.root, args.out)
    print(json.dumps({'salary': report['salary'], 'overlap':report['overlap_historical_split_80_20_seed42']}))


if __name__ == '__main__':
    main()
