"""Independent arithmetic and overlap semantics for read-only quality diagnostics."""
import numpy as np
import pandas as pd
from scripts.check_data_quality import normalized_description, overlap_report, salary_checks


def test_normalization_and_cross_split_denominators():
    assert normalized_description('A <b>JOB</b> &amp; Skills') == 'a job & skills'
    report = overlap_report(['a', 'a', 'b', '', 'b'], [0, 2], [1, 3, 4])
    assert report == {'nonempty_rows':4, 'repeated_groups':2, 'rows_in_repeated_groups':4,
                      'excess_rows':2, 'cross_split_groups':2,
                      'test_rows_with_train_group':2, 'largest_group':2}


def test_annualization_retains_vendor_values_and_unknown_period():
    raw = pd.DataFrame({'job_id':range(8), 'min_salary':[20.]*8, 'max_salary':[40.]*8,
                        'med_salary':[np.nan]*8,
                        'pay_period':['HOURLY','WEEKLY','BIWEEKLY','MONTHLY','YEARLY','UNKNOWN','HOURLY','YEARLY'],
                        'normalized_salary':[np.nan]*6+[12345., 30.]})
    cleaned = pd.DataFrame({'job_id':[6, 7], 'normalized_salary':[12345., 30.], 'salary_tier':['Low','Low']})
    result = salary_checks(raw, cleaned)
    assert result['implementation_matches_independent_formula']
    assert result['annualization_fills'] == 5
    assert result['vendor_vs_midpoint_mismatches_over_1cent'] == 1
    assert result['cleaned_vs_raw_normalized_mismatches'] == 0
    assert result['saved_tier_mismatches_fixed_boundaries'] == 0


def test_tier_provenance_uses_exact_thirds():
    raw = pd.DataFrame({'job_id':range(4), 'min_salary':[10000.,62400.,109200.,200000.],
                        'max_salary':[10000.,62400.,109200.,200000.], 'med_salary':[np.nan]*4,
                        'pay_period':['YEARLY']*4, 'normalized_salary':[10000.,62400.,109200.,200000.]})
    cleaned = pd.DataFrame({'job_id':range(4), 'normalized_salary':raw.normalized_salary,
                            'salary_tier':['Low','Low','Mid','High']})
    report = salary_checks(raw, cleaned)
    assert report['source_quantile_probabilities'] == [1/3, 2/3]
    assert report['saved_salary_exact_tercile_thresholds'] == [62400., 109200.]
    assert report['decimal_33_66_percent_thresholds_for_comparison'] != [62400.,109200.]


def test_report_refuses_canonical_or_existing_output(tmp_path):
    import pytest
    from scripts.check_data_quality import run
    root = tmp_path / 'canonical'
    root.mkdir()
    protected = root / 'outputs'
    protected.mkdir()
    existing = tmp_path / 'existing'
    existing.mkdir()
    sentinel = existing / 'sentinel'
    sentinel.write_text('preserved')
    for destination in (root, protected / 'new', existing):
        with pytest.raises(ValueError):
            run(root, destination)
    assert sentinel.read_text() == 'preserved'
    assert not (protected / 'new').exists()
