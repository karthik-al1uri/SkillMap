"""Split feasibility must preserve groups and distinguish the holdout estimands."""
import numpy as np
import pandas as pd
import pytest
from scripts.check_grouped_feasibility import description_groups, normalized_employers, grouped_design, run


def jobs(n=90):
    return pd.DataFrame({'job_id':range(n), 'salary_tier':np.tile(['Low','Mid','High'],n//3),
                         'company_name':[f'employer{i//3}' for i in range(n)]})


def test_normalization_and_missing_groups_are_individual():
    data=jobs(6)
    data['company_name']=[' Example  INC ', 'example inc',None,'', 'OTHER', ' Other ']
    groups=normalized_employers(data)
    assert groups[0]==groups[1] and groups[2]!=groups[3] and groups[4]==groups[5]
    source=pd.DataFrame({'job_id':[0,1,2], 'normalized_description_group':['x','x',np.nan]})
    descriptions=description_groups(data,source)
    assert descriptions[0]==descriptions[1]
    assert len(set(descriptions[2:]))==4
    with pytest.raises(ValueError):
        description_groups(data,pd.concat([source,source]))


def test_outer_and_inner_groups_disjoint_and_assignments_stable():
    data=jobs()
    groups=pd.Series([f'g{i//3}' for i in range(len(data))])
    first,a,inner=grouped_design(data,groups,groups)
    second,b,inner2=grouped_design(data,groups,groups)
    assert first==second
    assert np.array_equal(a,b) and np.array_equal(inner,inner2)
    for fold in [first['outer_selected'],*first['outer_all_folds'],*first['inner_folds']]:
        assert fold['shared_groups']==0 and fold['all_classes_present']
        assert fold['shared_normalized_description_groups']==0
    assert set(inner[a=='train'])=={0,1,2}
    assert set(inner[a=='test'])=={-1}


def test_existing_and_canonical_output_refused(tmp_path):
    root=tmp_path/'root';root.mkdir()
    for out in (root/'new',root,tmp_path):
        with pytest.raises(ValueError):
            run(root,tmp_path/'absent.csv',out)


def test_memberships_equal_explicit_canonical_numeric_sgkf():
    from sklearn.model_selection import StratifiedGroupKFold
    from scripts.check_grouped_feasibility import TIER_TO_CODE
    data = jobs(150)
    groups = pd.Series([f'g{i//2}' for i in range(len(data))])
    report, assignment, inner = grouped_design(data, groups, groups)
    y = data.salary_tier.map({'Low': 0, 'Mid': 1, 'High': 2}).to_numpy(dtype=int)
    expected_train, expected_test = next(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(data, y, groups))
    assert np.array_equal(np.flatnonzero(assignment == 'test'), expected_test)
    assert np.array_equal(np.flatnonzero(assignment == 'train'), expected_train)
    for fold, (_, valid) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=2027).split(expected_train, y[expected_train], groups.iloc[expected_train])):
        assert np.array_equal(np.flatnonzero(inner == fold), expected_train[valid])
    assert report['label_encoding'] == TIER_TO_CODE
