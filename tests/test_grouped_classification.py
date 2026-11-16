"""Synthetic checks for split isolation, CV-only selection and fixed-prediction metrics."""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from scripts.evaluate_grouped_classification import (CANDIDATES, build_pipeline, choose_by_inner_cv,
    frozen_splits, group_bootstrap, main, metrics)


def labels_groups():
    groups=np.repeat(np.arange(60).astype(str),3)
    y=np.repeat(np.tile([0,1,2],20),3)
    return y,groups


def test_outer_inner_group_isolation_and_reproducibility():
    y,groups=labels_groups()
    train,test,folds=frozen_splits(y,groups)
    assert not set(groups[train]) & set(groups[test])
    assert sorted(np.concatenate([v for _,v in folds]).tolist()) == sorted(train.tolist())
    for fit,valid in folds:
        assert not set(groups[fit]) & set(groups[valid])
        assert not set(fit) & set(test)
        assert not set(valid) & set(test)
    again=frozen_splits(y,groups)
    np.testing.assert_array_equal(test,again[1])


def test_dummy_cv_selection_never_receives_outer_test_and_ties_are_ordered():
    y,groups=labels_groups()
    train,test,folds=frozen_splits(y,groups)
    observed=[]
    def fit(name,fold,fit,valid):
        assert not set(fit) & set(test) and not set(valid) & set(test)
        model=DummyClassifier(strategy="constant",constant=int(name)).fit(np.zeros((len(fit),1)),y[fit])
        prediction=model.predict(np.zeros((len(valid),1)))
        result=metrics(y[valid],prediction,model.predict_proba(np.zeros((len(valid),1))))
        observed.append((name,fold))
        return result["f1_macro"]
    winner,rows=choose_by_inner_cv(("0","1","2"),folds,fit)
    assert winner==max(rows,key=lambda r:r["mean_f1_macro"])["candidate"]
    assert len(observed)==9
    tied,_=choose_by_inner_cv(("first","second"),folds,lambda *args:.5)
    assert tied=="first"
    with pytest.raises(TypeError):
        choose_by_inner_cv(("0",),folds,fit,test_y=y[test])


def test_metrics_use_fixed_class_order_and_bootstrap_is_reproducible():
    y=np.array([0,0,1,1,2,2]); pred=np.array([0,1,1,1,2,0])
    probabilities=np.full((6,3),.1)
    probabilities[np.arange(6),pred]=.8
    result=metrics(y,pred,probabilities)
    assert result["confusion_matrix"]==[[1,1,0],[0,2,0],[1,0,1]]
    assert result["accuracy"]==pytest.approx(4/6)
    assert result["f1_macro"]==pytest.approx((.5+.8+2/3)/3)
    groups=np.array(["a","a","b","b","c","c"])
    first=group_bootstrap(y,pred,probabilities,groups,repetitions=10)
    assert first==group_bootstrap(y,pred,probabilities,groups,repetitions=10)
    assert first["repetitions"]==10
    assert first["intervals"]["accuracy"]["valid_repetitions"]==10
    with pytest.raises(ValueError):
        group_bootstrap(y,pred,probabilities,groups,201)


def test_same_feature_control_and_fold_local_vocabulary():
    from src import classification as cl
    n=18
    data=pd.DataFrame({"matched_skills":[["python"]]*n,"experience_level":["Unknown"]*n,
        "employee_count":[20]*n,"state":["CO"]*n,"industry":["Software"]*n,
        "job_title":["software engineer"]*n,"company_name":["example systems"]*n})
    model=build_pipeline(cl,"nominal_logistic",["python"])
    model.fit(data,np.tile([0,1,2],6))
    held=data.iloc[:3].copy()
    held["job_title"]="heldouttoken"
    held["state"]="ZZ"
    model.predict(held)
    assert "heldouttoken" not in model["features"].vectorizers_["job_title"].vocabulary_
    assert "ZZ" not in model["features"].top_["states"]
    assert CANDIDATES==("nominal_logistic","ordinal_logistic_it","fixed_xgboost")
    assert all(build_pipeline(cl,name,["python"])["features"].use_text for name in CANDIDATES)


def test_output_guard_refuses_canonical_and_existing(tmp_path):
    root=tmp_path / "root";root.mkdir()
    groups=tmp_path / "groups.csv";groups.write_text("job_id,normalized_description_group\n")
    for out in (root / "run",tmp_path):
        with pytest.raises(SystemExit):
            main(["--root",str(root),"--groups",str(groups),"--out-dir",str(out)])
