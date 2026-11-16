"""Tiny fixed-data checks: distributions, bootstrap, thresholds and source guards."""
import numpy as np
import pandas as pd
import pytest
from scripts.validate_cached_live import (build, cap_companies, classification_metrics, company_groups,
                                         confusion, evaluate_muse, prediction_bootstrap, threshold_status)


def test_confusion_and_distribution_are_fixed_label_order():
    matrix = confusion(["Low", "Low", "High"], ["Low", "Mid", "Mid"])
    result = classification_metrics(matrix)
    assert result["accuracy"] == 1/3
    assert result["per_class_f1"] == {"Low": 2/3, "Mid": 0., "High": 0.}
    assert result["actual_counts"] == {"Low": 2, "Mid": 0, "High": 1}
    assert result["predicted_counts"] == {"Low": 1, "Mid": 2, "High": 0}
    assert result["per_class_recall"] == {"Low": .5, "Mid": 0., "High": 0.}


def test_prediction_cluster_bootstrap_is_deterministic_and_not_row_bootstrap():
    a = np.array(["Low"]*4 + ["High"]*4)
    p = np.array(["Low"]*4 + ["Low"]*4)
    groups = np.array(["one"]*4 + ["two"]*4)
    first = prediction_bootstrap(a, p, groups, n_boot=200, seed=42)
    assert first == prediction_bootstrap(a, p, groups, n_boot=200, seed=42)
    assert first["accuracy"]["low"] == 0.
    assert first["accuracy"]["high"] == 1.
    assert first["employer_groups"] == 2


def test_missing_employers_are_not_one_artificial_cluster():
    frame = pd.DataFrame({"company_name": [None, "", " Example  Corp ", "example corp"]})
    groups = company_groups(frame)
    assert len(set(groups)) == 3
    assert groups[2] == groups[3]


@pytest.mark.parametrize("support,confidence,lift,expected", [
    (.05, .6, 2., True), (.05, .5, 2., False), (.05, .6, 1.5, False), (.049, .6, 2., False)])
def test_rule_thresholds_use_unrounded_strict_criteria(support, confidence, lift, expected):
    assert threshold_status(dict(support=support, confidence=confidence, lift=lift), .05)[0] is expected


def test_eligible_denominator_and_undefined_bootstrap_metrics():
    frame = pd.DataFrame({"id": list("123456"), "company_name": ["A", "A", "B", "B", "C", "D"],
                          "matched_skills": [["a", "b"], ["a", "b"], ["x", "y"], ["x", "y"], [], ["a"]]})
    rules = pd.DataFrame([dict(antecedents="a", consequents="b", lift=2.)])
    rows, report = evaluate_muse(frame, rules, n_boot=200)
    row = rows.iloc[0]
    assert report["all_postings"] == 6 and report["eligible_postings"] == 4
    assert report["empty_baskets_all_postings"] == 1
    assert row["support"] == .5 and row["confidence"] == 1. and row["lift"] == 2.
    assert row["support_ci_low"] == 0. and row["support_ci_high"] == 1.
    assert row["lift_defined_replicates"] < 200
    assert row["lift_defined_replicates"] > 0
    again, _ = evaluate_muse(frame, rules, n_boot=200)
    pd.testing.assert_frame_equal(rows, again)


def test_company_cap_applies_before_eligibility_and_keeps_latest_then_id():
    frame = pd.DataFrame({"id": ["3", "2", "1"], "company_name": ["A"]*3,
                          "fetch_date": ["2026-10-14"]*3,
                          "publication_date": ["2026-01-01", "2026-02-01", "2026-02-01"],
                          "matched_skills": [["a", "b"], ["a", "b"], []]})
    result = cap_companies(frame, cap=1)
    assert result.id.tolist() == ["1"]
    assert result.matched_skills.tolist() == [[]]


def test_output_guard_runs_before_inputs(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError):
        build(root, root / "output")
    with pytest.raises(FileExistsError):
        build(root, tmp_path)
