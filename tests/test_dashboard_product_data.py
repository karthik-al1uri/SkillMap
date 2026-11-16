"""Product data preserves job grain and derives only descriptive counts from facts."""
import copy
import json

import pytest
from src.dashboard_data import build_product_data


def wrap(data, status="artifact_derived"):
    return {"available": data is not None, "status": status, "source_ids": ["source_001"], "data": data}


@pytest.fixture
def facts():
    jobs = [dict(job_id="a", normalized_salary=100., salary_tier="Low", location="Boulder, CO", matched_skills=["python", "sql", "python"]),
            dict(job_id="b", normalized_salary=100000., salary_tier="High", location="Denver, CO", matched_skills=["python"]),
            dict(job_id="c", normalized_salary=900., salary_tier="Mid", location=None, matched_skills=[])]
    labels = [dict(job_id="a", normalized_salary=100., salary_tier="Low", kmeans_cluster=7, kmeans_cluster_name="Observed Tech"),
              dict(job_id="b", normalized_salary=100000., salary_tier="High", kmeans_cluster=7, kmeans_cluster_name="Observed Tech"),
              dict(job_id="c", normalized_salary=900., salary_tier="Mid", kmeans_cluster=-1, kmeans_cluster_name="No matched skills")]
    return {"schema": "skillmap.dashboard.v1", "generated_utc": "saved", "sources": {"source_001": {"path": "do-not-open.csv", "status": "artifact_derived"}},
            "provenance": {"job_grain": "unique_job_id"}, "caveats": ["cached"],
            "metrics": {"n_jobs": wrap(3), "skill_counts": wrap({"python": 2, "sql": 1, "absent": 0}),
                        "median_salary": wrap(900.), "missing": wrap(None)},
            "tables": {"jobs": wrap(jobs), "06_cluster_labels": wrap(labels),
                       "08_skill_comparison": wrap([dict(skill="python", is_technical=True), dict(skill="sql", is_technical="False")]),
                       "05_classification_results": wrap([{"model": "historical", "f1_macro": .99}], "historical_exploratory"),
                       "04_association_rules": wrap([])},
            "evaluations": {"grouped": wrap(None), "live": wrap(None), "clustering": wrap(None)}}


def test_job_grain_salary_and_roles_not_weighted_by_skills(facts):
    payload = build_product_data(facts)
    assert payload["metrics"]["median_salary"] == 900.
    tech = next(role for role in payload["roles"] if role["id"] == 7)
    assert tech["count"] == 2 and tech["median_salary"] == 50050.
    assert tech["high_share"] == .5
    assert tech["top_skills"][0] == dict(skill="python", index=0, count=2, share=1., overall_share=2/3)
    assert len(payload["rows"]) == 3
    assert payload["rows"][0][3] == [0, 1]
    assert payload["roles"][payload["rows"][0][1]]["name"] == "Observed Tech"
    assert payload["roles"][payload["rows"][2][1]]["name"] == "No matched skills"
    assert payload["skills"][2]["is_technical"] is None
    json.dumps(payload, allow_nan=False)


def test_network_actual_counts_determinism_technical_flags_and_neighbors(facts):
    first = build_product_data(facts)
    assert first == build_product_data(facts)
    edges = first["networks"]["all"]["edges"]
    assert edges == [dict(source=0, target=1, count=1, jaccard=.5)]
    assert [node["id"] for node in first["networks"]["technical"]["nodes"]] == [0]
    assert first["networks"]["technical"]["edges"] == []
    assert first["skill_neighbors"]["0"] == [dict(skillIndex=1, count=1, share_among_selected=.5)]
    for node in first["networks"]["all"]["nodes"]:
        assert first["roles"][node["role"]]["name"] == "Observed Tech"


@pytest.mark.parametrize("change", ["duplicate_job", "infinite_salary", "duplicate_cluster", "salary_mismatch", "skill_count_mismatch"])
def test_reject_duplicate_or_inconsistent_salary_grain(facts, change):
    jobs = facts["tables"]["jobs"]["data"]
    labels = facts["tables"]["06_cluster_labels"]["data"]
    if change == "duplicate_job": jobs.append(copy.deepcopy(jobs[0]))
    elif change == "infinite_salary": jobs[0]["normalized_salary"] = float("inf")
    elif change == "duplicate_cluster": labels.append(copy.deepcopy(labels[0]))
    elif change == "salary_mismatch": labels[0]["normalized_salary"] = 10.
    else: facts["metrics"]["skill_counts"]["data"]["python"] = 3
    with pytest.raises(ValueError): build_product_data(facts)


def test_missing_optional_evaluations_do_not_invent_metrics(facts):
    payload = build_product_data(facts)
    assert payload["models"]["grouped"] is None
    assert payload["models"]["historical"][0]["f1_macro"] == .99
    assert payload["live"]["evaluation"] is None
    assert payload["robustness"] is None
    assert payload["metrics"]["missing"] is None
    assert not payload["availability"]["evaluations"]["grouped"]["available"]
    assert "data" not in payload["provenance"]["table_provenance"]["jobs"]


def test_grouped_cv_winner_uses_frozen_tie_order_and_rejects_conflict(facts):
    rows = [dict(candidate="first", mean_f1_macro=.6), dict(candidate="second", mean_f1_macro=.6)]
    facts["evaluations"]["grouped"] = wrap({"report": {"winner": "first", "cv_rows": rows},
        "frozen_design": {"candidate_order": ["first", "second"]},
        "cv_selection": {"winner": "first", "rows": rows}, "validated_training_cv_winner": "first"})
    assert build_product_data(facts)["models"]["grouped"]["report"]["winner"] == "first"
    facts["evaluations"]["grouped"]["data"]["report"]["winner"] = "second"
    with pytest.raises(ValueError, match="winner"): build_product_data(facts)


def test_empty_jobs_and_unknown_saved_roles(facts):
    facts["tables"]["06_cluster_labels"] = wrap(None)
    payload = build_product_data(facts)
    assert len(payload["roles"]) == 1 and payload["roles"][0]["name"] == "Unknown"
    assert all(row[1] == 0 for row in payload["rows"])
    empty = {"schema": "skillmap.dashboard.v1", "tables": {"jobs": wrap([])},
             "metrics": {"n_jobs": wrap(0), "skill_counts": wrap({})}, "evaluations": {}}
    payload = build_product_data(empty)
    assert payload["rows"] == [] and payload["roles"] == []
    assert payload["networks"]["all"] == {"nodes": [], "edges": []}
