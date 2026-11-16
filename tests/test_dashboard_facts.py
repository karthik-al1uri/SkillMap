"""Facts stay at job grain, preserve evidence status, and reject bad provenance metadata."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.build_dashboard_facts import Sources, build, validate_grouped


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "project"
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    (root / "outputs").mkdir()
    jobs = pd.DataFrame([dict(job_id="1", job_title="many skills", company_name="A", location="CO",
                              experience_level="Entry", matched_skills="['a', 'b', 'c', 'a']",
                              salary_tier="Low", normalized_salary=100.),
                         dict(job_id="2", job_title="zero skills", company_name="B", location="CA",
                              experience_level="Senior", matched_skills="[]",
                              salary_tier="High", normalized_salary=100000.)])
    jobs.to_csv(processed / "cleaned_jobs.csv", index=False)
    pd.DataFrame({"skill": ["a", "b", "c"]}).to_csv(processed / "skill_vocabulary.csv", index=False)
    pd.DataFrame([dict(model="historical winner", f1_macro=.99, roc_auc_ovr=.99)]).to_csv(root / "outputs/05_classification_results.csv", index=False)
    jobs[["job_id", "job_title", "salary_tier", "normalized_salary"]].assign(kmeans_cluster=0, kmeans_cluster_name="Example", dbscan_cluster=0).to_csv(root / "outputs/06_cluster_labels.csv", index=False)
    pd.DataFrame([dict(id="1", company_name="A", publication_date="2026-01-01", fetch_date="2026-10-14", matched_skills="['a']")]).to_csv(processed / "muse_cleaned_jobs.csv", index=False)
    pd.DataFrame({"fetch_date": ["2026-11-01"]}).to_csv(processed / "live_cleaned_jobs.csv", index=False)
    (root / "src").mkdir()
    (root / "src/classification.py").write_text("F1_TARGET = .75\nAUC_TARGET = .80\n")
    (root / "src/clustering.py").write_text("SILHOUETTE_TARGET = .50\n")
    (root / "docs").mkdir()
    (root / "docs/phase6b_classification_contract.md").write_text("Freeze Low <= $62,400; Mid > $62,400 and <= $109,200; High > $109,200.\n")
    return root


def test_unique_job_salary_not_skill_weighted_and_hashes(root, tmp_path):
    out = tmp_path / "facts.json"
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    bundle = build(root, out)
    assert bundle["schema"] == "skillmap.dashboard.v1"
    assert bundle["tables"]["cap"]["source_ids"]
    assert bundle["metrics"]["median_salary"]["data"] == 50050.
    assert bundle["metrics"]["salary_summary"]["data"]["mean"] == 50050.
    assert len(bundle["tables"]["jobs"]["data"]) == 2
    assert len(bundle["tables"]["job_skills"]["data"]) == 3
    assert bundle["metrics"]["mean_skills"]["data"] == 1.5
    assert bundle["metrics"]["zero_skills"]["data"] == 1
    assert bundle["metrics"]["skill_counts"]["data"] == {"a": 1, "b": 1, "c": 1}
    assert bundle["metrics"]["live_snapshot_dates"]["data"] == ["2026-10-14", "2026-11-01"]
    assert bundle["metrics"]["model_targets"]["data"] == {"f1_macro": .75, "roc_auc": .8, "silhouette": .5}
    assert bundle["metrics"]["salary_tier_boundaries"]["data"] == {"low_max": 62400, "mid_max": 109200}
    assert bundle["provenance"]["source_verification"]["all_unchanged"]
    for name in ["tables", "metrics", "evaluations"]:
        for value in bundle[name].values():
            assert {"available", "status", "source_ids", "data"} <= set(value)
            assert all(source_id in bundle["sources"] for source_id in value["source_ids"])
    for source in bundle["sources"].values():
        path = Path(source["path"])
        assert source["bytes"] == path.stat().st_size
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert json.loads(out.read_text())["metrics"]["median_salary"]["data"] == 50050.


def test_missing_grouped_never_uses_historical_scores(root, tmp_path):
    result = build(root, tmp_path / "facts.json", grouped=tmp_path / "missing")
    historical = result["tables"]["05_classification_results"]
    assert historical["available"] and historical["status"] == "historical_exploratory"
    assert historical["data"][0]["f1_macro"] == .99
    assert result["evaluations"]["grouped"]["data"] is None
    assert not result["evaluations"]["grouped"]["available"]
    assert result["evaluations"]["live"]["data"] is None


def grouped_data():
    rows = [dict(candidate="first", mean_f1_macro=.6, fold_f1_macro=[.5,.7]),
            dict(candidate="second", mean_f1_macro=.6, fold_f1_macro=[.6,.6])]
    report = dict(status="complete", winner="first", cv_rows=rows, outer_metrics={"f1_macro": .4})
    design = dict(candidate_order=["first", "second"])
    selection = dict(winner="first", rows=rows)
    return report, design, selection, pd.DataFrame(rows)


def test_training_cv_winner_tie_order_and_reject_inconsistent_report():
    report, design, selection, rows = grouped_data()
    assert validate_grouped(report, design, selection, rows) == "first"
    report["winner"] = "second"
    with pytest.raises(ValueError, match="winner"):
        validate_grouped(report, design, selection, rows)


def test_grouped_scores_remain_separate_from_historical(root, tmp_path):
    path = tmp_path / "grouped"
    path.mkdir()
    report, design, selection, rows = grouped_data()
    for name, value in [("report.json", report), ("frozen_design.json", design), ("cv_selection.json", selection)]:
        (path / name).write_text(json.dumps(value))
    rows.to_csv(path / "cv_selection.csv", index=False)
    facts = build(root, tmp_path / "facts.json", grouped=path)
    grouped = facts["evaluations"]["grouped"]
    assert grouped["status"] == "grouped_reused_data_evaluation"
    assert grouped["data"]["validated_training_cv_winner"] == "first"
    assert grouped["data"]["report"]["outer_metrics"]["f1_macro"] == .4
    assert facts["tables"]["05_classification_results"]["data"][0]["f1_macro"] == .99


@pytest.mark.parametrize("change", ["duplicate_job", "nonfinite_salary", "duplicate_cluster", "salary_inflation"])
def test_reject_bad_job_or_cluster_grain(root, tmp_path, change):
    jobs_path = root / "data/processed/cleaned_jobs.csv"
    cluster_path = root / "outputs/06_cluster_labels.csv"
    jobs, clusters = pd.read_csv(jobs_path), pd.read_csv(cluster_path)
    if change == "duplicate_job":
        pd.concat([jobs, jobs.iloc[:1]]).to_csv(jobs_path, index=False)
    elif change == "nonfinite_salary":
        jobs.loc[0, "normalized_salary"] = float("inf")
        jobs.to_csv(jobs_path, index=False)
    elif change == "duplicate_cluster":
        pd.concat([clusters, clusters.iloc[:1]]).to_csv(cluster_path, index=False)
    else:
        clusters.loc[0, "normalized_salary"] = 200.
        clusters.to_csv(cluster_path, index=False)
    with pytest.raises(ValueError):
        build(root, tmp_path / "facts.json")
    assert not (tmp_path / "facts.json").exists()


def test_sources_detect_changes_after_read(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"x":1}')
    sources = Sources()
    sources.read(path, "artifact_derived", "json")
    path.write_text('{"x":2}')
    with pytest.raises(ValueError, match="changed"):
        sources.verify()


def test_output_guards(root, tmp_path):
    with pytest.raises(ValueError):
        build(root, root / "facts.json")
    out = tmp_path / "existing.json"
    out.write_text("preserved")
    with pytest.raises(FileExistsError):
        build(root, out)
    assert out.read_text() == "preserved"
