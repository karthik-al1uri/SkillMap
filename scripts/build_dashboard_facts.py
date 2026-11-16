"""Build the sole, provenance-bearing JSON input for the SkillMap dashboard."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SCHEMA = "skillmap.dashboard.v1"
HISTORICAL = "historical_exploratory"
GROUPED = "grouped_reused_data_evaluation"
CACHED = "cached_live"
DERIVED = "artifact_derived"
CAP = 25
FROZEN = "frozen_contract"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Sources:
    """Hash each actually consumed file before reading, then verify it again at completion."""
    def __init__(self):
        self.records = {}
        self.by_path = {}

    def read(self, path, status, kind="csv", **kwargs):
        path = Path(path).resolve(strict=True)
        key = str(path)
        if key not in self.by_path:
            source_id = f"source_{len(self.records)+1:03d}"
            self.by_path[key] = source_id
            self.records[source_id] = {"path": key, "bytes": path.stat().st_size,
                                       "sha256": sha256(path), "status": status}
        source_id = self.by_path[key]
        if self.records[source_id]["status"] != status:
            raise ValueError(f"Conflicting source status for {path}")
        if kind == "json":
            value = json.loads(path.read_text())
        elif kind == "text":
            value = path.read_text()
        else:
            value = pd.read_csv(path, **kwargs)
        return value, source_id

    def verify(self):
        for value in self.records.values():
            path = Path(value["path"])
            if not path.is_file() or path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
                raise ValueError(f"Consumed source changed during fact generation: {path}")
        return {"consumed_files": len(self.records), "all_unchanged": True,
                "verification": "bytes and SHA-256 before/after; only actually consumed files are listed"}


def entry(data, source_ids, status, reason=None):
    result = {"available": data is not None, "status": status, "source_ids": list(source_ids),
              "data": clean_json(data)}
    if reason is not None:
        result["reason"] = reason
    return result


def missing(status, reason):
    return entry(None, [], status, reason)


def records(frame):
    return clean_json(frame.to_dict("records"))


def parse_skills(values):
    parsed = []
    for value in values:
        skills = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(skills, (list, tuple)) or not all(isinstance(s, str) for s in skills):
            raise ValueError("matched_skills must contain lists of strings")
        parsed.append(list(dict.fromkeys(skills)))
    return parsed


def parse_kmeans_sweep(text):
    match = re.search(r"K-Means sweep\s*\n=+\s*\n(.*?)\nChosen k", text, re.S)
    if not match:
        return None
    try:
        frame = pd.read_csv(io.StringIO(match.group(1)), sep=r"\s+")
        required = {"k", "inertia", "silhouette", "davies_bouldin"}
        if not required <= set(frame):
            return None
        return records(frame)
    except (ValueError, pd.errors.ParserError):
        return None


def validate_grouped(report, design, selection, csv_rows):
    """Reject any winner inconsistent with frozen candidate order and inner-CV scores."""
    order = design.get("candidate_order")
    rows = selection.get("rows")
    if not isinstance(order, list) or not order or len(set(order)) != len(order):
        raise ValueError("Grouped frozen candidate order is missing or duplicated")
    if not isinstance(rows, list) or len(rows) != len(order):
        raise ValueError("Grouped CV candidate rows do not match frozen order")
    by_candidate = {row["candidate"]: row for row in rows}
    if set(by_candidate) != set(order) or len(by_candidate) != len(rows):
        raise ValueError("Grouped CV candidate set differs from frozen design")
    scores = {}
    for candidate in order:
        score = float(by_candidate[candidate]["mean_f1_macro"])
        if not np.isfinite(score):
            raise ValueError("Grouped CV score is not finite")
        folds = by_candidate[candidate].get("fold_f1_macro")
        if folds is not None:
            fold_values = np.asarray(folds, dtype=float)
            if not len(fold_values) or not np.isfinite(fold_values).all() or not np.isclose(fold_values.mean(), score, rtol=0, atol=1e-12):
                raise ValueError("Grouped mean CV score disagrees with fold scores")
        scores[candidate] = score
    winner = max(order, key=lambda name: scores[name])
    if selection.get("winner") != winner or report.get("winner") != winner:
        raise ValueError("Grouped winner disagrees with maximum training CV or frozen tie order")
    report_rows = report.get("cv_rows")
    if not isinstance(report_rows, list) or len(report_rows) != len(order):
        raise ValueError("Grouped report CV rows missing")
    for values in [report_rows, csv_rows.to_dict("records")]:
        observed = {row["candidate"]: float(row["mean_f1_macro"]) for row in values}
        if set(observed) != set(scores) or len(values) != len(order):
            raise ValueError("Grouped selection metadata candidate sets differ")
        if any(not np.isclose(observed[c], scores[c], rtol=0, atol=1e-12) for c in order):
            raise ValueError("Grouped selection metadata scores differ")
    if report.get("status") != "complete":
        raise ValueError("Grouped evaluation is not complete")
    return winner


def load_grouped(path, sources):
    if path is None:
        return missing(GROUPED, "Optional grouped evaluation directory not provided")
    path = Path(path)
    names = {"report": "report.json", "frozen_design": "frozen_design.json",
             "cv_selection": "cv_selection.json", "cv_selection_rows": "cv_selection.csv"}
    absent = [name for name in names.values() if not (path / name).is_file()]
    if absent:
        return missing(GROUPED, "Missing optional grouped inputs: " + ", ".join(absent))
    values, ids = {}, []
    for key, name in names.items():
        value, source_id = sources.read(path / name, GROUPED, "csv" if name.endswith("csv") else "json")
        values[key] = value
        ids.append(source_id)
    values["validated_training_cv_winner"] = validate_grouped(values["report"], values["frozen_design"],
                                                             values["cv_selection"], values["cv_selection_rows"])
    values["cv_selection_rows"] = records(values["cv_selection_rows"])
    return entry(values, ids, GROUPED)


def load_optional_evaluation(path, sources, status, names):
    if path is None:
        return missing(status, "Optional evaluation directory not provided")
    path = Path(path)
    if not (path / names["report"]).is_file():
        return missing(status, "Optional report unavailable: " + str(path / names["report"]))
    values, ids = {}, []
    for key, name in names.items():
        if not (path / name).is_file():
            values[key] = missing(status, "Optional input unavailable: " + name)
            continue
        value, source_id = sources.read(path / name, status, "csv" if name.endswith("csv") else "json")
        values[key] = records(value) if isinstance(value, pd.DataFrame) else value
        ids.append(source_id)
    return entry(values, ids, status)


def optional_csv(path, sources, status):
    if not Path(path).is_file():
        return missing(status, "Saved input unavailable: " + str(path))
    frame, source_id = sources.read(path, status, dtype={"job_id": str, "id": str})
    return entry(records(frame), [source_id], status)


def validate_clusters(labels, jobs):
    if "job_id" not in labels or labels["job_id"].isna().any() or labels["job_id"].duplicated().any():
        raise ValueError("Cluster labels must have unique non-null job_id grain")
    joined = labels.merge(jobs[["job_id", "salary_tier", "normalized_salary"]], on="job_id", how="left",
                           validate="one_to_one", suffixes=("", "_job"), indicator=True)
    if joined["_merge"].ne("both").any():
        raise ValueError("Cluster label references an unknown job_id")
    if "normalized_salary" in labels:
        salary = pd.to_numeric(joined["normalized_salary"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(salary).all() or not np.allclose(salary, joined["normalized_salary_job"], rtol=0, atol=1e-6):
            raise ValueError("Cluster salary differs from unique-job salary")
    if "salary_tier" in labels and joined["salary_tier"].ne(joined["salary_tier_job"]).any():
        raise ValueError("Cluster tier differs from unique-job tier")
    return labels


def policy_metrics(root, sources):
    policy = {}
    files = [("classification", root / "src/classification.py"),
             ("clustering", root / "src/clustering.py"),
             ("contract", root / "docs/phase6b_classification_contract.md")]
    for key, path in files:
        if path.is_file():
            policy[key] = sources.read(path, FROZEN, "text")
    targets = {}
    ids = []
    for file_key, constant, metric_key in [("classification", "F1_TARGET", "f1_macro"),
                                          ("classification", "AUC_TARGET", "roc_auc"),
                                          ("clustering", "SILHOUETTE_TARGET", "silhouette")]:
        if file_key not in policy:
            continue
        text, source_id = policy[file_key]
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in node.targets):
                value = float(ast.literal_eval(node.value))
                if not np.isfinite(value):
                    raise ValueError("Frozen target must be finite")
                targets[metric_key] = value
                if source_id not in ids:
                    ids.append(source_id)
    model_targets = entry(targets, ids, FROZEN) if len(targets) == 3 else missing(FROZEN, "Reference target policy files/constants unavailable")
    boundaries = missing(FROZEN, "Frozen salary-tier contract unavailable")
    if "contract" in policy:
        text, source_id = policy["contract"]
        match = re.search(r"Freeze Low <= \$([\d,]+); Mid > \$[\d,]+ and <= \$([\d,]+); High", text)
        if match:
            boundaries = entry({"low_max": int(match.group(1).replace(",", "")),
                                "mid_max": int(match.group(2).replace(",", ""))}, [source_id], FROZEN)
        else:
            boundaries = missing(FROZEN, "Frozen salary-tier boundaries could not be parsed")
    return {"model_targets": model_targets, "salary_tier_boundaries": boundaries}


def build(root, out, rule_views=None, grouped=None, live_validation=None,
          cluster_robustness=None, salary_quality=None):
    root, out = Path(root).resolve(strict=True), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Facts output must be outside canonical input repository")
    if out.exists():
        raise FileExistsError("Facts output must name a new file")
    sources = Sources()
    _, generator_id = sources.read(Path(__file__), DERIVED, "text")
    jobs, jobs_id = sources.read(root / "data/processed/cleaned_jobs.csv", DERIVED, dtype={"job_id": str})
    if jobs["job_id"].isna().any() or jobs["job_id"].duplicated().any():
        raise ValueError("Jobs must have unique non-null job_id grain")
    salary = pd.to_numeric(jobs["normalized_salary"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(salary).all():
        raise ValueError("Every unique job must have finite normalized_salary")
    jobs["normalized_salary"] = salary
    jobs["matched_skills"] = parse_skills(jobs["matched_skills"])
    vocabulary, vocab_id = sources.read(root / "data/processed/skill_vocabulary.csv", DERIVED)
    if vocabulary["skill"].isna().any() or vocabulary["skill"].duplicated().any():
        raise ValueError("Vocabulary skills must be unique and non-null")
    known = set(vocabulary["skill"])
    if any(not set(skills) <= known for skills in jobs["matched_skills"]):
        raise ValueError("A job contains matched skills outside the saved vocabulary")
    pairs = [{"job_id": row.job_id, "skill_name": skill} for row in jobs.itertuples(index=False)
             for skill in row.matched_skills]
    counts = pd.Series([row["skill_name"] for row in pairs], dtype="object").value_counts()
    skill_counts = {skill: int(counts.get(skill, 0)) for skill in vocabulary["skill"]}
    n_skills = jobs["matched_skills"].map(len)
    tables = {"jobs": entry(records(jobs), [jobs_id], DERIVED),
              "job_skills": entry(pairs, [jobs_id, vocab_id], DERIVED),
              "cap": entry(CAP, [generator_id], DERIVED),
              "n_skills_vocab": entry(len(vocabulary), [vocab_id], DERIVED)}
    raw_metrics = {"n_jobs": len(jobs), "n_skills_vocab": len(vocabulary), "zero_skills": int(n_skills.eq(0).sum()),
                   "mean_skills": float(n_skills.mean()), "median_salary": float(np.median(salary)),
                   "tier_counts": jobs["salary_tier"].value_counts().to_dict(), "skill_counts": skill_counts,
                   "salary_summary": {"min": float(salary.min()), "max": float(salary.max()),
                                      "mean": float(salary.mean()), "median": float(np.median(salary))}}
    metrics = {name: entry(value, [vocab_id] if name == "n_skills_vocab" else [jobs_id, vocab_id], DERIVED)
               for name, value in raw_metrics.items()}
    metrics.update(policy_metrics(root, sources))
    rule_names = {"04_association_rules": "top10_lift.csv", "04_high_salary_rules": "top5_high_tier.csv",
                  "04_technical_rules_low_support": "technical_only.csv"}
    rule_report_id = None
    if rule_views is not None and (Path(rule_views) / "report.json").is_file():
        rule_report, rule_report_id = sources.read(Path(rule_views) / "report.json", DERIVED, "json")
        if rule_report.get("inputs_unchanged") is not True:
            raise ValueError("External rule-view report does not verify unchanged inputs")
        prior = rule_report.get("input_sha256", {}).get("data/processed/cleaned_jobs.csv")
        if prior is not None and prior != sources.records[jobs_id]["sha256"]:
            raise ValueError("External rule views refer to different cleaned jobs")
    for key, filename in rule_names.items():
        if rule_views is not None:
            if rule_report_id is None:
                tables[key] = missing(DERIVED, "External rule views require a verified report.json")
                continue
            tables[key] = optional_csv(Path(rule_views) / filename, sources, DERIVED)
            if tables[key]["available"]:
                tables[key]["source_ids"].append(rule_report_id)
        else:
            tables[key] = optional_csv(root / "outputs" / (key + ".csv"), sources, HISTORICAL)
    for key in ["05_classification_results", "06_cluster_labels", "08_skill_comparison", "08_rules_validation"]:
        path = root / "outputs" / (key + ".csv")
        if key == "06_cluster_labels" and path.is_file():
            labels, source_id = sources.read(path, HISTORICAL, dtype={"job_id": str})
            labels = validate_clusters(labels, jobs)
            tables[key] = entry(records(labels), [source_id, jobs_id], HISTORICAL)
        else:
            tables[key] = optional_csv(path, sources, HISTORICAL)
    summary_path = root / "outputs/06_summary.txt"
    if summary_path.is_file():
        summary, source_id = sources.read(summary_path, HISTORICAL, "text")
        parsed = parse_kmeans_sweep(summary)
        tables["kmeans_sweep"] = entry(parsed, [source_id], HISTORICAL,
                                       None if parsed is not None else "Saved sweep could not be parsed")
    else:
        tables["kmeans_sweep"] = missing(HISTORICAL, "Saved clustering summary unavailable")
    muse_path = root / "data/processed/muse_cleaned_jobs.csv"
    if muse_path.is_file():
        muse, source_id = sources.read(muse_path, CACHED, dtype={"id": str})
        muse["matched_skills"] = parse_skills(muse["matched_skills"])
        if muse.duplicated(["id", "fetch_date"]).any():
            raise ValueError("Muse snapshots must be unique by id and fetch_date")
        ordered = muse.sort_values(["publication_date", "id"], ascending=[False, True], na_position="last")
        kept = ordered.groupby(["fetch_date", "company_name"], dropna=False).head(CAP)
        snapshots = {str(date): records(frame.sort_values("id")) for date, frame in kept.groupby("fetch_date")}
        sizes = {str(date): int(len(frame)) for date, frame in muse.groupby("fetch_date")}
        tables["snapshots"] = entry(snapshots, [source_id], CACHED)
        tables["snapshot_sizes"] = entry(sizes, [source_id], CACHED)
        metrics["dates"] = entry(sorted(sizes), [source_id], CACHED)
    else:
        tables["snapshots"] = missing(CACHED, "Cached Muse cleaned snapshots unavailable")
        tables["snapshot_sizes"] = missing(CACHED, "Cached Muse cleaned snapshots unavailable")
        metrics["dates"] = missing(CACHED, "Cached Muse cleaned snapshots unavailable")
    date_ids = list(metrics["dates"]["source_ids"])
    all_live_dates = set(metrics["dates"]["data"] or [])
    adzuna_path = root / "data/processed/live_cleaned_jobs.csv"
    if adzuna_path.is_file():
        adzuna_dates, source_id = sources.read(adzuna_path, CACHED, usecols=["fetch_date"], dtype={"fetch_date": str})
        all_live_dates.update(adzuna_dates["fetch_date"].dropna().astype(str))
        date_ids.append(source_id)
    metrics["live_snapshot_dates"] = entry(sorted(all_live_dates), date_ids, CACHED) if date_ids else missing(CACHED, "No cached live dates available")
    evaluations = {
        "grouped": load_grouped(grouped, sources),
        "live": load_optional_evaluation(live_validation, sources, CACHED,
                 {"report": "report.json", "uncapped_fixed_rules": "muse_uncapped_fixed_rules.csv",
                  "capped_fixed_rules": "muse_company_capped_fixed_rules.csv",
                  "historical_saved_comparison": "historical_saved_comparison.csv"}),
        "clustering": load_optional_evaluation(cluster_robustness, sources, DERIVED,
                       {"report": "report.json", "summary": "summary.csv"}),
        "salary_quality": load_optional_evaluation(salary_quality, sources, DERIVED,
                           {"report": "data_quality.json"})}
    if evaluations["grouped"]["available"]:
        expected = evaluations["grouped"]["data"]["frozen_design"].get("source_hashes", {})
        referenced = expected.get(str((root / "data/processed/cleaned_jobs.csv").resolve()))
        if referenced is not None and referenced != sources.records[jobs_id]["sha256"]:
            raise ValueError("Grouped evaluation refers to different cleaned jobs")
    verification = sources.verify()
    bundle = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "sources": sources.records,
              "provenance": {"input_root": str(root), "generator_source_id": generator_id, "source_verification": verification,
                             "job_grain": "one row per unique job_id; salary summaries never join job-skill facts",
                             "job_skill_grain": "distinct (job_id, skill_name); zero-skill jobs remain in denominator",
                             "snapshot_cap": {"cap": CAP, "group": ["fetch_date", "company_name"],
                                              "order": "publication_date DESC then id ASC; no source rows overwritten"},
                             "optional_sources": {"rule_views": str(rule_views) if rule_views is not None else None,
                                                  "grouped": str(grouped) if grouped is not None else None,
                                                  "live_validation": str(live_validation) if live_validation is not None else None,
                                                  "cluster_robustness": str(cluster_robustness) if cluster_robustness is not None else None,
                                                  "salary_quality": str(salary_quality) if salary_quality is not None else None}},
              "tables": tables, "metrics": metrics, "evaluations": evaluations,
              "caveats": ["Historical saved classifier scores are exploratory; original test scores influenced earlier selection.",
                          "Grouped evaluation reuses development data and is not independent external validation.",
                          "Optional missing evaluations remain unavailable; historical scores never replace grouped results.",
                          "All cluster labels and robustness diagnostics are exploratory and do not prove discrete natural roles.",
                          "Cross-source skill shares and a single cached date do not establish temporal demand changes.",
                          "Salary tiers retain historical globally derived boundaries 62400/109200, not fold-learned labels.",
                          "External evaluation reports are reused without prediction, model scoring, training or re-mining.",
                          "Renderer must consume this bundle alone; source paths are provenance, not rendering inputs."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(clean_json(bundle), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rule-views", type=Path)
    parser.add_argument("--grouped", type=Path)
    parser.add_argument("--live-validation", type=Path)
    parser.add_argument("--cluster-robustness", type=Path)
    parser.add_argument("--salary-quality", type=Path)
    args = parser.parse_args(argv)
    bundle = build(args.root, args.out, args.rule_views, args.grouped, args.live_validation,
                   args.cluster_robustness, args.salary_quality)
    print(json.dumps({"schema": bundle["schema"], "n_jobs": bundle["metrics"]["n_jobs"]["data"],
                      "sources": len(bundle["sources"]), "out": str(args.out)}))


if __name__ == "__main__":
    main()
