"""Compact dashboard product data derived solely from a SkillMap facts bundle."""
from collections import Counter
from itertools import combinations
import math

import networkx as nx
import numpy as np

FACTS_SCHEMA = "skillmap.dashboard.v1"
PRODUCT_SCHEMA = "skillmap.dashboard.product.v1"
TIERS = ["Low", "Mid", "High"]
UNKNOWN_ROLE_ID = -2147483648


def _value(wrapper):
    if not isinstance(wrapper, dict) or not wrapper.get("available", False):
        return None
    return wrapper.get("data")


def _meta(wrapper):
    if not isinstance(wrapper, dict):
        return {"available": False, "status": "unavailable", "source_ids": [], "reason": "Not supplied"}
    return {key: value for key, value in wrapper.items() if key != "data"}


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite numeric data") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numeric data")
    return number


def _flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _grouped(data):
    if data is None:
        return None
    report = data.get("report")
    design = data.get("frozen_design")
    selector = data.get("cv_selection")
    if not all(isinstance(value, dict) for value in (report, design, selector)):
        raise ValueError("Available grouped evaluation requires report, design and CV selector")
    order = design.get("candidate_order")
    rows = selector.get("rows")
    if not isinstance(order, list) or not order or len(set(order)) != len(order) or not isinstance(rows, list):
        raise ValueError("Grouped candidate order or selector is invalid")
    scores = {row["candidate"]: _finite_number(row["mean_f1_macro"], "Grouped CV score") for row in rows}
    if len(rows) != len(order) or set(scores) != set(order):
        raise ValueError("Grouped candidates differ from frozen design")
    winner = max(order, key=lambda name: scores[name])
    if report.get("winner") != winner or selector.get("winner") != winner:
        raise ValueError("Grouped winner conflicts with training CV or candidate-order tie break")
    if data.get("validated_training_cv_winner", winner) != winner:
        raise ValueError("Grouped validated winner conflicts with selector")
    report_rows = report.get("cv_rows")
    if report_rows is not None:
        report_scores = {row["candidate"]: _finite_number(row["mean_f1_macro"], "Grouped report CV score") for row in report_rows}
        if len(report_rows) != len(order) or set(report_scores) != set(scores) or any(abs(report_scores[c]-scores[c]) > 1e-12 for c in order):
            raise ValueError("Grouped report scores conflict with selector")
    return {"report": report, "frozen_design": design, "cv_selection": selector,
            "validated_training_cv_winner": winner}


def _role_assignments(jobs, labels):
    by_job = {str(job["job_id"]): job for job in jobs}
    label_map, role_names = {}, {}
    for label in labels or []:
        job_id = str(label.get("job_id"))
        if job_id in label_map:
            raise ValueError("Duplicate cluster job_id would inflate salary joins")
        if job_id not in by_job:
            raise ValueError("Cluster label references unknown job_id")
        job = by_job[job_id]
        if "normalized_salary" in label:
            salary = _finite_number(label["normalized_salary"], "Cluster salary")
            if abs(salary-float(job["normalized_salary"])) > 1e-6:
                raise ValueError("Cluster salary does not match unique-job salary")
        if "salary_tier" in label and label["salary_tier"] != job["salary_tier"]:
            raise ValueError("Cluster tier does not match unique-job tier")
        raw_id = label.get("kmeans_cluster")
        if raw_id is None:
            role_id, name = UNKNOWN_ROLE_ID, "Unknown"
        else:
            numeric_id = _finite_number(raw_id, "Cluster id")
            if numeric_id != int(numeric_id):
                raise ValueError("Saved cluster id must be integral")
            role_id = int(numeric_id)
            name = label.get("kmeans_cluster_name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Saved cluster requires its observed role name")
        if role_id in role_names and role_names[role_id] != name:
            raise ValueError("Saved cluster id has conflicting role names")
        role_names[role_id] = name
        label_map[job_id] = role_id
    for job_id in by_job:
        if job_id not in label_map:
            label_map[job_id] = UNKNOWN_ROLE_ID
            role_names[UNKNOWN_ROLE_ID] = "Unknown"
    ids = sorted(role_names)
    index = {role_id: i for i, role_id in enumerate(ids)}
    return ids, role_names, [index[label_map[str(job["job_id"])]] for job in jobs]


def _network(selected, skills, pair_counts, role_skill_counts):
    selected = list(selected)
    edges = set()
    for source in selected:
        neighbors = []
        for target in selected:
            if source == target:
                continue
            key = (min(source, target), max(source, target))
            shared = pair_counts.get(key, 0)
            if shared:
                jaccard = shared / (skills[source]["count"] + skills[target]["count"] - shared)
                neighbors.append((target, shared, jaccard))
        neighbors.sort(key=lambda item: (-item[2], -item[1], item[0]))
        for target, _, _ in neighbors[:2]:
            edges.add((min(source, target), max(source, target)))
    graph = nx.Graph()
    graph.add_nodes_from(selected)
    edge_data = []
    for source, target in sorted(edges):
        shared = int(pair_counts[(source, target)])
        jaccard = shared / (skills[source]["count"] + skills[target]["count"] - shared)
        graph.add_edge(source, target, weight=jaccard)
        edge_data.append({"source": source, "target": target, "count": shared, "jaccard": jaccard})
    positions = nx.spring_layout(graph, seed=42, weight="weight", iterations=50) if selected else {}
    nodes = []
    for index in selected:
        memberships = [(role, int(counts[index])) for role, counts in enumerate(role_skill_counts)]
        role = min(memberships, key=lambda item: (-item[1], item[0]))[0] if memberships else None
        x, y = positions[index]
        nodes.append({"id": index, "label": skills[index]["name"], "count": skills[index]["count"],
                      "role": role, "x": round(float(x), 6), "y": round(float(y), 6)})
    return {"nodes": nodes, "edges": edge_data}


def build_product_data(facts):
    """Return compact plot/filter data without opening source paths or evaluating models."""
    if facts.get("schema") != FACTS_SCHEMA:
        raise ValueError("Expected skillmap.dashboard.v1 facts")
    tables, evaluations = facts.get("tables", {}), facts.get("evaluations", {})
    metrics = {key: _value(value) for key, value in facts.get("metrics", {}).items()}
    jobs = _value(tables.get("jobs"))
    jobs = [] if jobs is None else jobs
    ids = [str(job.get("job_id")) for job in jobs]
    if any(job.get("job_id") is None for job in jobs) or len(ids) != len(set(ids)):
        raise ValueError("Jobs must have unique non-null job_id grain")
    for job in jobs:
        _finite_number(job.get("normalized_salary"), "Job salary")
        if job.get("salary_tier") not in TIERS:
            raise ValueError("Job tier must be an observed Low/Mid/High label")
        if not isinstance(job.get("matched_skills"), list) or not all(isinstance(s, str) for s in job["matched_skills"]):
            raise ValueError("Jobs require saved matched-skill lists")
    if metrics.get("n_jobs") is not None and int(metrics["n_jobs"]) != len(jobs):
        raise ValueError("Job count conflicts with unique-job facts")
    vocabulary = metrics.get("skill_counts")
    if vocabulary is None:
        if jobs:
            raise ValueError("Available jobs require the facts skill-count vocabulary")
        vocabulary = {}
    if not isinstance(vocabulary, dict):
        raise ValueError("Skill counts must be an ordered mapping")
    skill_index = {name: i for i, name in enumerate(vocabulary)}
    technical = {}
    for row in _value(tables.get("08_skill_comparison")) or []:
        name, flag = row.get("skill"), _flag(row.get("is_technical"))
        if name in technical and technical[name] != flag:
            raise ValueError("Conflicting technical eligibility in saved comparison")
        technical[name] = flag
    skill_rows = []
    for job in jobs:
        if any(name not in skill_index for name in job["matched_skills"]):
            raise ValueError("Job skill is absent from the facts vocabulary")
        skill_rows.append(sorted({skill_index[name] for name in job["matched_skills"]}))
    actual_counts = Counter(index for row in skill_rows for index in row)
    skills = []
    for name, index in skill_index.items():
        count = int(vocabulary[name])
        if count < 0 or count != vocabulary[name] or count != actual_counts[index]:
            raise ValueError("Skill counts conflict with distinct job-skill facts")
        skills.append({"index": index, "name": name, "count": count,
                       "share": count/len(jobs) if jobs else None,
                       "is_technical": technical.get(name)})
    role_ids, role_names, role_indices = _role_assignments(jobs, _value(tables.get("06_cluster_labels")))
    role_skill_counts = np.zeros((len(role_ids), len(skills)), dtype=int)
    roles = []
    for role_index, role_id in enumerate(role_ids):
        member_indices = [i for i, value in enumerate(role_indices) if value == role_index]
        member_count = len(member_indices)
        for job_index in member_indices:
            role_skill_counts[role_index, skill_rows[job_index]] += 1
        top = sorted(range(len(skills)), key=lambda index: (-role_skill_counts[role_index, index], index))
        top = [index for index in top if role_skill_counts[role_index, index] > 0][:8]
        roles.append({"id": role_id, "name": role_names[role_id], "count": member_count,
                      "median_salary": float(np.median([float(jobs[i]["normalized_salary"]) for i in member_indices])),
                      "high_share": sum(jobs[i]["salary_tier"] == "High" for i in member_indices) / member_count,
                      "top_skills": [{"skill": skills[index]["name"], "index": index,
                                      "count": int(role_skill_counts[role_index,index]),
                                      "share": int(role_skill_counts[role_index,index])/member_count,
                                      "overall_share": skills[index]["share"]} for index in top]})
    location_values = [job.get("location") for job in jobs]
    if any(value is not None and not isinstance(value, str) for value in location_values):
        raise ValueError("Location must be a saved label or null")
    location_counts = Counter(location_values)
    locations = sorted(location_counts, key=lambda value: (-location_counts[value], value is None, value or ""))
    location_index = {label: index for index, label in enumerate(locations)}
    rows = [[float(job["normalized_salary"]), role_indices[i], location_index[location_values[i]],
             skill_rows[i], TIERS.index(job["salary_tier"])] for i, job in enumerate(jobs)]
    pair_counts = Counter()
    for row in skill_rows:
        pair_counts.update(combinations(row, 2))
    ranked = sorted(range(len(skills)), key=lambda index: (-skills[index]["count"], index))
    all_selected = [index for index in ranked if skills[index]["count"] > 0][:24]
    tech_selected = [index for index in ranked if skills[index]["count"] > 0 and skills[index]["is_technical"] is True][:24]
    networks = {"all": _network(all_selected, skills, pair_counts, role_skill_counts),
                "technical": _network(tech_selected, skills, pair_counts, role_skill_counts)}
    neighbors = {}
    for index in sorted(set(all_selected + tech_selected)):
        observed = []
        for other in ranked:
            if other == index:
                continue
            shared = pair_counts.get((min(index,other), max(index,other)), 0)
            if shared:
                observed.append({"skillIndex": other, "count": int(shared),
                                 "share_among_selected": shared / skills[index]["count"]})
        observed.sort(key=lambda row: (-row["count"], row["skillIndex"]))
        neighbors[str(index)] = observed[:6]
    grouped = _grouped(_value(evaluations.get("grouped")))
    live = _value(evaluations.get("live"))
    robustness = _value(evaluations.get("clustering"))
    availability = {"tables": {key: _meta(value) for key, value in tables.items()},
                    "metrics": {key: _meta(value) for key, value in facts.get("metrics", {}).items()},
                    "evaluations": {key: _meta(value) for key, value in evaluations.items()}}
    provenance = {"schema": facts["schema"], "generated_utc": facts.get("generated_utc"),
                  "sources": facts.get("sources", {}), "facts_provenance": facts.get("provenance", {}),
                  "table_provenance": availability["tables"], "metric_provenance": availability["metrics"],
                  "evaluation_provenance": availability["evaluations"], "caveats": facts.get("caveats", []),
                  "salary_grain": "unique_job_id", "row_columns": ["salary", "roleIndex", "locationIndex", "skillIndices", "tierIndex"],
                  "role_index": "rows roleIndex and network node role index into roles array; roles.id retains saved cluster id",
                  "technical_eligibility": "saved 08_skill_comparison is_technical flag only; absent/unknown flags remain null",
                  "network_definition": "Descriptive shared-posting counts over unique jobs, not Apriori/significance/certified communities; up to24 prevalence-ranked skills, strongest2 positive Jaccard neighbors per node union (max48 edges), layout seed42",
                  "neighbor_share_definition": "shared-posting count / unique jobs containing selected skill; neighbors may be outside graph",
                  "role_names": "saved K-Means archetype names; Unknown only for jobs without an assigned saved archetype"}
    return {"schema": PRODUCT_SCHEMA, "metrics": metrics, "roles": roles, "locations": locations,
            "skills": skills, "rows": rows, "tiers": list(TIERS), "networks": networks, "skill_neighbors": neighbors,
            "models": {"grouped": grouped, "historical": _value(tables.get("05_classification_results"))},
            "live": {"evaluation": live.get("report") if live else None,
                     "comparison": _value(tables.get("08_skill_comparison")),
                     "rules_validation": _value(tables.get("08_rules_validation")),
                     "fixed_rules_uncapped": live.get("uncapped_fixed_rules") if live else None,
                     "fixed_rules_capped": live.get("capped_fixed_rules") if live else None},
            "robustness": robustness.get("report") if robustness else None,
            "rules": {"general": _value(tables.get("04_association_rules")),
                      "high": _value(tables.get("04_high_salary_rules")),
                      "technical": _value(tables.get("04_technical_rules_low_support"))},
            "availability": availability, "provenance": provenance}
