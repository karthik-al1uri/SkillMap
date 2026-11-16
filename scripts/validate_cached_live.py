"""Descriptive cached live validation: fixed predictions/rules, no training or requests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ("Low", "Mid", "High")
PRIMARY_SUPPORT = .05
SENSITIVITY_SUPPORT = .02
MIN_CONFIDENCE = .5
MIN_LIFT = 1.5
MIN_SKILLS = 2
COMPANY_CAP = 25
BOOTSTRAPS = 200
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


def company_groups(frame):
    """Use normalized employers; missing employers form separate per-row groups."""
    names = frame["company_name"].fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return np.array([f"company:{name}" if name else f"missing-row:{i}" for i, name in enumerate(names)])


def confusion(actual, predicted):
    codes = {label: i for i, label in enumerate(LABELS)}
    if not set(actual) <= set(LABELS) or not set(predicted) <= set(LABELS):
        raise ValueError("Salary labels must be Low/Mid/High")
    matrix = np.zeros((3, 3), dtype=np.int64)
    for a, p in zip(actual, predicted):
        matrix[codes[a], codes[p]] += 1
    return matrix


def classification_metrics(matrix):
    matrix = np.asarray(matrix)
    true, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
    f1 = np.divide(2 * tp, true + predicted, out=np.zeros(3, dtype=float), where=true + predicted > 0)
    recall = np.divide(tp, true, out=np.zeros(3, dtype=float), where=true > 0)
    total = matrix.sum()
    return {"accuracy": float(tp.sum() / total) if total else None,
            "macro_f1": float(f1.mean()), "per_class_f1": dict(zip(LABELS, f1.tolist())),
            "per_class_recall": dict(zip(LABELS, recall.tolist())),
            "actual_counts": dict(zip(LABELS, true.astype(int).tolist())),
            "predicted_counts": dict(zip(LABELS, predicted.astype(int).tolist())),
            "actual_shares": dict(zip(LABELS, (true/total).tolist())) if total else {},
            "predicted_shares": dict(zip(LABELS, (predicted/total).tolist())) if total else {},
            "confusion_rows_actual_columns_predicted": matrix.astype(int).tolist(), "labels": list(LABELS)}


def intervals(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"low": float(np.quantile(finite, .025)) if len(finite) else None,
            "high": float(np.quantile(finite, .975)) if len(finite) else None,
            "defined_replicates": len(finite)}


def prediction_bootstrap(actual, predicted, groups, n_boot=BOOTSTRAPS, seed=SEED):
    """Resample employers with replacement, retaining each employer's complete fixed rows."""
    unique, inverse = np.unique(groups, return_inverse=True)
    aggregate = np.zeros((len(unique), 3, 3), dtype=np.int64)
    for i, group in enumerate(inverse):
        aggregate[group] += confusion([actual[i]], [predicted[i]])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        weights = rng.multinomial(len(unique), np.repeat(1 / len(unique), len(unique)))
        values.append(classification_metrics(np.tensordot(weights, aggregate, axes=(0, 0))))
    return {"accuracy": intervals([row["accuracy"] for row in values]),
            "macro_f1": intervals([row["macro_f1"] for row in values]),
            "employer_groups": len(unique), "replicates": n_boot, "seed": seed,
            "method": "percentile employer-cluster bootstrap; sample G employers G times; fixed predictions, no refit",
            "zero_division": "Class F1/recall are zero when their denominator is zero; macro always averages Low/Mid/High",
            "limitations": "Intervals condition on cached model and sampled employers; not a probability-sample population guarantee"}


def validate_adzuna(predictions, cleaned, training):
    if cleaned["id"].duplicated().any():
        raise ValueError("Employer join requires unique cleaned IDs")
    joined = predictions.merge(cleaned[["id", "company_name"]].rename(columns={"company_name": "joined_company"}),
                               on="id", how="left", validate="many_to_one", indicator=True)
    cached_name = joined.get("company_name", pd.Series("", index=joined.index)).fillna("")
    joined["company_name"] = joined["joined_company"].fillna(cached_name)
    actual, predicted = joined["actual_tier"].to_numpy(), joined["predicted_tier"].to_numpy()
    result = classification_metrics(confusion(actual, predicted))
    result["n"] = len(joined)
    result["roc_auc"] = {"value": None, "reason": "Cached predictions contain no class probabilities or decision scores"}
    counts = pd.Series(actual).value_counts().reindex(LABELS, fill_value=0)
    oracle = counts.idxmax()
    y = training["salary_tier"]
    historical_train, _ = train_test_split(y, test_size=.2, stratify=y, random_state=42)
    train_counts = historical_train.value_counts().reindex(LABELS, fill_value=0)
    majority = train_counts.idxmax()
    result["comparators"] = {
        "observed_majority_oracle": {"label": oracle, "status": "Selected using evaluation labels; descriptive oracle, not a training baseline",
                                      **classification_metrics(confusion(actual, np.repeat(oracle, len(actual))))},
        "historical_training_majority": {"label": majority,
                                         "provenance": "Reconstructed frozen 80/20 stratified split of saved cleaned rows; random_state=42; no model fit",
                                         "training_counts": {str(k): int(v) for k, v in train_counts.items()},
                                         "training_n": len(historical_train),
                                         **classification_metrics(confusion(actual, np.repeat(majority, len(actual))))},
        "all_cleaned_population_counts_not_training_only": {label: int((y == label).sum()) for label in LABELS}}
    groups = company_groups(joined)
    result["employer_join"] = {"matched_rows": int(joined["_merge"].eq("both").sum()),
                               "unmatched_rows": int(joined["_merge"].ne("both").sum()),
                               "company_name_disagreements": int((joined["joined_company"].notna() &
                                   joined["joined_company"].ne(cached_name)).sum()),
                               "missing_employer_rows_unique_fallback": int(pd.Series(joined["company_name"]).fillna("").astype(str).str.strip().eq("").sum()),
                               "prediction_duplicate_ids": int(joined["id"].duplicated().sum())}
    result["bootstrap"] = prediction_bootstrap(actual, predicted, groups)
    estimated = as_bool(joined["salary_is_predicted"])
    result["salary_label_status"] = {"estimated_rows": int(estimated.sum()), "advertised_rows": int((~estimated).sum()),
                                     "limitation": "Adzuna-estimated salary tiers are imperfect targets, not independently verified compensation"}
    result["salary_label_subgroups"] = {}
    for name, mask in [("estimated", estimated), ("advertised", ~estimated)]:
        result["salary_label_subgroups"][name] = {"n": int(mask.sum()),
            **classification_metrics(confusion(actual[mask], predicted[mask]))} if mask.any() else {"n": 0, "metrics": None}
    return result


def cap_companies(frame, cap=COMPANY_CAP):
    ordered = frame.sort_values(["publication_date", "id"], ascending=[False, True], na_position="last")
    kept = ordered.groupby(["fetch_date", "company_name"], dropna=False).head(cap).index
    return frame.loc[frame.index.isin(kept)].sort_values("id").reset_index(drop=True)


def rule_arrays(frame, rules):
    baskets = [frozenset(values) for values in frame["matched_skills"]]
    a, b = [], []
    for row in rules.itertuples(index=False):
        antecedent = frozenset(s.strip() for s in row.antecedents.split(","))
        consequent = frozenset(s.strip() for s in row.consequents.split(","))
        a.append([antecedent <= basket for basket in baskets])
        b.append([consequent <= basket for basket in baskets])
    return np.array(a, dtype=bool).T, np.array(b, dtype=bool).T


def probability_metrics(a, b, both, n):
    a, b, both = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(both, dtype=float)
    support = both / n
    confidence = np.divide(both, a, out=np.full(a.shape, np.nan), where=a > 0)
    lift = np.divide(both * n, a * b, out=np.full(a.shape, np.nan), where=(a*b) > 0)
    max_lift = np.divide(n, b, out=np.full(b.shape, np.nan), where=b > 0)
    return {"support": support, "confidence": confidence, "lift": lift,
            "consequent_support": b / n, "max_lift": max_lift}


def rule_bootstrap(frame, a, b, n_boot=BOOTSTRAPS, seed=SEED):
    groups, inverse = np.unique(company_groups(frame), return_inverse=True)
    n_rules = a.shape[1]
    aggregate = np.zeros((len(groups), 3 * n_rules + 1), dtype=float)
    for index, group in enumerate(inverse):
        aggregate[group] += np.r_[a[index], b[index], (a[index] & b[index]), 1]
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("support", "confidence", "lift", "consequent_support", "max_lift")}
    for _ in range(n_boot):
        weights = rng.multinomial(len(groups), np.repeat(1 / len(groups), len(groups)))
        counts = weights @ aggregate
        metrics = probability_metrics(counts[:n_rules], counts[n_rules:2*n_rules],
                                      counts[2*n_rules:3*n_rules], counts[-1])
        for name, value in metrics.items():
            values[name].append(value)
    return {name: np.array(value) for name, value in values.items()}, len(groups)


def threshold_status(row, support):
    holds = {"support": row["support"] >= support,
             "confidence": row["confidence"] > MIN_CONFIDENCE,
             "lift": row["lift"] > MIN_LIFT}
    failed = [name for name, value in holds.items() if not value]
    measurable = all(np.isfinite(row[name]) for name in holds)
    return not failed and measurable, ", ".join(failed) if measurable else "not measurable"


def evaluate_muse(frame, rules, n_boot=BOOTSTRAPS, seed=SEED):
    eligible = frame[frame["matched_skills"].map(len) >= MIN_SKILLS].copy()
    if eligible.empty:
        raise ValueError("No eligible Muse transactions")
    a, b = rule_arrays(eligible, rules)
    metrics = probability_metrics(a.sum(axis=0), b.sum(axis=0), (a & b).sum(axis=0), len(eligible))
    boot, group_count = rule_bootstrap(eligible, a, b, n_boot, seed)
    rows = []
    for index, rule in enumerate(rules.itertuples(index=False)):
        row = {"antecedents": rule.antecedents, "consequents": rule.consequents,
               "rule": f"{rule.antecedents} -> {rule.consequents}",
               "training_saved_lift_historical": float(rule.lift),
               "antecedent_count": int(a[:, index].sum()), "consequent_count": int(b[:, index].sum()),
               "itemset_count": int((a[:, index] & b[:, index]).sum()), "denominator": len(eligible),
               **{name: float(value[index]) for name, value in metrics.items()}}
        row["passes_primary_005"], row["failure_reasons_primary"] = threshold_status(row, PRIMARY_SUPPORT)
        row["passes_sensitivity_002"], row["failure_reasons_sensitivity"] = threshold_status(row, SENSITIVITY_SUPPORT)
        row["lift_ceiling_blocks_threshold"] = bool(row["max_lift"] <= MIN_LIFT)
        for name, samples in boot.items():
            ci = intervals(samples[:, index])
            row[f"{name}_ci_low"] = ci["low"]
            row[f"{name}_ci_high"] = ci["high"]
            row[f"{name}_defined_replicates"] = ci["defined_replicates"]
        for name, threshold in [("primary_005", PRIMARY_SUPPORT), ("sensitivity_002", SENSITIVITY_SUPPORT)]:
            passing = (boot["support"][:, index] >= threshold) & (boot["confidence"][:, index] > MIN_CONFIDENCE) & (boot["lift"][:, index] > MIN_LIFT)
            row[f"bootstrap_pass_fraction_{name}"] = float(passing.mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    report = {"all_postings": len(frame), "eligible_postings": len(eligible),
              "excluded_under_two_skills": len(frame)-len(eligible),
              "empty_baskets_all_postings": int(frame["matched_skills"].map(len).eq(0).sum()),
              "empty_baskets_eligible": int(eligible["matched_skills"].map(len).eq(0).sum()),
              "employers_all": len(np.unique(company_groups(frame))), "employers_eligible": group_count,
              "fixed_rules": len(rules), "primary_confirmed": int(result["passes_primary_005"].sum()),
              "sensitivity_confirmed": int(result["passes_sensitivity_002"].sum()),
              "confirmed_primary_rules": result.loc[result.passes_primary_005, "rule"].tolist(),
              "confirmed_sensitivity_rules": result.loc[result.passes_sensitivity_002, "rule"].tolist(),
              "bootstrap_replicates": n_boot, "bootstrap_seed": seed,
              "bootstrap_method": "percentile employer-cluster bootstrap of eligible cached postings; fixed rules, no mining/refit",
              "undefined_replicates": {name: int((~np.isfinite(value)).sum()) for name, value in boot.items()},
              "limitations": "Intervals condition on cached employers and fixed rules; undefined metrics are excluded from percentile CIs and counted; undefined replicates fail pass-fraction criteria"}
    return result, report


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build(root, out, fetch_date="2026-10-14"):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Output directory must be outside canonical repository")
    if out.exists():
        raise FileExistsError("Output directory must be fresh")
    relative = ["outputs/07_live_predictions.csv", "data/processed/live_cleaned_jobs.csv",
                "data/processed/cleaned_jobs.csv", "outputs/04_association_rules.csv",
                "data/processed/muse_cleaned_jobs.csv", "outputs/08_rules_validation.csv",
                "outputs/08_muse_summary.txt"]
    before = {name: sha256(root / name) for name in relative}
    predictions = pd.read_csv(root / relative[0], dtype={"id": str})
    cleaned = pd.read_csv(root / relative[1], dtype={"id": str})
    training = pd.read_csv(root / relative[2], usecols=["salary_tier"])
    rules = pd.read_csv(root / relative[3])
    muse = pd.read_csv(root / relative[4], dtype={"id": str})
    all_dates = sorted(muse["fetch_date"].astype(str).unique())
    muse = muse[muse["fetch_date"].eq(fetch_date)].copy()
    if muse.empty:
        raise ValueError(f"Missing cached Muse snapshot: {fetch_date}")
    muse["matched_skills"] = muse["matched_skills"].map(ast.literal_eval)
    mismatches = int(muse["n_skills"].ne(muse["matched_skills"].map(len)).sum())
    adzuna = validate_adzuna(predictions, cleaned, training)
    uncapped, uncapped_report = evaluate_muse(muse, rules)
    capped_frame = cap_companies(muse)
    capped, capped_report = evaluate_muse(capped_frame, rules)
    saved_validation = pd.read_csv(root / relative[5])
    comparison = uncapped[["rule", "passes_primary_005", "lift"]].merge(
        saved_validation[["rule", "validated", "validated_company_capped", "muse_lift", "fetch_date"]], on="rule", how="left", validate="one_to_one")
    comparison = comparison.merge(capped[["rule", "passes_primary_005"]].rename(columns={"passes_primary_005": "capped_passes_primary_005"}), on="rule")
    summary = (root / relative[6]).read_text()
    saved_support_line = next((line for line in summary.splitlines() if line.startswith("Transactions:")), None)
    historical_comparison = {"status": "Comparison with historical saved results, not a new mining run",
                             "saved_support_line": saved_support_line,
                             "saved_rule_count": len(saved_validation),
                             "saved_dates": sorted(saved_validation["fetch_date"].astype(str).unique()),
                             "joined_fixed_rules": int(comparison["validated"].notna().sum()),
                             "primary_confirmation_disagreements": int(comparison["passes_primary_005"].ne(as_bool(comparison["validated"])).sum()),
                             "capped_primary_confirmation_disagreements": int(comparison["capped_passes_primary_005"].ne(as_bool(comparison["validated_company_capped"])).sum()),
                             "maximum_lift_difference_vs_rounded_saved": float((comparison["lift"]-comparison["muse_lift"]).abs().max())}
    report = {"source_status": {"mode": "cached_fixed_predictions_and_rules_no_training_no_requests_no_database_writes",
                                "fetch_date": fetch_date, "available_muse_dates": all_dates,
                                "n_skills_disagreements": mismatches,
                                "fixed_training_rules": "historical top20 selection, not independent untouched rules"},
              "predeclared_contract": {"primary_support_inclusive": PRIMARY_SUPPORT,
                                       "sensitivity_support_inclusive": SENSITIVITY_SUPPORT,
                                       "confidence_strictly_greater_than": MIN_CONFIDENCE,
                                       "lift_strictly_greater_than": MIN_LIFT,
                                       "muse_eligibility_min_skills": MIN_SKILLS,
                                       "company_cap_before_eligibility": COMPANY_CAP,
                                       "cap_order": "publication_date descending, id ascending, separately per fetch_date and company",
                                       "support_denominator": "all qualifying postings in the selected source/capped snapshot; no skill-column filtering changes denominator",
                                       "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": SEED,
                                       "selection": "none; .02 is sensitivity, never chosen using confirmation counts"},
              "adzuna": adzuna, "muse_uncapped": uncapped_report, "muse_company_capped": capped_report,
              "historical_saved_comparison": historical_comparison,
              "input_sha256": before,
              "limitations": ["Cached exploratory model predictions do not establish independent external model validation.",
                              "Employer bootstrap conditions on cached source and predictions; sampling uncertainty is not model-selection uncertainty.",
                              "Estimated salary labels, source/employer selection, and description differences limit generalization.",
                              "One cached date and different source populations do not establish temporal demand trends.",
                              "Fixed rule metrics only; no Apriori search, threshold adaptation or retraining."]}
    report["inputs_unchanged"] = before == {name: sha256(root / name) for name in relative}
    out.mkdir(parents=True)
    uncapped.to_csv(out / "muse_uncapped_fixed_rules.csv", index=False)
    capped.to_csv(out / "muse_company_capped_fixed_rules.csv", index=False)
    comparison.to_csv(out / "historical_saved_comparison.csv", index=False)
    (out / "report.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False)+"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch-date", default="2026-10-14")
    args = parser.parse_args(argv)
    report = build(args.root, args.out, args.fetch_date)
    print(json.dumps({"adzuna_accuracy": report["adzuna"]["accuracy"], "adzuna_macro_f1": report["adzuna"]["macro_f1"],
                      "muse_uncapped": report["muse_uncapped"], "muse_company_capped": report["muse_company_capped"],
                      "inputs_unchanged": report["inputs_unchanged"]}))


if __name__ == "__main__":
    main()
