"""Build bounded, denominator-explicit association views from saved transactions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from src.preprocessing import FORCED_SKILLS

GENERAL_SUPPORT = .05
TECHNICAL_SUPPORT = .02
MIN_CONFIDENCE = .5
MIN_LIFT = 1.5
METRICS = ["antecedent support", "consequent support", "support", "confidence", "lift"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def items(value):
    return frozenset(x.strip() for x in str(value).split(",") if x.strip())


def score_rules(rules, baskets, high):
    """Score mining statistics on baskets; High-rate statistics on their supplied population."""
    rows = []
    high = np.asarray(high, dtype=bool)
    for record in rules.to_dict("records"):
        a, c = items(record["antecedents"]), items(record["consequents"])
        ma = np.fromiter((a <= basket for basket in baskets), dtype=bool)
        mc = np.fromiter((c <= basket for basket in baskets), dtype=bool)
        both = ma & mc
        support = both.mean()
        confidence = both.sum() / ma.sum() if ma.any() else np.nan
        lift = confidence / mc.mean() if mc.any() else np.nan
        record.update({"antecedent support": ma.mean(), "consequent support": mc.mean(),
                       "support": support, "confidence": confidence, "lift": lift,
                       "jobs_with_itemset": int(both.sum()),
                       "high_tier_rate": high[both].mean() if both.any() else np.nan})
        record["high_tier_lift"] = record["high_tier_rate"] / high.mean() if high.any() else np.nan
        rows.append(record)
    return pd.DataFrame(rows, columns=list(dict.fromkeys(list(rules.columns) + METRICS +
                         ["jobs_with_itemset", "high_tier_rate", "high_tier_lift"])))


def passes(rules, min_support):
    return rules[(rules["support"] >= min_support) &
                 (rules["confidence"] > MIN_CONFIDENCE) & (rules["lift"] > MIN_LIFT)]


def rank_general(rules, n=10):
    return rules.sort_values(["lift", "support", "antecedents", "consequents"],
                             ascending=[False, False, True, True]).head(n)


def rank_high(rules, n=5):
    return rules.sort_values(["high_tier_rate", "lift", "jobs_with_itemset", "antecedents", "consequents"],
                             ascending=[False, False, False, True, True]).head(n)


def strict_technical(rules, technical=FORCED_SKILLS):
    technical = frozenset(technical)
    mask = [(items(a) | items(c)) <= technical for a, c in
            zip(rules["antecedents"], rules["consequents"])]
    return rules.loc[np.asarray(mask, dtype=bool)]


def mine_technical(baskets, technical=FORCED_SKILLS):
    """Mine only forced technical columns, retaining every job in the support denominator."""
    technical = [s for s in technical if any(s in b for b in baskets)]
    encoded = pd.DataFrame({s: [s in b for b in baskets] for s in technical}, dtype=bool)
    empty = pd.DataFrame(columns=["antecedents", "consequents"] + METRICS)
    if not technical or not len(baskets):
        return empty
    frequent = apriori(encoded, min_support=TECHNICAL_SUPPORT, use_colnames=True)
    if frequent.empty or not any(frequent["itemsets"].map(len) > 1):
        return empty
    rules = association_rules(frequent, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = passes(rules, TECHNICAL_SUPPORT)
    for column in ("antecedents", "consequents"):
        rules[column] = rules[column].map(lambda x: ", ".join(sorted(x)))
    return strict_technical(rules)


def discrepancies(saved, exact):
    out = {}
    for column in METRICS:
        delta = (saved[column].astype(float).to_numpy() - exact[column].astype(float).to_numpy())
        out[column] = {"beyond_4_decimal_rounding": int((np.abs(delta) > .000050001).sum()),
                       "max_absolute_delta": float(np.nanmax(np.abs(delta))) if len(delta) else 0.}
    return out


def build(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Output directory must be outside the input repository")
    if out.exists():
        raise FileExistsError("Output directory must be fresh")
    names = ["data/processed/cleaned_jobs.csv", "outputs/04_association_rules.csv",
             "outputs/04_high_salary_rules.csv", "outputs/04_technical_rules_low_support.csv"]
    before = {name: file_hash(root / name) for name in names}
    jobs = pd.read_csv(root / names[0])
    baskets = [frozenset(ast.literal_eval(value)) for value in jobs["matched_skills"]]
    high = jobs["salary_tier"].eq("High").to_numpy()
    saved_all, saved_high, saved_tech = [pd.read_csv(root / name) for name in names[1:]]
    general = score_rules(saved_all, baskets, high)
    high_baskets = [basket for basket, flag in zip(baskets, high) if flag]
    within_high = score_rules(saved_high, high_baskets, np.ones(high.sum(), dtype=bool))
    all_for_high = score_rules(saved_high, baskets, high)
    for name in ["jobs_with_itemset", "high_tier_rate", "high_tier_lift"]:
        within_high[name] = all_for_high[name]
    technical = score_rules(mine_technical(baskets), baskets, high)
    exact_saved_tech = score_rules(saved_tech, baskets, high)
    general_valid = passes(general, GENERAL_SUPPORT)
    high_valid = passes(within_high, GENERAL_SUPPORT)
    top10, top5 = rank_general(general_valid), rank_high(high_valid)
    tech_view = rank_general(passes(technical, TECHNICAL_SUPPORT), len(technical))
    out.mkdir(parents=True)
    top10.to_csv(out / "top10_lift.csv", index=False)
    top5.to_csv(out / "top5_high_tier.csv", index=False)
    tech_view.to_csv(out / "technical_only.csv", index=False)
    report = {"population": {"all_jobs": len(jobs), "high_jobs": int(high.sum()),
                             "zero_skill_jobs_in_denominator": sum(not b for b in baskets)},
              "thresholds": {"general_support_inclusive": GENERAL_SUPPORT,
                             "technical_support_inclusive": TECHNICAL_SUPPORT,
                             "confidence_strictly_greater_than": MIN_CONFIDENCE,
                             "lift_strictly_greater_than": MIN_LIFT},
              "definitions": {"general_support": "full itemset count / ALL jobs, including empty baskets",
                              "high_mining_support": "full itemset count among High jobs / ALL High jobs",
                              "high_tier_rate": "High jobs with full itemset / ALL jobs with full itemset",
                              "high_tier_lift": "high_tier_rate / overall High share",
                              "technical_only": "ALL skills in antecedents and consequents are forced technical skills",
                              "selection": "top High-tier rules ranked by high_tier_rate, within-tier lift, then all-job itemset count"},
              "coverage": {"saved_general_candidates": len(saved_all), "saved_high_candidates": len(saved_high),
                           "saved_any_technical_candidates": len(saved_tech),
                           "saved_candidates_strictly_technical": len(strict_technical(saved_tech)),
                           "technical_only_remined_columns": len(FORCED_SKILLS),
                           "general_note": "saved CSV contains historical top 20, not all mined rules; top10 is within this saved candidate set"},
              "views": {"top10_rows": len(top10), "top5_high_rows": len(top5), "technical_only_rows": len(tech_view)},
              "threshold_failures_recomputed": {"general": len(general)-len(general_valid),
                                                "high": len(within_high)-len(high_valid),
                                                "saved_any_technical": len(exact_saved_tech)-len(passes(exact_saved_tech, TECHNICAL_SUPPORT))},
              "rounding_checks": {"general": discrepancies(saved_all, general),
                                  "high": discrepancies(saved_high, within_high),
                                  "saved_any_technical": discrepancies(saved_tech, exact_saved_tech)},
              "top_general": top10.iloc[0].to_dict() if len(top10) else None,
              "top_high": top5.iloc[0].to_dict() if len(top5) else None,
              "input_sha256": before,
              "limitations": ["No complete general Apriori rerun; ranking covers saved candidates.",
                              "Technical-only rerun uses saved Phase 6 extraction, not proposed Phase 6b changes.",
                              "Descriptive salary association is not causal or independently validated."]}
    report["inputs_unchanged"] = before == {name: file_hash(root / name) for name in names}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args.root, args.out)
    print(json.dumps({k: report[k] for k in ("population", "views", "threshold_failures_recomputed", "inputs_unchanged")}))


if __name__ == "__main__":
    main()
