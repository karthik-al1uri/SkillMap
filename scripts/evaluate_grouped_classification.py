"""Predefined bounded grouped classifier comparison; canonical artifacts are read-only."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

CANDIDATES = ("nominal_logistic", "ordinal_logistic_it", "fixed_xgboost")
OUTER_SEED = 2026
INNER_SEED = 2027
WALL_SECONDS = 1200


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_module(root):
    sys.path.insert(0, str(root))
    from src import classification as cl
    if Path(cl.__file__).resolve() != (root / "src/classification.py").resolve():
        raise RuntimeError("Classification module must come from the explicit canonical root")
    return cl


def load_inputs(root, groups_path):
    """Call the canonical loader using scoped read-only SQLite connections."""
    cl = canonical_module(root)
    original = cl.sqlite3.connect
    def read_only_connection(path, *args, **kwargs):
        return original(Path(path).resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    with patch.object(cl.sqlite3, "connect", read_only_connection):
        data = cl.load_model_data(root).reset_index(drop=True)
    skills = cl.load_skill_vocabulary(root)
    if len(skills) != 245 or len(set(skills)) != 245 or not data.job_id.is_unique:
        raise ValueError("Frozen unique job IDs and 245-skill vocabulary required")
    table = pd.read_csv(groups_path, dtype={"job_id": str, "normalized_description_group": str})
    if not table.job_id.is_unique:
        raise ValueError("Group candidate job IDs must be unique")
    lookup = table.set_index("job_id").normalized_description_group
    mapped = data.job_id.astype(str).map(lookup)
    missing = mapped.isna() | mapped.fillna("").str.strip().eq("")
    groups = mapped.where(~missing, "singleton-job:" + data.job_id.astype(str)).astype(str).to_numpy()
    y = data.salary_tier.map(cl.TIER_TO_CODE)
    if y.isna().any() or set(y.unique()) != {0, 1, 2}:
        raise ValueError("All three frozen salary tiers required")
    return cl, data, y.to_numpy(dtype=int), groups, skills, int(missing.sum())


def frozen_splits(y, groups):
    """Outer fold zero, then exactly three grouped training folds; assert isolation."""
    positions = np.arange(len(y))
    train, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED).split(positions, y, groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Outer group overlap")
    inner = []
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=INNER_SEED).split(train, y[train], groups[train]):
        fit, valid = train[a], train[b]
        if set(groups[fit]) & set(groups[valid]) or set(fit) & set(test) or set(valid) & set(test):
            raise AssertionError("Inner group/test overlap")
        if set(y[fit]) != {0,1,2} or set(y[valid]) != {0,1,2}:
            raise ValueError("Every inner fold must contain all classes")
        inner.append((fit, valid))
    if set(y[train]) != {0,1,2} or set(y[test]) != {0,1,2}:
        raise ValueError("Outer split must contain all classes")
    return train, test, inner


def class_balance(y, indices):
    return {name: int(np.sum(y[indices] == code)) for code, name in enumerate(("Low", "Mid", "High"))}


def choose_by_inner_cv(candidate_names, splits, fit_fold):
    """Callback sees training/validation rows only; explicit candidate-order tie break."""
    if not 1 <= len(candidate_names) <= 3:
        raise ValueError("At most three predefined candidates")
    rows = []
    for name in candidate_names:
        scores = [float(fit_fold(name, fold, fit, valid)) for fold, (fit, valid) in enumerate(splits)]
        rows.append({"candidate": name, "fold_f1_macro": scores, "mean_f1_macro": float(np.mean(scores))})
    winner = max(rows, key=lambda row: row["mean_f1_macro"])["candidate"]
    return winner, rows


def metrics(y, pred, probabilities):
    labels = [0,1,2]
    auc = None
    if set(np.unique(y)) == set(labels):
        auc = float(np.mean([roc_auc_score(y == k, probabilities[:, k]) for k in labels]))
    return {"f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "f1_per_class": dict(zip(("Low", "Mid", "High"), f1_score(y, pred, labels=labels, average=None, zero_division=0).tolist())),
            "accuracy": float(accuracy_score(y, pred)), "roc_auc_ovr_macro": auc,
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def group_bootstrap(y, pred, probabilities, groups, repetitions=200):
    """Percentile intervals on fixed predictions, resampling whole description groups."""
    if not 1 <= repetitions <= 200:
        raise ValueError("Bootstrap capped at 200")
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2028)
    records = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[group] for group in selected])
        result = metrics(y[take], pred[take], probabilities[take])
        records.append({"f1_macro": result["f1_macro"], "accuracy": result["accuracy"],
                        "roc_auc_ovr_macro": result["roc_auc_ovr_macro"],
                        **{"f1_" + key: value for key,value in result["f1_per_class"].items()}})
    intervals = {}
    for key in records[0]:
        values = [row[key] for row in records if row[key] is not None]
        intervals[key] = {"lower_2.5": float(np.quantile(values, .025)), "upper_97.5": float(np.quantile(values, .975)), "valid_repetitions": len(values)} if values else None
    return {"repetitions": repetitions, "seed": 2028, "unit": "normalized description group", "intervals": intervals}


def build_pipeline(cl, name, skills):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    import mord
    from xgboost import XGBClassifier
    estimators = {
        "nominal_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=2026)),
        "ordinal_logistic_it": make_pipeline(StandardScaler(), mord.LogisticIT(alpha=1., max_iter=1000)),
        "fixed_xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.1, subsample=1.,
                                       objective="multi:softprob", eval_metric="mlogloss", random_state=2026, n_jobs=1),
    }
    return Pipeline([("features", cl.PostingFeatures(skills, use_text=True)), ("model", estimators[name])])


def internal_fit(request_path):
    """One sequential isolated process gives the orchestrator a hard fit timeout."""
    with request_path.open("rb") as stream:
        request = pickle.load(stream)
    cl = canonical_module(Path(request["root"]))
    pipeline = build_pipeline(cl, request["candidate"], request["skills"])
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(request["fit_data"], request["fit_y"])
        pred = pipeline.predict(request["valid_data"]).astype(int)
        probabilities = pipeline.predict_proba(request["valid_data"])
    result = {"metrics": metrics(request["valid_y"], pred, probabilities), "pred": pred,
              "probabilities": probabilities, "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured]}
    if request["save_model"]:
        result["pipeline"] = pipeline
    with Path(request["response"]).open("xb") as stream:
        pickle.dump(result, stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root, group_path, out = args.root.resolve(strict=True), args.groups.resolve(strict=True), args.out_dir.resolve()
    if out == root or root in out.parents or out.exists():
        parser.error("Use a fresh output directory outside the input repository")
    cl, data, y, groups, skills, missing_groups = load_inputs(root, group_path)
    train, test, inner = frozen_splits(y, groups)
    out.mkdir(parents=True)
    manifest = pd.DataFrame({"job_id": data.job_id, "normalized_description_group": groups,
                             "salary_tier": data.salary_tier, "outer_role": "test", "inner_validation_fold": -1})
    manifest.loc[train, "outer_role"] = "train"
    for fold, (_, valid) in enumerate(inner):
        manifest.loc[valid, "inner_validation_fold"] = fold
    manifest.to_csv(out / "split_manifest.csv", index=False)
    sources = [root / relative for relative in ("src/classification.py", "src/preprocessing.py", "data/processed/cleaned_jobs.csv", "data/processed/skill_vocabulary.csv", "data/processed/skillmap.db")] + [group_path, Path(__file__).resolve()]
    design = {"evaluation_status": "exploratory_grouped_resplit_of_previously_used_development_data", "outer": {"splitter": "StratifiedGroupKFold", "n_splits":5,"shuffle":True,"seed":2026,"test_fold":0},
              "inner":{"splitter":"StratifiedGroupKFold","n_splits":3,"shuffle":True,"seed":2027},
              "candidate_order": CANDIDATES, "candidate_parameters": {"nominal_logistic":{"scaler":"StandardScaler","max_iter":2000,"seed":2026},
               "ordinal_logistic_it":{"scaler":"StandardScaler","alpha":1.,"max_iter":1000},
               "fixed_xgboost":{"n_estimators":200,"max_depth":4,"learning_rate":.1,"subsample":1.,"objective":"multi:softprob","n_jobs":1,"seed":2026}},
              "shared_features":{"PostingFeatures":"canonical main", "use_text":True,"skills":245,"preprocessing":"fit separately inside each inner training fold"},
              "selection":"maximum mean inner macro F1; ties resolved by candidate order; outer test excluded", "group_unit":"normalized_description_group", "singleton_fallback_rows":missing_groups,
              "rows":len(data), "groups":len(set(groups)), "outer_train_rows":len(train),"outer_test_rows":len(test),"outer_train_groups":len(set(groups[train])),"outer_test_groups":len(set(groups[test])),
              "outer_group_overlap":len(set(groups[train]) & set(groups[test])), "outer_test_job_id_sha256":hashlib.sha256(json.dumps(sorted(data.iloc[test].job_id.tolist())).encode()).hexdigest(), "outer_train_class_counts":class_balance(y,train),"outer_test_class_counts":class_balance(y,test),
              "inner_folds":[{"fold":i,"fit_rows":len(fit),"validation_rows":len(valid),"fit_class_counts":class_balance(y,fit),"validation_class_counts":class_balance(y,valid),"group_overlap":len(set(groups[fit]) & set(groups[valid]))} for i,(fit,valid) in enumerate(inner)],
              "limits":{"wall_seconds":1200,"training_processes":1,"numerical_threads":1,"inner_fits":9,"final_refits":1,"bootstrap_repetitions":200,"bootstrap_seed":2028},
              "salary_tiers":{"Low":"<=62400","Mid":">62400 and <=109200","High":">109200","provenance":"Historical global empirical quantiles; not estimated within these new training folds"},
              "source_hashes":{str(path):sha256(path) for path in sources}, "split_manifest_sha256":sha256(out / "split_manifest.csv"),
              "limitations":["New grouped split of previously used development data; not an independent or external dataset.","Normalized-description grouping does not eliminate all employer or near-template overlap.","Bootstrap intervals condition on the selected fit and fixed predictions; they omit training and model-selection uncertainty.","Salary thresholds originate in globally derived historical tiers; no retiering or canonical data changes.","Only three predefined models and one outer fold; no nested full outer-CV or hyperparameter search."]}
    dump_json(out / "frozen_design.json", design)
    started = time.monotonic()
    milestones = []
    def fit_one(name, fold, fit, valid, save_model=False):
        remaining = WALL_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("20-minute comparison cap reached")
        number = len(milestones)
        prefix = out / f"fit_{number:02d}_{name}_{fold}"
        request_path, response_path = prefix.with_suffix(".request.pkl"), prefix.with_suffix(".response.pkl")
        request = {"root":str(root),"candidate":name,"skills":skills,"fit_data":data.iloc[fit],"fit_y":y[fit],"valid_data":data.iloc[valid],"valid_y":y[valid],"save_model":save_model,"response":str(response_path)}
        with request_path.open("xb") as stream:
            pickle.dump(request, stream)
        milestone = {"candidate":name,"fold":fold,"fit_rows":len(fit),"validation_rows":len(valid),"status":"started","elapsed_start_seconds":time.monotonic()-started}
        milestones.append(milestone)
        dump_json(out / "fit_milestones.json", milestones)
        result = subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--internal-fit",str(request_path)], capture_output=True,text=True,timeout=max(.01,remaining),check=False)
        milestone.update({"returncode":result.returncode,"stderr":result.stderr,"elapsed_end_seconds":time.monotonic()-started})
        if result.returncode != 0:
            milestone["status"] = "failed"
            dump_json(out / "fit_milestones.json", milestones)
            raise RuntimeError(f"Fit failed: {result.stderr[-2000:]}")
        with response_path.open("rb") as stream:
            fitted_result = pickle.load(stream)
        milestone.update({"status":"complete","metrics":fitted_result["metrics"],"warnings":fitted_result["warnings"]})
        dump_json(out / "fit_milestones.json", milestones)
        return fitted_result
    try:
        winner, cv_rows = choose_by_inner_cv(CANDIDATES, inner, lambda name,fold,fit,valid: fit_one(name,fold,fit,valid)["metrics"]["f1_macro"])
        dump_json(out / "cv_selection.json", {"winner":winner,"selection":"inner CV only","rows":cv_rows})
        pd.DataFrame([{**row,"fold_f1_macro":json.dumps(row["fold_f1_macro"])} for row in cv_rows]).to_csv(out / "cv_selection.csv",index=False)
        final = fit_one(winner,"outer_refit",train,test,True)
        predictions = pd.DataFrame({"job_id":data.iloc[test].job_id.to_numpy(),"normalized_description_group":groups[test],"actual_code":y[test],"predicted_code":final["pred"], **{f"prob_{name}":final["probabilities"][:,i] for i,name in enumerate(("Low","Mid","High"))}})
        predictions.to_csv(out / "outer_predictions.csv",index=False)
        with (out / "selected_model.pkl").open("xb") as stream:
            pickle.dump(final["pipeline"],stream)
        uncertainty = group_bootstrap(y[test],final["pred"],final["probabilities"],groups[test])
        report = {"status":"complete","winner":winner,"cv_rows":cv_rows,"outer_metrics":final["metrics"],"bootstrap":uncertainty,"elapsed_seconds":time.monotonic()-started,"design_sha256":sha256(out / "frozen_design.json"),"split_manifest_sha256":sha256(out / "split_manifest.csv"),"limitations":design["limitations"]}
        assert report["split_manifest_sha256"] == design["split_manifest_sha256"]
        dump_json(out / "report.json",report)
        print(json.dumps(report,indent=2))
    except Exception as error:
        if milestones and milestones[-1]["status"] == "started":
            milestones[-1].update({"status":"interrupted","error":repr(error)})
            dump_json(out / "fit_milestones.json",milestones)
        dump_json(out / "report.json",{"status":"incomplete","error":repr(error),"elapsed_seconds":time.monotonic()-started,"completed_fits":sum(x["status"]=="complete" for x in milestones)})
        raise
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--internal-fit":
        internal_fit(Path(sys.argv[2]))
    else:
        main()
