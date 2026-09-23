"""SkillMap Stage 5 — salary tier classification (Logistic Regression, Random Forest, XGBoost).

Predicts `salary_tier` (Low / Mid / High) from posting features:
    - matched_skills: multi-hot over the 100-skill vocabulary
    - experience_level: ordinal (Internship 0 ... Executive 5, Unknown -1)
    - company_size: ordinal bucket of the company's employee_count (from the warehouse)
    - state: one-hot of the top 20 states, everything else (incl. no state) = Other
    - industry: multi-hot of the top 15 industries plus an Other flag (jobs have up to 3)

Top states and industries are chosen from the training split only.

Model 4 (XGBoost + title TF-IDF) adds TF-IDF weights of the TITLE_TFIDF_MAX_FEATURES
most frequent job-title words (English stop words removed, fit on the training split).
Its hyperparameters are chosen by 3-fold cross-validated grid search on the training
split only, so the test set is never used for tuning.

Run from the project root with:
    python -m src.classification
"""
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.preprocessing import find_project_root, load_cleaned_jobs

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_STATES = 20
TOP_N_INDUSTRIES = 15
TITLE_TFIDF_MAX_FEATURES = 50
TITLE_MODEL_NAME = "XGBoost + title TF-IDF"
# Grid searched for Model 4 with 3-fold CV (macro F1) on the training split.
TITLE_MODEL_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
    "colsample_bytree": [0.5, 1.0],
}
F1_TARGET = 0.75
AUC_TARGET = 0.80

TIER_LABELS = ["Low", "Mid", "High"]
TIER_TO_CODE = {label: code for code, label in enumerate(TIER_LABELS)}

# The data has "Associate" and "Mid-Senior level" instead of separate Mid and Senior levels.
EXPERIENCE_TO_CODE = {
    "Internship": 0,
    "Entry level": 1,
    "Associate": 2,          # Mid
    "Mid-Senior level": 3,   # Senior
    "Director": 4,
    "Executive": 5,
    "Unknown": -1,
}

# employee_count upper bounds for company_size codes 1-8; missing or 0 employees -> 0 (unknown).
COMPANY_SIZE_BINS = [0, 10, 50, 200, 500, 1000, 5000, 10000, np.inf]
COMPANY_SIZE_LABELS = ["unknown", "1-10", "11-50", "51-200", "201-500", "501-1000",
                       "1001-5000", "5001-10000", "10001+"]

# Chart styling: light surface, text inks, one bar hue, blue sequential ramp for heatmaps.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BAR_COLOR = "#2a78d6"
HEAT_RAMP = LinearSegmentedColormap.from_list(
    "heat_blue", ["#f0f6fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def load_model_data(project_root: Path) -> pd.DataFrame:
    """Load cleaned_jobs.csv and join state and employee_count from the warehouse on job_id."""
    jobs = load_cleaned_jobs(project_root / "data" / "processed" / "cleaned_jobs.csv")
    with sqlite3.connect(project_root / "data" / "processed" / "skillmap.db") as conn:
        extra = pd.read_sql(
            "SELECT j.job_id, l.state, c.employee_count "
            "FROM v_jobs j "
            "JOIN dim_location l USING (location_id) "
            "JOIN dim_company c USING (company_id)", conn)
    df = jobs.merge(extra, on="job_id", how="left", validate="one_to_one")
    logger.info("Loaded %d jobs (%d with employee_count, %d with state)",
                len(df), int(df["employee_count"].notna().sum()), int(df["state"].notna().sum()))
    return df


def load_skill_vocabulary(project_root: Path) -> list:
    """Return the 100 vocabulary skills (from Stage 2) that define the skill feature columns."""
    return pd.read_csv(project_root / "data" / "processed" / "skill_vocabulary.csv")["skill"].tolist()


def encode_company_size(employee_count: pd.Series) -> pd.Series:
    """Bucket employee_count into ordinal codes 0-8 (0 = unknown or 0 employees)."""
    codes = pd.cut(employee_count, bins=COMPANY_SIZE_BINS, labels=False, right=True)
    # Bin 0 is (0, 10] -> code 1; NaN (missing or exactly 0) -> 0
    return (codes + 1).fillna(0).astype(int)


def split_industries(industry: str) -> list:
    """Split the '; '-joined industry string into a list; 'Unknown' becomes an empty list."""
    return [] if industry == "Unknown" else [i.strip() for i in industry.split(";") if i.strip()]


def choose_top_categories(train: pd.DataFrame) -> dict:
    """Pick the top states and industries by frequency in the training split."""
    top_states = train["state"].value_counts().head(TOP_N_STATES).index.tolist()
    top_industries = (train["industry"].map(split_industries).explode().dropna()
                      .value_counts().head(TOP_N_INDUSTRIES).index.tolist())
    return {"states": top_states, "industries": top_industries}


def build_features(df: pd.DataFrame, skills: list, top: dict) -> pd.DataFrame:
    """Build the numeric feature matrix described in the module docstring."""
    features = {
        "experience_level": df["experience_level"].map(EXPERIENCE_TO_CODE).fillna(-1).astype(int),
        "company_size": encode_company_size(df["employee_count"]),
    }
    skill_sets = df["matched_skills"].map(set)
    for skill in skills:
        features[f"skill_{skill}"] = skill_sets.map(lambda s, k=skill: k in s).astype(int)

    state = df["state"].where(df["state"].isin(top["states"]), "Other")
    for s in [*top["states"], "Other"]:
        features[f"state_{s}"] = (state == s).astype(int)

    industries = df["industry"].map(split_industries)
    for ind in top["industries"]:
        features[f"industry_{ind}"] = industries.map(lambda xs, k=ind: k in xs).astype(int)
    features["industry_Other"] = industries.map(
        lambda xs: not xs or any(x not in top["industries"] for x in xs)).astype(int)
    return pd.DataFrame(features, index=df.index)


def build_title_features(train_titles: pd.Series, test_titles: pd.Series,
                         X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Append job-title TF-IDF columns to the base features.

    The vectorizer keeps the TITLE_TFIDF_MAX_FEATURES most frequent title words (English
    stop words removed) and is fit on training titles only. Returns sparse train/test
    matrices, the combined feature names, and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=TITLE_TFIDF_MAX_FEATURES, stop_words="english")
    title_train = vectorizer.fit_transform(train_titles)
    title_test = vectorizer.transform(test_titles)
    A_train = sp.hstack([sp.csr_matrix(X_train.to_numpy(dtype=float)), title_train]).tocsr()
    A_test = sp.hstack([sp.csr_matrix(X_test.to_numpy(dtype=float)), title_test]).tocsr()
    names = [*X_train.columns, *[f"title_{w}" for w in vectorizer.get_feature_names_out()]]
    return A_train, A_test, names, vectorizer


def tune_title_model(A_train, y_train) -> tuple:
    """Grid-search XGBoost hyperparameters for Model 4 with 3-fold CV on the training split.

    Returns the refit best estimator and a dict with the best params and CV macro F1.
    """
    search = GridSearchCV(
        XGBClassifier(subsample=0.9, objective="multi:softprob", eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1),
        TITLE_MODEL_GRID, scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE))
    search.fit(A_train, y_train)
    logger.info("%s grid search: best CV macro F1 %.3f with %s",
                TITLE_MODEL_NAME, search.best_score_, search.best_params_)
    return search.best_estimator_, {"best_params": search.best_params_, "cv_f1_macro": search.best_score_}


def build_models() -> dict:
    """Return the three untrained classifiers, keyed by display name."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6, objective="multi:softprob",
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Score a fitted model: macro F1 (train and test), one-vs-rest macro ROC-AUC, accuracy, confusion matrix."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
        "accuracy": accuracy_score(y_test, pred),
        "train_f1_macro": f1_score(y_train, model.predict(X_train), average="macro"),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1, 2]),
        "report": classification_report(y_test, pred, target_names=TIER_LABELS, digits=3),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test) -> tuple:
    """Fit the three base-feature models and return (fitted models, metrics per model)."""
    models, metrics = build_models(), {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
        logger.info("%s: F1 macro %.3f, ROC-AUC %.3f", name,
                    metrics[name]["f1_macro"], metrics[name]["roc_auc_ovr"])
    return models, metrics


def train_title_model(A_train, y_train, A_test, y_test) -> tuple:
    """Tune, fit, and evaluate Model 4 on base + title TF-IDF features.

    Returns (fitted model, metrics, tuning info).
    """
    model, tuning = tune_title_model(A_train, y_train)
    metrics = evaluate_model(model, A_train, y_train, A_test, y_test)
    logger.info("%s: F1 macro %.3f, ROC-AUC %.3f", TITLE_MODEL_NAME,
                metrics["f1_macro"], metrics["roc_auc_ovr"])
    return model, metrics, tuning


def results_table(metrics: dict) -> pd.DataFrame:
    """Return one row per model with the headline metrics, best F1 first."""
    rows = [{"model": name,
             "features": "base + title TF-IDF" if name == TITLE_MODEL_NAME else "base",
             "f1_macro": m["f1_macro"],
             "roc_auc_ovr": m["roc_auc_ovr"],
             "accuracy": m["accuracy"],
             "train_f1_macro": m["train_f1_macro"],
             "meets_f1_target": m["f1_macro"] > F1_TARGET,
             "meets_auc_target": m["roc_auc_ovr"] > AUC_TARGET}
            for name, m in metrics.items()]
    table = pd.DataFrame(rows).sort_values(["f1_macro", "roc_auc_ovr"], ascending=False)
    num = ["f1_macro", "roc_auc_ovr", "accuracy", "train_f1_macro"]
    table[num] = table[num].round(4)
    return table.reset_index(drop=True)


def feature_importance(model, feature_names: list) -> pd.Series:
    """Return importances for a fitted model, sorted descending.

    Tree models use their built-in importances (mean decrease in impurity for Random Forest,
    gain for XGBoost). Logistic Regression uses the mean absolute standardized coefficient
    across the three classes.
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model[-1].coef_).mean(axis=0)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def pretty_feature(name: str) -> str:
    """Turn a feature column name into a readable axis label."""
    for prefix, label in [("skill_", "skill"), ("state_", "state"), ("industry_", "industry"),
                          ("title_", "title word")]:
        if name.startswith(prefix):
            return f"{label}: {name[len(prefix):]}"
    return name.replace("_", " ")


def plot_feature_importance(importances: pd.Series, model_name: str, path: Path, top_n: int = 20) -> None:
    """Save a horizontal bar chart of the top `top_n` feature importances."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([pretty_feature(n) for n in top.index], top.values, color=BAR_COLOR, height=0.7)
    ax.set_xlabel(IMPORTANCE_LABEL.get(model_name, "Importance"), color=TEXT_SECONDARY)
    ax.set_title(f"{model_name}: top {top_n} features for predicting salary tier",
                 loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.tick_params(axis="y", labelcolor=TEXT_PRIMARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


IMPORTANCE_LABEL = {
    "Random Forest": "Importance (mean decrease in impurity)",
    "XGBoost": "Importance (share of total gain)",
    TITLE_MODEL_NAME: "Importance (share of total gain)",
    "Logistic Regression": "Mean |standardized coefficient| across classes",
}


def plot_confusion_matrices(metrics: dict, path: Path) -> None:
    """Save one row-normalized confusion matrix per model, annotated with % and counts."""
    names = list(metrics)
    ncols = 2 if len(names) == 4 else len(names)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5 * nrows), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(np.atleast_1d(axes).ravel(), names):
        cm = metrics[name]["confusion_matrix"]
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=HEAT_RAMP, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ink = SURFACE if share[i, j] > 0.55 else TEXT_PRIMARY
                ax.text(j, i, f"{share[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center",
                        color=ink, fontsize=11)
        ax.set_xticks(range(3), TIER_LABELS)
        ax.set_yticks(range(3), TIER_LABELS)
        ax.set_xlabel("Predicted tier", color=TEXT_SECONDARY)
        ax.set_ylabel("Actual tier", color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}\nF1 macro {metrics[name]['f1_macro']:.3f} · "
                     f"ROC-AUC {metrics[name]['roc_auc_ovr']:.3f}", color=TEXT_PRIMARY, fontsize=11)
    fig.suptitle("Confusion matrices on the 20% test set (row % of actual tier, with counts)",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_best_model(model, name: str, feature_names: list, top: dict, skills: list, path: Path,
                    title_vectorizer: TfidfVectorizer = None) -> None:
    """Pickle the best model with everything needed to rebuild its features.

    `title_vectorizer` is included when the best model uses title TF-IDF features.
    """
    bundle = {
        "model_name": name,
        "model": model,
        "feature_names": feature_names,
        "top_categories": top,
        "skills": skills,
        "experience_to_code": EXPERIENCE_TO_CODE,
        "company_size_bins": COMPANY_SIZE_BINS,
        "tier_labels": TIER_LABELS,
        "random_state": RANDOM_STATE,
        "title_vectorizer": title_vectorizer,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved %s (%s)", path, name)


def build_summary(df: pd.DataFrame, X: pd.DataFrame, n_train: int, n_test: int, top: dict,
                  table: pd.DataFrame, metrics: dict, best: str, importances: pd.Series,
                  title_words: list = None, tuning: dict = None) -> str:
    """Build the plain-text Stage 5 summary."""
    line = "=" * 80
    best_row = table.iloc[0]
    size_counts = (encode_company_size(df["employee_count"]).value_counts().sort_index()
                   .rename(index=dict(enumerate(COMPANY_SIZE_LABELS))))
    parts = [
        "SkillMap — Stage 5 Classification Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Target: salary_tier (Low=0, Mid=1, High=2). Split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} "
        f"stratified, random_state={RANDOM_STATE}",
        f"Rows: {len(df):,} (train {n_train:,}, test {n_test:,}). Features: {X.shape[1]}",
        "",
        f"{line}\nFeatures\n{line}",
        f"matched_skills multi-hot: {sum(c.startswith('skill_') for c in X.columns)} columns",
        f"experience_level ordinal: {EXPERIENCE_TO_CODE}",
        "company_size ordinal from warehouse employee_count: " + ", ".join(
            f"{k}={i}" for i, k in enumerate(COMPANY_SIZE_LABELS)),
        size_counts.to_string(),
        f"state one-hot, top {TOP_N_STATES} + Other (Other includes jobs with no state): {top['states']}",
        f"industry multi-hot, top {TOP_N_INDUSTRIES} + Other: {top['industries']}",
        *([f"Model 4 adds title TF-IDF, top {len(title_words)} words (stop words removed): {title_words}",
           f"Model 4 hyperparameters (3-fold CV on train, macro F1 {tuning['cv_f1_macro']:.3f}): "
           f"{tuning['best_params']}"] if title_words else []),
        "",
        f"{line}\nResults (test set)\n{line}",
        table.to_string(index=False),
        "",
        f"Best model (highest macro F1): {best}",
        f"F1 macro {best_row['f1_macro']:.3f} vs target > {F1_TARGET}: "
        f"{'MET' if best_row['meets_f1_target'] else 'NOT MET'}",
        f"ROC-AUC (OvR macro) {best_row['roc_auc_ovr']:.3f} vs target > {AUC_TARGET}: "
        f"{'MET' if best_row['meets_auc_target'] else 'NOT MET'}",
        "",
        f"-- {best}: per-class report --",
        metrics[best]["report"],
        "",
    ]
    for name, m in metrics.items():
        cm = pd.DataFrame(m["confusion_matrix"], index=[f"actual {t}" for t in TIER_LABELS],
                          columns=[f"pred {t}" for t in TIER_LABELS])
        parts += [f"-- Confusion matrix: {name} --", cm.to_string(), ""]
    parts += [f"-- {best}: top 20 features --",
              importances.head(20).round(4).to_string(), ""]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None) -> dict:
    """Run Stage 5 end to end and write the model, results, plots, and summary to outputs/."""
    root = project_root or find_project_root()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_model_data(root)
    skills = load_skill_vocabulary(root)
    y = df["salary_tier"].map(TIER_TO_CODE)
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    top = choose_top_categories(train_df)
    X_train = build_features(train_df, skills, top)
    X_test = build_features(test_df, skills, top)
    logger.info("Features: %d columns; train %d, test %d", X_train.shape[1], len(X_train), len(X_test))

    models, metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
    A_train, A_test, title_names, vectorizer = build_title_features(
        train_df["job_title"], test_df["job_title"], X_train, X_test)
    models[TITLE_MODEL_NAME], metrics[TITLE_MODEL_NAME], tuning = train_title_model(
        A_train, y_train, A_test, y_test)

    table = results_table(metrics)
    table.to_csv(output_dir / "05_classification_results.csv", index=False)
    best = table.iloc[0]["model"]
    uses_title = best == TITLE_MODEL_NAME
    feature_names = title_names if uses_title else list(X_train.columns)

    importances = feature_importance(models[best], feature_names)
    plot_feature_importance(importances, best, output_dir / "05_feature_importance.png")
    plot_confusion_matrices(metrics, output_dir / "05_confusion_matrix.png")
    save_best_model(models[best], best, feature_names, top, skills, output_dir / "05_best_model.pkl",
                    title_vectorizer=vectorizer if uses_title else None)

    title_words = list(vectorizer.get_feature_names_out())
    summary = build_summary(df, X_train, len(X_train), len(X_test), top, table, metrics, best, importances,
                            title_words=title_words, tuning=tuning)
    (output_dir / "05_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved %s", output_dir / "05_summary.txt")
    return {"models": models, "metrics": metrics, "table": table, "best": best,
            "importances": importances, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "top": top, "summary": summary,
            "A_train": A_train, "A_test": A_test, "title_vectorizer": vectorizer, "tuning": tuning}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
