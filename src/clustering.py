"""SkillMap Stage 6 — K-Means and DBSCAN clustering of job postings into role archetypes.

Each job is a multi-hot vector over the Stage 2 skill vocabulary (`matched_skills`, as in
Stage 5). Jobs with no matched skill are all-zero vectors, so they are left out of
clustering and labelled -1 ("No matched skills").

K-Means is run for k = 3..10 and k is chosen by silhouette score. DBSCAN's eps is tuned
from the k-distance graph. On binary vectors, Euclidean distances can only be sqrt(n) for
whole numbers n, so the graph is a staircase. Each step between the 10th and 90th
percentile of the k-distances is a candidate eps. The chosen candidate gives >= 2 clusters
with <= DBSCAN_MAX_NOISE noise and the highest silhouette.
Both are scored with silhouette (target > 0.50) and Davies-Bouldin (target < 1.0). Silhouette is computed on a fixed random sample of
SILHOUETTE_SAMPLE jobs, because the exact score is O(n^2).

Outlier detection: an Isolation Forest on matched_skills multi-hot + experience_level ordinal +
normalized_salary flags the OUTLIER_CONTAMINATION most anomalous postings. To test whether they
are salary mismatches (high pay for a low-skill profile or the reverse), each job's log salary is
z-scored within its (K-Means cluster, experience_level) group.

t-SNE (perplexity 30, 1,000 iterations) embeds the same job sample as the PCA plot, so the two
2D views can be compared. Each view is scored by kNN overlap (how many of a job's PURITY_NEIGHBORS
nearest neighbours in 2D are also its nearest neighbours in the 245-D space) and by purity (the
share of those 2D neighbours with the same K-Means cluster or salary tier). High-tier hotspots are
t-SNE regions where >= HOTSPOT_MIN_HIGH_SHARE of each job's HOTSPOT_NEIGHBORS neighbours are High tier,
grouped with DBSCAN on the t-SNE coordinates.

Run from the project root with:
    python -m src.clustering
"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.classification import EXPERIENCE_TO_CODE
from src.preprocessing import FORCED_SKILLS, find_project_root, load_cleaned_jobs

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
K_RANGE = range(3, 11)
SILHOUETTE_SAMPLE = 10_000
SILHOUETTE_TARGET = 0.50
DAVIES_BOULDIN_TARGET = 1.0
DBSCAN_MIN_SAMPLES = 20
DBSCAN_MAX_NOISE = 0.5        # a small eps can "win" on silhouette by calling most jobs noise
TOP_SKILLS = 5
TOP_TITLES = 5
MIN_SKILL_PREVALENCE = 0.15   # stops rare skills with huge lift from naming a cluster
GENERALIST_NAME = "Generalist (few listed skills)"
NO_SKILLS_LABEL = -1
NO_SKILLS_NAME = "No matched skills"
TIER_LABELS = ["Low", "Mid", "High"]
OUTLIER_CONTAMINATION = 0.01
OUTLIER_N_ESTIMATORS = 300
MISMATCH_Z = 2.0
OUTLIER_SVD_COMPONENTS = 10   # with 245 binary columns a random split rarely lands on salary
PLOT_SAMPLE = 8000
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000          # sklearn >= 1.7 calls n_iter max_iter
PURITY_NEIGHBORS = 15
HOTSPOT_NEIGHBORS = 30
HOTSPOT_MIN_HIGH_SHARE = 0.7
HOTSPOT_EPS = 3.0             # DBSCAN eps in t-SNE units
HOTSPOT_MIN_SAMPLES = 20

SKILL_THEMES = {
    "Healthcare & Clinical": [
        "patient care", "nursing", "registered nurse", "bls", "cpr", "acls", "healthcare",
        "medication administration", "patient education", "case management"],
    "Data, Tech & Engineering": [
        "python", "sql", "data analysis", "engineering", "troubleshooting", "research",
        "quality assurance", "analytical skills", "computer skills", "software development",
        "computer science", "data management", *FORCED_SKILLS],
    "Sales, Retail & Customer Service": [
        "sales", "marketing", "customer service", "retail", "merchandising", "cash handling",
        "product knowledge", "business development", "negotiation", "relationship building"],
    "Management & Leadership": [
        "management", "leadership", "team management", "team leadership", "performance management",
        "supervision", "supervisory experience", "budgeting", "budget management", "planning",
        "coaching", "mentoring", "decision making", "project management"],
    "Operations, Trades & Physical Work": [
        "safety", "maintenance", "lifting", "physical stamina", "physical strength", "standing",
        "bending", "quality control", "food safety", "food preparation", "inventory management",
        "driver's license", "flexible schedule"],
    "Finance, Compliance & Administration": [
        "accounting", "compliance", "regulatory compliance", "risk management", "reporting",
        "documentation", "confidentiality", "excel", "microsoft office", "microsoft word",
        "data entry", "scheduling"],
    "Education & Training": ["education", "training", "teaching", "professional development"],
    "Entry-Level & Credentials": ["high school diploma", "bachelor's degree", "master's degree"],
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NOISE_COLOR = "#c9c8c2"
TIER_COLORS = {"Low": "#86b6ef", "Mid": "#256abf", "High": "#0d366b"}   # one hue, light to dark: tiers are ordered
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_skill_matrix(project_root: Path) -> tuple:
    """Load cleaned jobs and return (jobs, multi-hot DataFrame over the vocabulary skills)."""
    jobs = load_cleaned_jobs(project_root / "data" / "processed" / "cleaned_jobs.csv")
    skills = pd.read_csv(project_root / "data" / "processed" / "skill_vocabulary.csv")["skill"].tolist()
    sets = jobs["matched_skills"].map(set)
    matrix = pd.DataFrame({s: sets.map(lambda x, k=s: k in x) for s in skills}, index=jobs.index).astype(np.int8)
    logger.info("Loaded %d jobs x %d skills", *matrix.shape)
    return jobs, matrix


def score_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Return silhouette (on a fixed sample) and Davies-Bouldin for labels, ignoring noise (-1)."""
    mask = labels >= 0
    if len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan}
    Xm, lm = X[mask], labels[mask]
    return {
        "silhouette": silhouette_score(Xm, lm, sample_size=min(SILHOUETTE_SAMPLE, len(lm)),
                                       random_state=RANDOM_STATE),
        "davies_bouldin": davies_bouldin_score(Xm, lm),
    }


def kmeans_sweep(X: np.ndarray, k_range=K_RANGE) -> tuple:
    """Fit K-Means for each k and return (metrics table, fitted models by k)."""
    rows, models = [], {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        models[k] = model
        rows.append({"k": k, "inertia": model.inertia_, **score_clustering(X, model.labels_)})
        logger.info("K-Means k=%d: silhouette %.3f, Davies-Bouldin %.3f",
                    k, rows[-1]["silhouette"], rows[-1]["davies_bouldin"])
    return pd.DataFrame(rows), models


def k_distances(X: np.ndarray, min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """Return each point's distance to its `min_samples`-th nearest neighbour, sorted ascending."""
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def eps_candidates(kdist: np.ndarray) -> list:
    """Return the distinct k-distance steps between the 10th and 90th percentiles as eps candidates.

    Squared distances between binary vectors are whole numbers n, so each step is exactly sqrt(n).
    A tiny tolerance is added so points exactly on the step count as neighbours.
    """
    low, high = np.quantile(kdist, [0.1, 0.9])
    steps = np.unique(np.rint(kdist ** 2).astype(int))
    return [float(np.sqrt(n) + 1e-6) for n in steps if low - 1e-6 <= np.sqrt(n) <= high + 1e-6]


def tune_dbscan(X: np.ndarray, kdist: np.ndarray, min_samples: int = DBSCAN_MIN_SAMPLES) -> tuple:
    """Run DBSCAN at every eps candidate and pick the best one.

    The pick is the highest silhouette among runs with >= 2 clusters and noise <= DBSCAN_MAX_NOISE.
    If no run qualifies, fall back to the lowest-noise run with >= 2 clusters, or else the first run.
    Returns (best result dict, table of all candidates).
    """
    results = [run_dbscan(X, eps, min_samples) for eps in eps_candidates(kdist)]
    table = pd.DataFrame([{k: r[k] for k in ["eps", "n_clusters", "noise_share", "largest_cluster_share",
                                              "silhouette", "davies_bouldin"]} for r in results])
    multi = [r for r in results if r["n_clusters"] >= 2]
    ok = [r for r in multi if r["noise_share"] <= DBSCAN_MAX_NOISE]
    if ok:
        best = max(ok, key=lambda r: r["silhouette"])
    elif multi:
        best = min(multi, key=lambda r: r["noise_share"])
    else:
        best = results[0]
    logger.info("DBSCAN chosen eps=%.2f", best["eps"])
    return best, table


def run_dbscan(X: np.ndarray, eps: float, min_samples: int = DBSCAN_MIN_SAMPLES) -> dict:
    """Fit DBSCAN and return labels, cluster count, noise share, largest-cluster share, and scores."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    clustered = labels >= 0
    sizes = np.bincount(labels[clustered]) if clustered.any() else np.array([0])
    result = {
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": int(labels.max() + 1),
        "noise_share": float(1 - clustered.mean()),
        "largest_cluster_share": float(sizes.max() / max(clustered.sum(), 1)),
        **score_clustering(X, labels),
    }
    logger.info("DBSCAN eps=%.2f min_samples=%d: %d clusters, %.1f%% noise, silhouette %.3f, DB %.3f",
                eps, min_samples, result["n_clusters"], 100 * result["noise_share"],
                result["silhouette"], result["davies_bouldin"])
    return result


def defining_skills(matrix: pd.DataFrame, labels: np.ndarray, top_n: int = TOP_SKILLS) -> dict:
    """Return each cluster's top skills, ranked by lift (in-cluster share / overall share).

    Eligible skills appear in at least MIN_SKILL_PREVALENCE of the cluster's jobs and are
    over-represented there (lift > 1), so rare skills and generic skills don't dominate.
    A cluster with no such skill (typically postings listing few skills overall) gets its
    most frequent skills instead, with `over_represented` False.
    """
    overall = matrix.mean()
    out = {}
    for c in sorted(set(labels) - {NO_SKILLS_LABEL}):
        share = matrix[labels == c].mean()
        lift = share / overall
        eligible = lift[(share >= MIN_SKILL_PREVALENCE) & (lift > 1)]
        over = not eligible.empty
        top = (eligible.sort_values(ascending=False) if over else share.sort_values(ascending=False)).head(top_n)
        out[c] = pd.DataFrame({"share_in_cluster": share[top.index], "share_overall": overall[top.index],
                               "lift": lift[top.index], "over_represented": over})
    return out


def name_cluster(skills_table: pd.DataFrame) -> str:
    """Name a cluster after the SKILL_THEMES theme with the highest lift-weighted share of its top skills.

    Clusters with no over-represented skill are named GENERALIST_NAME.
    """
    if not skills_table["over_represented"].iloc[0]:
        return GENERALIST_NAME
    scores = {}
    for skill, row in skills_table.iterrows():
        for theme, members in SKILL_THEMES.items():
            if skill in members:
                scores[theme] = scores.get(theme, 0) + row["lift"] * row["share_in_cluster"]
    if not scores:
        return "General Professional"
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0]


def name_clusters(tables: dict) -> dict:
    """Name every cluster, adding the defining skill to names that would otherwise repeat."""
    names = {c: name_cluster(t) for c, t in tables.items()}
    counts = pd.Series(names).value_counts()
    for c, name in names.items():
        if counts[name] > 1:
            names[c] = f"{name} ({tables[c].index[0]})"
    return names


def build_profiles(jobs: pd.DataFrame, matrix: pd.DataFrame, labels: np.ndarray, names: dict,
                   tables: dict) -> list:
    """Return one profile dict per cluster: size, defining skills, top titles, and salary stats."""
    profiles = []
    for c in sorted(set(labels)):
        members = jobs[labels == c]
        tiers = members["salary_tier"].value_counts(normalize=True).reindex(TIER_LABELS, fill_value=0)
        profiles.append({
            "cluster": c,
            "name": names.get(c, NO_SKILLS_NAME),
            "size": len(members),
            "share": len(members) / len(jobs),
            "skills": tables.get(c),
            "titles": members["job_title"].value_counts().head(TOP_TITLES),
            "avg_salary": members["normalized_salary"].mean(),
            "median_salary": members["normalized_salary"].median(),
            "tiers": tiers,
            "mean_skills": matrix[labels == c].sum(axis=1).mean(),
        })
    return profiles


def format_profiles(profiles: list, title: str) -> str:
    """Render cluster profiles as plain text."""
    lines = [title, "=" * 80]
    for p in profiles:
        lines += ["", f"Cluster {p['cluster']}: {p['name']}",
                  f"  Jobs: {p['size']:,} ({p['share']:.1%}); mean skills per job {p['mean_skills']:.1f}",
                  f"  Salary: mean ${p['avg_salary']:,.0f}, median ${p['median_salary']:,.0f}",
                  "  Tier mix: " + ", ".join(f"{t} {v:.0%}" for t, v in p["tiers"].items())]
        if p["skills"] is not None:
            header = ("Defining skills" if p["skills"]["over_represented"].iloc[0]
                      else "No over-represented skills; most frequent skills")
            lines.append(f"  {header} (share in cluster vs overall, lift):")
            lines += [f"    - {s}: {r.share_in_cluster:.0%} vs {r.share_overall:.0%} (lift {r.lift:.2f})"
                      for s, r in p["skills"].iterrows()]
        lines.append("  Most common job titles:")
        lines += [f"    - {t} ({n})" for t, n in p["titles"].items()]
    return "\n".join(lines) + "\n"


def compare_representations(matrix: np.ndarray, k: int) -> pd.DataFrame:
    """Score K-Means at `k` on the multi-hot matrix and on TF-IDF -> 10-dim SVD -> L2 (cosine-like).

    This is a sensitivity check: does a denser representation reveal more separated clusters?
    """
    tfidf = normalize(TfidfTransformer().fit_transform(matrix))
    svd = normalize(TruncatedSVD(n_components=10, random_state=RANDOM_STATE).fit_transform(tfidf))
    rows = []
    for name, X in [("multi-hot (used)", matrix), ("TF-IDF -> SVD(10) -> L2", svd)]:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        rows.append({"representation": name, "k": k, **score_clustering(X, labels)})
    return pd.DataFrame(rows)


def _style_axes(ax) -> None:
    """Apply the shared recessive axis styling."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_elbow(sweep: pd.DataFrame, best_k: int, path: Path) -> None:
    """Save side-by-side inertia (elbow) and silhouette curves over k, marking the chosen k."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, col, label in [(axes[0], "inertia", "Inertia (within-cluster sum of squares)"),
                           (axes[1], "silhouette", "Silhouette score")]:
        _style_axes(ax)
        ax.plot(sweep["k"], sweep[col], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        chosen = sweep.loc[sweep["k"] == best_k, col].iloc[0]
        ax.scatter([best_k], [chosen], s=160, facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.5, zorder=3)
        ax.annotate(f"chosen k = {best_k}", (best_k, chosen), xytext=(14, 0), textcoords="offset points",
                    va="center", color=TEXT_PRIMARY, fontsize=10)
        ax.set_xlabel("Number of clusters (k)", color=TEXT_SECONDARY)
        ax.set_title(label, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=10)
        ax.set_xticks(list(sweep["k"]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.3g}" if col == "silhouette"
                                                                          else "{x:,.0f}"))
    fig.suptitle("K-Means on multi-hot skill vectors: k = 3 to 10 (k chosen by silhouette)",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_kdistance(kdist: np.ndarray, eps: float, path: Path) -> None:
    """Save the sorted k-distance graph used to choose DBSCAN's eps."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.plot(np.arange(len(kdist)), kdist, color=CATEGORICAL[0], linewidth=2)
    ax.axhline(eps, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.annotate(f"chosen eps = {eps:.2f}", (0, eps), xytext=(8, 6), textcoords="offset points",
                color=TEXT_PRIMARY)
    ax.set_xlabel("Jobs sorted by distance", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Distance to {DBSCAN_MIN_SAMPLES}th nearest neighbour", color=TEXT_SECONDARY)
    ax.set_title("DBSCAN k-distance graph", loc="left", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_pca(X: np.ndarray, km_labels: np.ndarray, km_names: dict, db_labels: np.ndarray, path: Path,
             sample: int = 8000) -> float:
    """Save a 2-panel PCA(2) scatter coloured by K-Means cluster and by DBSCAN cluster.

    Plots a fixed random sample of jobs for legibility. Returns the variance explained by the 2 components.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)   # same sample as plot_sample()
    coords = pca.transform(X[idx])
    jitter = rng.normal(scale=0.03, size=coords.shape)   # binary data stacks on identical points
    coords = coords + jitter
    explained = pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    _style_axes(ax)
    for i, c in enumerate(sorted(km_names)):
        m = km_labels[idx] == c
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.45, color=CATEGORICAL[i % len(CATEGORICAL)],
                   edgecolor="none", label=f"{c}: {km_names[c]}")
    ax.set_title(f"K-Means (k = {len(km_names)})", loc="left", color=TEXT_PRIMARY, fontsize=12)
    leg = ax.legend(loc="upper right", frameon=False, markerscale=2.5, labelcolor=TEXT_SECONDARY, fontsize=9)

    ax = axes[1]
    _style_axes(ax)
    db = db_labels[idx]
    noise = db < 0
    ax.scatter(coords[noise, 0], coords[noise, 1], s=8, alpha=0.35, color=NOISE_COLOR, edgecolor="none",
               label=f"Noise ({noise.mean():.0%})")
    sizes = pd.Series(db[~noise]).value_counts()
    for i, c in enumerate(sizes.index[:len(CATEGORICAL) - 1]):
        m = db == c
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.6, color=CATEGORICAL[i], edgecolor="none",
                   label=f"Cluster {c} ({m.mean():.0%})")
    rest = ~noise & ~np.isin(db, sizes.index[:len(CATEGORICAL) - 1])
    if rest.any():
        ax.scatter(coords[rest, 0], coords[rest, 1], s=8, alpha=0.6, color=CATEGORICAL[-1], edgecolor="none",
                   label=f"Other clusters ({rest.mean():.0%})")
    ax.set_title(f"DBSCAN ({int(db_labels.max() + 1)} clusters + noise)", loc="left",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.legend(loc="upper right", frameon=False, markerscale=2.5, labelcolor=TEXT_SECONDARY, fontsize=9)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)", color=TEXT_SECONDARY)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)", color=TEXT_SECONDARY)
    fig.suptitle(f"Job postings in 2D (PCA of multi-hot skills, {explained:.1%} of variance; "
                 f"{len(idx):,} sampled jobs)", x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)
    return explained


def outlier_features(jobs: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the Isolation Forest inputs: skill multi-hot, experience_level ordinal, normalized_salary.

    No scaling is needed: Isolation Forest picks split points uniformly between each feature's
    min and max, so it is invariant to per-feature scale.
    """
    features = matrix.copy()
    features["experience_level"] = jobs["experience_level"].map(EXPERIENCE_TO_CODE).fillna(-1).astype(int)
    features["normalized_salary"] = jobs["normalized_salary"]
    return features


def detect_outliers(features: pd.DataFrame, contamination: float = OUTLIER_CONTAMINATION) -> tuple:
    """Fit an Isolation Forest and return (flagged mask, anomaly score; higher = more anomalous)."""
    iso = IsolationForest(n_estimators=OUTLIER_N_ESTIMATORS, contamination=contamination,
                          random_state=RANDOM_STATE, n_jobs=-1).fit(features)
    flags = iso.predict(features) == -1
    scores = -iso.score_samples(features)
    logger.info("Isolation Forest (contamination=%.2f): %d of %d jobs flagged",
                contamination, int(flags.sum()), len(flags))
    return flags, scores


def salary_mismatch_z(jobs: pd.DataFrame, km_labels: np.ndarray) -> pd.Series:
    """Return each job's log-salary z-score within its (K-Means cluster, experience_level) group.

    The group is the job's skill profile plus seniority, so a large positive z is high pay for that
    profile and a large negative z is low pay for it.
    """
    log_salary = np.log(jobs["normalized_salary"])
    groups = [km_labels, jobs["experience_level"].to_numpy()]
    mean = log_salary.groupby(groups).transform("mean")
    std = log_salary.groupby(groups).transform("std")
    return ((log_salary - mean) / std).fillna(0)


def compact_outlier_features(features: pd.DataFrame, n_skills: pd.Series) -> np.ndarray:
    """Return a sensitivity feature set: skills as SVD components + skill count, experience, salary.

    With 245 binary skill columns, a random Isolation Forest split rarely lands on salary. Here the
    skills are compressed to OUTLIER_SVD_COMPONENTS columns so salary is about 1 in 13 of the features.
    """
    skills = features.drop(columns=["experience_level", "normalized_salary"]).to_numpy(dtype=float)
    svd = TruncatedSVD(n_components=OUTLIER_SVD_COMPONENTS, random_state=RANDOM_STATE).fit_transform(skills)
    return np.column_stack([svd, n_skills, features["experience_level"], features["normalized_salary"]])


def outlier_table(jobs: pd.DataFrame, flags: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Return the flagged jobs with the 06_outliers.csv columns, most anomalous first."""
    out = jobs.loc[flags, ["job_id", "job_title", "company_name", "normalized_salary", "salary_tier",
                           "matched_skills"]].copy()
    out["matched_skills"] = out["matched_skills"].map("; ".join)
    out["anomaly_score"] = scores[flags].round(4)
    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def describe_outliers(jobs: pd.DataFrame, matrix: pd.DataFrame, flags: np.ndarray, scores: np.ndarray,
                      z: pd.Series, sensitivity_flags: np.ndarray) -> str:
    """Build the outlier section of the Stage 6 summary.

    Compares flagged jobs with all jobs (skills, salary, tier, experience), correlates the anomaly
    score with each input, tests the salary-mismatch explanation, and reports the sensitivity run.
    """
    line = "=" * 80
    n_skills = matrix.sum(axis=1)
    exp_code = jobs["experience_level"].map(EXPERIENCE_TO_CODE).fillna(-1)
    mismatch = z.abs() > MISMATCH_Z

    def profile(mask: np.ndarray) -> dict:
        """Summarize a subset of jobs."""
        sub = jobs[mask]
        tiers = sub["salary_tier"].value_counts(normalize=True).reindex(TIER_LABELS, fill_value=0)
        return {"jobs": int(mask.sum()), "mean_skills": n_skills[mask].mean(),
                "median_salary": sub["normalized_salary"].median(),
                **{f"{t}_share": tiers[t] for t in TIER_LABELS},
                "mismatch_share": mismatch[mask].mean(),
                "high_for_profile": (z[mask] > MISMATCH_Z).mean(),
                "low_for_profile": (z[mask] < -MISMATCH_Z).mean()}

    everyone = np.ones(len(jobs), dtype=bool)
    comparison = pd.DataFrame([{"group": "all jobs", **profile(everyone)},
                               {"group": "flagged (main)", **profile(flags)},
                               {"group": "flagged (sensitivity)", **profile(sensitivity_flags)}])
    correlations = pd.Series({
        "n_matched_skills": pd.Series(scores).corr(n_skills.reset_index(drop=True), method="spearman"),
        "normalized_salary": pd.Series(scores).corr(jobs["normalized_salary"].reset_index(drop=True),
                                                    method="spearman"),
        "experience_level": pd.Series(scores).corr(exp_code.reset_index(drop=True), method="spearman"),
    })
    lift = (matrix[flags].mean() / matrix.mean()).where(matrix[flags].mean() >= 0.25).dropna()
    experience = pd.DataFrame({
        "all jobs": jobs["experience_level"].value_counts(normalize=True),
        "flagged": jobs.loc[flags, "experience_level"].value_counts(normalize=True),
    }).fillna(0).sort_values("all jobs", ascending=False)
    overlap = (flags & sensitivity_flags).sum() / max(flags.sum(), 1)
    driver = correlations.abs().idxmax()
    return "\n".join([
        f"{line}\nOutlier detection (Isolation Forest)\n{line}",
        f"Features: {matrix.shape[1]} matched_skills multi-hot + experience_level ordinal + normalized_salary. "
        f"contamination={OUTLIER_CONTAMINATION}, {OUTLIER_N_ESTIMATORS} trees, random_state={RANDOM_STATE}.",
        f"Flagged {int(flags.sum()):,} of {len(jobs):,} jobs -> outputs/06_outliers.csv "
        f"(anomaly_score = -score_samples, higher = more anomalous; range "
        f"{scores[flags].min():.3f}-{scores[flags].max():.3f}).",
        f"Salary mismatch = |z| > {MISMATCH_Z} for log salary within the job's (K-Means cluster, "
        f"experience_level) group.",
        "",
        comparison.round(3).to_string(index=False),
        "",
        "Spearman correlation of anomaly score with each input:",
        correlations.round(3).to_string(),
        f"Strongest driver: {driver}.",
        "",
        "Experience level mix:",
        experience.round(3).to_string(),
        "",
        "Skills most over-represented among flagged jobs (in >= 25% of them; lift vs all jobs):",
        lift.sort_values(ascending=False).head(10).round(2).to_string(),
        "",
        f"Sensitivity run (skills -> SVD({OUTLIER_SVD_COMPONENTS}) + skill count + experience + salary): "
        f"{int(sensitivity_flags.sum()):,} flagged, {overlap:.0%} overlap with the main run.",
        "",
        f"Conclusion: the anomaly score is driven by {driver} (Spearman {correlations[driver]:.2f}). "
        f"Flagged jobs list {comparison.loc[1, 'mean_skills']:.1f} skills on average vs "
        f"{comparison.loc[0, 'mean_skills']:.1f} overall. Salary mismatches make up "
        f"{comparison.loc[1, 'mismatch_share']:.1%} of flagged jobs vs {comparison.loc[0, 'mismatch_share']:.1%} "
        f"of all jobs ({comparison.loc[2, 'mismatch_share']:.1%} in the sensitivity run).",
    ]) + "\n"


def plot_sample(n: int, sample: int = PLOT_SAMPLE) -> np.ndarray:
    """Return the fixed random sample of row indices shown in the 2D plots."""
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.choice(n, size=min(sample, n), replace=False)


def run_tsne(X: np.ndarray) -> np.ndarray:
    """Embed X in 2D with t-SNE (perplexity TSNE_PERPLEXITY, TSNE_MAX_ITER iterations, PCA init)."""
    coords = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_MAX_ITER, init="pca",
                  learning_rate="auto", random_state=RANDOM_STATE, n_jobs=-1).fit_transform(X)
    logger.info("t-SNE on %d jobs x %d skills", *X.shape)
    return coords


def knn_indices(coords: np.ndarray, k: int = PURITY_NEIGHBORS) -> np.ndarray:
    """Return each point's k nearest neighbours (excluding itself) as an (n, k) index array."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    return idx[:, 1:]


def compare_embeddings(X_sample: np.ndarray, pca_coords: np.ndarray, tsne_coords: np.ndarray,
                       km_labels: np.ndarray, tiers: np.ndarray) -> pd.DataFrame:
    """Score the full space, PCA 2D, and t-SNE 2D by kNN overlap with the full space and by label purity.

    knn_overlap: share of a job's PURITY_NEIGHBORS nearest neighbours that are also its nearest
    neighbours in the 245-D multi-hot space (1.0 for the full space itself). kmeans_purity and
    tier_purity: share of the neighbours with the same K-Means cluster / salary tier. The chance row
    is the sum of squared label shares (the purity of random neighbours).
    """
    full = knn_indices(X_sample)
    rows = []
    for name, coords in [("multi-hot (245-D)", X_sample), ("PCA 2D", pca_coords), ("t-SNE 2D", tsne_coords)]:
        idx = full if coords is X_sample else knn_indices(coords)
        overlap = np.mean([len(set(a) & set(b)) for a, b in zip(idx, full)]) / PURITY_NEIGHBORS
        rows.append({"space": name, "knn_overlap": overlap,
                     "kmeans_purity": float((km_labels[idx] == km_labels[:, None]).mean()),
                     "tier_purity": float((tiers[idx] == tiers[:, None]).mean())})
    chance = {"space": "chance", "knn_overlap": PURITY_NEIGHBORS / len(X_sample),
              **{col: float((pd.Series(lab).value_counts(normalize=True) ** 2).sum())
                 for col, lab in [("kmeans_purity", km_labels), ("tier_purity", tiers)]}}
    return pd.DataFrame(rows + [chance])


def high_tier_hotspots(tsne_coords: np.ndarray, tiers: np.ndarray, matrix_sample: pd.DataFrame,
                       titles: pd.Series, km_labels: np.ndarray, km_names: dict,
                       overall_share: pd.Series) -> pd.DataFrame:
    """Find t-SNE regions dominated by High-tier jobs and describe each one.

    A job is in a hotspot when >= HOTSPOT_MIN_HIGH_SHARE of its HOTSPOT_NEIGHBORS t-SNE neighbours are
    High tier. Those jobs are grouped with DBSCAN (HOTSPOT_EPS, HOTSPOT_MIN_SAMPLES) on the t-SNE
    coordinates. Each hotspot gets its size, High share, majority K-Means cluster, top skills by lift
    (in >= 30% of its jobs), most common titles, and centre (for plot labels). Largest first.
    """
    high = tiers == "High"
    _, idx = NearestNeighbors(n_neighbors=HOTSPOT_NEIGHBORS + 1).fit(tsne_coords).kneighbors(tsne_coords)
    hot = np.flatnonzero(high[idx[:, 1:]].mean(axis=1) >= HOTSPOT_MIN_HIGH_SHARE)
    if len(hot) == 0:
        return pd.DataFrame()
    labels = DBSCAN(eps=HOTSPOT_EPS, min_samples=HOTSPOT_MIN_SAMPLES).fit_predict(tsne_coords[hot])
    rows = []
    for c in sorted(set(labels) - {-1}):
        members = hot[labels == c]
        share = matrix_sample.iloc[members].mean()
        lift = (share / overall_share).where(share >= 0.3).dropna().sort_values(ascending=False)
        majority = pd.Series(km_labels[members]).value_counts()
        rows.append({
            "jobs": len(members),
            "high_share": high[members].mean(),
            "kmeans_majority": f"{km_names[majority.index[0]]} ({majority.iloc[0] / len(members):.0%})",
            "top_skills": ", ".join(lift.index[:5]),
            "top_titles": "; ".join(titles.iloc[members].value_counts().head(3).index),
            "x": tsne_coords[members, 0].mean(),
            "y": tsne_coords[members, 1].mean(),
        })
    return pd.DataFrame(rows).sort_values("jobs", ascending=False).reset_index(drop=True)


def plot_tsne_vs_pca(pca_coords: np.ndarray, tsne_coords: np.ndarray, km_labels: np.ndarray, km_names: dict,
                     tiers: np.ndarray, explained: float, path: Path, hotspots: pd.DataFrame = None) -> None:
    """Save a 2x2 grid: PCA (top) and t-SNE (bottom), coloured by K-Means cluster (left) and salary tier (right).

    Points are drawn in a shuffled order so no group is painted over the others. The t-SNE salary
    panel labels each High-tier hotspot with its top skill and High share.
    """
    order = np.random.default_rng(RANDOM_STATE).permutation(len(tiers))
    cluster_ids = sorted(km_names)
    cluster_color = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(cluster_ids)}
    colorings = [
        ("K-Means cluster", np.array([cluster_color[c] for c in km_labels]),
         [(cluster_color[c], f"{c}: {km_names[c]}") for c in cluster_ids]),
        ("Salary tier", np.array([TIER_COLORS[t] for t in tiers]),
         [(TIER_COLORS[t], t) for t in TIER_LABELS]),
    ]
    views = [(f"PCA ({explained:.1%} of variance)", pca_coords),
             (f"t-SNE (perplexity {TSNE_PERPLEXITY}, {TSNE_MAX_ITER:,} iterations)", tsne_coords)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for r, (view, coords) in enumerate(views):
        for c, (coloring, colors, legend) in enumerate(colorings):
            ax = axes[r, c]
            _style_axes(ax)
            ax.scatter(coords[order, 0], coords[order, 1], s=6, alpha=0.55, c=colors[order], edgecolor="none")
            ax.set_title(f"{view} — coloured by {coloring}", loc="left", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            handles = [ax.scatter([], [], s=40, color=col, label=lab) for col, lab in legend]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(handles),
                      frameon=False, labelcolor=TEXT_SECONDARY, fontsize=9)
    if hotspots is not None and not hotspots.empty:
        ax = axes[1, 1]
        shown = hotspots[hotspots["jobs"] >= HOTSPOT_MIN_SAMPLES]
        key = []
        for i, (_, h) in enumerate(shown.iterrows(), start=1):
            ax.scatter([h["x"]], [h["y"]], s=260, facecolor=SURFACE, edgecolor=TEXT_PRIMARY, linewidth=1.2,
                       zorder=3)
            ax.text(h["x"], h["y"], str(i), ha="center", va="center", fontsize=9, color=TEXT_PRIMARY, zorder=4)
            key.append(f"{i}  {' + '.join(h['top_skills'].split(', ')[:2])}: "
                       f"{h['high_share']:.0%} High, {h['jobs']} jobs")
        ax.text(0.01, 0.99, "High-tier hotspots\n" + "\n".join(key), transform=ax.transAxes, ha="left",
                va="top", fontsize=9, color=TEXT_PRIMARY, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, alpha=0.92))
    fig.suptitle(f"PCA vs t-SNE of multi-hot skill vectors ({len(tiers):,} sampled jobs); "
                 f"labels mark High-tier hotspots",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("Saved %s", path)


def format_embedding_section(purity: pd.DataFrame, hotspots: pd.DataFrame, n_sample: int) -> str:
    """Build the PCA vs t-SNE section of the Stage 6 summary."""
    line = "=" * 80
    row = purity.set_index("space")
    cols = ["jobs", "high_share", "kmeans_majority", "top_skills", "top_titles"]
    return "\n".join([
        f"{line}\nPCA vs t-SNE (outputs/06_clusters_tsne.png)\n{line}",
        f"t-SNE: perplexity {TSNE_PERPLEXITY}, {TSNE_MAX_ITER:,} iterations, PCA init, random_state={RANDOM_STATE}, "
        f"on the same {n_sample:,}-job sample as the PCA plot. k = {PURITY_NEIGHBORS} neighbours.",
        purity.round(3).to_string(index=False),
        f"kNN overlap with the 245-D space: t-SNE {row.loc['t-SNE 2D', 'knn_overlap']:.1%} vs PCA "
        f"{row.loc['PCA 2D', 'knn_overlap']:.1%}, so t-SNE keeps far more of each job's local neighbourhood.",
        f"PCA has the higher K-Means purity ({row.loc['PCA 2D', 'kmeans_purity']:.2f} vs "
        f"{row.loc['t-SNE 2D', 'kmeans_purity']:.2f}): K-Means splits along the top principal directions, so "
        f"PCA draws its partition cleanly. Neither 2D view separates salary tiers much better than the full space "
        f"(tier purity {row.loc['t-SNE 2D', 'tier_purity']:.2f} t-SNE, {row.loc['PCA 2D', 'tier_purity']:.2f} PCA, "
        f"{row.loc['multi-hot (245-D)', 'tier_purity']:.2f} full, {row.loc['chance', 'tier_purity']:.2f} chance).",
        "",
        f"High-tier hotspots in t-SNE (>= {HOTSPOT_MIN_HIGH_SHARE:.0%} of {HOTSPOT_NEIGHBORS} neighbours High; "
        f"DBSCAN eps {HOTSPOT_EPS}, min_samples {HOTSPOT_MIN_SAMPLES}):",
        hotspots[cols].round(3).to_string(index=False) if not hotspots.empty else "none",
    ]) + "\n"


def build_summary(n_jobs: int, n_clustered: int, n_skills: int, sweep: pd.DataFrame, best_k: int, kmeans_scores: dict,
                  dbscan: dict, dbscan_table: pd.DataFrame, comparison: pd.DataFrame, profiles: list,
                  explained: float, extra_sections: list = None) -> str:
    """Build the plain-text Stage 6 summary, ending with any `extra_sections` (pre-formatted text)."""
    line = "=" * 80

    def verdict(value, target, higher_is_better):
        """Return MET / NOT MET for a metric against its target."""
        if np.isnan(value):
            return "n/a"
        ok = value > target if higher_is_better else value < target
        return "MET" if ok else "NOT MET"

    evaluation = pd.DataFrame([
        {"method": f"K-Means (k={best_k})", "clusters": best_k, "noise_share": 0.0,
         "silhouette": kmeans_scores["silhouette"], "davies_bouldin": kmeans_scores["davies_bouldin"]},
        {"method": f"DBSCAN (eps={dbscan['eps']:.2f}, min_samples={dbscan['min_samples']})",
         "clusters": dbscan["n_clusters"], "noise_share": dbscan["noise_share"],
         "silhouette": dbscan["silhouette"], "davies_bouldin": dbscan["davies_bouldin"]},
    ])
    evaluation["silhouette_target"] = [verdict(v, SILHOUETTE_TARGET, True) for v in evaluation["silhouette"]]
    evaluation["db_target"] = [verdict(v, DAVIES_BOULDIN_TARGET, False) for v in evaluation["davies_bouldin"]]
    parts = [
        "SkillMap — Stage 6 Clustering Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Vectors: multi-hot over {n_skills} matched_skills. Jobs: {n_jobs:,}; clustered {n_clustered:,} "
        f"({n_jobs - n_clustered:,} with no matched skill labelled {NO_SKILLS_LABEL}).",
        f"Silhouette on a fixed random sample of {SILHOUETTE_SAMPLE:,} jobs (DBSCAN: non-noise jobs only).",
        "",
        f"{line}\nK-Means sweep\n{line}",
        sweep.round(4).to_string(index=False),
        f"Chosen k = {best_k} (highest silhouette).",
        "",
        f"{line}\nDBSCAN\n{line}",
        f"min_samples = {dbscan['min_samples']}. eps candidates = k-distance steps between the 10th and "
        f"90th percentiles (outputs/06_kdistance_plot.png):",
        dbscan_table.round(4).to_string(index=False),
        f"Chosen eps = {dbscan['eps']:.2f}: highest silhouette with >= 2 clusters and <= "
        f"{DBSCAN_MAX_NOISE:.0%} noise.",
        f"Clusters: {dbscan['n_clusters']}, noise: {dbscan['noise_share']:.1%}, "
        f"largest cluster holds {dbscan['largest_cluster_share']:.1%} of clustered jobs",
        "",
        f"{line}\nEvaluation vs targets (silhouette > {SILHOUETTE_TARGET}, Davies-Bouldin < "
        f"{DAVIES_BOULDIN_TARGET})\n{line}",
        evaluation.round(4).to_string(index=False),
        "",
        "Sensitivity: same k on a denser representation",
        comparison.round(4).to_string(index=False),
        "",
        f"PCA 2D explains {explained:.1%} of variance (outputs/06_clusters_pca.png).",
        "",
        f"{line}\nK-Means clusters (details in outputs/06_cluster_profiles.txt)\n{line}",
    ]
    for p in profiles:
        skills = ", ".join(p["skills"].index) if p["skills"] is not None else "-"
        parts.append(f"{p['cluster']:>2} {p['name']:<45} {p['size']:>6,} jobs  "
                     f"mean ${p['avg_salary']:>9,.0f}  High {p['tiers']['High']:.0%}  | {skills}")
    for section in extra_sections or []:
        parts += ["", section.rstrip("\n")]
    return "\n".join(parts) + "\n"


def run_pipeline(project_root: Path = None) -> dict:
    """Run Stage 6 end to end and write labels, plots, profiles, and summary to outputs/."""
    root = project_root or find_project_root()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs, matrix = load_skill_matrix(root)
    has_skills = (matrix.sum(axis=1) > 0).to_numpy()
    X = matrix.to_numpy(dtype=float)[has_skills]
    logger.info("Clustering %d jobs with >= 1 skill (%d without skills set aside)",
                len(X), int((~has_skills).sum()))

    sweep, km_models = kmeans_sweep(X)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    km_fit = km_models[best_k].labels_
    plot_elbow(sweep, best_k, output_dir / "06_elbow_plot.png")

    kdist = k_distances(X)
    dbscan, dbscan_table = tune_dbscan(X, kdist)
    plot_kdistance(kdist, dbscan["eps"], output_dir / "06_kdistance_plot.png")

    km_labels = np.full(len(jobs), NO_SKILLS_LABEL)
    km_labels[has_skills] = km_fit
    db_labels = np.full(len(jobs), NO_SKILLS_LABEL)
    db_labels[has_skills] = dbscan["labels"]

    tables = defining_skills(matrix, km_labels)
    names = name_clusters(tables)
    profiles = build_profiles(jobs, matrix, km_labels, names, tables)
    (output_dir / "06_cluster_profiles.txt").write_text(
        format_profiles(profiles, f"SkillMap — K-Means cluster profiles (k = {best_k})"), encoding="utf-8")

    labels_out = pd.DataFrame({
        "job_id": jobs["job_id"],
        "job_title": jobs["job_title"],
        "kmeans_cluster": km_labels,
        "kmeans_cluster_name": [names.get(c, NO_SKILLS_NAME) for c in km_labels],
        "dbscan_cluster": db_labels,   # -1 = noise or no matched skills
        "salary_tier": jobs["salary_tier"],
        "normalized_salary": jobs["normalized_salary"],
    })
    labels_out.to_csv(output_dir / "06_cluster_labels.csv", index=False)
    logger.info("Saved %s", output_dir / "06_cluster_labels.csv")

    explained = plot_pca(X, km_fit, names, dbscan["labels"], output_dir / "06_clusters_pca.png")
    sample = plot_sample(len(X))
    X_sample = X[sample]
    pca_coords = PCA(n_components=2, random_state=RANDOM_STATE).fit(X).transform(X_sample)
    tsne_coords = run_tsne(X_sample)
    tiers_sample = jobs.loc[has_skills, "salary_tier"].to_numpy()[sample]
    purity = compare_embeddings(X_sample, pca_coords, tsne_coords, km_fit[sample], tiers_sample)
    hotspots = high_tier_hotspots(tsne_coords, tiers_sample, matrix[has_skills].iloc[sample],
                                  jobs.loc[has_skills, "job_title"].iloc[sample], km_fit[sample], names,
                                  matrix.mean())
    jitter = np.random.default_rng(RANDOM_STATE).normal(scale=0.03, size=pca_coords.shape)
    plot_tsne_vs_pca(pca_coords + jitter, tsne_coords, km_fit[sample], names, tiers_sample, explained,
                     output_dir / "06_clusters_tsne.png", hotspots)
    comparison = compare_representations(X, best_k)
    kmeans_scores = sweep.loc[sweep["k"] == best_k, ["silhouette", "davies_bouldin"]].iloc[0].to_dict()

    features = outlier_features(jobs, matrix)
    outlier_flags, anomaly_scores = detect_outliers(features)
    sensitivity_flags, _ = detect_outliers(compact_outlier_features(features, matrix.sum(axis=1)))
    outliers = outlier_table(jobs, outlier_flags, anomaly_scores)
    outliers.to_csv(output_dir / "06_outliers.csv", index=False)
    logger.info("Saved %s", output_dir / "06_outliers.csv")
    z = salary_mismatch_z(jobs, km_labels)
    outlier_section = describe_outliers(jobs, matrix, outlier_flags, anomaly_scores, z, sensitivity_flags)

    summary = build_summary(len(jobs), len(X), X.shape[1], sweep, best_k, kmeans_scores, dbscan, dbscan_table,
                            comparison, profiles, explained,
                            [format_embedding_section(purity, hotspots, len(sample)), outlier_section])
    (output_dir / "06_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved %s", output_dir / "06_summary.txt")
    return {"jobs": jobs, "matrix": matrix, "X": X, "has_skills": has_skills, "sweep": sweep,
            "best_k": best_k, "km_labels": km_labels, "dbscan": dbscan, "dbscan_table": dbscan_table,
            "db_labels": db_labels,
            "kdist": kdist, "names": names, "profiles": profiles, "comparison": comparison,
            "tsne_coords": tsne_coords, "purity": purity, "hotspots": hotspots, "outliers": outliers, "outlier_flags": outlier_flags, "anomaly_scores": anomaly_scores,
            "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
