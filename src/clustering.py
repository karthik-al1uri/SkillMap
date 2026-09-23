"""SkillMap Stage 6 — K-Means and DBSCAN clustering of job postings into role archetypes.

Each job is a multi-hot vector over the 100-skill vocabulary (`matched_skills`, as in
Stage 5). Jobs with no matched skill are all-zero vectors, so they are left out of
clustering and labelled -1 ("No matched skills").

K-Means is run for k = 3..10 and k is chosen by silhouette score. DBSCAN's eps is tuned
from the k-distance graph. On binary vectors, Euclidean distances can only be sqrt(n) for
whole numbers n, so the graph is a staircase. Each step between the 10th and 90th
percentile of the k-distances is a candidate eps. The chosen candidate gives >= 2 clusters
with <= DBSCAN_MAX_NOISE noise and the highest silhouette.
Both are scored with silhouette (target > 0.50) and Davies-Bouldin (target < 1.0). Silhouette is computed on a fixed random sample of
SILHOUETTE_SAMPLE jobs, because the exact score is O(n^2).

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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.preprocessing import find_project_root, load_cleaned_jobs

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
K_RANGE = range(3, 11)
SILHOUETTE_SAMPLE = 10_000
SILHOUETTE_TARGET = 0.50
DAVIES_BOULDIN_TARGET = 1.0
DBSCAN_MIN_SAMPLES = 20
DBSCAN_MAX_NOISE = 0.5        # candidate eps values labelling more than this share as noise are rejected
TOP_SKILLS = 5
TOP_TITLES = 5
MIN_SKILL_PREVALENCE = 0.15   # a defining skill must appear in >= 15% of the cluster's jobs
GENERALIST_NAME = "Generalist (few listed skills)"
NO_SKILLS_LABEL = -1
NO_SKILLS_NAME = "No matched skills"
TIER_LABELS = ["Low", "Mid", "High"]

# Skill -> theme used to name clusters from their defining skills.
SKILL_THEMES = {
    "Healthcare & Clinical": [
        "patient care", "nursing", "registered nurse", "bls", "cpr", "acls", "healthcare",
        "medication administration", "patient education", "case management"],
    "Data, Tech & Engineering": [
        "python", "sql", "data analysis", "engineering", "troubleshooting", "research",
        "quality assurance", "analytical skills", "computer skills"],
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

# Chart styling: light surface, text inks, categorical slots in fixed order, noise gray.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NOISE_COLOR = "#c9c8c2"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_skill_matrix(project_root: Path) -> tuple:
    """Load cleaned jobs and return (jobs, multi-hot DataFrame over the 100 vocabulary skills)."""
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
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
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


def build_summary(n_jobs: int, n_clustered: int, sweep: pd.DataFrame, best_k: int, kmeans_scores: dict,
                  dbscan: dict, dbscan_table: pd.DataFrame, comparison: pd.DataFrame, profiles: list,
                  explained: float) -> str:
    """Build the plain-text Stage 6 summary."""
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
        f"Vectors: multi-hot over 100 matched_skills. Jobs: {n_jobs:,}; clustered {n_clustered:,} "
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
    comparison = compare_representations(X, best_k)
    kmeans_scores = sweep.loc[sweep["k"] == best_k, ["silhouette", "davies_bouldin"]].iloc[0].to_dict()
    summary = build_summary(len(jobs), len(X), sweep, best_k, kmeans_scores, dbscan, dbscan_table,
                            comparison, profiles, explained)
    (output_dir / "06_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved %s", output_dir / "06_summary.txt")
    return {"jobs": jobs, "matrix": matrix, "X": X, "has_skills": has_skills, "sweep": sweep,
            "best_k": best_k, "km_labels": km_labels, "dbscan": dbscan, "dbscan_table": dbscan_table,
            "db_labels": db_labels,
            "kdist": kdist, "names": names, "profiles": profiles, "comparison": comparison,
            "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
