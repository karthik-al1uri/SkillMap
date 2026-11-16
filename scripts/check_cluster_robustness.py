"""Bounded offline comparison of saved skill vectors; never rewrites canonical artifacts."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

SEEDS = (42, 43, 44)
K_VALUES = (3, 5, 7)
MAX_ROWS = 6000
SCORE_ROWS = 1000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved_features(root, max_rows=MAX_ROWS):
    """Reuse saved matched skills; sample uniformly without salary-based selection."""
    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("max_rows must be in 1..6000")
    root = Path(root)
    jobs_path = root / "data/processed/cleaned_jobs.csv"
    vocab_path = root / "data/processed/skill_vocabulary.csv"
    jobs = pd.read_csv(jobs_path, usecols=["job_id", "matched_skills"])
    skills = pd.read_csv(vocab_path)["skill"].tolist()
    if len(skills) != 245 or len(set(skills)) != 245:
        raise ValueError("Expected the frozen unique 245-skill vocabulary")
    lookup = {skill: i for i, skill in enumerate(skills)}
    rows, cols = [], []
    for i, text in enumerate(jobs.matched_skills):
        for skill in set(ast.literal_eval(text)):
            if skill not in lookup:
                raise ValueError(f"Unknown saved skill: {skill}")
            rows.append(i)
            cols.append(lookup[skill])
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(len(jobs), len(skills)))
    eligible = np.flatnonzero(np.asarray(matrix.sum(axis=1)).ravel() > 0)
    selected = np.sort(np.random.default_rng(42).choice(eligible, size=min(max_rows, len(eligible)), replace=False))
    return jobs.iloc[selected].job_id.to_numpy(), matrix[selected], {
        "total_saved_rows": len(jobs), "nonzero_rows": len(eligible), "zero_skill_rows_excluded": len(jobs)-len(eligible),
        "sample_rows": len(selected), "sample_seed": 42, "sampling": "uniform without replacement among nonzero skill vectors",
        "sources": {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in (jobs_path, vocab_path)},
    }


def score_labels(X, labels, score_indices, metric):
    sample_labels = labels[score_indices]
    if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(sample_labels):
        return None
    return float(silhouette_score(X[score_indices], sample_labels, metric=metric))


def compare(X, k_values=K_VALUES, seeds=SEEDS, components=32, score_rows=SCORE_ROWS):
    """Same-sample seed stability, fixed score rows and common original-space metrics."""
    if len(X.shape) != 2 or X.shape[0] < 4 or X.shape[0] > MAX_ROWS:
        raise ValueError("Expected 4..6000 rows")
    if not seeds or len(seeds) > 3 or len(k_values) > 3 or any(k < 2 or k >= X.shape[0] or k > 7 for k in k_values):
        raise ValueError("Bounded sweep requires <=3 seeds and <=3 k values in 2..7")
    if not 2 <= score_rows <= SCORE_ROWS or not 1 <= components <= 32:
        raise ValueError("Score sample <=1000 and SVD components <=32 required")
    score_indices = np.sort(np.random.default_rng(42).choice(X.shape[0], size=min(score_rows, X.shape[0]), replace=False))
    dimensions = min(components, X.shape[1]-1, X.shape[0]-1)
    if dimensions < 1:
        raise ValueError("Need at least two input features")
    with threadpool_limits(limits=1):
        svd = TruncatedSVD(n_components=dimensions, random_state=42, n_iter=5)
        projected = normalize(svd.fit_transform(X), norm="l2")
        representations = {"binary": X, "svd_l2": projected}
        runs, summaries, label_columns = [], [], {}
        for name, features in representations.items():
            for k in k_values:
                labels_by_seed, group_rows = [], []
                for seed in seeds:
                    model = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=100, algorithm="lloyd").fit(features)
                    labels = model.labels_
                    sizes = np.bincount(labels, minlength=k)
                    record = {"representation": name, "k": k, "seed": seed,
                              "cosine_silhouette_original": score_labels(X, labels, score_indices, "cosine"),
                              "cosine_silhouette_representation": score_labels(features, labels, score_indices, "cosine"),
                              "euclidean_silhouette_representation": score_labels(features, labels, score_indices, "euclidean"),
                              "cluster_sizes": sizes.tolist(), "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
                              "min_cluster_share": float(sizes.min()/len(labels)), "max_cluster_share": float(sizes.max()/len(labels)),
                              "iterations": int(model.n_iter_), "iteration_limit_reached": bool(model.n_iter_ >= 100)}
                    runs.append(record)
                    group_rows.append(record)
                    labels_by_seed.append(labels)
                    label_columns[f"{name}_k{k}_seed{seed}"] = labels
                ari = [float(adjusted_rand_score(a,b)) for a,b in itertools.combinations(labels_by_seed, 2)]
                summaries.append({"representation": name, "k": k,
                                  "mean_cosine_silhouette_original": float(np.mean([r["cosine_silhouette_original"] for r in group_rows])),
                                  "mean_cosine_silhouette_representation": float(np.mean([r["cosine_silhouette_representation"] for r in group_rows])),
                                  "mean_euclidean_silhouette_representation": float(np.mean([r["euclidean_silhouette_representation"] for r in group_rows])),
                                  "mean_pairwise_seed_ari": float(np.mean(ari)) if ari else None,
                                  "min_pairwise_seed_ari": min(ari) if ari else None,
                                  "min_cluster_share": min(r["min_cluster_share"] for r in group_rows),
                                  "max_cluster_share": max(r["max_cluster_share"] for r in group_rows)})
    return {"runs": runs, "summary": summaries, "svd_components": dimensions,
            "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "score_rows": len(score_indices), "score_sample_indices": score_indices.tolist()}, label_columns


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    out = args.out_dir.resolve()
    if out == root or root in out.parents:
        parser.error("--out-dir must be outside the input repository")
    if out.exists():
        parser.error("--out-dir must be fresh")
    ids, X, provenance = load_saved_features(root, args.max_rows)
    metrics, columns = compare(X)
    out.mkdir(parents=True)
    report = {"provenance": provenance, "configuration": {"k": K_VALUES, "seeds": SEEDS, "n_init": 3, "max_iter": 100, "threads": 1},
              "limitations": ["Exploratory same-sample seed stability; not resampling, grouped or external validation.",
                              "SVD is fit on the exploratory sample, not a classification feature preprocessor.",
                              "KMeans on L2-normalized SVD features is not spherical KMeans; centroids are not renormalized.",
                              "Representation-specific silhouettes are not directly comparable across spaces; original-space cosine is a common reference.",
                              "Saved extracted skills and 245-skill vocabulary were reused; extraction was not rerun.",
                              "No classification labels/salary were used to choose representations, sample, k or seeds."], **metrics}
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    pd.DataFrame(metrics["summary"]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame({"job_id": ids, **columns}).to_csv(out / "labels.csv", index=False)
    print(json.dumps({"output": str(out), "provenance": provenance, "summary": metrics["summary"]}, indent=2))
    return out


if __name__ == "__main__":
    main()
