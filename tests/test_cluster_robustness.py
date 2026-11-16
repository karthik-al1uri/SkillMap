"""Tiny checks for bounded saved-feature clustering experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scripts.check_cluster_robustness import compare, load_saved_features, main


def matrix():
    return sparse.csr_matrix(np.tile([[1,1,0,0], [1,0,0,0], [0,0,1,1], [0,0,1,0]], (12,1)), dtype=float)


def test_compare_fixed_score_rows_stability_and_size_totals():
    metrics, labels = compare(matrix(), k_values=(2,), seeds=(42,43), components=2, score_rows=30)
    assert metrics["score_rows"] == 30
    assert len(metrics["runs"]) == 4
    assert set(labels) == {"binary_k2_seed42", "binary_k2_seed43", "svd_l2_k2_seed42", "svd_l2_k2_seed43"}
    assert all(sum(run["cluster_sizes"]) == 48 for run in metrics["runs"])
    assert all(run["min_cluster_share"] > 0 for run in metrics["runs"])
    assert all(summary["mean_pairwise_seed_ari"] == 1 for summary in metrics["summary"])
    assert all(summary["mean_cosine_silhouette_original"] > .5 for summary in metrics["summary"])


@pytest.mark.parametrize("kwargs", [{"score_rows":1001}, {"components":33}, {"k_values":(8,)}, {"seeds":(1,2,3,4)}])
def test_computational_caps(kwargs):
    with pytest.raises(ValueError):
        compare(matrix(), **kwargs)


def test_saved_features_preserve_zero_exclusion_vocabulary_and_inputs(tmp_path):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"skill":[f"s{i}" for i in range(245)]}).to_csv(processed / "skill_vocabulary.csv", index=False)
    path = processed / "cleaned_jobs.csv"
    pd.DataFrame({"job_id":[1,2,3,4], "matched_skills":["['s1']", "[]", "['s2']", "['s1','s2']"]}).to_csv(path,index=False)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    ids, X, provenance = load_saved_features(tmp_path)
    assert ids.tolist() == [1,3,4]
    assert X.shape == (3,245)
    assert provenance["zero_skill_rows_excluded"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError):
        load_saved_features(tmp_path, 6001)


def test_output_refuses_canonical_and_existing_destination(tmp_path):
    root=tmp_path / "root"
    root.mkdir()
    for out in (root / "new", tmp_path):
        with pytest.raises(SystemExit):
            main(["--root", str(root), "--out-dir", str(out)])
