"""Tiny synthetic checks of evaluation separation and output preservation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from src import classification as cl


def postings(n=18):
    """Return tiny synthetic postings with all required feature fields."""
    return pd.DataFrame({
        "matched_skills": [["python"] if i % 2 else [] for i in range(n)],
        "experience_level": ["Entry level"] * n,
        "employee_count": [20] * n,
        "state": ["CO"] * n,
        "industry": ["Software"] * n,
        "job_title": ["software engineer"] * n,
        "company_name": ["example systems"] * n,
    })


def test_held_out_categories_and_words_never_enter_features():
    train, held = postings(), postings(3)
    held.loc[:, "state"] = "ZZ"
    held.loc[:, "industry"] = "HeldoutIndustry"
    held.loc[:, "job_title"] = "heldouttitleword"
    held.loc[:, "company_name"] = "heldoutcompanyword"
    encoder = cl.PostingFeatures(["python"], use_text=True).fit(train)
    before = encoder.transform(train)
    encoded = encoder.transform(held)
    assert "ZZ" not in encoder.top_["states"]
    assert "HeldoutIndustry" not in encoder.top_["industries"]
    assert "heldouttitleword" not in encoder.vectorizers_["job_title"].vocabulary_
    assert "heldoutcompanyword" not in encoder.vectorizers_["company_name"].vocabulary_
    assert np.array_equal(before, encoder.transform(train))
    names = list(encoder.get_feature_names_out())
    assert np.all(encoded[:, names.index("state_Other")] == 1)
    assert np.all(encoded[:, names.index("industry_Other")] == 1)
    assert not hasattr(clone(encoder), "top_")


def test_cv_fits_feature_metadata_on_each_fold(monkeypatch):
    data = postings()
    data["state"] = [f"state{i}" for i in range(len(data))]
    data["job_title"] = [f"uniquetoken{i}" for i in range(len(data))]
    y = np.tile([0, 1, 2], 6)
    observed = []
    original_fit = cl.PostingFeatures.fit

    def recording_fit(self, X, y=None):
        result = original_fit(self, X, y)
        observed.append((set(X.index), set(self.top_["states"]), set(self.vectorizers_["job_title"].vocabulary_)))
        return result

    monkeypatch.setattr(cl.PostingFeatures, "fit", recording_fit)
    candidates = {"tiny": (DummyClassifier(strategy="prior"), True, {})}
    cl.select_by_training_cv(data, y, ["python"], candidates=candidates,
                             cv=StratifiedKFold(3, shuffle=True, random_state=42))
    folds = [record for record in observed if len(record[0]) < len(data)]
    assert len(folds) == 3
    for indices, states, words in folds:
        assert states == set(data.loc[list(indices), "state"])
        assert words == set(data.loc[list(indices), "job_title"])


def test_winner_uses_cv_scores_and_accepts_no_test_argument():
    data = postings(30)
    y = np.array([0] * 18 + [1] * 6 + [2] * 6)
    candidates = {
        "always_high": (DummyClassifier(strategy="constant", constant=2), False, {}),
        "majority": (DummyClassifier(strategy="most_frequent"), False, {}),
    }
    fitted, winner, scores, _ = cl.select_by_training_cv(data, y, ["python"], candidates=candidates)
    assert winner == "majority"
    assert winner == scores.loc[scores.cv_f1_macro.idxmax(), "model"]
    # Deliberately inverted held-out labels favor the losing model; selection stays frozen.
    held_y = np.array([2] * 18 + [1] * 6 + [0] * 6)
    metrics = cl.evaluate_selected_candidates(fitted, data, y, data, held_y)
    assert metrics["always_high"]["f1_macro"] > metrics[winner]["f1_macro"]
    assert winner == "majority"
    with pytest.raises(TypeError):
        cl.select_by_training_cv(data, y, ["python"], test_df=data)


def test_fresh_output_required_and_existing_artifacts_preserved(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    old = root / "outputs"
    old.mkdir()
    model = old / "05_best_model.pkl"
    model.write_bytes(b"historical")
    for destination in [None, old, old / "new"]:
        with pytest.raises(ValueError):
            cl.prepare_output_dir(root, destination)
    with pytest.raises(FileExistsError):
        cl.prepare_output_dir(root, tmp_path)
    assert model.read_bytes() == b"historical"
    fresh = cl.prepare_output_dir(root, tmp_path / "isolated-run")
    assert fresh.is_dir()


def test_serialization_rebuilds_live_inference_features(tmp_path):
    import pickle
    from src.live_pipeline import build_live_features
    data = postings()
    data["experience_level"] = "Unknown"
    data["employee_count"] = np.nan
    data["industry"] = "Unknown"
    y = np.tile([0, 1, 2], 6)
    control = cl.fit_ordinal_control(data, y, ["python"])
    assert cl.ORDINAL_CONTROL_NAME not in cl.selection_candidates()
    features, model = control["features"], control["model"]
    path = tmp_path / "model.pkl"
    cl.save_best_model(model, cl.ORDINAL_CONTROL_NAME, features.feature_names_, features.top_,
                       ["python"], path, text_vectorizers=features.vectorizers_)
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    live = data[["matched_skills", "state", "job_title", "company_name"]]
    rebuilt = build_live_features(live, bundle)
    np.testing.assert_allclose(rebuilt.toarray(), features.transform(data))
    np.testing.assert_array_equal(bundle["model"].predict(rebuilt), control.predict(data))
    np.testing.assert_allclose(bundle["model"].predict_proba(rebuilt), control.predict_proba(data))


def test_module_and_notebook_share_provenance(tmp_path):
    import json
    from pathlib import Path
    train, test = postings(), postings(3)
    train["job_id"], test["job_id"] = range(18), range(18, 21)
    cl.write_evaluation_manifest(tmp_path, train, test, "synthetic")
    manifest = json.loads((tmp_path / "05_evaluation_manifest.json").read_text())
    assert manifest["train_job_ids"] == list(range(18))
    assert manifest["test_job_ids"] == [18, 19, 20]
    assert manifest["selected_model"] == "synthetic"
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/05_classification.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "cl.write_evaluation_manifest(OUTPUT_DIR, train_df, test_df, best)" in source
    assert "cl.fit_ordinal_control(train_df, y_train, skills)" in source
