"""Small offline checks for snapshot preservation and rule interpretation."""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src import muse_pipeline as mp


@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    """Fail immediately if a tested path attempts a request."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Network requests are prohibited in these tests")
    monkeypatch.setattr(mp.requests, "get", forbidden)


def snapshot(date, ids=("1", "2")):
    """Create a small complete cleaned snapshot."""
    return pd.DataFrame([
        dict(id=i, job_title="engineer", company_name="Example", location="Boulder, CO",
             state="CO", experience_level="Senior", categories="Engineering",
             matched_skills=["python", "sql"], n_skills=2, fetch_date=date,
             publication_date="2026-10-01") for i in ids
    ], columns=mp.CLEANED_COLUMNS)


def test_csv_snapshot_replacement_preserves_other_date(tmp_path):
    """Replacing one date removes its stale rows and leaves another date intact."""
    path = tmp_path / "snapshots.csv"
    old = snapshot("2026-10-14")
    mp.save_cleaned(old, path)
    mp.save_cleaned(snapshot("2026-11-10"), path)
    replacement = snapshot("2026-11-10", ("3",))
    first = mp.save_cleaned(replacement, path)
    second = mp.save_cleaned(replacement, path)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(mp.load_muse_cleaned(path, "2026-10-14"), old)
    assert mp.load_muse_cleaned(path, "2026-11-10")["id"].tolist() == ["3"]


def test_sqlite_snapshot_replacement_and_cap_preserve_rows(tmp_path):
    """Idempotent replacement and read-time caps preserve other snapshots."""
    db = tmp_path / "snapshots.db"
    mp.load_into_warehouse(snapshot("2026-10-14"), db)
    mp.load_into_warehouse(snapshot("2026-11-10"), db)
    replacement = snapshot("2026-11-10", ("3",))
    mp.load_into_warehouse(replacement, db)
    result = mp.load_into_warehouse(replacement, db)
    assert result["snapshots"] == {"2026-10-14": 2, "2026-11-10": 1}
    before = db.read_bytes()
    assert len(mp.load_snapshot(db, "2026-10-14", 1)) == 1
    assert len(mp.load_snapshot(db, "2026-10-14")) == 2
    assert mp.warehouse_snapshots(db) == ["2026-10-14", "2026-11-10"]
    assert db.read_bytes() == before


def test_read_helpers_do_not_create_missing_database(tmp_path):
    """Read helpers fail on absent files instead of creating databases."""
    db = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        mp.warehouse_snapshots(db)
    with pytest.raises(sqlite3.OperationalError):
        mp.load_snapshot(db, "2026-10-14")
    assert not db.exists()


def validation(onehot, min_support=0.05):
    """Score a single fixed rule without mining or loading project data."""
    train = pd.DataFrame([dict(antecedents="a", consequents="b", support=.1,
                               confidence=.7, lift=2.0)])
    empty = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    return mp.validate_rules(train, empty, onehot, empty, onehot, min_support=min_support)


@pytest.mark.parametrize("n_b,blocked", [(6, False), (8, True)])
def test_lift_ceiling_uses_actual_strict_threshold(n_b, blocked):
    """A ceiling below two can still exceed the actual 1.5 threshold."""
    onehot = pd.DataFrame({"a": [True] * 5 + [False] * 5,
                           "b": [True] * n_b + [False] * (10 - n_b)})
    row = validation(onehot).iloc[0]
    assert bool(row["lift_ceiling_blocks_threshold"]) is blocked
    assert bool(row["lift_threshold_holds"]) is (not blocked)
    assert row["confidence_holds"]


def test_failure_reasons_distinguish_support_confidence_lift():
    """Report separate failed criteria, including the actual fallback support."""
    onehot = pd.DataFrame({"a": [True, True, True, False, False],
                           "b": [True, False, False, True, True]})
    row = validation(onehot, min_support=.3).iloc[0]
    assert row["failure_reasons"] == "support, confidence, lift"
    assert not row["support_holds"]
    assert validation(onehot, min_support=.2).iloc[0]["support_holds"]


def test_key_finding_does_not_claim_ceiling_proves_failure():
    """Reduced headroom is not mathematical impossibility or temporal change."""
    onehot = pd.DataFrame({"a": [True] * 5 + [False] * 5,
                           "b": [True] * 6 + [False] * 4})
    v = validation(onehot)
    shifts = (pd.DataFrame({"skill": ["python"]}), pd.DataFrame({"skill": ["sql"]}))
    text = mp.key_finding(v, shifts, snapshot("2026-10-14"), ("Example", 1.0))
    assert "0 cannot exceed the lift threshold" in text
    assert "without making confirmation impossible" in text
    assert "prevent attributing these differences" in text
    assert "Most of the rest did not break" not in text


def test_notebook_uses_snapshot_module_contract():
    """Notebook uses date-resolved cached fetches and snapshot-preserving writes."""
    path = Path(__file__).resolve().parents[1] / "notebooks/08_muse_pipeline.ipynb"
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "fetch_date = mp.resolve_fetch_date(paths)" in source
    assert "mp.fetch_all(paths, fetch_date, allow_fetch=False)" in source
    assert 'mp.save_cleaned(muse, paths["cleaned"])' in source
    assert 'mp.load_muse_cleaned(paths["cleaned"], fetch_date)' in source
    assert 'muse.to_csv(paths["cleaned"]' not in source
    assert "validation.assign(fetch_date=fetch_date).to_csv" in source
    assert "comparison.assign(fetch_date=fetch_date).to_csv" in source


def test_isolated_run_preserves_canonical_snapshots_and_warehouse(tmp_path):
    import hashlib
    root = tmp_path / "canonical"
    paths = mp.muse_paths(root)
    paths["pages_dir"].mkdir(parents=True)
    paths["cleaned"].parent.mkdir(parents=True)
    (paths["pages_dir"] / "engineering_00.json").write_text('{"results": [], "page_count": 1}')
    paths["fetch_log"].write_text('[{"time": "2026-10-14T12:00:00"}]')
    mp.save_cleaned(snapshot("2026-10-14"), paths["cleaned"])
    mp.load_into_warehouse(snapshot("2026-10-14"), paths["db"])
    with sqlite3.connect(paths["db"]) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('preserved')")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    isolated, out = mp.prepare_run_paths(root, tmp_path / "run")
    assert mp.resolve_fetch_date(isolated) == "2026-10-14"
    mp.save_cleaned(snapshot("2026-11-10"), isolated["cleaned"])
    mp.load_into_warehouse(snapshot("2026-11-10"), isolated["db"])
    assert mp.warehouse_snapshots(isolated["db"]) == ["2026-10-14", "2026-11-10"]
    assert set(mp.load_muse_cleaned(isolated["cleaned"])["fetch_date"]) == {"2026-10-14", "2026-11-10"}
    with sqlite3.connect(isolated["db"]) as conn:
        assert conn.execute("SELECT value FROM unrelated").fetchone() == ("preserved",)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    for destination in (None, root / "run", out):
        with pytest.raises((ValueError, FileExistsError)):
            mp.prepare_run_paths(root, destination)
    with pytest.raises(ValueError, match="cache-only"):
        mp.run_pipeline(root, new_snapshot=True, output_dir=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
