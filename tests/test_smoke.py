"""Offline synthetic integration checks; never invoke canonical pipeline entry points."""
import csv
import hashlib
import socket
import sqlite3
from pathlib import Path
import pandas as pd
import pytest
from src import preprocessing as pp
from src import association as assoc
from scripts.build_mining_facts import main as build_facts


def test_200_row_offline_smoke_and_facts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden in smoke tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    root = tmp_path / "fixture"
    processed = root / "data" / "processed"
    output = root / "outputs"
    processed.mkdir(parents=True)
    output.mkdir()
    texts = ["python sql"]*80 + ["excel"]*60 + [""]*60
    vocab = pd.DataFrame([{"skill": s, "forms": [s], "pattern": pp.skill_pattern([s])}
                          for s in ["python", "sql", "excel"]])
    frame = pd.DataFrame({"description": texts, "skills_desc": [""]*200})
    skills = pp.match_description_skills(frame, vocab)
    tiers, _ = pp.assign_salary_tier(pd.Series([40000.]*70 + [80000.]*60 + [140000.]*70))
    jobs = pd.DataFrame({"job_id": range(200), "matched_skills": skills, "salary_tier": tiers})
    cleaned = processed / "cleaned_jobs.csv"
    jobs.to_csv(cleaned, index=False)
    encoded = assoc.encode_transactions(skills.tolist())
    rules = assoc.generate_rules(assoc.mine_frequent_itemsets(encoded))
    assert len(rules) == 2
    assert all(rules.lift == 2.5)
    assoc.format_rules(rules).to_csv(output / "04_association_rules.csv", index=False)
    db = processed / "skillmap.db"
    with sqlite3.connect(db) as conn:
        jobs.drop(columns="matched_skills").to_sql("job_postings", conn, index=False)
        conn.execute("CREATE VIEW v_jobs AS SELECT * FROM job_postings")
    pd.DataFrame([{"model": "synthetic saved model", "f1_macro": .5, "roc_auc_ovr": .6,
                   "accuracy": .5}]).to_csv(output / "05_classification_results.csv", index=False)
    pd.DataFrame([{"actual_tier": "Low", "predicted_tier": "Low"},
                  {"actual_tier": "High", "predicted_tier": "Mid"}]).to_csv(output / "07_live_predictions.csv", index=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    result = build_facts(["--root", str(root), "--out", str(tmp_path / "facts.md")])
    text = result.read_text()
    assert "200 saved rows" in text and "0.500000" in text
    assert "historical/exploratory" in text and "not model reproduction" in text
    assert "SHA-256" in text and "Missing input:" in text
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_facts_refuses_canonical_destination_and_overwrite(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(SystemExit):
        build_facts(["--root", str(root), "--out", str(root / "facts.md")])
    existing = tmp_path / "existing.md"
    existing.write_text("preserved")
    with pytest.raises(SystemExit):
        build_facts(["--root", str(root), "--out", str(existing)])
    assert existing.read_text() == "preserved"
