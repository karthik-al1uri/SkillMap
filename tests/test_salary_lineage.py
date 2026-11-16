"""Salary-lineage schema, exact source preservation and read-only export contracts."""
import csv
import gzip
import json

import pandas as pd
import pytest

from scripts.export_salary_lineage import COLUMNS, REPOST_KEYS, digest, export
from src.preprocessing import DUPLICATE_POSTING_KEYS


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path, *, salary="62400.00", tier="Low", missing=False, duplicate=False):
    root = tmp_path / "source"
    clean = root / "data/processed/cleaned_jobs.csv"
    raw = root / "data/raw/linkedin_postings/postings.csv"
    write_csv(clean, ["job_id", "normalized_salary", "salary_tier"], [
        {"job_id": "2", "normalized_salary": "109200.00", "salary_tier": "Mid"},
        {"job_id": "1", "normalized_salary": salary, "salary_tier": tier},
    ])
    common = {"company_name": "Private employer", "title": "Private title", "description": "Private description",
              "normalized_salary": "62400.000", "min_salary": "25.0000", "max_salary": "35.0000",
              "med_salary": "", "pay_period": "HOURLY", "currency": "USD"}
    rows = [{"job_id": "1", **common}, {"job_id": "3", **common}]
    if not missing:
        rows.append({**common, "job_id": "2", "normalized_salary": "109200.0", "pay_period": "YEARLY"})
    if duplicate:
        rows.append({"job_id": "1", **common})
    write_csv(raw, ["job_id", *common], rows)
    return root, clean, raw


def read_export(path):
    with gzip.open(path / "salary_lineage.v1.csv.gz", "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_source_strings_ids_schema_duplicates_and_unknown_status(tmp_path):
    root, clean, raw = fixture(tmp_path)
    before = (digest(clean), digest(raw))
    manifest = export(root, tmp_path / "export")
    columns, rows = read_export(tmp_path / "export")
    assert columns == list(COLUMNS)
    assert [r["job_id"] for r in rows] == ["1", "2"]
    assert rows[0]["canonical_normalized_salary"] == "62400.00"
    assert rows[0]["source_normalized_salary"] == "62400.000"
    assert rows[0]["source_min_salary"] == "25.0000"
    assert rows[0]["source_med_salary"] == ""
    assert rows[0]["source_record"] == "1" and rows[1]["source_record"] == "3"
    assert rows[0]["raw_repost_group_size"] == "2"
    assert rows[1]["raw_repost_group_size"] == "1"
    assert rows[0]["raw_repost_group_sha256"] != rows[1]["raw_repost_group_sha256"]
    assert all(r["salary_estimation_status"] == "unknown" for r in rows)
    assert all("Private" not in value for r in rows for value in r.values())
    assert tuple(DUPLICATE_POSTING_KEYS) == REPOST_KEYS
    assert manifest["rows"] == manifest["linked_rows"] == manifest["known_salary_origin_rows"] == 2
    assert (digest(clean), digest(raw)) == before


def test_regeneration_is_byte_identical_and_manifest_has_verifiable_sources(tmp_path):
    root, clean, raw = fixture(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    export(root, a)
    export(root, b)
    for filename in ["salary_lineage.v1.csv.gz", "manifest.v1.json"]:
        assert (a / filename).read_bytes() == (b / filename).read_bytes()
    m = json.loads((a / "manifest.v1.json").read_text())
    assert m["inputs"]["retained"]["sha256"] == digest(clean)
    assert m["inputs"]["source"]["sha256"] == digest(raw)
    assert m["output"]["sha256"] == digest(a / "salary_lineage.v1.csv.gz")


def test_unavailable_source_is_explicit_never_imputed(tmp_path):
    root, _, _ = fixture(tmp_path, missing=True)
    m = export(root, tmp_path / "export")
    _, rows = read_export(tmp_path / "export")
    missing = rows[1]
    assert missing["source_status"] == missing["salary_origin"] == "unavailable"
    assert missing["canonical_normalized_salary"] == "109200.00"
    assert missing["source_record"] == missing["source_normalized_salary"] == ""
    assert m["linked_rows"] == 1


@pytest.mark.parametrize("salary,tier", [("62400.01", "Low"), ("109200.01", "Mid"), ("100000", "Mid"), ("NaN", "Low")])
def test_tier_or_salary_discrepancy_refused_before_output(tmp_path, salary, tier):
    root, _, _ = fixture(tmp_path, salary=salary, tier=tier)
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export(root, destination)
    assert not destination.exists()


def test_ambiguous_ids_and_protected_destinations_refused(tmp_path):
    root, clean, raw = fixture(tmp_path, duplicate=True)
    for destination in [root, root / "outputs/new", root / "data/processed/cleaned_jobs.csv", tmp_path, tmp_path / "bad-source"]:
        with pytest.raises(ValueError):
            export(root, destination)
    assert not (tmp_path / "bad-source").exists()
    assert clean.exists() and raw.exists()


def test_missing_normalized_source_remains_unknown(tmp_path):
    root, _, raw = fixture(tmp_path)
    frame = pd.read_csv(raw, dtype=str, keep_default_na=False)
    frame.loc[frame.job_id.eq("1"), "normalized_salary"] = ""
    frame.to_csv(raw, index=False)
    m = export(root, tmp_path / "export")
    _, rows = read_export(tmp_path / "export")
    assert rows[0]["source_status"] == "linked"
    assert rows[0]["salary_origin"] == "unavailable"
    assert rows[0]["canonical_normalized_salary"] == "62400.00"
    assert m["known_salary_origin_rows"] == 1


def test_optional_repository_provenance_destination(tmp_path):
    root, _, _ = fixture(tmp_path)
    destination = root / "data/processed/salary_lineage/run-v1"
    export(root, destination)
    with pytest.raises(ValueError):
        export(root, destination)
