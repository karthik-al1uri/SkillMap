"""Export salary-lineage.v1 without changing data, salaries, tiers, or deduplication.

Run: python -m scripts.export_salary_lineage --root . --out-dir /tmp/new-lineage-v1
The destination must be fresh and external, or beneath data/processed/salary_lineage/.
The gzip CSV contains only job IDs, logical CSV record numbers (header excluded),
salary fields and hashed repost keys; no descriptions, names, or locations. Source
file hashes make record pointers verifiable. Repost groups use the existing exact
company/title/description/normalized_salary keys over ALL raw rows: counts are
source-key matches, not deduplication decisions or potential-template matches.

Canonical fields are copied verbatim. Source salary strings are preserved. Unknown
estimation status stays unknown. This version verifies vendor-normalized salaries;
missing source records/values are explicit unknowns, never inferred or imputed.
No warehouse changes, model features, or automatic pipeline hooks are introduced.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "salary-lineage.v1"
REPOST_KEYS = ("company_name", "title", "description", "normalized_salary")
SALARY_FIELDS = ("normalized_salary", "min_salary", "max_salary", "med_salary", "pay_period", "currency")
COLUMNS = (
    "job_id", "canonical_normalized_salary", "canonical_salary_tier",
    "source_status", "source_record", "salary_origin", "salary_estimation_status",
    *("source_" + field for field in SALARY_FIELDS),
    "raw_repost_group_sha256", "raw_repost_group_size",
    "raw_repost_representative_job_id",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def amount(value):
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Invalid salary value") from error
    if not result.is_finite():
        raise ValueError("Salary must be finite")
    return result


def fixed_tier(value):
    value = amount(value)
    return "Low" if value <= 62400 else "Mid" if value <= 109200 else "High"


def repost_keys(source):
    """Hash pandas-parsed exact keys, preserving existing null/numeric semantics."""
    result = []
    for values in source[list(REPOST_KEYS)].itertuples(index=False, name=None):
        key = [None if pd.isna(value) else value for value in values]
        encoded = json.dumps(key, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        result.append(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    return pd.Series(result, index=source.index, dtype=str)


def export(root, out_dir):
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    permitted = root / "data/processed/salary_lineage"
    if out_dir.exists() or out_dir == root or out_dir in root.parents:
        raise ValueError("Destination must be a fresh directory, never an input or ancestor")
    if root in out_dir.parents and permitted not in out_dir.parents:
        raise ValueError("Repository output is restricted to data/processed/salary_lineage/<fresh-run>")
    inputs = {
        "retained": root / "data/processed/cleaned_jobs.csv",
        "source": root / "data/raw/linkedin_postings/postings.csv",
    }
    before = {name: digest(path) for name, path in inputs.items()}
    clean = pd.read_csv(inputs["retained"], usecols=["job_id", "normalized_salary", "salary_tier"],
                        dtype=str, keep_default_na=False)
    source = pd.read_csv(inputs["source"], usecols=["job_id", *REPOST_KEYS],
                         dtype={"job_id": str}, low_memory=False)
    if clean.job_id.eq("").any() or not clean.job_id.is_unique or not source.job_id.is_unique:
        raise ValueError("Job IDs must be nonempty and unique; ambiguous source mapping refused")
    if source.job_id.isna().any() or source.job_id.eq("").any():
        raise ValueError("Source job IDs must be nonempty")
    if any(fixed_tier(salary) != tier for salary, tier in zip(clean.normalized_salary, clean.salary_tier)):
        raise ValueError("Canonical tier does not match frozen 62400/109200 boundaries")
    keys = repost_keys(source)
    sizes = keys.value_counts()
    representatives = source.assign(key=keys).groupby("key", sort=False).job_id.min()
    pointers = {job_id: (record, key) for record, (job_id, key) in
                enumerate(zip(source.job_id, keys), start=1)}
    # A separate string read preserves original source lexical values, including precision.
    original = pd.read_csv(inputs["source"], usecols=["job_id", *SALARY_FIELDS],
                           dtype=str, keep_default_na=False).set_index("job_id")
    rows = []
    for job in clean.sort_values("job_id", kind="stable").itertuples(index=False):
        found = job.job_id in pointers
        raw = original.loc[job.job_id] if found else None
        known_salary = found and raw["normalized_salary"] != ""
        if known_salary and abs(amount(raw["normalized_salary"]) - amount(job.normalized_salary)) > Decimal("0.0051"):
            raise ValueError(f"Source/canonical salary discrepancy for job {job.job_id}")
        record, key = pointers[job.job_id] if found else ("", "")
        row = {
            "job_id": job.job_id, "canonical_normalized_salary": job.normalized_salary,
            "canonical_salary_tier": job.salary_tier,
            "source_status": "linked" if found else "unavailable",
            "source_record": record,
            "salary_origin": "postings.normalized_salary" if known_salary else "unavailable",
            "salary_estimation_status": "unknown",
            **{"source_" + field: raw[field] if found else "" for field in SALARY_FIELDS},
            "raw_repost_group_sha256": key,
            "raw_repost_group_size": int(sizes[key]) if found else "",
            "raw_repost_representative_job_id": representatives[key] if found else "",
        }
        rows.append(row)
    if any(digest(path) != before[name] for name, path in inputs.items()):
        raise RuntimeError("An input changed during export")
    out_dir.mkdir(parents=True, exist_ok=False)
    output = out_dir / "salary_lineage.v1.csv.gz"
    with output.open("xb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION, "columns": list(COLUMNS), "rows": len(rows),
        "linked_rows": sum(row["source_status"] == "linked" for row in rows),
        "known_salary_origin_rows": sum(row["salary_origin"] != "unavailable" for row in rows),
        "estimation_status": "unknown: source does not identify estimated versus advertised salary",
        "tier_contract": {"Low": "<=62400", "Mid": ">62400 and <=109200", "High": ">109200",
                          "provenance": "historical global exact terciles, frozen; not fold-derived"},
        "salary_contract": "Canonical/source strings copied unchanged; comparison permits existing cent rounding (absolute difference <=0.0051); no calculation, conversion or imputation",
        "repost_keys": list(REPOST_KEYS),
        "repost_scope": "All raw rows, before filters; group size is not number removed; no deduplication performed",
        "source_record_contract": "1-based logical CSV record, header excluded; valid only with matching source SHA256",
        "missing_provenance": "Missing records and source normalized values remain unavailable; no fallback, annualization or imputation origin inferred",
        "inputs": {name: {"path": str(path.relative_to(root)), "sha256": before[name]} for name, path in inputs.items()},
        "generator_sha256": digest(Path(__file__)),
        "pandas_version": pd.__version__,
        "output": {"path": output.name, "sha256": digest(output), "bytes": output.stat().st_size},
    }
    with (out_dir / "manifest.v1.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.root, args.out_dir)
    print(json.dumps({"schema_version": manifest["schema_version"], "rows": manifest["rows"],
                      "linked_rows": manifest["linked_rows"], "output_bytes": manifest["output"]["bytes"]}))


if __name__ == "__main__":
    main()
