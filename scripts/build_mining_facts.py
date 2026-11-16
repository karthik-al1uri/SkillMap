"""Generate an auditable fact sheet without running models or modifying inputs."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


TIERS = ("Low", "Mid", "High")


def sha256(path):
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    """Read a saved CSV, without evaluating its contents as code."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def generate_facts(root):
    """Return Markdown with saved results, independent arithmetic, and source hashes."""
    root = Path(root).resolve(strict=True)
    sources = {}
    lines = ["# SkillMap mining facts", "",
             f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Input root: `{root}`", "",
             "## Evidence scope", "",
             "No model training, prediction, clustering, extraction, or API fetch was performed. "
             "Saved evaluation metrics are historical/exploratory: the existing test set influenced "
             "model selection. They are not a new independent validation. Artifact counts and "
             "prediction arithmetic below were checked during fact generation. Missing files are "
             "reported rather than inferred.", ""]

    def source(relative):
        """Register the exact input artifact and return its path when present."""
        path = root / relative
        if not path.is_file():
            lines.extend([f"Missing input: `{relative}`.", ""])
            return None
        sources[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    lines += ["## Newly checked artifact counts and arithmetic", ""]
    for relative in ("data/processed/cleaned_jobs.csv", "data/processed/live_cleaned_jobs.csv",
                     "data/processed/muse_cleaned_jobs.csv"):
        path = source(relative)
        if path:
            rows = read_rows(path)
            lines.append(f"- `{relative}`: {len(rows):,} saved rows.")
            if rows and "matched_skills" in rows[0]:
                counts = [len(ast.literal_eval(row["matched_skills"])) for row in rows]
                lines.append(f"  Mean matched skills: {sum(counts)/len(counts):.6f}; "
                             f"zero-skill rows: {counts.count(0):,}.")
            if rows and "fetch_date" in rows[0]:
                dates = sorted({row["fetch_date"] for row in rows})
                lines.append(f"  Snapshot dates: {', '.join(dates)}.")
    path = source("outputs/07_live_predictions.csv")
    if path:
        rows = read_rows(path)
        if rows:
            correct = sum(r["actual_tier"] == r["predicted_tier"] for r in rows)
            f1 = []
            for tier in TIERS:
                actual = sum(r["actual_tier"] == tier for r in rows)
                predicted = sum(r["predicted_tier"] == tier for r in rows)
                tp = sum(r["actual_tier"] == tier and r["predicted_tier"] == tier for r in rows)
                f1.append(2 * tp / (actual + predicted) if actual + predicted else 0)
            lines.append(f"- Saved Adzuna predictions ({len(rows):,}): recalculated accuracy "
                         f"{correct/len(rows):.6f}; macro F1 {sum(f1)/3:.6f}. "
                         "This verifies saved prediction arithmetic, not model reproduction; "
                         "estimated salaries remain imperfect labels.")
    path = source("outputs/08_rules_validation.csv")
    if path:
        rows = read_rows(path)
        confirmed = sum(r["validated"].lower() == "true" for r in rows)
        capped = sum(r.get("validated_company_capped", "").lower() == "true" for r in rows)
        blocked = sum(float(r["muse_max_lift"]) <= 1.5 for r in rows if r.get("muse_max_lift"))
        lines.append(f"- Saved Muse validation: {confirmed}/{len(rows)} uncapped, "
                     f"{capped}/{len(rows)} company-capped confirmations. "
                     f"Rules with maximum possible lift ≤1.5: {blocked}. "
                     "A ceiling below 2 alone does not prevent passing the >1.5 threshold.")
    path = source("data/processed/skillmap.db")
    if path:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            integrity = conn.execute("PRAGMA quick_check").fetchall()
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            lines.append(f"- SQLite quick check: `{integrity}`; foreign-key violations: {len(violations)}.")
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
            for name in ("v_jobs", "job_postings", "live_job_postings", "muse_job_postings"):
                if name in names:
                    count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    lines.append(f"  `{name}`: {count:,} rows.")
    lines += ["", "## Historical saved classifier evaluation (not rerun)", ""]
    path = source("outputs/05_classification_results.csv")
    if path:
        lines += ["| Model | Macro F1 | ROC-AUC | Accuracy |", "|---|---:|---:|---:|"]
        for row in read_rows(path):
            model = row["model"].replace("|", "\\|")
            lines.append(f"| {model} | {row['f1_macro']} | {row['roc_auc_ovr']} | {row['accuracy']} |")
    lines += ["", "## Historical saved analysis summaries (not rerun)", ""]
    for relative in ("outputs/02_summary.txt", "outputs/04_summary.txt", "outputs/06_summary.txt",
                     "outputs/07_live_summary.txt", "outputs/08_muse_summary.txt"):
        if source(relative):
            lines.append(f"- `{relative}` is preserved as historical evidence. "
                         "Its prose conclusions are not independently validated by this generator.")
    lines += ["", "## Input provenance", "", "Hashes identify the exact files read. "
              "They do not establish which source commit originally produced these saved artifacts.", "",
              "| Artifact | Bytes | SHA-256 |", "|---|---:|---|"]
    for name, meta in sources.items():
        lines.append(f"| `{name}` | {meta['bytes']} | `{meta['sha256']}` |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Require a fresh external output path; refuse any project-tree output or overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="repository containing saved inputs")
    parser.add_argument("--out", type=Path, required=True, help="new Markdown file outside input repository")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    out = args.out.resolve()
    if out == root or root in out.parents:
        parser.error("--out must be outside --root; canonical artifacts are read-only")
    if out.exists():
        parser.error("--out already exists; refusing to overwrite")
    text = generate_facts(root)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return out


if __name__ == "__main__":
    main()
