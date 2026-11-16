"""Export existing SkillMap facts as the distributable React dashboard dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "frontend/public/data/skillmap.json"
DEFAULT_FACTS = Path("/private/tmp/skillmap-phase3/facts/dashboard-facts-final.json")


def canonical_python():
    """Use the repository environment when available; npm may invoke system python3."""
    python = REPO / ".venv/bin/python"
    if python.is_file() and os.environ.get("SKILLMAP_EXPORT_PYTHON") != "1":
        env = os.environ.copy()
        env.update(SKILLMAP_EXPORT_PYTHON="1", PYTHONDONTWRITEBYTECODE="1")
        os.execve(str(python), [str(python), "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_sources(sources):
    for source in sources.values():
        path = Path(source["path"])
        if not path.is_file() or path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
            raise ValueError(f"Recorded facts source is missing or changed: {path}. Rebuild the facts bundle explicitly; export does not recreate experiments.")


def export(facts_path, out, check=False):
    facts_path, out = Path(facts_path).resolve(), Path(out).absolute()
    if not facts_path.is_file():
        raise ValueError("Saved facts bundle not found. Pass --facts /path/to/dashboard-facts.json or set SKILLMAP_FACTS. No data is fabricated or fetched.")
    designated = out == DEFAULT_OUT
    if out.is_symlink() or (designated and out.resolve() != out):
        raise ValueError("Generated output may not follow a symlink")
    resolved_out = out.resolve()
    if resolved_out == facts_path:
        raise ValueError("The source facts bundle is read-only")
    if not designated:
        if resolved_out == REPO or REPO in resolved_out.parents:
            raise ValueError("Only frontend/public/data/skillmap.json may be generated inside the repository")
        if resolved_out.exists():
            raise ValueError("An external --out must name a new file; only the designated frontend dataset may be regenerated")
    before_hash = digest(facts_path)
    before_bytes = facts_path.stat().st_size
    facts = json.loads(facts_path.read_text())
    if check:
        verify_sources(facts.get("sources", {}))
    # Dependencies are already installed in the canonical environment; no installation occurs.
    sys.path.insert(0, str(REPO))
    from src.dashboard_data import build_product_data
    payload = build_product_data(facts)
    payload["provenance"]["sourcefacts_sha256"] = before_hash
    if facts.get("provenance", {}).get("recovery_notice"):
        payload["provenance"]["recovery_notice"] = facts["provenance"]["recovery_notice"]
    payload["provenance"]["facts_source"] = {"path": str(facts_path), "bytes": before_bytes,
                                               "sha256": before_hash, "schema": facts["schema"]}
    payload["provenance"]["export_contract"] = {"salary_grain": "unique_job_id", "rows": len(payload["rows"]),
                                                  "source_count": len(facts.get("sources", {})),
                                                  "source_checks_requested": check,
                                                  "execution": "Existing facts only; no training, model evaluation, APIs, database reads or experiment recreation"}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if facts_path.stat().st_size != before_bytes or digest(facts_path) != before_hash:
        raise ValueError("Source facts changed during export; output was not replaced")
    if check:
        verify_sources(facts.get("sources", {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=".skillmap-export-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"out": str(out), "rows": len(payload["rows"]), "sources": len(facts.get("sources", {})),
            "sourcefacts_sha256": before_hash, "bytes": out.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=Path(os.environ.get("SKILLMAP_FACTS", str(DEFAULT_FACTS))))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Verify all original source sizes/SHA-256 before and after export")
    args = parser.parse_args(argv)
    try:
        result = export(args.facts, args.out, args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    canonical_python()
    main()
