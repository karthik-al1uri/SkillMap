"""SkillMap Stage 8 — live skill validation on The Muse job postings.

Stage 7's Adzuna descriptions were ~500-character snippets (2.0 skills per posting), too short
to test skill co-occurrence. The Muse public API returns full HTML descriptions without an API
key. This stage fetches current tech postings, extracts skills with the Stage 2 matcher, loads
them into the warehouse, checks which Stage 4 association rules still hold, and compares skill
demand with the 2023-2024 training data.

API notes (found when probing the API on 2026-10-14):
  - Network errors and HTTP 429/5xx are retried with exponential backoff (honouring
    Retry-After); a category stops, keeping its cached pages, if a page still fails.
  - CloudFront blocks generic client User-Agents (HTTP 403), so requests send a descriptive
    USER_AGENT that identifies the project.
  - The Muse category names differ from the requested ones; MUSE_CATEGORIES maps them.
    "Data Science" exists but holds only 2 jobs.
  - Results are not sorted by date (`descending` does not help), so postings span 2024-2026;
    the summary reports publication dates.

Data quality (as in Stage 2): reposted ads (same company, title, and description) are dropped
after the id dedup. A few employers dominate the results (SpaceX alone is about a third), and
shared employer boilerplate creates spurious rules, so validation is repeated with at most
MAX_PER_COMPANY postings per company as a sensitivity check.

Snapshots: each fetch is a snapshot keyed by fetch_date. Pages are cached under
data/raw/muse_live/pages/<fetch_date>/ (the original 2026-10-14 fetch sits directly in pages/),
and the cleaned CSV and warehouse table hold one set of rows per fetch_date. Reprocessing a
snapshot replaces only its own rows, so running twice is safe. The default run uses cached
pages only and makes no API calls.

Company cap: analysis queries keep at most `max_per_company` postings per company and snapshot
(most recent first); no rows are deleted.

Run from the project root with:
    python -m src.muse_pipeline                       # latest cached snapshot, no API calls
    python -m src.muse_pipeline --new-snapshot        # fetch today's snapshot (live API calls)
    python -m src.muse_pipeline --fetch-date 2026-10-14 --max-per-company 50
"""
import argparse
import json
import logging
import re
import sqlite3
import time
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src import association as assoc
from src.association import rule_key, rule_metrics
from src.preprocessing import FORCED_SKILLS, find_project_root, load_cleaned_jobs, match_description_skills
from src.warehouse import STATE_CODES, US_STATES

logger = logging.getLogger(__name__)

API_URL = "https://www.themuse.com/api/public/jobs"
USER_AGENT = "SkillMap/1.0 (CU Boulder data mining course project)"
MUSE_CATEGORIES = {
    "Data Science": "Data Science",
    "Engineering": "Software Engineering",
    "IT": "Computer and IT",
    "Analytics": "Data and Analytics",
}
PAGES = range(0, 25)
SLEEP_SECONDS = 0.5
REQUEST_TIMEOUT = 30
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
SOURCE = "muse_live"

LEVEL_MAP = {
    "Internship": "Internship",
    "Entry Level": "Entry",
    "Mid Level": "Mid-Senior",
    "Senior Level": "Senior",
    "Manager": "Director",
    "Management": "Director",
    "Executive": "Executive",
}

MIN_SKILLS = 2
MIN_SUPPORT = 0.05
FALLBACK_SUPPORT = 0.02
MIN_RULES = 10
TOP_SKILLS = 20
TOP_SHIFTS = 3
TECHNICAL_SKILLS = set(FORCED_SKILLS)
DUPLICATE_KEYS = ["company", "name", "description"]   # Stage 2 also keyed on salary; Muse has none
MAX_PER_COMPANY = 25
ADZUNA_SKILLS_PER_POSTING = 2.0   # Stage 7, for comparison when the Adzuna files are missing

CLEANED_COLUMNS = ["id", "job_title", "company_name", "location", "state", "experience_level",
                   "categories", "matched_skills", "n_skills", "fetch_date", "publication_date"]
TABLE_COLUMNS = ["id", "job_title", "company_name", "location", "state", "experience_level",
                 "matched_skills", "n_skills", "source", "fetch_date", "publication_date"]
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MUSE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS muse_job_postings (
    id                TEXT NOT NULL,
    job_title         TEXT,
    company_name      TEXT,
    location          TEXT,
    state             TEXT,
    experience_level  TEXT,
    matched_skills    TEXT,
    n_skills          INTEGER,
    source            TEXT DEFAULT 'muse_live',
    fetch_date        TEXT NOT NULL,
    publication_date  TEXT,
    PRIMARY KEY (id, fetch_date)
);
"""

SNAPSHOT_SQL = """
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY fetch_date, company_name
                                 ORDER BY publication_date DESC, id) AS company_rank
    FROM muse_job_postings
    WHERE fetch_date = ?
)
WHERE ? IS NULL OR company_rank <= ?
ORDER BY id
"""


def muse_paths(root: Path) -> dict:
    """Return the raw and processed file paths used by Stage 8."""
    raw = root / "data" / "raw" / "muse_live"
    return {
        "raw_dir": raw,
        "pages_dir": raw / "pages",
        "fetch_log": raw / "fetch_log.json",
        "raw_json": raw / "raw_jobs.json",
        "raw_csv": raw / "live_jobs.csv",
        "cleaned": root / "data" / "processed" / "muse_cleaned_jobs.csv",
        "db": root / "data" / "processed" / "skillmap.db",
    }


def legacy_fetch_date(paths: dict):
    """Date of the original snapshot cached directly in pages/, or None if there is none.

    Taken from the last fetch-log entry without a fetch_date (entries from later snapshots carry one).
    """
    flat = sorted(paths["pages_dir"].glob("*.json")) if paths["pages_dir"].is_dir() else []
    if not flat:
        return None
    log = json.loads(paths["fetch_log"].read_text()) if paths["fetch_log"].exists() else []
    legacy = [e["time"][:10] for e in log if "fetch_date" not in e]
    if legacy:
        return legacy[-1]
    return datetime.fromtimestamp(max(p.stat().st_mtime for p in flat)).strftime("%Y-%m-%d")


def list_snapshots(paths: dict) -> list:
    """Sorted fetch_dates that have cached pages."""
    dates = {d.name for d in paths["pages_dir"].glob("*")
             if d.is_dir() and DATE_DIR.match(d.name) and any(d.glob("*.json"))}
    legacy = legacy_fetch_date(paths)
    if legacy:
        dates.add(legacy)
    return sorted(dates)


def snapshot_dir(paths: dict, fetch_date: str) -> Path:
    """Page cache directory of one snapshot (the legacy snapshot lives directly in pages/)."""
    dated = paths["pages_dir"] / fetch_date
    if not dated.is_dir() and fetch_date == legacy_fetch_date(paths):
        return paths["pages_dir"]
    return dated


def resolve_fetch_date(paths: dict, fetch_date: str = None, new_snapshot: bool = False) -> str:
    """Return the snapshot to process: today for a new snapshot, else `fetch_date` or the latest cached one."""
    if new_snapshot:
        return datetime.now().strftime("%Y-%m-%d")
    snapshots = list_snapshots(paths)
    if fetch_date is None:
        if not snapshots:
            raise FileNotFoundError(f"No cached Muse snapshot in {paths['pages_dir']}; run with --new-snapshot")
        return snapshots[-1]
    if fetch_date not in snapshots:
        raise FileNotFoundError(f"No cached snapshot for {fetch_date}; available: {', '.join(snapshots) or 'none'}")
    return fetch_date


def page_file(directory: Path, category: str, page: int) -> Path:
    """Return the cache file for one category page inside a snapshot directory."""
    return directory / f"{category.lower().replace(' ', '_')}_{page:02d}.json"


def page_path(paths: dict, category: str, page: int, fetch_date: str = None) -> Path:
    """Return the cache file for one category page of a snapshot (default: the latest cached one)."""
    return page_file(snapshot_dir(paths, fetch_date or resolve_fetch_date(paths)), category, page)


def get_page(category: str, page: int, log: list) -> tuple:
    """Request one page, retrying network errors and HTTP RETRY_STATUSES with exponential backoff.

    A 429 response's Retry-After header (seconds) is honoured when present. Every attempt is
    appended to `log`. Returns (parsed JSON body or None, reason string when the page failed).
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "category": category,
                 "page": page, "attempt": attempt}
        try:
            response = requests.get(API_URL, params={"category": category, "page": page},
                                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                                    timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.append({**entry, "status": f"error: {type(exc).__name__}"})
            reason = f"page {page}: {type(exc).__name__} after {attempt} attempt(s)"
        else:
            log.append({**entry, "status": response.status_code})
            if response.status_code == 200:
                try:
                    return response.json(), None
                except ValueError:
                    return None, f"page {page}: HTTP 200 but the body is not valid JSON"
            reason = f"page {page} returned HTTP {response.status_code}"
            if response.status_code not in RETRY_STATUSES:
                return None, reason
            retry_after = response.headers.get("Retry-After", "")
            if response.status_code == 429 and retry_after.isdigit():
                wait = max(wait, float(retry_after))
        if attempt < MAX_ATTEMPTS:
            logger.warning("%s; retrying in %.0f s", reason, wait)
            time.sleep(wait)
    return None, reason


def fetch_category(paths: dict, category: str, log: list, fetch_date: str = None,
                   allow_fetch: bool = True) -> dict:
    """Fetch PAGES of one requested category, caching each page and stopping past the last page.

    With allow_fetch=False only cached pages are read. Returns {"category", "muse_category",
    "pages", "jobs", "total", "stopped"}.
    """
    muse_category = MUSE_CATEGORIES[category]
    directory = snapshot_dir(paths, fetch_date or resolve_fetch_date(paths))
    jobs, pages, total, stopped = 0, 0, None, "all pages fetched"
    for page in PAGES:
        cache = page_file(directory, category, page)
        if cache.exists():
            body = json.loads(cache.read_text())
        elif not allow_fetch:
            stopped = f"page {page} not cached"
            break
        else:
            if log:
                time.sleep(SLEEP_SECONDS)
            body, failure = get_page(muse_category, page, log)
            if body is None:
                stopped = failure
                logger.warning("%s: stopped at %s", category, failure)
                break
            cache.write_text(json.dumps(body))
        pages += 1
        jobs += len(body.get("results", []))
        total = body.get("total")
        if page + 1 >= body.get("page_count", 0):
            stopped = f"last page ({body.get('page_count', 0)} pages exist)"
            break
    logger.info("%s (%s): %d pages, %d jobs, %s", category, muse_category, pages, jobs, stopped)
    return {"category": category, "muse_category": muse_category, "pages": pages, "jobs": jobs,
            "total": total, "stopped": stopped}


def fetch_all(paths: dict, fetch_date: str = None, allow_fetch: bool = False) -> pd.DataFrame:
    """Read one snapshot's pages for every requested category and return one row per category.

    fetch_date defaults to the latest cached snapshot. Only with allow_fetch=True are missing
    pages requested from the API (cached pages are always reused, so a rerun makes no calls).
    """
    fetch_date = fetch_date or resolve_fetch_date(paths)
    pages_dir = snapshot_dir(paths, fetch_date)
    pages_dir.mkdir(parents=True, exist_ok=True)
    log = json.loads(paths["fetch_log"].read_text()) if paths["fetch_log"].exists() else []
    new_log = []
    stats = [fetch_category(paths, category, new_log, fetch_date, allow_fetch) for category in MUSE_CATEGORIES]
    new_log = [{**entry, "fetch_date": fetch_date} for entry in new_log]
    if new_log:
        paths["fetch_log"].write_text(json.dumps(log + new_log, indent=1))
    logger.info("Fetch: %d new API calls (%d logged in total)", len(new_log), len(log) + len(new_log))
    return pd.DataFrame(stats)


def combine_pages(paths: dict, fetch_date: str = None) -> tuple:
    """Merge one snapshot's cached pages (default: latest) into unique postings (by id) and save raw_jobs.json.

    Returns (unique postings, total fetched including duplicates). Each posting gets a
    `fetched_categories` list of the requested categories it was found under.
    """
    directory = snapshot_dir(paths, fetch_date or resolve_fetch_date(paths))
    by_id, total = {}, 0
    for category in MUSE_CATEGORIES:
        for page in PAGES:
            cache = page_file(directory, category, page)
            if not cache.exists():
                break
            for job in json.loads(cache.read_text()).get("results", []):
                total += 1
                entry = by_id.setdefault(job["id"], {**job, "fetched_categories": []})
                if category not in entry["fetched_categories"]:
                    entry["fetched_categories"].append(category)
    postings = list(by_id.values())
    paths["raw_json"].write_text(json.dumps(postings, indent=1))
    logger.info("%d postings fetched, %d unique after dedup", total, len(postings))
    return postings, total


def html_to_text(html: str) -> str:
    """Strip HTML tags with BeautifulSoup, keeping words separated, and collapse whitespace."""
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())


def extract_postings(postings: list) -> pd.DataFrame:
    """Flatten postings: id, title, company, locations, plain-text description, levels, categories,
    publication date, and landing page. List fields stay Python lists."""
    rows = [{
        "id": str(job["id"]),
        "name": job.get("name"),
        "company": (job.get("company") or {}).get("name"),
        "locations": [loc.get("name") for loc in job.get("locations", [])],
        "description": html_to_text(job.get("contents")),
        "levels": [lvl.get("name") for lvl in job.get("levels", [])],
        "categories": [cat.get("name") for cat in job.get("categories", [])],
        "fetched_categories": job.get("fetched_categories", []),
        "publication_date": job.get("publication_date"),
        "landing_page": (job.get("refs") or {}).get("landing_page"),
    } for job in postings]
    df = pd.DataFrame(rows)
    df["description_words"] = df["description"].str.split().str.len()
    return df


def drop_reposts(raw: pd.DataFrame) -> tuple:
    """Drop reposted ads (same DUPLICATE_KEYS), keeping the first; return (deduplicated frame, rows dropped)."""
    dup = raw.duplicated(DUPLICATE_KEYS)
    logger.info("Dropped %d reposted duplicates (same %s)", int(dup.sum()), ", ".join(DUPLICATE_KEYS))
    return raw[~dup].reset_index(drop=True), int(dup.sum())


def cap_per_company(muse: pd.DataFrame, cap: int = MAX_PER_COMPANY) -> pd.DataFrame:
    """In-memory version of the SNAPSHOT_SQL cap: the `cap` most recent postings per company and fetch_date."""
    if not cap:
        return muse.reset_index(drop=True)
    ordered = muse.sort_values(["publication_date", "id"], ascending=[False, True], na_position="last")
    keep = ordered.groupby(["fetch_date", "company_name"], dropna=False).head(cap).index
    return muse.loc[muse.index.isin(keep)].sort_values("id").reset_index(drop=True)


def save_raw_csv(raw: pd.DataFrame, path: Path) -> None:
    """Save the extracted postings with list columns joined by '; '."""
    out = raw.copy()
    for col in ["locations", "levels", "categories", "fetched_categories"]:
        out[col] = out[col].map("; ".join)
    out.to_csv(path, index=False)


def map_experience(levels: list) -> str:
    """Map The Muse levels to an experience_level via LEVEL_MAP (first mapped level wins; else Unknown)."""
    for level in levels:
        if level in LEVEL_MAP:
            return LEVEL_MAP[level]
    return "Unknown"


def location_state(location: str):
    """Best-effort US state code from one Muse location ("Austin, TX", "New York, New York", "Remote")."""
    for part in reversed([p.strip() for p in str(location).split(",")]):
        code = part.upper() if part.upper() in STATE_CODES else US_STATES.get(part.lower())
        if code:
            return code
    return None


def first_state(locations: list):
    """Return the state of the first location that has one, or None."""
    for location in locations:
        state = location_state(location)
        if state:
            return state
    return None


def preprocess(raw: pd.DataFrame, vocab: pd.DataFrame, fetch_date: str) -> pd.DataFrame:
    """Extract skills with the Stage 2 matcher and build the cleaned Muse table (CLEANED_COLUMNS)."""
    df = raw.assign(skills_desc="")
    matched = match_description_skills(df, vocab)
    out = pd.DataFrame({
        "id": raw["id"],
        "job_title": raw["name"].fillna("").str.lower().str.strip(),
        "company_name": raw["company"],
        "location": raw["locations"].map("; ".join),
        "state": raw["locations"].map(first_state),
        "experience_level": raw["levels"].map(map_experience),
        "categories": raw["categories"].map("; ".join),
        "matched_skills": matched,
        "n_skills": matched.str.len(),
        "fetch_date": fetch_date,
        "publication_date": raw["publication_date"].str[:10],
    })
    return out[CLEANED_COLUMNS]


def load_muse_cleaned(path: Path, fetch_date: str = None) -> pd.DataFrame:
    """Read muse_cleaned_jobs.csv back with matched_skills parsed into lists (optionally one snapshot)."""
    df = pd.read_csv(path, dtype={"id": str, "fetch_date": str})
    if fetch_date is not None:
        df = df[df["fetch_date"] == fetch_date].reset_index(drop=True)
    df["matched_skills"] = df["matched_skills"].apply(literal_eval)
    return df


def save_cleaned(muse: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Upsert snapshots into muse_cleaned_jobs.csv: replace rows of the fetch_dates in `muse`, keep the others.

    Returns all snapshots, with matched_skills as lists.
    """
    out = muse
    if path.exists():
        old = load_muse_cleaned(path)
        out = pd.concat([old[~old["fetch_date"].isin(muse["fetch_date"])], muse], ignore_index=True)
    out = out.sort_values(["fetch_date", "id"]).reset_index(drop=True)
    out[CLEANED_COLUMNS].to_csv(path, index=False)
    logger.info("%s: %d rows over snapshots %s", path.name, len(out), out["fetch_date"].value_counts().sort_index().to_dict())
    return out


def ensure_muse_table(conn: sqlite3.Connection) -> None:
    """Create muse_job_postings, migrating the earlier one-snapshot table (id-only key) in place."""
    info = conn.execute("PRAGMA table_info(muse_job_postings)").fetchall()
    pk = [row[1] for row in sorted(info, key=lambda r: r[5]) if row[5]]
    if info and pk != ["id", "fetch_date"]:
        old_cols = [row[1] for row in info if row[1] in TABLE_COLUMNS]
        conn.execute("ALTER TABLE muse_job_postings RENAME TO muse_job_postings_old")
        conn.executescript(MUSE_TABLE_SQL)
        conn.execute(f"INSERT INTO muse_job_postings ({', '.join(old_cols)}) "
                     f"SELECT {', '.join(old_cols)} FROM muse_job_postings_old")
        conn.execute("DROP TABLE muse_job_postings_old")
        logger.info("Migrated muse_job_postings to a (id, fetch_date) key")
    conn.executescript(MUSE_TABLE_SQL)


def load_into_warehouse(muse: pd.DataFrame, db_path: Path) -> dict:
    """Upsert the snapshots in `muse` into muse_job_postings (one row per posting and fetch_date).

    Rows of other fetch_dates are kept, so loading the same snapshot twice is safe. matched_skills
    is stored as a JSON array string. Returns the row count and mean skills of the loaded
    snapshots plus {fetch_date: rows} for the whole table.
    """
    table = muse.assign(matched_skills=muse["matched_skills"].map(json.dumps), source=SOURCE)
    table = table.reindex(columns=TABLE_COLUMNS)
    dates = sorted(table["fetch_date"].unique())
    marks = ", ".join("?" * len(dates))
    with sqlite3.connect(db_path) as conn:
        ensure_muse_table(conn)
        conn.execute(f"DELETE FROM muse_job_postings WHERE fetch_date IN ({marks})", dates)
        table.to_sql("muse_job_postings", conn, if_exists="append", index=False)
        count, avg_skills = conn.execute(f"SELECT COUNT(*), AVG(n_skills) FROM muse_job_postings "
                                         f"WHERE fetch_date IN ({marks})", dates).fetchone()
        snapshots = dict(conn.execute("SELECT fetch_date, COUNT(*) FROM muse_job_postings "
                                      "GROUP BY fetch_date ORDER BY fetch_date").fetchall())
    logger.info("muse_job_postings: %d rows for %s, mean %.2f skills; snapshots %s",
                count, ", ".join(dates), avg_skills, snapshots)
    return {"rows": count, "avg_skills": avg_skills, "snapshots": snapshots}


def warehouse_snapshots(db_path: Path) -> list:
    """Sorted fetch_dates in muse_job_postings (empty if the table does not exist)."""
    with sqlite3.connect(db_path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'muse_job_postings'").fetchone():
            return []
        return [r[0] for r in conn.execute("SELECT DISTINCT fetch_date FROM muse_job_postings ORDER BY 1")]


def load_snapshot(db_path: Path, fetch_date: str = None, max_per_company: int = None) -> pd.DataFrame:
    """Query one snapshot (default: latest) from muse_job_postings, keeping at most `max_per_company`
    postings per company (most recent first; None or 0 = no cap). matched_skills is parsed into lists.
    """
    fetch_date = fetch_date or warehouse_snapshots(db_path)[-1]
    cap = max_per_company or None
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(SNAPSHOT_SQL, conn, params=(fetch_date, cap, cap))
    df["matched_skills"] = df["matched_skills"].map(json.loads)
    return df.drop(columns="company_rank")


def mine_muse_rules(muse: pd.DataFrame) -> tuple:
    """Apriori on postings with >= MIN_SKILLS skills at MIN_SUPPORT, retrying at FALLBACK_SUPPORT if < MIN_RULES.

    Returns (one-hot matrix, formatted rules, min_support used, number of qualifying postings).
    """
    transactions = muse.loc[muse["n_skills"] >= MIN_SKILLS, "matched_skills"].tolist()
    onehot = assoc.encode_transactions(transactions)
    support = MIN_SUPPORT
    rules = assoc.generate_rules(assoc.mine_frequent_itemsets(onehot, support))
    if len(rules) < MIN_RULES:
        logger.info("Only %d rules at min_support=%.2f; retrying at %.2f", len(rules), support, FALLBACK_SUPPORT)
        support = FALLBACK_SUPPORT
        rules = assoc.generate_rules(assoc.mine_frequent_itemsets(onehot, support))
    return onehot, assoc.format_rules(rules), support, len(transactions)


def validate_rules(train_rules: pd.DataFrame, muse_rules: pd.DataFrame, onehot: pd.DataFrame,
                   capped_rules: pd.DataFrame = None, capped_onehot: pd.DataFrame = None) -> pd.DataFrame:
    """Check each Stage 4 training rule against the Muse data.

    validated = the rule is among the Muse rules (passes min_support, confidence, and lift filters).
    lift_holds = on Muse data the rule still has confidence > 0.5 and lift > 1.5, ignoring support.
    muse_max_lift = 1 / support of the consequent in Muse data: the highest lift the rule could
    reach. A consequent in most postings caps lift near 1, whatever the co-occurrence.
    muse_lift etc. are taken from the Muse rules when present, otherwise computed directly.
    With `capped_rules` / `capped_onehot`, the same check is repeated on the company-capped sample.
    """
    def lookup(rules: pd.DataFrame) -> dict:
        """Map rule key -> rule row."""
        return {rule_key(a, c): r for a, c, r in zip(rules["antecedents"], rules["consequents"],
                                                    rules.itertuples(index=False))}

    def metrics(found, onehot_: pd.DataFrame, ante: str, cons: str) -> dict:
        """Rule metrics from the mined rule if found, otherwise computed directly."""
        return ({"support": found.support, "confidence": found.confidence, "lift": found.lift} if found
                else rule_metrics(onehot_, ante, cons))

    muse, capped = lookup(muse_rules), lookup(capped_rules) if capped_rules is not None else {}
    rows = []
    for t in train_rules.itertuples(index=False):
        key = rule_key(t.antecedents, t.consequents)
        found = muse.get(key)
        m = metrics(found, onehot, t.antecedents, t.consequents)
        cons = [x.strip() for x in t.consequents.split(",")]
        cons_support = onehot[cons].all(axis=1).mean() if all(c in onehot for c in cons) else np.nan
        row = {
            "rule": key, "training_lift": t.lift, "muse_lift": m["lift"], "validated": found is not None,
            "training_support": t.support, "muse_support": m["support"],
            "training_confidence": t.confidence, "muse_confidence": m["confidence"],
            "muse_consequent_support": cons_support,
            "muse_max_lift": 1 / cons_support if cons_support else np.nan,
            "lift_holds": bool(m["confidence"] > assoc.MIN_CONFIDENCE and m["lift"] > assoc.MIN_LIFT)
            if pd.notna(m["lift"]) else False,
        }
        if capped_onehot is not None:
            c = capped.get(key)
            row["validated_company_capped"] = c is not None
            row["muse_lift_company_capped"] = metrics(c, capped_onehot, t.antecedents, t.consequents)["lift"]
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes(include=float).columns
    out[num] = out[num].round(4)
    return out.sort_values(["validated", "lift_holds", "muse_lift"], ascending=False).reset_index(drop=True)


def skill_comparison(muse: pd.DataFrame, train: pd.DataFrame, tech_ids: set,
                     capped: pd.DataFrame = None) -> pd.DataFrame:
    """Share of postings listing each skill: Muse vs all training jobs vs training tech jobs.

    Training tech jobs = the Stage 6 "Data, Tech & Engineering" cluster, the closest training
    counterpart to the Muse tech categories. change_vs_tech_pp is the Muse share minus the
    training tech share, in percentage points. With `capped` (the company-capped sample),
    muse_share_capped is added.
    """
    def shares(lists: pd.Series) -> pd.Series:
        """Share of postings listing each skill."""
        return lists.explode().dropna().value_counts() / len(lists)

    tech = train[train["job_id"].isin(tech_ids)]
    out = pd.DataFrame({"muse_share": shares(muse["matched_skills"]),
                        "train_share": shares(train["matched_skills"]),
                        "train_tech_share": shares(tech["matched_skills"])}).fillna(0)
    if capped is not None:
        out["muse_share_capped"] = shares(capped["matched_skills"]).reindex(out.index).fillna(0)
    out["is_technical"] = out.index.isin(TECHNICAL_SKILLS)
    out["change_vs_tech_pp"] = 100 * (out["muse_share"] - out["train_tech_share"])
    out["ratio_vs_tech"] = out["muse_share"] / out["train_tech_share"].replace(0, np.nan)
    out.index.name = "skill"
    return out.sort_values("muse_share", ascending=False).round(4).reset_index()


def demand_shifts(comparison: pd.DataFrame, n: int = TOP_SHIFTS, min_share: float = 0.05) -> tuple:
    """Return the n largest rises and n largest drops vs training tech jobs, among skills in >= min_share
    of postings in either source."""
    common = comparison[(comparison["muse_share"] >= min_share) | (comparison["train_tech_share"] >= min_share)]
    ranked = common.sort_values("change_vs_tech_pp")
    return ranked.tail(n).iloc[::-1], ranked.head(n)


def adzuna_reference(root: Path) -> dict:
    """Return Stage 7 Adzuna description length (words) and skills per posting, if the files exist."""
    raw = root / "data" / "raw" / "adzuna_live" / "live_jobs.csv"
    cleaned = root / "data" / "processed" / "live_cleaned_jobs.csv"
    out = {"words": np.nan, "skills": ADZUNA_SKILLS_PER_POSTING}
    if raw.exists():
        out["words"] = pd.read_csv(raw)["description"].fillna("").str.split().str.len().mean()
    if cleaned.exists():
        out["skills"] = pd.read_csv(cleaned)["matched_skills"].apply(literal_eval).str.len().mean()
    return out


def key_finding(validation: pd.DataFrame, shifts: tuple, muse: pd.DataFrame, top_share: tuple,
                cap: int = MAX_PER_COMPANY) -> str:
    """Compose the 2-3 sentence key finding from the computed results."""
    n, validated = len(validation), int(validation["validated"].sum())
    capped = int(validation["validated_company_capped"].sum())
    confirmed = validation[validation["validated"]].sort_values("muse_lift", ascending=False)["rule"]
    failed = validation[~validation["lift_holds"]]
    ceiling = failed[failed["muse_max_lift"] < 2]
    conf_up = (ceiling["muse_confidence"] > ceiling["training_confidence"]).sum()
    ceiling_skills = sorted({r.split(" -> ")[1] for r in ceiling["rule"]})
    recent = (pd.to_datetime(muse["publication_date"]) >= pd.Timestamp(muse["fetch_date"].iloc[0])
              - pd.Timedelta(days=90)).mean()
    rises, drops = shifts
    return " ".join([
        f"Only {validated} of {n} top training rules are confirmed on current Muse tech postings, {recent:.0%} of "
        f"them published in the 90 days before the fetch ({capped} of {n} with at most {cap} postings "
        f"per company): "
        f"{', '.join(confirmed) if len(confirmed) else 'none'}.",
        f"Most of the rest did not break: {len(ceiling)} of the {len(failed)} failing rules point to skills "
        f"({', '.join(ceiling_skills)}) so common in tech postings that their lift is capped below 2, "
        f"and {conf_up} of those {len(ceiling)} actually have higher confidence than in training, so the skills "
        f"still co-occur but are no longer distinctive in a tech-only sample.",
        f"Skill demand vs training tech jobs rose most for {', '.join(rises['skill'])} and fell for "
        f"{', '.join(drops['skill'])}, but the sample is concentrated ({top_share[0]} is {top_share[1]:.0%} of postings), "
        f"so shifts are indicative only.",
    ])


def build_summary(fetch_date: str, stats: pd.DataFrame, total: int, n_unique: int, n_reposts: int,
                  raw: pd.DataFrame, muse: pd.DataFrame, train: pd.DataFrame, adzuna: dict, warehouse: dict,
                  rules: pd.DataFrame, support: float, n_qualify: int, validation: pd.DataFrame,
                  comparison: pd.DataFrame, n_calls: int, capped: dict) -> str:
    """Build the plain-text Stage 8 summary."""
    line = "=" * 80
    rises, drops = demand_shifts(comparison)
    pub = pd.to_datetime(muse["publication_date"])
    fetch_ts = pd.Timestamp(fetch_date)
    tech = comparison[comparison["is_technical"]].head(10)
    cols = ["skill", "muse_share", "train_tech_share", "train_share", "change_vs_tech_pp"]
    validated = validation[validation["validated"]].sort_values("muse_lift", ascending=False)
    cap = capped.get("cap", MAX_PER_COMPANY)
    snapshots = warehouse.get("snapshots", {fetch_date: warehouse["rows"]})
    parts = [
        "SkillMap — Stage 8 The Muse Live Skill Validation Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Fetch date: {fetch_date}",
        f"Source: {API_URL} (no API key; descriptive User-Agent). API page calls logged: {n_calls}",
        "",
        f"{line}\nFetch\n{line}",
        stats.to_string(index=False),
        "Requested categories are mapped to The Muse names; 'Data Science' exists but has only "
        f"{int(stats.loc[stats['category'] == 'Data Science', 'total'].iloc[0])} jobs.",
        f"Total postings fetched: {total:,}; unique after dedup by id: {n_unique:,}; "
        f"after dropping {n_reposts} reposted ads (same company, title, description): {len(raw):,}",
        f"Employers: {muse['company_name'].nunique()} companies; largest: "
        + ", ".join(f"{c} {k} ({k / len(muse):.0%})" for c, k in muse["company_name"].value_counts().head(5).items()),
        f"Publication dates: {pub.min():%Y-%m-%d} to {pub.max():%Y-%m-%d}; "
        f"{(pub.dt.year == fetch_ts.year).mean():.0%} published in {fetch_ts.year}, "
        f"{(pub >= fetch_ts - pd.Timedelta(days=90)).mean():.0%} in the 90 days before the fetch",
        f"Experience levels: {muse['experience_level'].value_counts().to_dict()}",
        f"State parsed for {muse['state'].notna().mean():.0%} of postings",
        "",
        f"{line}\nDescription length and skills\n{line}",
        f"Mean description length: Muse {raw['description_words'].mean():,.0f} words vs Adzuna "
        f"{adzuna['words']:,.0f} words (Stage 7 snippets)",
        f"Mean skills per posting: Muse {muse['n_skills'].mean():.1f} vs Adzuna {adzuna['skills']:.1f} vs training "
        f"{train['matched_skills'].str.len().mean():.1f}",
        f"Postings with >= 1 skill: {(muse['n_skills'] >= 1).mean():.1%}; with >= {MIN_SKILLS}: {n_qualify:,} "
        f"({n_qualify / len(muse):.1%})",
        "",
        f"{line}\nWarehouse (data/processed/skillmap.db, table muse_job_postings)\n{line}",
        f"SELECT COUNT(*), AVG(n_skills) FROM muse_job_postings WHERE fetch_date = '{fetch_date}' -> "
        f"{warehouse['rows']:,}, {warehouse['avg_skills']:.2f}",
        "Snapshots (fetch_date: rows): " + ", ".join(f"{d}: {k:,}" for d, k in snapshots.items()),
        "",
        f"{line}\nAssociation rule validation (outputs/08_rules_validation.csv)\n{line}",
        f"Transactions: {n_qualify:,} postings with >= {MIN_SKILLS} skills. min_support={support}, "
        f"confidence > {assoc.MIN_CONFIDENCE}, lift > {assoc.MIN_LIFT}: {len(rules)} Muse rules "
        f"(outputs/08_muse_association_rules.csv)",
        f"Training rules confirmed (also a Muse rule): {int(validation['validated'].sum())} of {len(validation)} "
        f"({validation['validated'].mean():.0%})",
        f"Training rules whose confidence > {assoc.MIN_CONFIDENCE} and lift > {assoc.MIN_LIFT} still hold on Muse "
        f"data, ignoring support: {int(validation['lift_holds'].sum())} of {len(validation)}",
        f"Company-capped sensitivity (at most {cap} most recent postings per company: {capped['n']:,} postings, "
        f"min_support={capped['support']}, {capped['n_rules']:,} rules): "
        f"{int(validation['validated_company_capped'].sum())} of {len(validation)} training rules confirmed",
        "muse_max_lift = 1 / support of the rule's consequent in Muse data (the highest lift possible).",
        "Top 5 validated rules:",
        validated.head(5)[["rule", "training_lift", "muse_lift", "muse_support", "muse_confidence"]].to_string(index=False)
        if len(validated) else "  none",
        "",
        "All training rules:",
        validation[["rule", "training_lift", "muse_lift", "muse_max_lift", "training_confidence", "muse_confidence",
                    "validated", "validated_company_capped"]].to_string(index=False),
        "",
        "Top 5 Muse rules by lift (high-lift multi-skill rules mostly reflect one employer's template text):",
        rules.head(5)[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False),
        "",
        f"{line}\nSkill demand (outputs/08_skill_comparison.csv)\n{line}",
        "Shares of postings. train_tech = Stage 6 'Data, Tech & Engineering' cluster (closest training counterpart).",
        f"Top {TOP_SKILLS} skills in Muse data:",
        comparison.head(TOP_SKILLS)[cols].to_string(index=False),
        "",
        "Most common technical skills in Muse data:",
        tech[cols].to_string(index=False),
        "",
        f"Top {TOP_SHIFTS} demand rises vs training tech jobs (skills in >= 5% of either):",
        rises[cols].to_string(index=False),
        f"Top {TOP_SHIFTS} demand drops vs training tech jobs:",
        drops[cols].to_string(index=False),
        "Caveat: shifts mix real demand change with source differences (The Muse vs LinkedIn, description "
        "style and length), so treat them as indicative.",
        "",
        f"{line}\nKey finding\n{line}",
        key_finding(validation, (rises, drops), muse,
                    (muse["company_name"].value_counts().index[0], muse["company_name"].value_counts(normalize=True).iloc[0]),
                    cap),
        "",
    ]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None, new_snapshot: bool = False, fetch_date: str = None,
                 max_per_company: int = MAX_PER_COMPANY) -> dict:
    """Run Stage 8 end to end on one snapshot.

    By default the latest cached snapshot is processed with no API calls; `fetch_date` picks another
    cached one. new_snapshot=True fetches today's snapshot (only missing pages are requested). The
    snapshot is upserted into the cleaned CSV and the warehouse; analysis runs on it, with the
    company-capped repeat keeping at most `max_per_company` postings per company (0 = no cap).
    """
    root = project_root or find_project_root()
    paths = muse_paths(root)
    output_dir = root / "outputs"
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    fetch_date = resolve_fetch_date(paths, fetch_date, new_snapshot)
    logger.info("Snapshot %s (%s)", fetch_date, "new fetch" if new_snapshot else "cache only")

    stats = fetch_all(paths, fetch_date, allow_fetch=new_snapshot)
    postings, total = combine_pages(paths, fetch_date)
    raw = extract_postings(postings)
    save_raw_csv(raw, paths["raw_csv"])
    n_unique = len(raw)
    raw, n_reposts = drop_reposts(raw)
    log = json.loads(paths["fetch_log"].read_text()) if paths["fetch_log"].exists() else []

    vocab = pd.read_csv(root / "data" / "processed" / "skill_vocabulary.csv")
    vocab["forms"] = vocab["forms"].apply(literal_eval)
    all_snapshots = save_cleaned(preprocess(raw, vocab, fetch_date), paths["cleaned"])

    warehouse = load_into_warehouse(all_snapshots, paths["db"])
    muse = load_snapshot(paths["db"], fetch_date)
    capped_muse = load_snapshot(paths["db"], fetch_date, max_per_company)
    warehouse.update(rows=len(muse), avg_skills=muse["n_skills"].mean())

    onehot, rules, support, n_qualify = mine_muse_rules(muse)
    rules.to_csv(output_dir / "08_muse_association_rules.csv", index=False)
    capped_onehot, capped_rules, capped_support, capped_n = mine_muse_rules(capped_muse)
    capped = {"n": capped_n, "support": capped_support, "n_rules": len(capped_rules), "cap": max_per_company}
    validation = validate_rules(pd.read_csv(output_dir / "04_association_rules.csv"), rules, onehot,
                                capped_rules, capped_onehot)
    validation.to_csv(output_dir / "08_rules_validation.csv", index=False)

    train = load_cleaned_jobs(root / "data" / "processed" / "cleaned_jobs.csv")
    labels = pd.read_csv(output_dir / "06_cluster_labels.csv")
    tech_ids = set(labels.loc[labels["kmeans_cluster_name"] == "Data, Tech & Engineering", "job_id"])
    comparison = skill_comparison(muse, train, tech_ids, capped_muse)
    comparison.to_csv(output_dir / "08_skill_comparison.csv", index=False)

    summary = build_summary(fetch_date, stats, total, n_unique, n_reposts, raw, muse, train, adzuna_reference(root),
                            warehouse, rules, support, n_qualify, validation, comparison, len(log), capped)
    (output_dir / "08_muse_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved %s", output_dir / "08_muse_summary.txt")
    return {"fetch_date": fetch_date, "stats": stats, "total": total, "n_unique": n_unique,
            "n_reposts": n_reposts, "capped": capped, "capped_rules": capped_rules, "raw": raw, "muse": muse,
            "warehouse": warehouse, "rules": rules, "support": support, "n_qualify": n_qualify,
            "validation": validation, "comparison": comparison, "summary": summary}


def parse_args(argv: list = None) -> argparse.Namespace:
    """Parse the Stage 8 command-line options."""
    parser = argparse.ArgumentParser(description="SkillMap Stage 8: The Muse live skill validation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--new-snapshot", action="store_true",
                       help="fetch today's snapshot from the API (live calls; cached pages are reused)")
    group.add_argument("--fetch-date", help="process this cached snapshot (YYYY-MM-DD) instead of the latest")
    parser.add_argument("--max-per-company", type=int, default=MAX_PER_COMPANY,
                        help=f"postings kept per company in the capped analysis (default {MAX_PER_COMPANY}; 0 = no cap)")
    parser.add_argument("--list-snapshots", action="store_true", help="list cached snapshots and exit")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.list_snapshots:
        root = find_project_root()
        logger.info("Cached snapshots: %s", ", ".join(list_snapshots(muse_paths(root))) or "none")
    else:
        run_pipeline(new_snapshot=args.new_snapshot, fetch_date=args.fetch_date,
                     max_per_company=args.max_per_company)
