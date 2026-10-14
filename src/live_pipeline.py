"""SkillMap Stage 7 — live validation on current Adzuna job postings.

Fetches current US IT job postings from the Adzuna API, preprocesses them with the Stage 2
skill vocabulary and salary tier thresholds, loads them into the warehouse, and checks whether
the training-data findings hold: association rules (Stage 4) and the salary-tier model (Stage 5).

API budget: the free tier allows CALL_BUDGET calls in total. Every request is recorded in
data/raw/adzuna_live/api_calls.json *before* it is sent, and no request is made once the
ledger reaches the budget. The health check result and every fetched page are cached, so
re-running this module or the notebook never calls the API again.

Credentials come from the gitignored .env file (see .env.example): Adzuna authenticates with
two query parameters, app_id (ADZUNA_APP_ID) and app_key (ADZUNA_API_KEY). An api_key-only
request returned HTTP 400. Credentials are never logged: request errors are re-raised without
the URL, and response text is scrubbed of both values.

Run from the project root with:
    python -m src.live_pipeline
"""
import json
import logging
import os
import pickle
import re
import sqlite3
import time
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import association as assoc
from src import classification as cl
from src.preprocessing import find_project_root, load_cleaned_jobs, match_description_skills
from src.warehouse import STATE_CODES, US_STATES

logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("ADZUNA_API_KEY")
APP_ID = os.getenv("ADZUNA_APP_ID")

API_URL = "https://api.adzuna.com/v1/api/jobs/us/search/{page}"
CATEGORY = "it-jobs"
CALL_BUDGET = 25
HEALTH_RESULTS_PER_PAGE = 10
RESULTS_PER_PAGE = 50
FETCH_PAGES = range(1, 25)          # 24 calls; with the health check, 25 in total
SLEEP_SECONDS = 1.0
REQUEST_TIMEOUT = 30

# Exact Stage 2 training thresholds (33rd / 66th percentiles of normalized_salary).
LOW_MAX = 62_400
MID_MAX = 109_200
TIER_LABELS = ["Low", "Mid", "High"]

MIN_SUPPORT = 0.05
FALLBACK_SUPPORT = 0.02
MIN_RULES = 20
MIN_SALARY_ROWS = 50
TOP_SKILLS = 10
SOURCE = "adzuna_live"

EXTRACT_COLUMNS = ["id", "title", "company", "location", "state", "salary_min", "salary_max", "description",
                   "created", "redirect_url", "salary_is_predicted"]
CLEANED_COLUMNS = ["id", "job_title", "company_name", "location", "matched_skills", "mean_salary",
                   "salary_tier", "has_salary", "fetch_date", "state", "salary_is_predicted"]

LIVE_TABLE_SQL = """
DROP TABLE IF EXISTS live_job_postings;
CREATE TABLE live_job_postings (
    posting_skill_id   INTEGER PRIMARY KEY,
    job_id             INTEGER NOT NULL,
    skill_id           INTEGER REFERENCES dim_skills (skill_id),
    location_id        INTEGER REFERENCES dim_location (location_id),
    company_id         INTEGER NOT NULL REFERENCES dim_company (company_id),
    time_id            INTEGER,
    salary_tier        TEXT CHECK (salary_tier IN ('Low', 'Mid', 'High')),
    experience_level   TEXT NOT NULL,
    normalized_salary  REAL,
    source             TEXT NOT NULL DEFAULT 'adzuna_live',
    fetch_date         TEXT,
    has_salary         INTEGER NOT NULL,
    UNIQUE (job_id, skill_id)
);
CREATE INDEX idx_live_job ON live_job_postings (job_id);
CREATE INDEX idx_live_skill ON live_job_postings (skill_id);
"""


def require_api_key() -> str:
    """Return the Adzuna API key from .env, raising if it is missing or still the placeholder."""
    if not API_KEY or API_KEY == "your_api_key_here":
        raise ValueError("ADZUNA_API_KEY not set in .env file")
    return API_KEY


def require_app_id() -> str:
    """Return the Adzuna app ID from .env, raising if it is missing, a placeholder, or not a single token.

    The check runs before any request, so a malformed value never costs an API call.
    """
    if not APP_ID or APP_ID == "your_app_id_here":
        raise ValueError("ADZUNA_APP_ID not set in .env file")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", APP_ID):
        raise ValueError("ADZUNA_APP_ID in .env is not a single ID (it contains spaces or punctuation); "
                         "copy only the Application ID from the Adzuna developer dashboard")
    return APP_ID


def live_paths(root: Path) -> dict:
    """Return the raw and processed file paths used by Stage 7."""
    raw = root / "data" / "raw" / "adzuna_live"
    return {
        "raw_dir": raw,
        "pages_dir": raw / "pages",
        "ledger": raw / "api_calls.json",
        "health": raw / "health_check.json",
        "raw_json": raw / "raw_jobs.json",
        "raw_csv": raw / "live_jobs.csv",
        "cleaned": root / "data" / "processed" / "live_cleaned_jobs.csv",
        "db": root / "data" / "processed" / "skillmap.db",
    }


def read_ledger(path: Path) -> list:
    """Return the list of recorded API calls (empty if none yet)."""
    return json.loads(path.read_text()) if path.exists() else []


def calls_remaining(path: Path) -> int:
    """Return how many calls are left in the CALL_BUDGET according to the ledger."""
    return CALL_BUDGET - len(read_ledger(path))


def api_get(page: int, results_per_page: int, ledger_path: Path) -> requests.Response:
    """Make one Adzuna search call, recording it in the ledger first and refusing past the budget.

    Network errors are re-raised as RuntimeError with the exception type only, because
    requests' own messages include the full URL and therefore the API key.
    """
    params = {"app_id": require_app_id(), "app_key": require_api_key(),
              "results_per_page": results_per_page, "category": CATEGORY}
    ledger = read_ledger(ledger_path)
    if len(ledger) >= CALL_BUDGET:
        raise RuntimeError(f"API call budget exhausted ({CALL_BUDGET} calls recorded in {ledger_path.name})")
    entry = {"call": len(ledger) + 1, "time": datetime.now().isoformat(timespec="seconds"),
             "page": page, "results_per_page": results_per_page, "status": None}
    ledger.append(entry)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    try:
        response = requests.get(API_URL.format(page=page), params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        entry["status"] = f"error: {type(exc).__name__}"
        ledger_path.write_text(json.dumps(ledger, indent=2))
        raise RuntimeError(f"Request for page {page} failed: {type(exc).__name__}") from None
    entry["status"] = response.status_code
    ledger_path.write_text(json.dumps(ledger, indent=2))
    logger.info("API call %d/%d: page %d, %d per page -> HTTP %d",
                entry["call"], CALL_BUDGET, page, results_per_page, response.status_code)
    return response


def scrub(text: str) -> str:
    """Remove the API key from text (e.g. an error body) before it is saved or shown."""
    for secret in (API_KEY, APP_ID):
        if secret:
            text = text.replace(secret, "***")
    return text


def health_check(paths: dict) -> dict:
    """Step 1: make ONE call to page 1 (HEALTH_RESULTS_PER_PAGE results) and record the outcome.

    The result is cached in health_check.json; if it exists, it is returned without a call.
    Returns {"status", "ok", "count", "first_title", "error", "time"}.
    """
    if paths["health"].exists():
        result = json.loads(paths["health"].read_text())
        logger.info("Health check cached from %s: HTTP %s", result["time"], result["status"])
        return result
    response = api_get(1, HEALTH_RESULTS_PER_PAGE, paths["ledger"])
    result = {"status": response.status_code, "ok": response.status_code == 200, "count": None,
              "first_title": None, "error": None, "time": datetime.now().isoformat(timespec="seconds")}
    if result["ok"]:
        body = response.json()
        results = body.get("results", [])
        result["count"] = body.get("count")
        result["first_title"] = results[0].get("title") if results else None
        result["ok"] = bool(results)
        if not results:
            result["error"] = "HTTP 200 but no results returned"
    else:
        result["error"] = scrub(response.text[:500])
    paths["health"].write_text(json.dumps(result, indent=2))
    return result


def fetch_pages(paths: dict, health: dict) -> dict:
    """Step 2: fetch FETCH_PAGES with RESULTS_PER_PAGE results each, caching every page.

    Only runs if the health check passed. Pages already cached are skipped, so an interrupted
    fetch can resume without repeating calls. Stops at the first non-200 response, when a page
    returns fewer than RESULTS_PER_PAGE results (no more data), or when the budget runs out.
    Returns {"pages_fetched", "pages_cached", "stopped_reason"}.
    """
    if not health["ok"]:
        raise RuntimeError(f"Health check failed (HTTP {health['status']}): {health['error']}")
    paths["pages_dir"].mkdir(parents=True, exist_ok=True)
    fetched, reason = 0, "all pages fetched"
    for page in FETCH_PAGES:
        page_path = paths["pages_dir"] / f"page_{page:02d}.json"
        if page_path.exists():
            if len(json.loads(page_path.read_text()).get("results", [])) < RESULTS_PER_PAGE:
                reason = f"page {page} (cached) was the last page of results"
                break
            continue
        if calls_remaining(paths["ledger"]) <= 0:
            reason = "API call budget exhausted"
            break
        if fetched:
            time.sleep(SLEEP_SECONDS)
        response = api_get(page, RESULTS_PER_PAGE, paths["ledger"])
        fetched += 1
        if response.status_code != 200:
            reason = f"page {page} returned HTTP {response.status_code}: {scrub(response.text[:200])}"
            break
        body = json.loads(scrub(response.text))   # redirect_url carries the app ID as utm_source
        page_path.write_text(json.dumps(body))
        if len(body.get("results", [])) < RESULTS_PER_PAGE:
            reason = f"page {page} was the last page of results"
            break
    cached = sorted(paths["pages_dir"].glob("page_*.json"))
    logger.info("Fetch: %d new calls, %d pages cached (%s)", fetched, len(cached), reason)
    return {"pages_fetched": fetched, "pages_cached": len(cached), "stopped_reason": reason}


def combine_pages(paths: dict) -> list:
    """Merge the cached pages into one list of raw postings (deduplicated by id) and save raw_jobs.json.

    Cached pages are scrubbed of credentials first (Adzuna puts the app ID in each redirect_url).
    """
    postings, seen = [], set()
    for page_path in sorted(paths["pages_dir"].glob("page_*.json")):
        text = page_path.read_text()
        if scrub(text) != text:
            page_path.write_text(scrub(text))
        for job in json.loads(scrub(text)).get("results", []):
            if job.get("id") not in seen:
                seen.add(job.get("id"))
                postings.append(job)
    paths["raw_json"].write_text(json.dumps(postings, indent=1))
    logger.info("Saved %d unique postings to %s", len(postings), paths["raw_json"])
    return postings


def posting_state(location: dict):
    """Return a posting's US state code from Adzuna's structured location.area
    ([country, state, county, city]), falling back to parsing display_name."""
    area = location.get("area") or []
    if len(area) >= 2 and area[1]:
        code = extract_state(area[1])
        if code:
            return code
    return extract_state(location.get("display_name", ""))


def extract_postings(postings: list) -> pd.DataFrame:
    """Flatten the raw postings into EXTRACT_COLUMNS (company and location use display_name).

    `state` comes from the structured location.area field (see posting_state).
    """
    rows = [{
        "id": job.get("id"),
        "title": job.get("title"),
        "company": (job.get("company") or {}).get("display_name"),
        "location": (job.get("location") or {}).get("display_name"),
        "state": posting_state(job.get("location") or {}),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "description": job.get("description"),
        "created": job.get("created"),
        "redirect_url": job.get("redirect_url"),
        "salary_is_predicted": job.get("salary_is_predicted"),
    } for job in postings]
    return pd.DataFrame(rows, columns=EXTRACT_COLUMNS)


def salary_counts(raw: pd.DataFrame) -> dict:
    """Count postings with salary_min, salary_max, both, and Adzuna-predicted salaries."""
    both = raw["salary_min"].notna() & raw["salary_max"].notna()
    predicted = pd.to_numeric(raw["salary_is_predicted"], errors="coerce").fillna(0).astype(int) == 1
    return {"total": len(raw), "with_salary_min": int(raw["salary_min"].notna().sum()),
            "with_salary_max": int(raw["salary_max"].notna().sum()), "with_both": int(both.sum()),
            "predicted_by_adzuna": int((both & predicted).sum())}


def load_vocabulary(root: Path) -> pd.DataFrame:
    """Load the Stage 2 skill vocabulary with `forms` parsed back into lists."""
    vocab = pd.read_csv(root / "data" / "processed" / "skill_vocabulary.csv")
    vocab["forms"] = vocab["forms"].apply(literal_eval)
    return vocab


def assign_tier(salary: pd.Series) -> pd.Series:
    """Bin salaries with the exact training thresholds (Low <= LOW_MAX < Mid <= MID_MAX < High); NaN stays NaN."""
    tiers = pd.cut(salary, bins=[-np.inf, LOW_MAX, MID_MAX, np.inf], labels=TIER_LABELS)
    return tiers.astype(object).where(salary.notna(), None)


def preprocess_live(raw: pd.DataFrame, vocab: pd.DataFrame, fetch_date: str) -> pd.DataFrame:
    """Step 3: extract skills, compute mean_salary and has_salary, normalize titles, and assign tiers."""
    df = raw.copy()
    df["skills_desc"] = ""
    matched = match_description_skills(df, vocab)
    both = df["salary_min"].notna() & df["salary_max"].notna()
    mean_salary = ((df["salary_min"] + df["salary_max"]) / 2).where(both)
    out = pd.DataFrame({
        "id": df["id"].astype(str),
        "job_title": df["title"].fillna("").str.lower().str.strip(),
        "company_name": df["company"],
        "location": df["location"],
        "matched_skills": matched,
        "mean_salary": mean_salary,
        "salary_tier": assign_tier(mean_salary),
        "has_salary": both,
        "fetch_date": fetch_date,
        "state": df["state"],
        "salary_is_predicted": pd.to_numeric(df["salary_is_predicted"], errors="coerce").fillna(0).astype(int) == 1,
    })
    return out[CLEANED_COLUMNS]


def load_live_cleaned(path: Path) -> pd.DataFrame:
    """Read live_cleaned_jobs.csv back with matched_skills parsed into lists."""
    df = pd.read_csv(path, dtype={"id": str})
    df["matched_skills"] = df["matched_skills"].apply(literal_eval)
    return df


def extract_state(location: str):
    """Best-effort US state code from an Adzuna location string (e.g. "Austin, Travis County" -> None,
    "Seattle, Washington" -> "WA"). Checks each comma-separated part for a state name or code."""
    for part in reversed([p.strip() for p in str(location).split(",")]):
        code = US_STATES.get(part.lower()) or (part.upper() if part.upper() in STATE_CODES else None)
        if code:
            return code
    return None


def load_into_warehouse(live: pd.DataFrame, db_path: Path) -> dict:
    """Step 4: (re)create live_job_postings at the job_postings grain (one row per job x skill).

    Jobs with no matched skill get one row with skill_id NULL. location_id is matched on
    (city, state) against dim_location, company_id on the lowercased name against dim_company
    (0 = Unknown otherwise). time_id is the posting's `created` date as YYYYMMDD. Returns the
    verification query results.
    """
    with sqlite3.connect(db_path) as conn:
        skills = pd.read_sql("SELECT skill_id, skill_name FROM dim_skills WHERE skill_type = 'extracted'", conn)
        locations = pd.read_sql("SELECT location_id, city, state FROM dim_location WHERE city IS NOT NULL", conn)
        companies = pd.read_sql("SELECT company_id, lower(company_name) AS name FROM dim_company "
                                "WHERE company_name IS NOT NULL", conn)
        skill_ids = dict(zip(skills["skill_name"], skills["skill_id"]))
        location_ids = {(c.lower(), s): i for i, c, s in locations.itertuples(index=False)}
        company_ids = companies.drop_duplicates("name").set_index("name")["company_id"].to_dict()

        rows = []
        for job in live.itertuples(index=False):
            city = str(job.location).split(",")[0].strip().lower()
            base = {
                "job_id": int(job.id),
                "location_id": location_ids.get((city, job.state if pd.notna(job.state) else None)),
                "company_id": company_ids.get(str(job.company_name).lower(), 0),
                "time_id": int(pd.Timestamp(job.created).strftime("%Y%m%d")) if pd.notna(job.created) else None,
                "salary_tier": job.salary_tier if job.has_salary else None,
                "experience_level": "Unknown",
                "normalized_salary": job.mean_salary if job.has_salary else None,
                "source": SOURCE,
                "fetch_date": job.fetch_date,
                "has_salary": int(job.has_salary),
            }
            for skill in job.matched_skills or [None]:
                rows.append({**base, "skill_id": skill_ids.get(skill) if skill else None})
        fact = pd.DataFrame(rows)
        conn.executescript(LIVE_TABLE_SQL)
        fact.to_sql("live_job_postings", conn, if_exists="append", index=False)
        n_rows, salary_rows = conn.execute("SELECT COUNT(*), SUM(has_salary) FROM live_job_postings").fetchone()
        n_jobs, salary_jobs = conn.execute(
            "SELECT COUNT(*), SUM(has_salary) FROM "
            "(SELECT DISTINCT job_id, has_salary FROM live_job_postings)").fetchone()
        matched_locations = conn.execute(
            "SELECT COUNT(DISTINCT job_id) FROM live_job_postings WHERE location_id IS NOT NULL").fetchone()[0]
        matched_companies = conn.execute(
            "SELECT COUNT(DISTINCT job_id) FROM live_job_postings WHERE company_id <> 0").fetchone()[0]
    result = {"rows": n_rows, "rows_with_salary": salary_rows or 0, "jobs": n_jobs,
              "jobs_with_salary": salary_jobs or 0, "jobs_matched_location": matched_locations,
              "jobs_matched_company": matched_companies}
    logger.info("live_job_postings: %d rows (%d jobs), %d jobs with salary", n_rows, n_jobs, result["jobs_with_salary"])
    return result


def rule_key(antecedents: str, consequents: str) -> str:
    """Return a canonical 'a, b -> c' key for a rule given format_rules-style item strings."""
    return f"{antecedents} -> {consequents}"


def rule_metrics(onehot: pd.DataFrame, antecedents: str, consequents: str) -> dict:
    """Compute support, confidence, and lift of a rule directly on a one-hot transaction matrix.

    Returns NaNs if a skill is absent from the matrix.
    """
    ante = [s.strip() for s in antecedents.split(",")]
    cons = [s.strip() for s in consequents.split(",")]
    if any(s not in onehot.columns for s in ante + cons):
        return {"support": np.nan, "confidence": np.nan, "lift": np.nan}
    a = onehot[ante].all(axis=1)
    both = a & onehot[cons].all(axis=1)
    support = both.mean()
    confidence = both.sum() / a.sum() if a.any() else np.nan
    cons_support = onehot[cons].all(axis=1).mean()
    return {"support": support, "confidence": confidence,
            "lift": confidence / cons_support if cons_support else np.nan}


def mine_live_rules(live: pd.DataFrame) -> tuple:
    """Step 5a: Apriori on live jobs with >= 1 skill at MIN_SUPPORT, retrying at FALLBACK_SUPPORT if < MIN_RULES rules.

    Returns (one-hot matrix, formatted rules, min_support used).
    """
    transactions = live.loc[live["matched_skills"].str.len() > 0, "matched_skills"].tolist()
    onehot = assoc.encode_transactions(transactions)
    support = MIN_SUPPORT
    rules = assoc.generate_rules(assoc.mine_frequent_itemsets(onehot, support))
    if len(rules) < MIN_RULES:
        logger.info("Only %d rules at min_support=%.2f; retrying at %.2f", len(rules), support, FALLBACK_SUPPORT)
        support = FALLBACK_SUPPORT
        rules = assoc.generate_rules(assoc.mine_frequent_itemsets(onehot, support))
    return onehot, assoc.format_rules(rules), support


def compare_rules(live_rules: pd.DataFrame, train_rules: pd.DataFrame, live_onehot: pd.DataFrame) -> pd.DataFrame:
    """Step 5b: classify each rule as in both, new in live, or disappeared from live.

    The training side is the Stage 4 top-20 rules. Training rules missing from the live rules get
    their live support / confidence / lift computed directly, to show why they disappeared.
    """
    train = train_rules.assign(key=[rule_key(a, c) for a, c in zip(train_rules["antecedents"],
                                                                   train_rules["consequents"])])
    live = live_rules.assign(key=[rule_key(a, c) for a, c in zip(live_rules["antecedents"],
                                                                 live_rules["consequents"])])
    merged = train[["key", "antecedents", "consequents", "support", "confidence", "lift"]].merge(
        live[["key", "antecedents", "consequents", "support", "confidence", "lift"]],
        on="key", how="outer", suffixes=("_train", "_live"))
    merged["antecedents"] = merged["antecedents_train"].fillna(merged["antecedents_live"])
    merged["consequents"] = merged["consequents_train"].fillna(merged["consequents_live"])
    in_train, in_live = merged["support_train"].notna(), merged["support_live"].notna()
    merged["status"] = np.select([in_train & in_live, in_live], ["both", "new_in_live"], "disappeared_in_live")
    for i in merged.index[in_train & ~in_live]:
        m = rule_metrics(live_onehot, merged.at[i, "antecedents"], merged.at[i, "consequents"])
        merged.loc[i, ["support_live", "confidence_live", "lift_live"]] = [m["support"], m["confidence"], m["lift"]]
    cols = ["status", "antecedents", "consequents", "support_train", "confidence_train", "lift_train",
            "support_live", "confidence_live", "lift_live"]
    order = {"both": 0, "disappeared_in_live": 1, "new_in_live": 2}
    out = merged[cols].sort_values(["status", "lift_live"], key=lambda s: s.map(order) if s.name == "status" else -s)
    num = [c for c in cols if c.startswith(("support", "confidence", "lift"))]
    out[num] = out[num].astype(float).round(4)
    return out.reset_index(drop=True)


def build_live_features(live: pd.DataFrame, bundle: dict) -> sp.csr_matrix:
    """Step 6a: build the best model's feature matrix for live postings.

    Uses Stage 5's own builders: matched_skills multi-hot, experience_level Unknown, company_size 0
    (unknown), state from Adzuna's location, industry Other, n_matched_skills, and
    the saved title and company TF-IDF vectorizers. Checks the column order matches training.
    """
    df = pd.DataFrame({
        "matched_skills": live["matched_skills"].tolist(),
        "experience_level": "Unknown",
        "employee_count": np.nan,
        "state": live["state"].where(live["state"].notna(), None).tolist(),
        "industry": "Unknown",
        "job_title": live["job_title"].fillna("").tolist(),
        "company_name": live["company_name"].fillna("").tolist(),
    })
    base = cl.build_features(df, bundle["skills"], bundle["top_categories"])
    parts, names = [sp.csr_matrix(base.to_numpy(dtype=float))], list(base.columns)
    for column, vectorizer in (bundle["text_vectorizers"] or {}).items():
        parts.append(vectorizer.transform(df[column]))
        prefix = "title" if column == "job_title" else "company"
        names += [f"{prefix}_{w}" for w in vectorizer.get_feature_names_out()]
    if names != list(bundle["feature_names"]):
        raise ValueError("Live feature columns do not match the training feature columns")
    return sp.hstack(parts).tocsr()


def predict_live(live: pd.DataFrame, model_path: Path) -> tuple:
    """Step 6b: predict salary tier for live postings with a salary and compare with the actual tier.

    Returns (predictions frame, metrics dict), or (None, {"skipped": reason}) if fewer than
    MIN_SALARY_ROWS postings have a salary.
    """
    rows = live[live["has_salary"]].reset_index(drop=True)
    if len(rows) < MIN_SALARY_ROWS:
        return None, {"skipped": f"only {len(rows)} postings have a salary (< {MIN_SALARY_ROWS})"}
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    X = build_live_features(rows, bundle)
    predicted = [bundle["tier_labels"][i] for i in bundle["model"].predict(X)]
    preds = pd.DataFrame({
        "id": rows["id"], "job_title": rows["job_title"], "company_name": rows["company_name"],
        "mean_salary": rows["mean_salary"], "actual_tier": rows["salary_tier"], "predicted_tier": predicted,
        "correct": rows["salary_tier"].to_numpy() == np.array(predicted),
        "salary_is_predicted": rows["salary_is_predicted"],
    })
    metrics = {
        "model_name": bundle["model_name"],
        "n": len(preds),
        "accuracy": accuracy_score(preds["actual_tier"], preds["predicted_tier"]),
        "f1_macro": f1_score(preds["actual_tier"], preds["predicted_tier"], labels=TIER_LABELS,
                             average="macro", zero_division=0),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(preds["actual_tier"], preds["predicted_tier"], labels=TIER_LABELS),
            index=[f"actual {t}" for t in TIER_LABELS], columns=[f"pred {t}" for t in TIER_LABELS]),
        "majority_baseline": preds["actual_tier"].value_counts(normalize=True).max(),
    }
    results = pd.read_csv(model_path.parent / "05_classification_results.csv").set_index("model")
    metrics["train_accuracy"] = results.loc[bundle["model_name"], "accuracy"]
    metrics["train_f1_macro"] = results.loc[bundle["model_name"], "f1_macro"]
    for flag, name in [(True, "adzuna_predicted"), (False, "advertised")]:
        sub = preds[preds["salary_is_predicted"] == flag]
        metrics[f"accuracy_{name}"] = accuracy_score(sub["actual_tier"], sub["predicted_tier"]) if len(sub) else np.nan
        metrics[f"n_{name}"] = len(sub)
    logger.info("Live predictions: %d postings, accuracy %.3f", len(preds), metrics["accuracy"])
    return preds, metrics


def skill_shares(skill_lists: pd.Series) -> pd.Series:
    """Return the share of postings listing each skill, descending."""
    return skill_lists.explode().dropna().value_counts() / len(skill_lists)


def tier_distribution(tiers: pd.Series) -> pd.Series:
    """Return the Low / Mid / High shares of a tier column, ignoring missing values."""
    return tiers.dropna().value_counts(normalize=True).reindex(TIER_LABELS, fill_value=0)


def key_finding(live: pd.DataFrame, train_tiers: pd.Series, comparison: pd.DataFrame, metrics: dict) -> str:
    """Compose the 2-3 sentence key finding from the computed results."""
    live_high = tier_distribution(live["salary_tier"])["High"]
    train_high = tier_distribution(train_tiers)["High"]
    n_train_rules = int((comparison["status"] != "new_in_live").sum())
    both = int((comparison["status"] == "both").sum())
    direction = "better" if live_high > train_high else "worse"
    sentences = [
        f"With the same thresholds, {live_high:.0%} of current Adzuna IT postings with a salary are High tier vs "
        f"{train_high:.0%} of the all-industry training data, so the live sample is {direction} paid.",
        f"{both} of the {n_train_rules} top training rules reappear in the live data; the live sample is IT-only "
        f"and Adzuna returns short description snippets, so far fewer skills are matched per posting.",
    ]
    if "skipped" in metrics:
        sentences.append(f"The salary model was not evaluated: {metrics['skipped']}.")
    else:
        relation = "below" if metrics["accuracy"] < metrics["majority_baseline"] else "above"
        sentences.append(
            f"The Stage 5 model classifies {metrics['accuracy']:.0%} of live postings correctly, {relation} the "
            f"{metrics['majority_baseline']:.0%} majority-class baseline, without the experience level, company size, "
            f"and industry features it was trained with; and {metrics['n_adzuna_predicted'] / metrics['n']:.0%} of the "
            f"live salaries are Adzuna estimates rather than advertised pay, so the 'actual' tiers are themselves "
            f"model outputs.")
    return " ".join(sentences)


def build_summary(fetch_date: str, health: dict, fetch: dict, counts: dict, live: pd.DataFrame,
                  train: pd.DataFrame, warehouse: dict, rules: pd.DataFrame, support: float,
                  comparison: pd.DataFrame, metrics: dict, ledger: list) -> str:
    """Step 7: build the plain-text Stage 7 summary."""
    line = "=" * 80
    n = len(live)
    live_shares, train_shares = skill_shares(live["matched_skills"]), skill_shares(train["matched_skills"])
    top_live = pd.DataFrame({"live_share": live_shares.head(TOP_SKILLS),
                             "train_share": train_shares.reindex(live_shares.head(TOP_SKILLS).index)})
    top_train = pd.DataFrame({"train_share": train_shares.head(TOP_SKILLS),
                              "live_share": live_shares.reindex(train_shares.head(TOP_SKILLS).index).fillna(0)})
    tiers = pd.DataFrame({"live (with salary)": tier_distribution(live["salary_tier"]),
                          "training": tier_distribution(train["salary_tier"])})
    status = comparison["status"].value_counts()
    has_salary = live["has_salary"]
    parts = [
        "SkillMap — Stage 7 Live Data Summary (Adzuna API)",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Fetch date: {fetch_date}",
        f"Source: Adzuna US search, category={CATEGORY}. API calls used: {len(ledger)} of {CALL_BUDGET}: "
        f"{sum(c['results_per_page'] != RESULTS_PER_PAGE for c in ledger)} health check(s) "
        f"(HTTP {', '.join(str(c['status']) for c in ledger if c['results_per_page'] != RESULTS_PER_PAGE)}) + "
        f"{sum(c['results_per_page'] == RESULTS_PER_PAGE for c in ledger)} pages of {RESULTS_PER_PAGE}.",
        "",
        f"{line}\nFetch\n{line}",
        f"Health check: HTTP {health['status']}, first title: {health['first_title']!r}, "
        f"Adzuna total matching jobs: {health['count']}",
        f"Pages cached: {fetch['pages_cached']} ({fetch['stopped_reason']})",
        f"Total postings fetched (unique ids): {counts['total']:,}",
        f"With salary_min: {counts['with_salary_min']:,}; with salary_max: {counts['with_salary_max']:,}; "
        f"with both: {counts['with_both']:,} ({counts['with_both'] / max(counts['total'], 1):.1%})",
        f"Postings with salary (has_salary): {int(has_salary.sum()):,} ({has_salary.mean():.1%}); "
        f"of these, Adzuna-estimated (salary_is_predicted=1): {int((has_salary & live['salary_is_predicted']).sum()):,}",
        f"State parsed for {live['state'].notna().mean():.1%} of postings.",
        f"Skills: mean {live['matched_skills'].str.len().mean():.1f} per posting "
        f"(training {train['matched_skills'].str.len().mean():.1f}); "
        f"{(live['matched_skills'].str.len() > 0).mean():.1%} have >= 1 (training "
        f"{(train['matched_skills'].str.len() > 0).mean():.1%}). Adzuna descriptions are short snippets, "
        f"mean {live['description_length'].mean():.0f} characters.",
        "",
        f"{line}\nTop {TOP_SKILLS} skills in live data (share of postings) vs training\n{line}",
        top_live.round(3).to_string(),
        "",
        f"Top {TOP_SKILLS} skills in training data vs live",
        top_train.round(3).to_string(),
        "",
        f"{line}\nSalary tier distribution (training thresholds: Low <= ${LOW_MAX:,}, Mid <= ${MID_MAX:,})\n{line}",
        tiers.round(3).to_string(),
        f"Live mean salary ${live['mean_salary'].mean():,.0f}, median ${live['mean_salary'].median():,.0f}; "
        f"training mean ${train['normalized_salary'].mean():,.0f}, median ${train['normalized_salary'].median():,.0f}",
        "",
        f"{line}\nWarehouse (data/processed/skillmap.db, table live_job_postings)\n{line}",
        f"SELECT COUNT(*), SUM(has_salary) FROM live_job_postings -> {warehouse['rows']:,}, "
        f"{warehouse['rows_with_salary']:,} (rows are job x skill, as in job_postings)",
        f"Per job: {warehouse['jobs']:,} jobs, {warehouse['jobs_with_salary']:,} with salary; "
        f"location matched to dim_location for {warehouse['jobs_matched_location']:,}, "
        f"company matched to dim_company for {warehouse['jobs_matched_company']:,}",
        "",
        f"{line}\nAssociation rules on live data (outputs/07_live_association_rules.csv)\n{line}",
        f"Transactions: {int((live['matched_skills'].str.len() > 0).sum()):,} postings with >= 1 skill. "
        f"min_support={support}, confidence > {assoc.MIN_CONFIDENCE}, lift > {assoc.MIN_LIFT}: {len(rules)} rules",
        "Top 3 live rules by lift:",
        rules.head(3)[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False),
        "",
        "Comparison with the Stage 4 top-20 training rules (outputs/07_rules_comparison.csv):",
        f"  in both: {status.get('both', 0)}; disappeared in live: {status.get('disappeared_in_live', 0)}; "
        f"new in live: {status.get('new_in_live', 0)}",
        comparison[comparison["status"] != "new_in_live"][
            ["status", "antecedents", "consequents", "lift_train", "lift_live", "support_live"]
        ].to_string(index=False),
        "",
        f"{line}\nSalary-tier prediction on live data (outputs/07_live_predictions.csv)\n{line}",
    ]
    if "skipped" in metrics:
        parts.append(f"Skipped (limitation): {metrics['skipped']}.")
    else:
        parts += [
            f"Model: {metrics['model_name']}. Postings with salary: {metrics['n']:,}",
            f"Accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['f1_macro']:.3f} "
            f"(training test set: accuracy {metrics['train_accuracy']:.3f}, macro F1 {metrics['train_f1_macro']:.3f}). "
            f"Majority-class baseline "
            f"{metrics['majority_baseline']:.3f}.",
            f"Accuracy on advertised salaries: {metrics['accuracy_advertised']:.3f} (n={metrics['n_advertised']}); "
            f"on Adzuna-estimated salaries: {metrics['accuracy_adzuna_predicted']:.3f} "
            f"(n={metrics['n_adzuna_predicted']})",
            "Features unavailable from Adzuna were set as instructed: experience_level Unknown, "
            "company_size 0, industry Other; state from Adzuna's structured location.",
            f"Caveat: {metrics['n_adzuna_predicted'] / metrics['n']:.0%} of the 'actual' salaries are Adzuna's own "
            f"estimates (salary_is_predicted=1), not advertised pay.",
            metrics["confusion_matrix"].to_string(),
        ]
    parts += ["", f"{line}\nKey finding\n{line}", key_finding(live, train["salary_tier"], comparison, metrics), "",
              "Follow-up: Stage 8 (src/muse_pipeline.py, outputs/08_muse_summary.txt) addresses the truncation "
              "limitation with full-length The Muse descriptions (about 800 words and 16 skills per posting) to "
              "re-test the association rules.", ""]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None) -> dict:
    """Run Stage 7 end to end (Steps 1-7), using cached API data whenever it exists."""
    root = project_root or find_project_root()
    paths = live_paths(root)
    output_dir = root / "outputs"
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)

    health = health_check(paths)
    if not health["ok"]:
        raise RuntimeError(f"Adzuna API health check failed: HTTP {health['status']}: {health['error']}")
    logger.info("Health check: HTTP %s, first result title: %r", health["status"], health["first_title"])
    logger.info("API responding — proceeding with full fetch")
    fetch = fetch_pages(paths, health)
    postings = combine_pages(paths)
    raw = extract_postings(postings)
    raw.to_csv(paths["raw_csv"], index=False)
    counts = salary_counts(raw)
    fetch_date = read_ledger(paths["ledger"])[-1]["time"][:10]

    live = preprocess_live(raw, load_vocabulary(root), fetch_date)
    live.to_csv(paths["cleaned"], index=False)
    live = load_live_cleaned(paths["cleaned"]).merge(raw[["id", "created"]].astype({"id": str}), on="id", how="left")
    live["description_length"] = raw["description"].fillna("").str.len().to_numpy()

    warehouse = load_into_warehouse(live, paths["db"])

    onehot, rules, support = mine_live_rules(live)
    rules.to_csv(output_dir / "07_live_association_rules.csv", index=False)
    train_rules = pd.read_csv(output_dir / "04_association_rules.csv")
    comparison = compare_rules(rules, train_rules, onehot)
    comparison.to_csv(output_dir / "07_rules_comparison.csv", index=False)

    preds, metrics = predict_live(live, output_dir / "05_best_model.pkl")
    if preds is not None:
        preds.to_csv(output_dir / "07_live_predictions.csv", index=False)

    train = load_cleaned_jobs(root / "data" / "processed" / "cleaned_jobs.csv")
    summary = build_summary(fetch_date, health, fetch, counts, live, train, warehouse, rules, support,
                            comparison, metrics, read_ledger(paths["ledger"]))
    (output_dir / "07_live_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved %s", output_dir / "07_live_summary.txt")
    return {"health": health, "fetch": fetch, "raw": raw, "counts": counts, "live": live,
            "warehouse": warehouse, "rules": rules, "support": support, "comparison": comparison,
            "predictions": preds, "metrics": metrics, "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
