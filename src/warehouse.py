"""SkillMap Stage 3 — SQLite star schema data warehouse and OLAP queries.

Loads `data/processed/cleaned_jobs.csv` (plus company, time, and industry attributes
from the raw LinkedIn Job Postings tables) into `data/processed/skillmap.db`.

Schema
------
Fact table `job_postings`, one row per (job, skill):
    job_id, skill_id, location_id, company_id, time_id,
    salary_tier, experience_level, normalized_salary
    Jobs with no skills get a single row with skill_id NULL, so every job is present.
    Salary is repeated on each of a job's rows: aggregate salaries per job through the
    `v_jobs` view (one row per job), not directly over the fact table.

Dimensions: dim_skills, dim_location, dim_company, dim_time, dim_industry.
Bridge: bridge_job_industry (job_id, industry_id). A job can have up to 3 industries.

Run from the project root with:
    python -m src.warehouse
"""
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.preprocessing import find_project_root, load_cleaned_jobs

logger = logging.getLogger(__name__)

UNKNOWN_COMPANY_ID = 0

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "puerto rico": "PR",
}
STATE_CODES = set(US_STATES.values())

SCHEMA_SQL = """
CREATE TABLE dim_skills (
    skill_id    INTEGER PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    skill_type  TEXT NOT NULL CHECK (skill_type IN ('extracted', 'category')),
    UNIQUE (skill_name, skill_type)
);
CREATE TABLE dim_location (
    location_id    INTEGER PRIMARY KEY,
    location_raw   TEXT NOT NULL UNIQUE,
    city           TEXT,
    state          TEXT,
    country        TEXT NOT NULL,
    location_type  TEXT NOT NULL CHECK (location_type IN ('city', 'metro', 'state', 'country'))
);
CREATE TABLE dim_company (
    company_id      INTEGER PRIMARY KEY,
    company_name    TEXT,
    company_size    INTEGER,
    employee_count  INTEGER,
    city            TEXT,
    state           TEXT,
    country         TEXT
);
CREATE TABLE dim_time (
    time_id       INTEGER PRIMARY KEY,
    date          TEXT NOT NULL,
    year          INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    month_name    TEXT NOT NULL,
    day           INTEGER NOT NULL,
    day_of_week   TEXT NOT NULL,
    week_of_year  INTEGER NOT NULL
);
CREATE TABLE dim_industry (
    industry_id    INTEGER PRIMARY KEY,
    industry_name  TEXT NOT NULL
);
CREATE TABLE bridge_job_industry (
    job_id       INTEGER NOT NULL,
    industry_id  INTEGER NOT NULL REFERENCES dim_industry (industry_id),
    PRIMARY KEY (job_id, industry_id)
);
CREATE TABLE job_postings (
    posting_skill_id   INTEGER PRIMARY KEY,
    job_id             INTEGER NOT NULL,
    skill_id           INTEGER REFERENCES dim_skills (skill_id),
    location_id        INTEGER NOT NULL REFERENCES dim_location (location_id),
    company_id         INTEGER NOT NULL REFERENCES dim_company (company_id),
    time_id            INTEGER NOT NULL REFERENCES dim_time (time_id),
    salary_tier        TEXT NOT NULL CHECK (salary_tier IN ('Low', 'Mid', 'High')),
    experience_level   TEXT NOT NULL,
    normalized_salary  REAL NOT NULL,
    UNIQUE (job_id, skill_id)
);
CREATE INDEX idx_fact_job ON job_postings (job_id);
CREATE INDEX idx_fact_skill ON job_postings (skill_id);
CREATE INDEX idx_fact_location ON job_postings (location_id);
CREATE INDEX idx_fact_company ON job_postings (company_id);
CREATE INDEX idx_fact_time ON job_postings (time_id);
CREATE INDEX idx_bridge_industry ON bridge_job_industry (industry_id);

-- One row per job: use this for any salary aggregate.
CREATE VIEW v_jobs AS
SELECT DISTINCT job_id, location_id, company_id, time_id,
       salary_tier, experience_level, normalized_salary
FROM job_postings;
"""

OLAP_QUERIES = {
    "Q1 — Top 10 most demanded skills (extracted from descriptions)": """
        SELECT s.skill_name,
               COUNT(*)                                             AS jobs,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM v_jobs), 1) AS pct_of_jobs,
               ROUND(AVG(f.normalized_salary))                      AS avg_salary,
               ROUND(100.0 * AVG(f.salary_tier = 'High'), 1)        AS pct_high_tier
        FROM job_postings f
        JOIN dim_skills s ON s.skill_id = f.skill_id
        WHERE s.skill_type = 'extracted'
        GROUP BY s.skill_name
        ORDER BY jobs DESC
        LIMIT 10
    """,
    "Q2 — Average salary by experience level": """
        SELECT experience_level,
               COUNT(*)                                       AS jobs,
               ROUND(AVG(normalized_salary))                  AS avg_salary,
               ROUND(MIN(normalized_salary))                  AS min_salary,
               ROUND(MAX(normalized_salary))                  AS max_salary,
               ROUND(100.0 * AVG(salary_tier = 'Low'), 1)     AS pct_low,
               ROUND(100.0 * AVG(salary_tier = 'Mid'), 1)     AS pct_mid,
               ROUND(100.0 * AVG(salary_tier = 'High'), 1)    AS pct_high
        FROM v_jobs
        GROUP BY experience_level
        ORDER BY avg_salary DESC
    """,
    "Q3 — Job count by industry and location (state), top 15": """
        SELECT i.industry_name,
               l.state,
               COUNT(*)                          AS jobs,
               ROUND(AVG(j.normalized_salary))   AS avg_salary
        FROM v_jobs j
        JOIN bridge_job_industry b ON b.job_id = j.job_id
        JOIN dim_industry i        ON i.industry_id = b.industry_id
        JOIN dim_location l        ON l.location_id = j.location_id
        WHERE l.state IS NOT NULL
        GROUP BY i.industry_name, l.state
        ORDER BY jobs DESC
        LIMIT 15
    """,
}


def parse_location(raw: str) -> dict:
    """Split a LinkedIn location string into city, state (2-letter code), country, and type.

    Handles "City, ST", "City, State, United States", "State, United States",
    "City, State Metropolitan Area", bare metro names ("Greater Boston"), and "United States".
    """
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    country = "United States"
    if parts and parts[-1] == "United States":
        parts = parts[:-1]
    if not parts:
        return {"city": None, "state": None, "country": country, "location_type": "country"}

    is_metro = bool(re.search(r"\b(Area|Metroplex|Greater)\b", raw))
    last = re.sub(r"\s+(Metropolitan\s+)?Area$", "", parts[-1]).strip()
    state = last.upper() if last.upper() in STATE_CODES else US_STATES.get(last.lower())

    if len(parts) == 1:
        if state:
            return {"city": None, "state": state, "country": country, "location_type": "state"}
        return {"city": None, "state": None, "country": country, "location_type": "metro"}
    return {"city": parts[0], "state": state, "country": country,
            "location_type": "metro" if is_metro else "city"}


def load_job_attributes(raw_dir: Path, job_ids: pd.Series) -> pd.DataFrame:
    """Return company_id and posting date (from original_listed_time) for `job_ids` from postings.csv."""
    postings = pd.read_csv(raw_dir / "linkedin_postings" / "postings.csv",
                           usecols=["job_id", "company_id", "original_listed_time"])
    postings = postings[postings["job_id"].isin(job_ids)]
    postings["company_id"] = postings["company_id"].fillna(UNKNOWN_COMPANY_ID).astype("int64")
    postings["date"] = pd.to_datetime(postings["original_listed_time"], unit="ms").dt.normalize()
    return postings[["job_id", "company_id", "date"]]


def build_dim_skills(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build dim_skills from matched_skills ('extracted') and skills_list ('category')."""
    frames = []
    for col, skill_type in [("matched_skills", "extracted"), ("skills_list", "category")]:
        names = sorted(cleaned[col].explode().dropna().unique())
        frames.append(pd.DataFrame({"skill_name": names, "skill_type": skill_type}))
    dim = pd.concat(frames, ignore_index=True)
    dim.insert(0, "skill_id", range(1, len(dim) + 1))
    return dim


def build_dim_location(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build dim_location from the distinct raw location strings."""
    raw = sorted(cleaned["location"].unique())
    dim = pd.DataFrame([{"location_raw": r, **parse_location(r)} for r in raw])
    dim.insert(0, "location_id", range(1, len(dim) + 1))
    return dim


def build_dim_company(raw_dir: Path, attrs: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build dim_company from companies.csv and the latest employee_counts row per company.

    Includes an 'Unknown' member (company_id 0) for postings without a company_id, and falls
    back to the posting's company_name for companies missing from companies.csv.
    """
    base = raw_dir / "linkedin_postings" / "companies"
    companies = pd.read_csv(base / "companies.csv",
                            usecols=["company_id", "name", "company_size", "city", "state", "country"])
    counts = (pd.read_csv(base / "employee_counts.csv")
              .sort_values("time_recorded")
              .drop_duplicates("company_id", keep="last")[["company_id", "employee_count"]])

    names = (attrs.merge(cleaned[["job_id", "company_name"]], on="job_id")
             .dropna(subset=["company_name"])
             .drop_duplicates("company_id")[["company_id", "company_name"]])
    dim = (pd.DataFrame({"company_id": sorted(attrs["company_id"].unique())})
           .merge(companies, on="company_id", how="left")
           .merge(counts, on="company_id", how="left")
           .merge(names, on="company_id", how="left"))
    dim["company_name"] = dim["name"].combine_first(dim["company_name"])
    dim.loc[dim["company_id"] == UNKNOWN_COMPANY_ID, "company_name"] = "Unknown"
    for col in ["company_size", "employee_count"]:
        dim[col] = dim[col].astype("Int64")
    return dim[["company_id", "company_name", "company_size", "employee_count", "city", "state", "country"]]


def build_dim_time(dates: pd.Series) -> pd.DataFrame:
    """Build dim_time with one row per distinct posting date; time_id is YYYYMMDD."""
    d = pd.Series(sorted(dates.unique()))
    return pd.DataFrame({
        "time_id": d.dt.strftime("%Y%m%d").astype(int),
        "date": d.dt.strftime("%Y-%m-%d"),
        "year": d.dt.year,
        "quarter": d.dt.quarter,
        "month": d.dt.month,
        "month_name": d.dt.month_name(),
        "day": d.dt.day,
        "day_of_week": d.dt.day_name(),
        "week_of_year": d.dt.isocalendar().week.astype(int).to_numpy(),
    })


def build_industry_tables(raw_dir: Path, job_ids: pd.Series) -> tuple:
    """Return (dim_industry, bridge_job_industry) for the jobs in `job_ids`."""
    base = raw_dir / "linkedin_postings"
    dim = pd.read_csv(base / "mappings" / "industries.csv").dropna(subset=["industry_name"])
    dim["industry_name"] = dim["industry_name"].str.strip()
    bridge = pd.read_csv(base / "jobs" / "job_industries.csv")
    bridge = bridge[bridge["job_id"].isin(job_ids) & bridge["industry_id"].isin(dim["industry_id"])]
    return dim, bridge.drop_duplicates()


def build_fact(cleaned: pd.DataFrame, attrs: pd.DataFrame, dim_skills: pd.DataFrame,
               dim_location: pd.DataFrame) -> pd.DataFrame:
    """Build the job_postings fact table at (job, skill) grain from both skill columns."""
    jobs = (cleaned.merge(attrs, on="job_id", how="left")
            .merge(dim_location[["location_id", "location_raw"]],
                   left_on="location", right_on="location_raw", how="left"))
    jobs["time_id"] = jobs["date"].dt.strftime("%Y%m%d").astype(int)

    skill_ids = dim_skills.set_index(["skill_type", "skill_name"])["skill_id"]
    pairs = []
    for col, skill_type in [("matched_skills", "extracted"), ("skills_list", "category")]:
        exploded = jobs[["job_id", col]].explode(col).dropna(subset=[col])
        exploded["skill_id"] = skill_ids.loc[skill_type].reindex(exploded[col]).to_numpy()
        pairs.append(exploded[["job_id", "skill_id"]])
    pairs = pd.concat(pairs, ignore_index=True)

    job_cols = ["job_id", "location_id", "company_id", "time_id",
                "salary_tier", "experience_level", "normalized_salary"]
    fact = jobs[job_cols].merge(pairs, on="job_id", how="left")
    fact["skill_id"] = fact["skill_id"].astype("Int64")
    return fact[["job_id", "skill_id", *job_cols[1:]]]


def build_warehouse(project_root: Path = None) -> Path:
    """Build data/processed/skillmap.db from cleaned_jobs.csv and the raw postings tables.

    Any existing database is replaced. Returns the database path.
    """
    root = project_root or find_project_root()
    raw_dir = root / "data" / "raw"
    db_path = root / "data" / "processed" / "skillmap.db"

    cleaned = load_cleaned_jobs(root / "data" / "processed" / "cleaned_jobs.csv")
    logger.info("Loaded cleaned_jobs.csv: %d jobs", len(cleaned))
    attrs = load_job_attributes(raw_dir, cleaned["job_id"])

    dim_skills = build_dim_skills(cleaned)
    dim_location = build_dim_location(cleaned)
    dim_company = build_dim_company(raw_dir, attrs, cleaned)
    dim_time = build_dim_time(attrs["date"])
    dim_industry, bridge = build_industry_tables(raw_dir, cleaned["job_id"])
    fact = build_fact(cleaned, attrs, dim_skills, dim_location)

    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        tables = [("dim_skills", dim_skills), ("dim_location", dim_location),
                  ("dim_company", dim_company), ("dim_time", dim_time),
                  ("dim_industry", dim_industry), ("bridge_job_industry", bridge),
                  ("job_postings", fact)]
        for name, df in tables:
            df.to_sql(name, conn, if_exists="append", index=False)
            logger.info("Loaded %s: %d rows", name, len(df))
        conn.execute("PRAGMA foreign_keys = ON")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"{len(violations)} foreign key violations, e.g. {violations[:3]}")
    logger.info("Saved %s", db_path)
    return db_path


def table_counts(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return row counts for every table and view in the warehouse."""
    names = pd.read_sql("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
                        "ORDER BY type, name", conn)
    names["rows"] = [conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names["name"]]
    return names


def run_olap_queries(conn: sqlite3.Connection) -> dict:
    """Run the three OLAP queries and return {title: result DataFrame}."""
    return {title: pd.read_sql(sql, conn) for title, sql in OLAP_QUERIES.items()}


def build_summary(conn: sqlite3.Connection, results: dict) -> str:
    """Build the plain-text Stage 3 summary: table sizes, dimension checks, and OLAP results."""
    line = "=" * 80
    loc_types = pd.read_sql("SELECT location_type, COUNT(*) AS locations, "
                            "SUM(state IS NULL) AS without_state FROM dim_location "
                            "GROUP BY location_type", conn)
    jobs_no_state = conn.execute("SELECT COUNT(*) FROM v_jobs j JOIN dim_location l "
                                 "USING (location_id) WHERE l.state IS NULL").fetchone()[0]
    unknown_company = conn.execute(f"SELECT COUNT(*) FROM v_jobs WHERE company_id = {UNKNOWN_COMPANY_ID}"
                                   ).fetchone()[0]
    no_size = conn.execute("SELECT COUNT(*) FROM v_jobs JOIN dim_company USING (company_id) "
                           "WHERE company_size IS NULL").fetchone()[0]
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM dim_time").fetchone()
    parts = [
        "SkillMap — Stage 3 Data Warehouse Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "Database: data/processed/skillmap.db",
        "",
        f"{line}\nTables\n{line}",
        table_counts(conn).to_string(index=False),
        "",
        "-- Dimension checks --",
        loc_types.to_string(index=False),
        f"Jobs whose location has no state (metro/country level): {jobs_no_state:,}",
        f"Jobs with unknown company: {unknown_company:,}",
        f"Jobs whose company has no company_size: {no_size:,}",
        f"Posting dates: {date_range[0]} to {date_range[1]}",
        "",
    ]
    for title, df in results.items():
        parts += [f"{line}\n{title}\n{line}", df.to_string(index=False), ""]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None) -> dict:
    """Build the warehouse, run the OLAP queries, and write outputs/03_summary.txt."""
    root = project_root or find_project_root()
    db_path = build_warehouse(root)
    with sqlite3.connect(db_path) as conn:
        results = run_olap_queries(conn)
        summary = build_summary(conn, results)
    summary_path = root / "outputs" / "03_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Saved %s", summary_path)
    return {"db_path": db_path, "results": results, "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
