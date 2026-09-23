"""SkillMap Stage 2 — cleaning, skill extraction, salary imputation, and salary tiering.

Builds `data/processed/cleaned_jobs.csv` from the LinkedIn Job Postings 2023-2024
dataset (`data/raw/linkedin_postings/`), plus a cleaned per-job skill list from the
1.3M LinkedIn Jobs & Skills dataset for later association mining, and benchmarks
the salary tiers against the Data Science Salaries dataset.

Postings only carry 35 coarse job-function categories as skills, so the top
TOP_N_SKILLS skills from the 1.3M dataset are used as a vocabulary and matched
against each posting's description to produce `matched_skills`.

Run from the project root with:
    python -m src.preprocessing
"""
import ast
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Multipliers that convert a pay_period amount to an annual amount.
# Matches how the dataset's own `normalized_salary` column is computed.
PAY_PERIOD_TO_ANNUAL = {
    "HOURLY": 2080,
    "WEEKLY": 52,
    "BIWEEKLY": 26,
    "MONTHLY": 12,
    "YEARLY": 1,
}

# Annual salaries outside this range are treated as data-entry errors and dropped
# (e.g. $0, hourly rates entered as yearly, or $535M). Set either bound to None to disable.
MIN_PLAUSIBLE_SALARY = 10_000
MAX_PLAUSIBLE_SALARY = 1_000_000

# Size of the skill vocabulary taken from linkedin_jobs_skills for description matching.
TOP_N_SKILLS = 100

# Frequent linkedin_jobs "skills" that are benefits or legal boilerplate, not skills.
# Excluded before taking the top N. Empty this set to keep them.
EXCLUDED_TERMS = {
    "paid time off", "401k", "life insurance", "dental insurance", "vision insurance",
    "health insurance", "equal opportunity employer",
    # Mostly match description boilerplate ("medical, dental, vision", "hiring manager",
    # EEO statements) rather than a skill requirement.
    "medical", "vision", "diversity", "hiring",
}

# Near-duplicate skills merged into one canonical skill (variant -> canonical).
SKILL_SYNONYMS = {
    "communication skills": "communication",
    "leadership skills": "leadership",
    "problemsolving skills": "problem solving",
    "problem solving skills": "problem solving",
    "microsoft office suite": "microsoft office",
    "microsoft excel": "excel",
    "high school diploma or equivalent": "high school diploma",
    "high school diploma or ged": "high school diploma",
    "bls certification": "bls",
    "cpr certification": "cpr",
    "valid driver's license": "driver's license",
    "retail experience": "retail",
    "organizational skills": "organization",
}

# A spelling variant is searched for only if it accounts for at least this share of its
# skill's job count, which drops rare typos such as "micro soft office".
MIN_VARIANT_SHARE = 0.01

TIER_QUANTILES = (1 / 3, 2 / 3)
TIER_LABELS = ["Low", "Mid", "High"]

OUTPUT_COLUMNS = [
    "job_id", "job_title", "company_name", "location", "experience_level",
    "industry", "skills_list", "matched_skills", "normalized_salary", "salary_tier",
]
LIST_COLUMNS = ["skills_list", "matched_skills"]

POSTINGS_COLUMNS = [
    "job_id", "title", "company_name", "location", "formatted_experience_level",
    "min_salary", "max_salary", "med_salary", "pay_period", "currency", "normalized_salary",
    "description", "skills_desc",
]
SALARY_COLUMNS = ["min_salary", "max_salary", "med_salary", "pay_period", "currency"]

# ds_salaries experience codes mapped to LinkedIn's formatted_experience_level.
DS_EXPERIENCE_MAP = {
    "EN": "Entry level",
    "MI": "Associate",
    "SE": "Mid-Senior level",
    "EX": "Director",
}


def find_project_root(start: Path = None) -> Path:
    """Return the first directory at or above `start` that contains CLAUDE.md."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "CLAUDE.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root (no CLAUDE.md found above cwd).")


def clean_skill_list(skills) -> list:
    """Lowercase, strip, and de-duplicate an iterable of skills, preserving first-seen order."""
    seen = {}
    for skill in skills:
        if not isinstance(skill, str):
            continue
        cleaned = " ".join(skill.lower().split())
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def load_postings(raw_dir: Path) -> pd.DataFrame:
    """Load the main postings table (one row per job_id) with the columns Stage 2 needs."""
    path = raw_dir / "linkedin_postings" / "postings.csv"
    df = pd.read_csv(path, usecols=POSTINGS_COLUMNS, low_memory=False)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def attach_salaries(df: pd.DataFrame, raw_dir: Path) -> tuple:
    """Left-join jobs/salaries.csv on job_id, filling salary fields missing from postings.

    Returns the joined frame and the number of salary values filled from salaries.csv.
    """
    salaries = pd.read_csv(raw_dir / "linkedin_postings" / "jobs" / "salaries.csv",
                           usecols=["job_id", *SALARY_COLUMNS])
    salaries = salaries.drop_duplicates("job_id")
    merged = df.merge(salaries, on="job_id", how="left", suffixes=("", "_sal"))
    filled = 0
    for col in SALARY_COLUMNS:
        before = merged[col].isna().sum()
        merged[col] = merged[col].combine_first(merged[f"{col}_sal"])
        filled += int(before - merged[col].isna().sum())
    merged = merged.drop(columns=[f"{c}_sal" for c in SALARY_COLUMNS])
    logger.info("Joined salaries.csv: %d salary values filled", filled)
    return merged, filled


def attach_skills(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Add a `skills_list` column of skill names per job from jobs/job_skills.csv + mappings/skills.csv."""
    base = raw_dir / "linkedin_postings"
    job_skills = pd.read_csv(base / "jobs" / "job_skills.csv")
    skill_map = pd.read_csv(base / "mappings" / "skills.csv")
    named = job_skills.merge(skill_map, on="skill_abr", how="left")
    lists = named.groupby("job_id")["skill_name"].agg(clean_skill_list).rename("skills_list")
    out = df.merge(lists, on="job_id", how="left")
    out["skills_list"] = out["skills_list"].apply(lambda v: v if isinstance(v, list) else [])
    logger.info("Attached skills: %d jobs have at least one skill",
                int((out["skills_list"].str.len() > 0).sum()))
    return out


def attach_industries(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Add an `industry` column (industry names joined with '; ') from job_industries + mappings."""
    base = raw_dir / "linkedin_postings"
    job_ind = pd.read_csv(base / "jobs" / "job_industries.csv")
    ind_map = pd.read_csv(base / "mappings" / "industries.csv")
    named = job_ind.merge(ind_map, on="industry_id", how="left").dropna(subset=["industry_name"])
    joined = (named.groupby("job_id")["industry_name"]
              .agg(lambda names: "; ".join(sorted(set(n.strip() for n in names))))
              .rename("industry"))
    out = df.merge(joined, on="job_id", how="left")
    out["industry"] = out["industry"].fillna("Unknown")
    logger.info("Attached industries: %d jobs have an industry", int((out["industry"] != "Unknown").sum()))
    return out


def normalize_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Create `job_title` as the lowercased, whitespace-collapsed posting title."""
    out = df.copy()
    out["job_title"] = out["title"].astype(str).str.lower().str.split().str.join(" ")
    return out


def annualize_salary(df: pd.DataFrame) -> tuple:
    """Fill `normalized_salary` from min/max/med salary and pay_period where it is missing.

    The annual amount is the min/max midpoint (or med_salary when no range is given)
    times the pay_period multiplier: hourly x 2080, weekly x 52, biweekly x 26,
    monthly x 12, yearly x 1. Returns the frame and the number of values filled.
    """
    out = df.copy()
    midpoint = out[["min_salary", "max_salary"]].mean(axis=1).fillna(out["med_salary"])
    multiplier = out["pay_period"].str.upper().map(PAY_PERIOD_TO_ANNUAL)
    computed = midpoint * multiplier
    before = out["normalized_salary"].isna().sum()
    out["normalized_salary"] = out["normalized_salary"].combine_first(computed)
    filled = int(before - out["normalized_salary"].isna().sum())
    logger.info("Annualized salary: %d missing normalized_salary values filled from pay_period", filled)
    return out, filled


def filter_salary_rows(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Drop rows with no salary, non-USD rows, and implausible annual salaries; record counts in `stats`."""
    out = df[~(df["normalized_salary"].isna() & df["min_salary"].isna())]
    stats["dropped_no_salary"] = len(df) - len(out)

    non_usd = out["currency"].notna() & (out["currency"] != "USD")
    stats["dropped_non_usd"] = int(non_usd.sum())
    stats["non_usd_currencies"] = out.loc[non_usd, "currency"].value_counts().to_dict()
    out = out[~non_usd]

    implausible = pd.Series(False, index=out.index)
    if MIN_PLAUSIBLE_SALARY is not None:
        implausible |= out["normalized_salary"] < MIN_PLAUSIBLE_SALARY
    if MAX_PLAUSIBLE_SALARY is not None:
        implausible |= out["normalized_salary"] > MAX_PLAUSIBLE_SALARY
    stats["dropped_implausible_salary"] = int(implausible.sum())
    out = out[~implausible]

    logger.info("Salary filter: -%d no salary, -%d non-USD, -%d implausible -> %d rows",
                stats["dropped_no_salary"], stats["dropped_non_usd"],
                stats["dropped_implausible_salary"], len(out))
    return out.copy()


def impute_salary_by_experience(df: pd.DataFrame) -> tuple:
    """Fill missing `normalized_salary` with the median for the row's experience_level.

    Rows whose group has no known salary fall back to the overall median.
    Returns the frame and the number of values imputed.
    """
    out = df.copy()
    missing = out["normalized_salary"].isna()
    group_median = out.groupby("experience_level")["normalized_salary"].transform("median")
    out["normalized_salary"] = (out["normalized_salary"]
                                .fillna(group_median)
                                .fillna(out["normalized_salary"].median()))
    imputed = int(missing.sum())
    logger.info("Imputed %d missing salaries by experience_level median", imputed)
    return out, imputed


def assign_salary_tier(salary: pd.Series) -> tuple:
    """Bin salaries into Low / Mid / High at the 33rd and 66th percentiles.

    Returns the tier Series and the (p33, p66) thresholds.
    """
    p33, p66 = salary.quantile(list(TIER_QUANTILES)).tolist()
    tiers = pd.cut(salary, bins=[-np.inf, p33, p66, np.inf], labels=TIER_LABELS)
    return tiers.astype(str), (p33, p66)


def skill_key(skill: str) -> str:
    """Return a spelling-insensitive key for a skill (letters, digits, and + # only)."""
    return re.sub(r"[^a-z0-9+#]", "", skill.lower())


def clean_skill_form(skill: str) -> str:
    """Strip leading bullets/symbols and trailing punctuation from a raw skill string."""
    skill = re.sub(r"^[^a-z0-9]+|[^a-z0-9+#]+$", "", skill.lower())
    return " ".join(skill.split())


def build_skill_vocabulary(jobs_skills: pd.DataFrame, top_n: int = TOP_N_SKILLS) -> pd.DataFrame:
    """Return the `top_n` most common skills in `jobs_skills` after merging variants.

    Raw skills are grouped when they differ only in spacing or punctuation (e.g. "problem
    solving", "problemsolving", "* problem solving.") or are listed in SKILL_SYNONYMS.
    Skills in EXCLUDED_TERMS are removed. Each skill is named after its most frequent
    form, and job counts are summed across its variants.

    Columns: skill, job_count, n_variants, forms (the spellings searched for in text).
    """
    counts = jobs_skills["skills_list"].explode().value_counts()
    variants = counts.rename("job_count").rename_axis("raw").reset_index()
    variants["form"] = variants["raw"].map(clean_skill_form)
    variants["key"] = variants["form"].map(skill_key)
    variants = variants[variants["key"] != ""]

    synonym_keys = {skill_key(k): skill_key(v) for k, v in SKILL_SYNONYMS.items()}
    variants["group"] = variants["key"].replace(synonym_keys)
    excluded_keys = {skill_key(t) for t in EXCLUDED_TERMS}
    variants = variants[~variants["group"].isin(excluded_keys)]

    forms = (variants.groupby(["group", "form"], sort=False)["job_count"].sum()
             .reset_index().sort_values("job_count", ascending=False))
    totals = forms.groupby("group")["job_count"].transform("sum")
    forms["searched"] = (forms["job_count"] >= MIN_VARIANT_SHARE * totals) & forms["form"].str.isascii()

    # Name each group after its most frequent form, unless that form is a synonym source.
    canonical = {skill_key(v): v for v in SKILL_SYNONYMS.values()}
    vocab = (forms.groupby("group", sort=False)
             .agg(skill=("form", "first"), job_count=("job_count", "sum"), n_variants=("form", "size"))
             .join(forms[forms["searched"]].groupby("group")["form"].agg(list).rename("forms")))
    vocab["skill"] = [canonical.get(g, name) for g, name in zip(vocab.index, vocab["skill"])]
    vocab = vocab.sort_values("job_count", ascending=False).head(top_n).reset_index(drop=True)
    logger.info("Built skill vocabulary: top %d of %d distinct raw skills (%d terms excluded)",
                len(vocab), len(counts), len(EXCLUDED_TERMS))
    return vocab


def skill_pattern(forms: list) -> str:
    """Build a regex matching any of a skill's `forms` as a whole phrase in lowercase text.

    Words may be separated by whitespace, hyphens, or nothing, so "problem solving",
    "problem-solving", and "problemsolving" all match. Custom boundaries are used instead
    of \\b so skills ending in symbols (e.g. "c++") still match.
    """
    alternatives = []
    for form in forms:
        words = [re.escape(w) for w in re.split(r"[\s\-]+", form) if w]
        alternatives.append(r"[\s\-]*".join(words))
    return r"(?<![a-z0-9])(?:" + "|".join(alternatives) + r")(?![a-z0-9])"


def match_description_skills(df: pd.DataFrame, vocab: pd.DataFrame) -> pd.Series:
    """Return a `matched_skills` list per row: vocabulary skills found in description + skills_desc.

    Skills are listed in vocabulary order (most common first).
    """
    text = (df["description"].fillna("") + " " + df["skills_desc"].fillna("")).str.lower()
    hits = pd.DataFrame(
        {row.skill: text.str.contains(skill_pattern(row.forms), regex=True) for row in vocab.itertuples()},
        index=df.index,
    )
    matched = hits.apply(lambda row: row.index[row.to_numpy()].tolist(), axis=1)
    logger.info("Matched description skills: %d of %d jobs have at least one, mean %.1f per job",
                int((matched.str.len() > 0).sum()), len(matched), matched.str.len().mean())
    return matched


def build_cleaned_jobs(raw_dir: Path, stats: dict, vocab: pd.DataFrame) -> pd.DataFrame:
    """Run the full postings cleaning pipeline and return the final cleaned_jobs frame.

    `vocab` is the skill vocabulary from build_skill_vocabulary, used for `matched_skills`.
    """
    df = load_postings(raw_dir)
    stats["postings_rows"] = len(df)
    before = len(df)
    df = df.drop_duplicates("job_id")
    stats["dropped_duplicate_job_id"] = before - len(df)

    df, stats["salary_values_from_salaries_csv"] = attach_salaries(df, raw_dir)
    df = attach_skills(df, raw_dir)
    df = attach_industries(df, raw_dir)
    df = normalize_titles(df)
    df["experience_level"] = df["formatted_experience_level"].fillna("Unknown")

    df, stats["salary_values_annualized"] = annualize_salary(df)
    df = filter_salary_rows(df, stats)
    df, stats["salary_values_imputed"] = impute_salary_by_experience(df)

    df["salary_tier"], stats["tier_thresholds"] = assign_salary_tier(df["normalized_salary"])
    df["normalized_salary"] = df["normalized_salary"].round(2)
    df["matched_skills"] = match_description_skills(df, vocab)
    stats["final_rows"] = len(df)
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def load_linkedin_jobs_skills(raw_dir: Path) -> pd.DataFrame:
    """Load linkedin_jobs/job_skills.csv and split each comma-separated string into a cleaned list."""
    path = raw_dir / "linkedin_jobs" / "job_skills.csv"
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    df = df.dropna(subset=["job_skills"]).drop_duplicates("job_link")
    df["skills_list"] = df["job_skills"].str.split(",").apply(clean_skill_list)
    df = df[df["skills_list"].str.len() > 0]
    return df[["job_link", "skills_list"]].reset_index(drop=True)


def benchmark_ds_salaries(raw_dir: Path, thresholds: tuple, cleaned: pd.DataFrame) -> dict:
    """Compare LinkedIn salary tiers against the Data Science Salaries dataset (US, USD).

    Returns tier thresholds and per-experience-level medians from both sources.
    """
    ds = pd.read_csv(raw_dir / "ds_salaries" / "DataScience_salaries_2025.csv")
    us = ds[(ds["company_location"] == "US") & (ds["employment_type"] == "FT")]
    p33, p66 = us["salary_in_usd"].quantile(list(TIER_QUANTILES)).tolist()
    ds_tiers = pd.cut(us["salary_in_usd"], bins=[-np.inf, *thresholds, np.inf], labels=TIER_LABELS)
    ds_medians = (us.assign(experience_level=us["experience_level"].map(DS_EXPERIENCE_MAP))
                  .groupby("experience_level")["salary_in_usd"].median())
    li_medians = cleaned.groupby("experience_level")["normalized_salary"].median()
    return {
        "ds_rows_total": len(ds),
        "ds_rows_us_fulltime": len(us),
        "ds_thresholds": (p33, p66),
        "ds_share_in_linkedin_tiers": ds_tiers.value_counts(normalize=True).reindex(TIER_LABELS).round(3).to_dict(),
        "median_by_experience": pd.DataFrame({"linkedin_postings": li_medians,
                                              "ds_salaries_us": ds_medians}).round(0),
    }


def load_cleaned_jobs(path: Path) -> pd.DataFrame:
    """Read cleaned_jobs.csv back, parsing the list columns from their string form into Python lists."""
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(ast.literal_eval)
    return df


def build_summary(cleaned: pd.DataFrame, jobs_skills: pd.DataFrame, vocab: pd.DataFrame,
                  stats: dict, benchmark: dict) -> str:
    """Build the plain-text Stage 2 preprocessing summary."""
    line = "=" * 80
    p33, p66 = stats["tier_thresholds"]
    ds33, ds66 = benchmark["ds_thresholds"]
    skill_counts = cleaned["skills_list"].explode().value_counts()
    jobs_skill_counts = jobs_skills["skills_list"].explode().value_counts()
    matched_counts = cleaned["matched_skills"].explode().value_counts()
    unmatched = [s for s in vocab["skill"] if s not in matched_counts.index]
    tier_by_skill = (cleaned[["matched_skills", "normalized_salary"]].explode("matched_skills")
                     .groupby("matched_skills")["normalized_salary"].agg(["count", "median"])
                     .query("count >= 200").sort_values("median", ascending=False))
    parts = [
        "SkillMap — Stage 2 Preprocessing Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"{line}\nLinkedIn Job Postings 2023-2024 -> data/processed/cleaned_jobs.csv\n{line}",
        f"Raw postings rows:                        {stats['postings_rows']:>8,}",
        f"Dropped duplicate job_id:                 {stats['dropped_duplicate_job_id']:>8,}",
        f"Salary values filled from salaries.csv:   {stats['salary_values_from_salaries_csv']:>8,}",
        f"Salary values annualized from pay_period: {stats['salary_values_annualized']:>8,}",
        f"Dropped: no normalized_salary/min_salary: {stats['dropped_no_salary']:>8,}",
        f"Dropped: non-USD currency:                {stats['dropped_non_usd']:>8,}  {stats['non_usd_currencies']}",
        f"Dropped: salary outside [${MIN_PLAUSIBLE_SALARY:,}, ${MAX_PLAUSIBLE_SALARY:,}]: "
        f"{stats['dropped_implausible_salary']:>6,}",
        f"Salaries imputed (experience median):     {stats['salary_values_imputed']:>8,}",
        f"Final rows:                               {stats['final_rows']:>8,}",
        "",
        "-- Salary tiers (33rd / 66th percentile of normalized_salary) --",
        f"Low  <= ${p33:,.0f}",
        f"Mid  <= ${p66:,.0f}",
        f"High >  ${p66:,.0f}",
        cleaned["salary_tier"].value_counts().reindex(TIER_LABELS).to_string(),
        "",
        "-- normalized_salary distribution --",
        cleaned["normalized_salary"].describe().round(0).to_string(),
        "",
        "-- Experience level --",
        cleaned["experience_level"].value_counts().to_string(),
        "",
        "-- Salary tier by experience level (row %) --",
        (pd.crosstab(cleaned["experience_level"], cleaned["salary_tier"], normalize="index")
         .reindex(columns=TIER_LABELS) * 100).round(1).to_string(),
        "",
        "-- Skills (job-function categories from mappings/skills.csv) --",
        f"Distinct skills: {len(skill_counts)}",
        f"Jobs with no skills: {int((cleaned['skills_list'].str.len() == 0).sum()):,}",
        f"Mean skills per job: {cleaned['skills_list'].str.len().mean():.2f}",
        skill_counts.head(15).to_string(),
        "",
        f"-- Matched skills (top {len(vocab)} linkedin_jobs skills found in descriptions) --",
        f"Jobs with >= 1 matched skill: {int((cleaned['matched_skills'].str.len() > 0).sum()):,} "
        f"of {len(cleaned):,}",
        f"Mean matched skills per job: {cleaned['matched_skills'].str.len().mean():.1f}",
        f"Vocabulary skills never matched: {unmatched if unmatched else 'none'}",
        f"Excluded non-skill terms: {sorted(EXCLUDED_TERMS)}",
        f"Synonyms merged: {SKILL_SYNONYMS}",
        "Spelling variants per skill are listed in data/processed/skill_vocabulary.csv",
        "Most frequent matched skills:",
        matched_counts.head(20).to_string(),
        "",
        "Highest median salary by matched skill (skills in >= 200 jobs):",
        tier_by_skill.head(10).round(0).to_string(),
        "",
        "Lowest median salary by matched skill (skills in >= 200 jobs):",
        tier_by_skill.tail(10).round(0).to_string(),
        "",
        "-- Top industries --",
        f"Jobs with Unknown industry: {int((cleaned['industry'] == 'Unknown').sum()):,}",
        cleaned["industry"].value_counts().head(10).to_string(),
        "",
        "-- Nulls in cleaned_jobs --",
        cleaned.isna().sum().to_string(),
        "",
        f"{line}\n1.3M LinkedIn Jobs & Skills -> data/processed/linkedin_jobs_skills.csv\n{line}",
        f"Jobs with skills: {len(jobs_skills):,}",
        f"Distinct skills:  {len(jobs_skill_counts):,}",
        f"Mean skills per job: {jobs_skills['skills_list'].str.len().mean():.1f}",
        "Top 20 skills:",
        jobs_skill_counts.head(20).to_string(),
        "",
        f"{line}\nBenchmark: Data Science Salaries 2020-2025 (US, full-time)\n{line}",
        f"Rows used: {benchmark['ds_rows_us_fulltime']:,} of {benchmark['ds_rows_total']:,}",
        f"ds_salaries 33rd / 66th percentile: ${ds33:,.0f} / ${ds66:,.0f}",
        f"Share of ds_salaries rows in LinkedIn tiers: {benchmark['ds_share_in_linkedin_tiers']}",
        "Median salary by experience level (ds EN/MI/SE/EX mapped to Entry/Associate/Mid-Senior/Director):",
        benchmark["median_by_experience"].to_string(),
        "",
    ]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None) -> dict:
    """Run Stage 2 end to end, writing cleaned CSVs to data/processed/ and the summary to outputs/.

    Returns a dict with the cleaned frames, stats, benchmark results, and summary text.
    """
    root = project_root or find_project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    output_dir = root / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    jobs_skills = load_linkedin_jobs_skills(raw_dir)
    vocab = build_skill_vocabulary(jobs_skills)
    cleaned = build_cleaned_jobs(raw_dir, stats, vocab)
    benchmark = benchmark_ds_salaries(raw_dir, stats["tier_thresholds"], cleaned)

    cleaned_path = processed_dir / "cleaned_jobs.csv"
    cleaned.to_csv(cleaned_path, index=False)
    logger.info("Saved %s (%d rows)", cleaned_path, len(cleaned))

    jobs_skills_path = processed_dir / "linkedin_jobs_skills.csv"
    jobs_skills.to_csv(jobs_skills_path, index=False)
    logger.info("Saved %s (%d rows)", jobs_skills_path, len(jobs_skills))

    vocab_path = processed_dir / "skill_vocabulary.csv"
    vocab.to_csv(vocab_path, index=False)
    logger.info("Saved %s (%d skills)", vocab_path, len(vocab))

    summary = build_summary(cleaned, jobs_skills, vocab, stats, benchmark)
    summary_path = output_dir / "02_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Saved %s", summary_path)

    return {"cleaned": cleaned, "jobs_skills": jobs_skills, "vocab": vocab, "stats": stats,
            "benchmark": benchmark, "summary": summary}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
