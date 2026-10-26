"""SkillMap live demo: fetch current tech postings from The Muse and extract skills on the spot.

Fetches PAGES pages (20 postings each) from The Muse public API, strips the HTML descriptions,
matches the 245-skill Stage 2 vocabulary, and prints one row per posting: title, company,
level, and top skills (technical skills first, then the most specific of the rest).

Read-only with respect to the project: raw pages go to a fresh temp folder, the vocabulary is
only read, and skillmap.db and outputs/ are never opened. Requests reuse the Stage 8
User-Agent and retry logic (src.muse_pipeline.get_page). If the API cannot be reached, the
demo falls back to the pages cached in Stage 8 and says so.

Run from the project root with:
    .venv/bin/python scripts/live_demo.py                 # live: 2 API calls
    .venv/bin/python scripts/live_demo.py --offline       # cached Stage 8 pages, no API calls
"""
import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from ast import literal_eval
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import muse_pipeline as mp  # noqa: E402
from src.preprocessing import match_description_skills  # noqa: E402

logger = logging.getLogger("live_demo")

DEFAULT_CATEGORY = "Software Engineering"
PAGES = 2
TOP_SKILLS = 5
TRAINING_SKILLS_PER_POSTING = 11.5   # Stage 2 mean, for comparison
CACHE_DIR = ROOT / "data" / "raw" / "muse_live" / "pages"
CACHE_PREFIX = {muse: requested.lower().replace(" ", "_") for requested, muse in mp.MUSE_CATEGORIES.items()}
LEVEL_LABELS = {"Mid-Senior": "Mid", "Director": "Manager"}   # short labels for the table
# Vocabulary skills that mostly match figurative phrases in corporate text ("driving key business
# outcomes" vs CDL driving). Still counted in n_skills; only shown after the other skills.
DISPLAY_LAST = {"driving"}


def parse_args() -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="Fetch live Muse postings and extract skills.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY,
                        help=f"The Muse category name (default: {DEFAULT_CATEGORY!r})")
    parser.add_argument("--pages", type=int, default=PAGES, help=f"pages to fetch, 20 postings each (default: {PAGES})")
    parser.add_argument("--offline", action="store_true", help="use cached Stage 8 pages instead of the API")
    parser.add_argument("--out-dir", type=Path, default=None, help="folder for raw pages (default: a new temp folder)")
    return parser.parse_args()


def load_vocabulary() -> pd.DataFrame:
    """Read the 245-skill Stage 2 vocabulary (read-only) with `forms` parsed into lists."""
    vocab = pd.read_csv(ROOT / "data" / "processed" / "skill_vocabulary.csv")
    vocab["forms"] = vocab["forms"].apply(literal_eval)
    return vocab


def fetch_live(category: str, pages: int, out_dir: Path) -> tuple:
    """Fetch `pages` pages from The Muse into out_dir; return (postings, API attempts made, failure or None)."""
    postings, log = [], []
    for page in range(pages):
        if page:
            time.sleep(mp.SLEEP_SECONDS)
        body, failure = mp.get_page(category, page, log)
        if body is None:
            return postings, len(log), failure
        (out_dir / f"page_{page:02d}.json").write_text(json.dumps(body))
        postings.extend(body.get("results", []))
    return postings, len(log), None


def load_cached(category: str, pages: int) -> list:
    """Return postings from the Stage 8 cached pages for `category` (read-only)."""
    prefix = CACHE_PREFIX.get(category, "engineering")
    postings = []
    for page in range(pages):
        path = CACHE_DIR / f"{prefix}_{page:02d}.json"
        if path.exists():
            postings.extend(json.loads(path.read_text()).get("results", []))
    return postings


def rank_skills(skills: list, vocab: pd.DataFrame) -> list:
    """Order skills for display: technical (forced) skills first, then the rarest (most specific) first,
    with DISPLAY_LAST skills at the end."""
    forced = dict(zip(vocab["skill"], vocab["forced"]))
    count = dict(zip(vocab["skill"], vocab["job_count"]))
    return sorted(skills, key=lambda s: (s in DISPLAY_LAST, not forced.get(s, False), count.get(s, 0)))


def build_table(postings: list, vocab: pd.DataFrame) -> pd.DataFrame:
    """Strip HTML, match skills, and return one row per unique posting."""
    raw = mp.extract_postings(postings).drop_duplicates("id").reset_index(drop=True)
    raw = raw.assign(skills_desc="")
    raw["skills"] = match_description_skills(raw, vocab)
    return pd.DataFrame({
        "title": raw["name"].fillna("").str.strip(),
        "company": raw["company"].fillna("?"),
        "level": raw["levels"].map(mp.map_experience).replace(LEVEL_LABELS),
        "skills": raw["skills"],
        "n_skills": raw["skills"].str.len(),
        "words": raw["description_words"],
        "published": raw["publication_date"].str[:10],
        "technical": raw["skills"].map(lambda s: [x for x in s if x in mp.TECHNICAL_SKILLS]),
    })


def clip(text: str, width: int) -> str:
    """Truncate text to width characters, marking cuts with an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def print_table(table: pd.DataFrame, vocab: pd.DataFrame) -> None:
    """Print the postings as a fixed-width table sized to the terminal."""
    term = max(shutil.get_terminal_size((140, 40)).columns, 100)
    w_no, w_title, w_company, w_level, w_n = 3, 34, 18, 8, 3
    w_skills = term - (w_no + w_title + w_company + w_level + w_n + 5 * 2)
    header = (f"{'#':>{w_no}}  {'Title':<{w_title}}  {'Company':<{w_company}}  {'Level':<{w_level}}  "
              f"{'n':>{w_n}}  Top skills")
    print(header)
    print("─" * min(term, len(header) + w_skills))
    for i, row in enumerate(table.itertuples(index=False), 1):
        skills = ", ".join(rank_skills(row.skills, vocab)[:TOP_SKILLS]) or "—"
        print(f"{i:>{w_no}}  {clip(row.title, w_title):<{w_title}}  {clip(row.company, w_company):<{w_company}}  "
              f"{clip(row.level, w_level):<{w_level}}  {row.n_skills:>{w_n}}  {clip(skills, w_skills)}")


def print_summary(table: pd.DataFrame) -> None:
    """Print aggregate stats: skills per posting, most common skills, and most common technical skills."""
    n = len(table)
    common = Counter(s for skills in table["skills"] for s in skills).most_common(8)
    tech = Counter(s for skills in table["technical"] for s in skills).most_common(8)
    print()
    print(f"Postings: {n}   companies: {table['company'].nunique()}   "
          f"published {table['published'].min()} → {table['published'].max()}")
    print(f"Mean description: {table['words'].mean():,.0f} words   mean skills per posting: "
          f"{table['n_skills'].mean():.1f}  (LinkedIn training data: {TRAINING_SKILLS_PER_POSTING})   "
          f"with ≥ 1 technical skill: {(table['technical'].str.len() > 0).mean():.0%}")
    print("Most common skills:     " + ", ".join(f"{s} ({k / n:.0%})" for s, k in common))
    print("Most common technical:  " + (", ".join(f"{s} ({k / n:.0%})" for s, k in tech) or "none"))


def main() -> None:
    """Fetch (or load cached) postings, extract skills, and print the demo table."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="skillmap_demo_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    source = f"The Muse API, live ({datetime.now():%Y-%m-%d %H:%M})"
    if args.offline:
        postings, source = load_cached(args.category, args.pages), "cached Stage 8 pages (offline mode)"
    else:
        print(f"Fetching {args.pages} page(s) of {args.category!r} from {mp.API_URL} ...")
        postings, attempts, failure = fetch_live(args.category, args.pages, out_dir)
        print(f"API requests: {attempts}; raw pages saved to {out_dir}")
        if failure:
            logger.warning("Live fetch stopped (%s)", failure)
            if not postings:
                postings, source = load_cached(args.category, args.pages), "cached Stage 8 pages (API unreachable)"
    if not postings:
        sys.exit("No postings available (API unreachable and no cached pages for this category).")

    vocab = load_vocabulary()
    table = build_table(postings, vocab)
    print(f"\nSkillMap live demo — {args.category} — source: {source}\n")
    print_table(table, vocab)
    print_summary(table)


if __name__ == "__main__":
    main()
