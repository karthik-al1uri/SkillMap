"""SkillMap dashboard: one self-contained interactive Plotly HTML page.

Reads data/processed/skillmap.db and the stage outputs in outputs/ and writes
outputs/dashboard.html with four tabs: association rules, clusters, model results, and live
trends. Every number comes from those files; nothing is fetched from the network.

Run from the project root with:
    .venv/bin/python scripts/build_dashboard.py
    .venv/bin/python scripts/build_dashboard.py --max-per-company 50 --out outputs/dashboard.html
"""
import argparse
import html
import io
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import muse_pipeline as mp  # noqa: E402
from src.classification import AUC_TARGET, F1_TARGET  # noqa: E402
from src.clustering import SILHOUETTE_TARGET  # noqa: E402

logger = logging.getLogger("build_dashboard")

TOP_RULES = 20
TOP_HIGH_RULES = 15
TOP_CLUSTER_SKILLS = 4
MIN_CLUSTER_SKILL_SHARE = 0.10
TOP_LIVE_SKILLS = 15
TOP_SHIFTS = 8
TOP_TREND_SKILLS = 8
TIERS = ["Low", "Mid", "High"]

LIGHT = {
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "tiers": ["#86b6ef", "#3987e5", "#1c5cab"],
    "seq": ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"],
}
DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "tiers": ["#256abf", "#3987e5", "#86b6ef"],
    "seq": ["#383835", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
}
C = LIGHT
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"]}


def theme_map() -> dict:
    """Light hex -> dark hex for every themed color, so the page can swap figure colors in dark mode."""
    pairs = {}
    for key, light in LIGHT.items():
        dark = DARK[key]
        for lo, hi in zip(light, dark) if isinstance(light, list) else [(light, dark)]:
            if pairs.setdefault(lo, hi) != hi:
                raise ValueError(f"{lo} maps to both {pairs[lo]} and {hi} in dark mode")
    return pairs


def base_layout(fig: go.Figure, height: int = 380, **kwargs) -> go.Figure:
    """Apply the shared chrome: surface, ink, hairline grid, compact margins, hover label."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=16, t=8, b=8), paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=C["ink2"]),
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["grid"], font=dict(color=C["ink"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        bargap=0.25, **kwargs)
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"], automargin=True)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"], automargin=True)
    return fig


def target_line(fig: go.Figure, value: float, text: str, axis: str = "x", side: str = "right") -> None:
    """Draw a thin dotted reference line, labelled above the plot (x) or at its left end (y)."""
    line = dict(color=C["muted"], width=1, dash="dot")
    font = dict(color=C["muted"], size=11)
    if axis == "x":
        fig.add_vline(value, line=line)
        fig.add_annotation(x=value, y=1, xref="x", yref="paper", yanchor="bottom", showarrow=False, text=text,
                           font=font, xanchor="left" if side == "right" else "right")
        fig.update_layout(margin_t=max(fig.layout.margin.t or 0, 28))
    else:
        fig.add_hline(value, line=line, annotation_text=text, annotation_position="bottom left", annotation_font=font)


def short(text: str, width: int = 48) -> str:
    """Truncate long labels for axis ticks."""
    return text if len(text) <= width else text[:width - 1] + "…"


# ---------- data ----------

def load_inputs(root: Path, cap: int) -> dict:
    """Read every file the dashboard uses; missing optional files come back as None."""
    out_dir, db = root / "outputs", root / "data" / "processed" / "skillmap.db"

    def csv(name: str):
        """Read outputs/<name> if it exists."""
        path = out_dir / name
        if not path.exists():
            logger.warning("Missing %s; its charts are skipped", path)
            return None
        return pd.read_csv(path)

    data = {name: csv(f"{name}.csv") for name in [
        "04_association_rules", "04_high_salary_rules", "05_classification_results", "06_cluster_labels",
        "08_skill_comparison", "08_rules_validation"]}
    summary = out_dir / "06_summary.txt"
    data["kmeans_sweep"] = parse_kmeans_sweep(summary.read_text()) if summary.exists() else None
    with sqlite3.connect(db) as conn:
        data["jobs"] = pd.read_sql_query("SELECT job_id, salary_tier FROM v_jobs", conn)
        data["job_skills"] = pd.read_sql_query(
            "SELECT DISTINCT f.job_id, s.skill_name FROM job_postings f "
            "JOIN dim_skills s ON s.skill_id = f.skill_id WHERE s.skill_type = 'extracted'", conn)
        data["n_skills_vocab"] = conn.execute(
            "SELECT COUNT(*) FROM dim_skills WHERE skill_type = 'extracted'").fetchone()[0]
    snapshots = mp.warehouse_snapshots(db)
    data["snapshots"] = {d: mp.load_snapshot(db, d, cap) for d in snapshots}
    data["snapshot_sizes"] = {d: len(mp.load_snapshot(db, d)) for d in snapshots}
    data["cap"] = cap
    return data


def parse_kmeans_sweep(text: str):
    """Pull the K-Means sweep table (k, inertia, silhouette, davies_bouldin) out of 06_summary.txt."""
    match = re.search(r"^\s*k\s+inertia\s+silhouette\s+davies_bouldin\s*\n((?:\s*\d+\s+[-\d.\s]+\n)+)", text, re.M)
    if not match:
        logger.warning("No K-Means sweep table in 06_summary.txt; silhouette chart skipped")
        return None
    return pd.read_csv(io.StringIO(match.group(1)), sep=r"\s+", names=["k", "inertia", "silhouette", "davies_bouldin"])


def skill_shares(lists: pd.Series) -> pd.Series:
    """Share of postings listing each skill."""
    return lists.explode().dropna().value_counts() / max(len(lists), 1)


# ---------- tab 1: association rules ----------

def rules_by_lift(rules: pd.DataFrame) -> go.Figure:
    """Top rules by lift, one bar per rule."""
    top = rules.sort_values("lift", ascending=False).head(TOP_RULES).iloc[::-1]
    labels = [f"{a} → {c}" for a, c in zip(top["antecedents"], top["consequents"])]
    fig = go.Figure(go.Bar(
        x=top["lift"], y=labels, orientation="h", marker=dict(color=C["series"][0], cornerradius=4),
        customdata=top[["support", "confidence"]],
        hovertemplate="%{y}<br>lift %{x:.2f}<br>support %{customdata[0]:.1%}<br>"
                      "confidence %{customdata[1]:.1%}<extra></extra>"))
    return base_layout(fig, height=max(380, 22 * len(top)), xaxis_title="lift", yaxis=dict(tickfont=dict(size=11)))


def rules_by_high_share(high: pd.DataFrame, jobs: pd.DataFrame) -> go.Figure:
    """High-tier rules ranked by the share of jobs with the itemset that are High tier, vs the overall share."""
    top = high.sort_values("high_tier_rate", ascending=False).head(TOP_HIGH_RULES).iloc[::-1]
    labels = [short(f"{a} → {c}") for a, c in zip(top["antecedents"], top["consequents"])]
    fig = go.Figure(go.Bar(
        x=top["high_tier_rate"], y=labels, orientation="h", marker=dict(color=C["series"][0], cornerradius=4),
        customdata=top[["jobs_with_itemset", "high_tier_lift"]],
        hovertemplate="%{y}<br>High tier %{x:.0%}<br>jobs %{customdata[0]:,}<br>"
                      "vs baseline ×%{customdata[1]:.2f}<extra></extra>"))
    base_layout(fig, height=max(380, 24 * len(top)), xaxis=dict(title="share High tier", tickformat=".0%"))
    share = (jobs["salary_tier"] == "High").mean()
    target_line(fig, share, f"all jobs {share:.0%}")
    return fig


# ---------- tab 2: clusters ----------

def cluster_frame(labels: pd.DataFrame) -> pd.DataFrame:
    """Clustered jobs (K-Means label >= 0) with a display name."""
    df = labels[labels["kmeans_cluster"] >= 0].copy()
    df["cluster"] = df["kmeans_cluster_name"]
    return df


def cluster_order(df: pd.DataFrame) -> list:
    """Cluster names ordered by median salary, highest first."""
    return df.groupby("cluster")["normalized_salary"].median().sort_values(ascending=False).index.tolist()


def cluster_tier_mix(df: pd.DataFrame) -> go.Figure:
    """Salary tier mix per cluster (100% stacked bars, ordinal blues)."""
    order = cluster_order(df)
    mix = pd.crosstab(df["cluster"], df["salary_tier"], normalize="index").reindex(index=order, columns=TIERS)
    counts = df["cluster"].value_counts()
    fig = go.Figure([go.Bar(
        y=order, x=mix[tier], name=tier, orientation="h", marker=dict(color=C["tiers"][i], line=dict(color=C["surface"], width=2)),
        customdata=counts.reindex(order),
        hovertemplate="%{y}<br>" + tier + " %{x:.0%} of %{customdata:,} jobs<extra></extra>")
        for i, tier in enumerate(TIERS)])
    return base_layout(fig, height=300, barmode="stack", xaxis=dict(tickformat=".0%"),
                       yaxis=dict(autorange="reversed"), legend_traceorder="normal")


def cluster_salary(df: pd.DataFrame) -> go.Figure:
    """Salary distribution per cluster (box plots)."""
    order = cluster_order(df)
    fig = go.Figure([go.Box(
        x=df.loc[df["cluster"] == name, "normalized_salary"], name=name, orientation="h", boxpoints=False,
        marker=dict(color=C["series"][0]), line=dict(width=1.5), fillcolor=C["seq"][1],
        hovertemplate="%{x:$,.0f}<extra></extra>") for name in order])
    return base_layout(fig, height=300, showlegend=False, xaxis=dict(title="annual salary", tickformat="$,.0s"),
                       yaxis=dict(autorange="reversed"))


def cluster_skills(df: pd.DataFrame, job_skills: pd.DataFrame) -> go.Figure:
    """Share of each cluster's jobs listing its most distinctive skills (highest lift vs all clustered jobs)."""
    js = job_skills.merge(df[["job_id", "cluster"]], on="job_id")
    share = js.groupby(["cluster", "skill_name"]).size().unstack(fill_value=0).div(df["cluster"].value_counts(), axis=0)
    overall = js["skill_name"].value_counts() / len(df)
    lift = share.div(overall, axis=1)
    order = cluster_order(df)
    skills = []
    for name in order:
        candidates = lift.loc[name][share.loc[name] >= MIN_CLUSTER_SKILL_SHARE].sort_values(ascending=False)
        skills += [s for s in candidates.index if s not in skills][:TOP_CLUSTER_SKILLS]
    z = share.loc[order, skills]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=skills, y=order, colorscale=[[i / (len(C["seq"]) - 1), c] for i, c in enumerate(C["seq"])],
        zmin=0, xgap=2, ygap=2, colorbar=dict(tickformat=".0%", thickness=10, outlinewidth=0),
        hovertemplate="%{y}<br>%{x}: %{z:.0%} of jobs<extra></extra>"))
    return base_layout(fig, height=300, xaxis=dict(tickangle=-35, showgrid=False),
                       yaxis=dict(showgrid=False, autorange="reversed"))


def silhouette_sweep(sweep: pd.DataFrame) -> go.Figure:
    """K-Means silhouette by k, against the target."""
    fig = go.Figure(go.Scatter(
        x=sweep["k"], y=sweep["silhouette"], mode="lines+markers", line=dict(color=C["series"][0], width=2),
        marker=dict(size=8, line=dict(color=C["surface"], width=2)),
        hovertemplate="k = %{x}<br>silhouette %{y:.3f}<extra></extra>"))
    base_layout(fig, height=300, showlegend=False, xaxis=dict(title="k", dtick=1), yaxis=dict(title="silhouette"))
    target_line(fig, SILHOUETTE_TARGET, f"target {SILHOUETTE_TARGET:g}", axis="y")
    return fig


# ---------- tab 3: model results ----------

def model_scores(results: pd.DataFrame) -> go.Figure:
    """Macro F1 and ROC-AUC per model, with the project targets."""
    df = results.sort_values("f1_macro")
    fig = go.Figure([go.Bar(y=df["model"], x=df[col], name=name, orientation="h",
                            marker=dict(color=C["series"][i], cornerradius=4),
                            hovertemplate="%{y}<br>" + name + " %{x:.3f}<extra></extra>")
                     for i, (col, name) in enumerate([("f1_macro", "F1 macro"), ("roc_auc_ovr", "ROC-AUC")])])
    base_layout(fig, height=max(320, 64 * len(df)), barmode="group", xaxis=dict(range=[0, 1]), bargroupgap=0.1)
    target_line(fig, F1_TARGET, f"F1 target {F1_TARGET:g}", side="left")
    target_line(fig, AUC_TARGET, f"AUC target {AUC_TARGET:g}")
    fig.update_layout(legend=dict(y=-0.08, yanchor="top"))
    return fig


def model_tiers(results: pd.DataFrame) -> go.Figure:
    """Per-tier F1 heatmap (model x tier)."""
    df = results.sort_values("f1_macro", ascending=False)
    cols = [c for c in ["f1_low", "f1_mid", "f1_high"] if c in df]
    z = df[cols].values
    fig = go.Figure(go.Heatmap(
        z=z, x=[c.split("_")[1].title() for c in cols], y=df["model"], text=z, texttemplate="%{text:.2f}",
        colorscale=[[i / (len(C["seq"]) - 1), c] for i, c in enumerate(C["seq"])], zmin=0, zmax=1,
        xgap=2, ygap=2, showscale=False, hovertemplate="%{y}<br>%{x} F1 %{z:.3f}<extra></extra>"))
    return base_layout(fig, height=max(280, 48 * len(df)), xaxis=dict(side="top", showgrid=False),
                       yaxis=dict(autorange="reversed", showgrid=False))


def model_overfit(results: pd.DataFrame) -> go.Figure:
    """Train vs test macro F1 per model (dumbbell)."""
    df = results.sort_values("f1_macro")
    fig = go.Figure()
    for model, train, test in zip(df["model"], df["train_f1_macro"], df["f1_macro"]):
        fig.add_trace(go.Scatter(x=[test, train], y=[model, model], mode="lines", line=dict(color=C["axis"], width=2),
                                 hoverinfo="skip", showlegend=False))
    for i, (col, name) in enumerate([("f1_macro", "test"), ("train_f1_macro", "train")]):
        fig.add_trace(go.Scatter(x=df[col], y=df["model"], mode="markers", name=name,
                                 marker=dict(size=10, color=C["series"][i], line=dict(color=C["surface"], width=2)),
                                 hovertemplate="%{y}<br>" + name + " F1 %{x:.3f}<extra></extra>"))
    return base_layout(fig, height=max(280, 48 * len(df)), xaxis=dict(title="F1 macro", range=[0, 1.02]))


# ---------- tab 4: live trends ----------

def live_vs_training(comparison: pd.DataFrame) -> go.Figure:
    """Top skills in the latest Muse snapshot: Muse share (all and company-capped) vs training tech jobs."""
    top = comparison.sort_values("muse_share", ascending=False).head(TOP_LIVE_SKILLS).iloc[::-1]
    series = [("muse_share", "Muse"), ("muse_share_capped", "Muse, capped"), ("train_tech_share", "Training tech")]
    fig = go.Figure()
    for i, (col, name) in enumerate((c, n) for c, n in series if c in top):
        fig.add_trace(go.Scatter(x=top[col], y=top["skill"], mode="markers", name=name,
                                 marker=dict(size=10, color=C["series"][i], line=dict(color=C["surface"], width=2)),
                                 hovertemplate="%{y}<br>" + name + " %{x:.0%}<extra></extra>"))
    return base_layout(fig, height=max(380, 24 * len(top)), xaxis=dict(title="share of postings", tickformat=".0%"))


def live_shifts(comparison: pd.DataFrame) -> go.Figure:
    """Largest rises and drops in skill share, Muse vs training tech jobs (percentage points)."""
    rises, drops = mp.demand_shifts(comparison, n=TOP_SHIFTS)
    df = pd.concat([drops, rises.iloc[::-1]]).drop_duplicates("skill")
    colors = [C["series"][0] if v > 0 else C["series"][7] for v in df["change_vs_tech_pp"]]
    fig = go.Figure(go.Bar(
        x=df["change_vs_tech_pp"], y=df["skill"], orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=df[["muse_share", "train_tech_share"]],
        hovertemplate="%{y}<br>%{x:+.1f} pp<br>Muse %{customdata[0]:.0%} · training tech %{customdata[1]:.0%}"
                      "<extra></extra>"))
    return base_layout(fig, height=max(380, 24 * len(df)), showlegend=False,
                       xaxis=dict(title="change (pp)", zeroline=True, zerolinewidth=1))


def live_rule_lift(validation: pd.DataFrame) -> go.Figure:
    """Training lift vs Muse lift for the top training rules, coloured by whether the rule is confirmed."""
    df = validation.dropna(subset=["muse_lift"])
    hi = max(df["training_lift"].max(), df["muse_lift"].max()) * 1.05
    fig = go.Figure(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=C["axis"], width=1),
                               hoverinfo="skip", showlegend=False))
    for i, (flag, name) in enumerate([(True, "confirmed"), (False, "not confirmed")]):
        part = df[df["validated"] == flag]
        fig.add_trace(go.Scatter(
            x=part["training_lift"], y=part["muse_lift"], mode="markers", name=name, text=part["rule"],
            marker=dict(size=10, color=C["series"][i], line=dict(color=C["surface"], width=2)),
            hovertemplate="%{text}<br>training %{x:.2f} · Muse %{y:.2f}<extra></extra>"))
    return base_layout(fig, height=380, xaxis=dict(title="training lift", range=[0, hi]),
                       yaxis=dict(title="Muse lift", range=[0, hi]))


def live_by_fetch_date(snapshots: dict) -> go.Figure:
    """Company-capped skill share per fetch_date for the top skills of the latest snapshot."""
    shares = pd.DataFrame({d: skill_shares(df["matched_skills"]) for d, df in snapshots.items()}).fillna(0)
    latest = max(snapshots)
    top = shares[latest].sort_values(ascending=False).head(TOP_TREND_SKILLS).index
    fig = go.Figure([go.Scatter(
        x=list(shares.columns), y=shares.loc[skill], name=skill, mode="lines+markers",
        line=dict(color=C["series"][i], width=2), marker=dict(size=8, line=dict(color=C["surface"], width=2)),
        hovertemplate=skill + "<br>%{x}: %{y:.0%}<extra></extra>") for i, skill in enumerate(top)])
    return base_layout(fig, height=380, hovermode="x unified", xaxis=dict(title="fetch date", type="category"),
                       yaxis=dict(title="share of postings", tickformat=".0%"))


# ---------- page ----------

def tile(label: str, value: str) -> str:
    """One stat tile."""
    return f'<div class="tile"><div class="tile-v">{html.escape(value)}</div><div class="tile-l">{html.escape(label)}</div></div>'


def card(title: str, fig: go.Figure, div_id: str, wide: bool = False) -> str:
    """A titled chart card."""
    body = pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id, config=PLOT_CONFIG)
    return f'<section class="card{" wide" if wide else ""}"><h3>{html.escape(title)}</h3>{body}</section>'


def note(title: str, text: str) -> str:
    """A titled card holding one short line instead of a chart."""
    return f'<section class="card"><h3>{html.escape(title)}</h3><p class="note">{html.escape(text)}</p></section>'


def build_tabs(data: dict) -> list:
    """Return [(tab id, tab label, tiles html, cards html)]."""
    tabs = []

    cards, tiles = [], []
    if data["04_association_rules"] is not None:
        rules = data["04_association_rules"]
        tiles += [tile("top rules", f"{len(rules)}"), tile("max lift", f"{rules['lift'].max():.2f}")]
        cards.append(card("Top rules by lift", rules_by_lift(rules), "rules-lift"))
    if data["04_high_salary_rules"] is not None:
        high = data["04_high_salary_rules"]
        tiles.append(tile("High-tier rules", f"{len(high)}"))
        cards.append(card("High-tier rules by share High", rules_by_high_share(high, data["jobs"]), "rules-high"))
    tabs.append(("rules", "Association rules", tiles, cards))

    cards, tiles = [], []
    if data["06_cluster_labels"] is not None:
        df = cluster_frame(data["06_cluster_labels"])
        tiles += [tile("clusters", f"{df['cluster'].nunique()}"), tile("jobs clustered", f"{len(df):,}")]
        cards += [card("Salary tier mix", cluster_tier_mix(df), "cl-tiers"),
                  card("Salary by cluster", cluster_salary(df), "cl-salary"),
                  card("Distinctive skills", cluster_skills(df, data["job_skills"]), "cl-skills")]
    if data["kmeans_sweep"] is not None:
        sweep = data["kmeans_sweep"]
        tiles.append(tile("best silhouette", f"{sweep['silhouette'].max():.3f}"))
        cards.append(card("K-Means silhouette by k", silhouette_sweep(sweep), "cl-sil"))
    tabs.append(("clusters", "Clusters", tiles, cards))

    cards, tiles = [], []
    if data["05_classification_results"] is not None:
        res = data["05_classification_results"]
        best = res.loc[res["f1_macro"].idxmax()]
        tiles += [tile("best F1 macro", f"{best['f1_macro']:.3f}"), tile("best ROC-AUC", f"{res['roc_auc_ovr'].max():.3f}"),
                  tile("best model", best["model"])]
        cards += [card("F1 and ROC-AUC", model_scores(res), "m-scores", wide=True),
                  card("F1 by tier", model_tiers(res), "m-tiers")]
        if "train_f1_macro" in res:
            cards.append(card("Train vs test F1", model_overfit(res), "m-overfit"))
    tabs.append(("models", "Model results", tiles, cards))

    cards, tiles = [], []
    snapshots = data["snapshots"]
    comp, val = data["08_skill_comparison"], data["08_rules_validation"]
    analysed = next((str(df["fetch_date"].iloc[0]) for df in (comp, val) if df is not None and "fetch_date" in df), None)
    suffix = f" · {analysed}" if analysed else ""
    if analysed:
        tiles.append(tile("fetch analysed", analysed))
        if analysed in data["snapshot_sizes"]:
            tiles.append(tile("postings", f"{data['snapshot_sizes'][analysed]:,}"))
    if snapshots:
        tiles += [tile("snapshots", f"{len(snapshots)}"),
                  tile("cap per company", f"{data['cap']}" if data["cap"] else "none")]
    if comp is not None:
        cards += [card("Live vs training skill share" + suffix, live_vs_training(comp), "lv-share"),
                  card("Largest shifts vs training tech" + suffix, live_shifts(comp), "lv-shift")]
    if val is not None:
        tiles.append(tile("rules confirmed", f"{int(val['validated'].sum())} of {len(val)}"))
        cards.append(card("Rule lift: training vs Muse" + suffix, live_rule_lift(val), "lv-rules"))
    if len(snapshots) > 1:
        cards.append(card("Skill share by fetch date", live_by_fetch_date(snapshots), "lv-trend"))
    elif snapshots:
        cards.append(note("Skill share by fetch date", f"One snapshot ({max(snapshots)}). Trends appear after "
                          "python -m src.muse_pipeline --new-snapshot."))
    tabs.append(("live", "Live trends", tiles, cards))
    return tabs


PAGE_CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--ring:rgba(11,11,11,.10);--accent:#2a78d6;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{max-width:1280px;margin:0 auto;padding:24px 16px 8px}
h1{font-size:20px;margin:0 0 2px;font-weight:650}
.sub{color:var(--muted);font-size:13px}
nav{max-width:1280px;margin:0 auto;padding:0 16px;display:flex;gap:4px;border-bottom:1px solid var(--grid);overflow-x:auto}
nav button{font:inherit;color:var(--ink2);background:none;border:0;border-bottom:2px solid transparent;
padding:10px 12px;cursor:pointer;white-space:nowrap}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
nav button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
main{max-width:1280px;margin:0 auto;padding:16px}
.panel[hidden]{display:none}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 14px;min-width:120px}
.tile-v{font-size:20px;font-weight:600;color:var(--ink)}
.tile-l{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 12px 4px;min-width:0}
.card.wide{grid-column:1/-1}
.card h3{margin:0 0 4px;font-size:14px;font-weight:600;color:var(--ink)}
.note{color:var(--muted);margin:8px 0 12px}
"""

PAGE_JS = """
const MAP = __MAP__;
const original = new Map();
function themed(obj, dark) {
  const s = JSON.stringify(obj);
  return JSON.parse(dark ? s.replace(/#[0-9a-f]{6}/gi, h => MAP[h.toLowerCase()] || h) : s);
}
function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
}
function applyTheme() {
  const dark = isDark();
  if (!dark && !original.size) return;
  document.querySelectorAll('.js-plotly-plot').forEach(gd => {
    if (!original.has(gd)) original.set(gd, {data: JSON.parse(JSON.stringify(gd.data)), layout: JSON.parse(JSON.stringify(gd.layout))});
    const o = original.get(gd);
    Plotly.react(gd, themed(o.data, dark), themed(o.layout, dark));
  });
}
function show(id) {
  document.querySelectorAll('nav button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === id));
  document.querySelectorAll('.panel').forEach(p => { p.hidden = p.id !== id; });
  document.querySelectorAll('#' + id + ' .js-plotly-plot').forEach(gd => Plotly.Plots.resize(gd));
  try { localStorage.setItem('skillmap-tab', id); } catch (e) {}
  history.replaceState(null, '', '#' + id);
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
let saved = null;
try { saved = localStorage.getItem('skillmap-tab'); } catch (e) {}
const hash = location.hash.slice(1);
show([hash, saved].find(id => id && document.getElementById(id)) || document.querySelector('nav button').dataset.tab);
applyTheme();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
"""


def render_page(data: dict) -> str:
    """Assemble the full HTML page."""
    tabs = build_tabs(data)
    n_jobs = len(data["jobs"])
    sub = f"{n_jobs:,} training jobs · {data['n_skills_vocab']} skills · {len(data['snapshots'])} live snapshot(s) · " \
          f"built {datetime.now():%Y-%m-%d %H:%M}"
    nav = "".join(f'<button role="tab" data-tab="{tid}" aria-controls="{tid}">{html.escape(label)}</button>'
                  for tid, label, _, _ in tabs)
    panels = "".join(f'<div class="panel" id="{tid}" role="tabpanel"><div class="tiles">{"".join(tiles)}</div>'
                     f'<div class="grid">{"".join(cards)}</div></div>' for tid, _, tiles, cards in tabs)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>SkillMap Dashboard</title>'
            f'<style>{PAGE_CSS}</style><script>{get_plotlyjs()}</script></head><body>'
            f'<header><h1>SkillMap</h1><div class="sub">{html.escape(sub)}</div></header>'
            f'<nav role="tablist">{nav}</nav><main>{panels}</main>'
            f'<script>{PAGE_JS.replace("__MAP__", json.dumps(theme_map()))}</script></body></html>')


def main(argv: list = None) -> Path:
    """Build outputs/dashboard.html (or --out) from the warehouse and stage outputs."""
    parser = argparse.ArgumentParser(description="Build the SkillMap Plotly dashboard")
    parser.add_argument("--root", type=Path, default=ROOT, help="project root (default: this repository)")
    parser.add_argument("--out", type=Path, help="output HTML (default: <root>/outputs/dashboard.html)")
    parser.add_argument("--max-per-company", type=int, default=mp.MAX_PER_COMPANY,
                        help=f"company cap for the by-fetch-date trend (default {mp.MAX_PER_COMPANY}; 0 = no cap)")
    args = parser.parse_args(argv)
    out = args.out or args.root / "outputs" / "dashboard.html"
    data = load_inputs(args.root, args.max_per_company)
    out.write_text(render_page(data), encoding="utf-8")
    logger.info("Saved %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
