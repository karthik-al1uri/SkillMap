"""SkillMap's self-contained six-view analytics product.

Render the existing provenance-aware facts bundle into a fresh external HTML file:
    python scripts/build_dashboard.py --facts /path/facts.json --out /tmp/new-skillmap.html
No database, model evaluation, training, external assets or network are required.
"""
import argparse
import html
import io
import json
import logging
import hashlib
import re
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
from scripts.build_rule_views import rank_general, rank_high, strict_technical  # noqa: E402

logger = logging.getLogger("build_dashboard")

TOP_RULES = 10
TOP_HIGH_RULES = 5
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

def load_inputs(facts_path: Path) -> dict:
    """Render only a generated facts bundle; never read canonical data or a database."""
    path = Path(facts_path).resolve(strict=True)
    facts = json.loads(path.read_text())
    if facts.get("schema") != "skillmap.dashboard.v1":
        raise ValueError("Expected a skillmap.dashboard.v1 facts bundle")
    sources = facts.get("sources", {})
    for section in ("tables", "metrics", "evaluations"):
        for key, wrapper in facts.get(section, {}).items():
            if any(source not in sources for source in wrapper.get("source_ids", [])):
                raise ValueError(f"Unknown provenance source for {section}.{key}")
    data = {"_facts": facts, "_facts_path": str(path),
            "_facts_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    scalars = {"snapshots", "snapshot_sizes", "cap", "n_skills_vocab"}
    for key, wrapper in facts.get("tables", {}).items():
        value = wrapper.get("data") if wrapper.get("available") else None
        data[key] = value if key in scalars else (pd.DataFrame(value) if value is not None else None)
    data["snapshots"] = {date: pd.DataFrame(rows) for date, rows in (data.get("snapshots") or {}).items()}
    data["snapshot_sizes"] = data.get("snapshot_sizes") or {}
    data["metrics"] = {key: value.get("data") if value.get("available") else None
                       for key, value in facts.get("metrics", {}).items()}
    data["evaluations"] = {key: value.get("data") if value.get("available") else None
                           for key, value in facts.get("evaluations", {}).items()}
    jobs = data.get("jobs")
    if jobs is not None and ("job_id" not in jobs or jobs.job_id.duplicated().any()):
        raise ValueError("Dashboard salary data must contain one row per job_id")
    grouped_report(data)  # Fail closed on inconsistent selection metadata.
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
    top = rank_general(rules, TOP_RULES).iloc[::-1]
    labels = [f"{a} → {c}" for a, c in zip(top["antecedents"], top["consequents"])]
    fig = go.Figure(go.Bar(
        x=top["lift"], y=labels, orientation="h", marker=dict(color=C["series"][0], cornerradius=4),
        customdata=top[["support", "confidence"]],
        hovertemplate="%{y}<br>lift %{x:.2f}<br>support %{customdata[0]:.1%}<br>"
                      "confidence %{customdata[1]:.1%}<extra></extra>"))
    return base_layout(fig, height=max(380, 22 * len(top)), xaxis_title="lift", yaxis=dict(tickfont=dict(size=11)))


def rules_by_high_share(high: pd.DataFrame, jobs: pd.DataFrame) -> go.Figure:
    """High-tier rules ranked by the share of jobs with the itemset that are High tier, vs the overall share."""
    top = rank_high(high, TOP_HIGH_RULES).iloc[::-1]
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


def silhouette_sweep(sweep: pd.DataFrame, target=None) -> go.Figure:
    """K-Means silhouette by k, against the target."""
    fig = go.Figure(go.Scatter(
        x=sweep["k"], y=sweep["silhouette"], mode="lines+markers", line=dict(color=C["series"][0], width=2),
        marker=dict(size=8, line=dict(color=C["surface"], width=2)),
        hovertemplate="k = %{x}<br>silhouette %{y:.3f}<extra></extra>"))
    base_layout(fig, height=300, showlegend=False, xaxis=dict(title="k", dtick=1), yaxis=dict(title="silhouette"))
    if target is not None:
        target_line(fig, target, f"target {target:g}", axis="y")
    return fig


# ---------- tab 3: model results ----------

def model_scores(results: pd.DataFrame, targets=None) -> go.Figure:
    """Macro F1 and ROC-AUC per model, with the project targets."""
    df = results.sort_values("f1_macro")
    fig = go.Figure([go.Bar(y=df["model"], x=df[col], name=name, orientation="h",
                            marker=dict(color=C["series"][i], cornerradius=4),
                            hovertemplate="%{y}<br>" + name + " %{x:.3f}<extra></extra>")
                     for i, (col, name) in enumerate([("f1_macro", "F1 macro"), ("roc_auc_ovr", "ROC-AUC")])])
    base_layout(fig, height=max(320, 64 * len(df)), barmode="group", xaxis=dict(range=[0, 1]), bargroupgap=0.1)
    targets = targets or {}
    if targets.get("f1_macro") is not None:
        target_line(fig, targets["f1_macro"], f"F1 target {targets['f1_macro']:g}", side="left")
    if targets.get("roc_auc_ovr") is not None:
        target_line(fig, targets["roc_auc_ovr"], f"AUC target {targets['roc_auc_ovr']:g}")
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
                       xaxis=dict(title="cross-source difference (pp)", zeroline=True, zerolinewidth=1))


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


def selected_model_row(results):
    """Use an explicit unique training-CV selection; never infer it from test scores."""
    if "selected_by_training_cv" not in results:
        return None
    flags = results["selected_by_training_cv"].astype(str).str.lower().eq("true")
    selected = results.loc[flags]
    return selected.iloc[0] if len(selected) == 1 else None


def grouped_report(data):
    """Use the unique winner selected by mean training CV; reject conflicting metadata."""
    grouped = (data.get("evaluations") or {}).get("grouped")
    if not grouped:
        return None
    report = grouped.get("report", {})
    if report.get("status") != "complete":
        return None
    rows = report.get("cv_rows", [])
    if not rows or report.get("winner") != max(rows, key=lambda row: row["mean_f1_macro"])["candidate"]:
        raise ValueError("Grouped model winner disagrees with training CV")
    selection = grouped.get("cv_selection") or {}
    if selection.get("winner", report["winner"]) != report["winner"]:
        raise ValueError("Conflicting grouped selection metadata")
    return report


def provenance_note(data, *keys, evaluations=()):
    facts = data.get("_facts", {})
    entries = [facts.get("tables", {}).get(key) for key in keys]
    entries += [facts.get("evaluations", {}).get(key) for key in evaluations]
    entries = [entry for entry in entries if entry]
    scopes = sorted({entry["status"].replace("_", " ") for entry in entries})
    ids = sorted({source for entry in entries for source in entry.get("source_ids", [])})
    return note("Evidence", (" / ".join(scopes) or "Unavailable evidence") +
                (". Sources: " + ", ".join(ids) if ids else ". No source supplied."))


def horizontal_values(labels, values, title, percent=False):
    fig = go.Figure(go.Bar(y=list(labels), x=list(values), orientation="h",
                          marker=dict(color=C["series"][0], cornerradius=4)))
    return base_layout(fig, height=max(300, 25*len(labels)), showlegend=False,
                       xaxis=dict(title=title, tickformat=".0%" if percent else None),
                       yaxis=dict(autorange="reversed"))


def confusion_figure(matrix, labels=TIERS):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, text=matrix, texttemplate="%{text}",
                              colorscale="Blues", showscale=False))
    return base_layout(fig, height=320, xaxis=dict(title="predicted", showgrid=False),
                       yaxis=dict(title="observed", autorange="reversed", showgrid=False))


def salary_summary(jobs):
    """Every salary observation is a job, regardless of its number of skills."""
    if jobs.job_id.duplicated().any():
        raise ValueError("Salary views require unique job_id grain")
    salaries = pd.to_numeric(jobs.normalized_salary, errors="raise")
    if salaries.isna().any():
        raise ValueError("Salary observations must be complete")
    return {"count": len(jobs), "median": float(salaries.median())}


def compact_provenance(data):
    facts = data["_facts"]
    return {"schema": facts["schema"], "generated_utc": facts.get("generated_utc"),
            "sources": facts.get("sources", {}), "metrics": facts.get("metrics", {}),
            "evaluations": facts.get("evaluations", {}),
            "provenance": {**facts.get("provenance", {}), "salary_grain": "unique_job_id"},
            "table_provenance": {key: {k: v for k, v in wrapper.items() if k != "data"}
                                 for key, wrapper in facts.get("tables", {}).items()},
            "bundle_sha256": data["_facts_sha256"]}


def cached_summary(result):
    intervals = result.get("bootstrap", {})
    def ci(key):
        row = intervals.get(key) or {}
        return f"{row['low']:.4f}–{row['high']:.4f}" if row.get("low") is not None else "unavailable"
    comparators = result.get("comparators", {})
    training, oracle = comparators.get("historical_training_majority"), comparators.get("observed_majority_oracle")
    text = f"95% employer-cluster intervals: accuracy {ci('accuracy')}; macro F1 {ci('macro_f1')}. "
    if training:
        text += f"Training-majority ({training['label']}): accuracy {training['accuracy']:.4f}, macro F1 {training['macro_f1']:.4f}. "
    if oracle:
        text += f"Observed-majority oracle ({oracle['label']}, uses evaluation labels): accuracy {oracle['accuracy']:.4f}, macro F1 {oracle['macro_f1']:.4f}. "
    labels = result.get("salary_label_status", {})
    if labels:
        text += f"Targets: {labels['estimated_rows']:,} estimated salaries; {labels['advertised_rows']:,} advertised. "
    text += "Intervals condition on cached predictions and employers. ROC-AUC is unavailable without saved probabilities."
    return text


def grouped_intervals(report):
    labels = {"f1_macro": "macro F1", "accuracy": "accuracy", "roc_auc_ovr_macro": "ROC-AUC"}
    rows = report.get("bootstrap", {}).get("intervals", {})
    return "; ".join(f"{label} {rows[key]['lower_2.5']:.4f}–{rows[key]['upper_97.5']:.4f}"
                     for key, label in labels.items() if rows.get(key)) + ". These omit fitting and selection uncertainty."


def build_tabs(data: dict) -> list:
    """Six analyst views over one generated provenance-aware facts bundle."""
    tabs = []
    metrics, evaluations = data.get("metrics", {}), data.get("evaluations", {})
    jobs, job_skills = data.get("jobs"), data.get("job_skills")
    grouped = grouped_report(data)
    live = (evaluations.get("live") or {}).get("report")
    dates = metrics.get("live_snapshot_dates") or sorted(data.get("snapshot_sizes", {}))
    targets = dict(metrics.get("model_targets") or {})
    if "roc_auc" in targets:
        targets["roc_auc_ovr"] = targets["roc_auc"]

    tiles, cards = [], []
    for key, label, fmt in [("n_jobs", "saved training jobs", lambda v: f"{v:,}"),
                             ("n_skills_vocab", "frozen skill vocabulary", str),
                             ("mean_skills", "mean skills per job", lambda v: f"{v:.2f}"),
                             ("median_salary", "median annual salary", lambda v: f"${v:,.0f}")]:
        if metrics.get(key) is not None:
            tiles.append(tile(label, fmt(metrics[key])))
    if grouped:
        tiles.append(tile("grouped holdout macro F1", f"{grouped['outer_metrics']['f1_macro']:.4f}"))
    cards += [note("Read the evidence by scope", "Artifact-derived counts describe saved jobs. Historical exploratory results "
                   "used an inspected test split. The new grouped evaluation reuses development data. Live results are cached snapshots, not current-market measurements."),
              note("Analyst guide", "Skills: prevalence and association. Salary: job-level distributions and descriptive associations. "
                   "Clusters: saved groupings and bounded robustness. Live: dated cross-source checks. Model: training-CV selection and grouped holdout uncertainty."),
              note("Cached snapshot dates", ", ".join(dates) if dates else "No cached snapshot dates supplied.")]
    facts = data.get("_facts", {})
    if facts.get("sources"):
        rows = "".join("<tr><td>"+html.escape(source_id)+"</td><td>"+html.escape(source.get("status", ""))+
                       "</td><td>"+html.escape(source.get("path", ""))+"</td><td><code>"+
                       html.escape(source.get("sha256", "unavailable"))+"</code></td></tr>"
                       for source_id, source in facts["sources"].items())
        cards.append('<section class="card wide"><h3>Source provenance</h3><div class="table-scroll"><table>'
                     '<thead><tr><th>Source</th><th>Scope</th><th>Artifact</th><th>SHA-256</th></tr></thead><tbody>'+rows+'</tbody></table></div></section>')
    tabs.append(("overview", "Overview", tiles, cards))

    tiles, cards = [], []
    counts = metrics.get("skill_counts")
    if counts:
        if isinstance(counts, list):
            counts = {row.get("skill", row.get("skill_name")): row.get("count", row.get("jobs")) for row in counts}
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        cards.append(card("Most prevalent extracted skills · all jobs", horizontal_values(
            [key for key, _ in top], [value / metrics["n_jobs"] for _, value in top], "share of all jobs", True), "skills-prevalence"))
    rules = data.get("04_association_rules")
    if rules is not None and len(rules):
        cards.append(card("Top 10 rules by lift", rules_by_lift(rules), "rules-lift"))
        tiles.append(tile("general candidates supplied", str(len(rules))))
    technical = data.get("04_technical_rules_low_support")
    if technical is not None:
        technical = strict_technical(technical) if len(technical) else technical
        tiles.append(tile("technical-only rules", str(len(technical))))
        cards.append(card("Technical-only rules", rules_by_lift(technical), "rules-tech") if len(technical)
                     else note("Technical-only rules", "No supplied rule contains exclusively forced-technical skills on both sides."))
    cards += [note("Rule definitions", "General support ≥5% includes empty baskets; technical support ≥2% uses all jobs. "
                   "Confidence >0.5 and lift >1.5 are applied before rounding. Technical-only requires every skill on both sides in the frozen technical set. Views rank supplied candidates."),
              provenance_note(data, "jobs", "job_skills", "04_association_rules", "04_technical_rules_low_support")]
    tabs.append(("skills", "Skills", tiles, cards))

    tiles, cards = [], []
    if jobs is not None and len(jobs):
        summary = salary_summary(jobs)
        tiles += [tile("jobs in salary views", f"{summary['count']:,}"),
                  tile("median annual salary", f"${summary['median']:,.0f}")]
        cards.append(card("Annual salary distribution · one observation per job", base_layout(go.Figure(
            go.Histogram(x=jobs.normalized_salary, nbinsx=40, marker=dict(color=C["series"][0]))),
            xaxis=dict(title="annual USD"), yaxis=dict(title="jobs")), "salary-distribution"))
        tiers = jobs.salary_tier.value_counts().reindex(TIERS, fill_value=0)
        cards.append(card("Salary-tier counts", horizontal_values(tiers.index, tiers, "jobs"), "salary-tiers"))
        if "experience_level" in jobs:
            by_experience = jobs.groupby("experience_level").normalized_salary.agg(["median", "size"]).sort_values("median")
            fig = go.Figure(go.Bar(y=by_experience.index, x=by_experience["median"], orientation="h", customdata=by_experience["size"],
                hovertemplate="%{y}<br>median $%{x:,.0f}<br>%{customdata:,} jobs<extra></extra>"))
            cards.append(card("Median salary by recorded experience", base_layout(fig, xaxis=dict(title="annual USD")), "salary-experience"))
    high = data.get("04_high_salary_rules")
    if high is not None and len(high) and jobs is not None:
        cards.append(card("Top 5 High-tier rules by share High", rules_by_high_share(high, jobs), "rules-high"))
    bounds = metrics.get("salary_tier_boundaries")
    if bounds:
        cards.append(note("Fixed salary task", f"Low ≤ ${bounds['low_max']:,.0f}; Mid above that and ≤ ${bounds['mid_max']:,.0f}; High above ${bounds['mid_max']:,.0f}. Boundaries are frozen historical task definitions."))
    cards += [note("Grain and interpretation", "Salary observations are unique jobs, never expanded through job-skill fact rows. "
                   "Salary associations are descriptive, not causal. Historical tier boundaries were derived from global dataset quantiles before the model split."),
              provenance_note(data, "jobs", "04_high_salary_rules", evaluations=("salary_quality",))]
    tabs.append(("salary", "Salary", tiles, cards))

    tiles, cards = [], []
    labels = data.get("06_cluster_labels")
    if labels is not None and len(labels):
        df = cluster_frame(labels)
        if df.job_id.duplicated().any():
            raise ValueError("Cluster salary views require unique jobs")
        tiles += [tile("historical clusters", str(df.cluster.nunique())), tile("jobs clustered", f"{len(df):,}")]
        cards += [card("Historical salary tier mix", cluster_tier_mix(df), "cl-tiers"),
                  card("Historical salary by cluster", cluster_salary(df), "cl-salary")]
        if job_skills is not None:
            cards.append(card("Historical distinctive skills", cluster_skills(df, job_skills), "cl-skills"))
    sweep = data.get("kmeans_sweep")
    if sweep is not None and len(sweep):
        cards.append(card("Historical K-Means silhouette by k", silhouette_sweep(sweep, targets.get("silhouette")), "cl-sil"))
    robustness = (evaluations.get("clustering") or {}).get("report")
    if robustness and robustness.get("summary"):
        fig = go.Figure()
        for i, name in enumerate(dict.fromkeys(row["representation"] for row in robustness["summary"])):
            rows = [row for row in robustness["summary"] if row["representation"] == name]
            fig.add_trace(go.Scatter(x=[row["k"] for row in rows], y=[row["mean_cosine_silhouette_original"] for row in rows],
                                    name=name, mode="lines+markers", line=dict(color=C["series"][i])))
        cards.append(card("Bounded robustness · common original-space cosine", base_layout(fig, xaxis=dict(title="k"), yaxis=dict(title="mean cosine silhouette")), "cl-robustness"))
    cards += [note("Clustering scope", "Saved clusters and bounded robustness are exploratory. Weak separation is not proof that meaningful clusters cannot exist. "
                   "Seed stability is not resampling or external validation; SVD/L2 features were not adopted for classification."),
              provenance_note(data, "06_cluster_labels", "kmeans_sweep", evaluations=("clustering",))]
    tabs.append(("clusters", "Clusters", tiles, cards))

    tiles, cards = [], [note("Cached evidence", "Snapshot dates: " + (", ".join(dates) or "unavailable") +
        ". One snapshot and different source/employer populations do not establish temporal demand trends. Estimated salary labels are imperfect targets.")]
    if live:
        a, uncapped, capped = live["adzuna"], live["muse_uncapped"], live["muse_company_capped"]
        tiles += [tile("cached Adzuna accuracy", f"{a['accuracy']:.4f}"), tile("cached Adzuna macro F1", f"{a['macro_f1']:.4f}"),
                  tile("Muse rules confirmed", f"{uncapped['primary_confirmed']} / {uncapped['fixed_rules']}"),
                  tile("Muse rules · company capped", f"{capped['primary_confirmed']} / {capped['fixed_rules']}")]
        fig = go.Figure([go.Bar(x=TIERS, y=[a[key][tier] for tier in TIERS], name=label)
                         for key, label in [("actual_counts", "observed salary tier"), ("predicted_counts", "cached predicted tier")]])
        cards += [card("Adzuna observed vs predicted distribution", base_layout(fig, barmode="group", yaxis=dict(title="postings")), "live-distribution"),
                  card("Adzuna confusion matrix", confusion_figure(a["confusion_rows_actual_columns_predicted"]), "live-confusion"),
                  note("Adzuna uncertainty and baselines", cached_summary(a)),
                  note("Muse populations and uncertainty", f"Uncapped eligible/all: {uncapped['eligible_postings']}/{uncapped['all_postings']}; "
                       f"capped eligible/all: {capped['eligible_postings']}/{capped['all_postings']}. "
                       "Rules are fixed historical candidates. Employer resampling, not independent model validation. "+
                       "Confirmed uncapped: "+", ".join(uncapped.get("confirmed_primary_rules", []))+"; capped: "+", ".join(capped.get("confirmed_primary_rules", [])))]
        rule_rows = (evaluations.get("live") or {}).get("uncapped_fixed_rules")
        if isinstance(rule_rows, list) and rule_rows:
            val = pd.DataFrame(rule_rows).rename(columns={"training_saved_lift_historical": "training_lift", "lift": "muse_lift", "passes_primary_005": "validated"})
            cards.append(card("Fixed historical rules · cached Muse lift", live_rule_lift(val), "lv-rules"))
    else:
        cards.append(note("Live evaluation unavailable", "No generated cached-validation report supplied; historical metrics are not substituted."))
    comparison = data.get("08_skill_comparison")
    if comparison is not None and len(comparison):
        cards += [card("Cached Muse vs training skill shares", live_vs_training(comparison), "lv-share"),
                  card("Cross-source share differences · historical snapshot", live_shifts(comparison), "lv-shift")]
    if len(data.get("snapshots", {})) > 1:
        cards.append(card("Cached skill share by fetch date · company capped", live_by_fetch_date(data["snapshots"]), "lv-by-date"))
    else:
        cards.append(note("Time coverage", "A single cached snapshot cannot establish a temporal market trend."))
    cards.append(provenance_note(data, "08_skill_comparison", evaluations=("live",)))
    tabs.append(("live", "Live", tiles, cards))

    tiles, cards = [], []
    if grouped:
        m = grouped["outer_metrics"]
        tiles += [tile("CV-selected model", grouped["winner"]), tile("grouped holdout macro F1", f"{m['f1_macro']:.4f}"),
                  tile("grouped holdout ROC-AUC", f"{m['roc_auc_ovr_macro']:.4f}")]
        cv = grouped["cv_rows"]
        cards += [card("Model selection · training grouped CV only", horizontal_values([row["candidate"] for row in cv],
                        [row["mean_f1_macro"] for row in cv], "mean inner-CV macro F1"), "model-cv"),
                  card("Selected model · per-class F1", horizontal_values(TIERS, [m["f1_per_class"][tier] for tier in TIERS], "F1"), "model-classes"),
                  card("Selected model · grouped holdout confusion", confusion_figure(m["confusion_matrix"]), "model-confusion"),
                  note("New grouped evaluation · reused development data", "The selected model comes only from training grouped CV. "
                       "Normalized-description groups are isolated; employers may overlap. This is not an untouched external test. "
                       "Conditional 95% group-bootstrap intervals: " + grouped_intervals(grouped))]
    else:
        cards.append(note("Grouped evaluation unavailable", "No complete grouped evaluation supplied; no model is chosen from historical test F1."))
    historical = data.get("05_classification_results")
    if historical is not None and len(historical):
        cards += [note("Historical exploratory results", "The old test split influenced development and model selection. "
                       "Historical scores and the new grouped comparison use different splits and search scope; score differences do not isolate grouping effects."),
                  card("Historical exploratory F1 and ROC-AUC", model_scores(historical, targets), "m-scores", wide=True),
                  card("Historical exploratory F1 by tier", model_tiers(historical), "m-tiers")]
        if "train_f1_macro" in historical:
            cards.append(card("Historical exploratory train vs test F1", model_overfit(historical), "m-overfit"))
    cards.append(provenance_note(data, "05_classification_results", evaluations=("grouped",)))
    tabs.append(("model", "Model", tiles, cards))
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
.table-scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--grid);vertical-align:top}td code{font-size:10px}p.note{overflow-wrap:anywhere}

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
  document.querySelectorAll('nav button').forEach(b => (b.setAttribute('aria-selected', b.dataset.tab === id), b.setAttribute('tabindex', b.dataset.tab === id ? '0' : '-1')));
  document.querySelectorAll('.panel').forEach(p => { p.hidden = p.id !== id; });
  document.querySelectorAll('#' + id + ' .js-plotly-plot').forEach(gd => Plotly.Plots.resize(gd));
  try { localStorage.setItem('skillmap-tab', id); } catch (e) {}
  history.replaceState(null, '', '#' + id);
}
document.querySelectorAll('nav button').forEach(b => {
  b.addEventListener('click', () => show(b.dataset.tab));
  b.addEventListener('keydown', e => {
    const buttons = [...document.querySelectorAll('nav button')];
    let index = buttons.indexOf(b);
    if (e.key === 'ArrowRight') index = (index + 1) % buttons.length;
    else if (e.key === 'ArrowLeft') index = (index + buttons.length - 1) % buttons.length;
    else if (e.key === 'Home') index = 0;
    else if (e.key === 'End') index = buttons.length - 1;
    else return;
    e.preventDefault(); buttons[index].focus(); show(buttons[index].dataset.tab);
  });
});
let saved = null;
try { saved = localStorage.getItem('skillmap-tab'); } catch (e) {}
const hash = location.hash.slice(1);
const aliases = {rules:'skills',models:'model'};
const validTabs = new Set([...document.querySelectorAll('nav button')].map(b => b.dataset.tab));
show([hash,saved].map(id => aliases[id] || id).find(id => validTabs.has(id)) || 'overview');
applyTheme();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
"""


def render_product_page(product: dict, provenance: dict) -> str:
    """Inline the product assets and facts-derived compact data; no CDN or build step."""
    assets = ROOT / "src" / "dashboard_assets"
    icons = {
        "overview": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
        "skills": '<circle cx="12" cy="12" r="3"/><circle cx="4" cy="5" r="2"/><circle cx="20" cy="5" r="2"/><circle cx="4" cy="20" r="2"/><circle cx="20" cy="20" r="2"/><path d="m6 6 4 4m4 0 4-4M6 18l4-4m4 0 4 4"/>',
        "salary": '<rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="12" cy="12" r="3"/><path d="M6 12h.01M18 12h.01"/>',
        "clusters": '<circle cx="8" cy="8" r="4"/><circle cx="17" cy="8" r="3"/><path d="M2 21v-2a6 6 0 0 1 12 0v2m2-7a5 5 0 0 1 6 5v2"/>',
        "live": '<path d="M2 12h4l3-8 5 16 3-8h5"/>',
        "model": '<path d="M4 20V4m0 16h17M8 16l4-5 4 2 5-8"/>',
    }
    nav = "".join(f'<button id="tab-{key}" role="tab" data-tab="{key}" aria-controls="{key}" '
                  f'aria-selected="{str(i == 0).lower()}" tabindex="{0 if i == 0 else -1}">'
                  f'<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor">{icon}</svg>'
                  f'{key.title()}</button>' for i, (key, icon) in enumerate(icons.items()))
    def safe_json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = (assets / "product.html").read_text()
    values = {"__CSS__": (assets / "product.css").read_text(), "__PLOTLY__": get_plotlyjs(),
              "__NAV__": nav, "__DATA__": safe_json(product), "__PROVENANCE__": safe_json(provenance),
              "__JS__": (assets / "product.js").read_text()}
    # Substitute template tokens in one pass: data cannot inject another template placeholder.
    return re.sub(r"__(?:CSS|PLOTLY|NAV|DATA|PROVENANCE|JS)__", lambda match: values[match.group()], page)


def render_page(data: dict) -> str:
    """Build compact, unique-job product data from the supplied facts and render it."""
    from src.dashboard_data import build_product_data
    product = build_product_data(data["_facts"])
    return render_product_page(product, compact_provenance(data))


def main(argv: list = None) -> Path:
    """Render generated facts into a fresh isolated HTML artifact."""
    parser = argparse.ArgumentParser(description="Build SkillMap from generated facts, without canonical reads")
    parser.add_argument("--facts", type=Path, required=True, help="generated dashboard facts JSON")
    parser.add_argument("--out", type=Path, required=True, help="fresh output HTML outside the repository")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT.resolve() or ROOT.resolve() in out.parents:
        parser.error("--out must be outside the source repository")
    if out.exists():
        parser.error("--out must name a new file; existing artifacts are preserved")
    data = load_inputs(args.facts)
    page = render_page(data)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(page)
    logger.info("Saved %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
