"""SkillMap Stage 4 — Apriori association rule mining and skill co-occurrence graph.

Mines skill association rules from the `matched_skills` of each job in
`data/processed/cleaned_jobs.csv`, first over all jobs and then over High salary tier
jobs only. High-tier rules are scored against all jobs, to show which skill combinations
are associated with higher pay rather than just common among well-paid jobs.

Run from the project root with:
    python -m src.association
"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.preprocessing import find_project_root, load_cleaned_jobs

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.5
TOP_N_RULES = 20
PLOT_MIN_LIFT = 1.2    # only edges at or above this lift are drawn in the network plot
NODE_MAX_AREA = 2400   # marker area (pt^2) of the most frequent skill
HIGH_TIER = "High"

RULE_COLUMNS = ["antecedents", "consequents", "antecedent support", "consequent support",
                "support", "confidence", "lift", "leverage", "conviction"]

# Chart styling: light chart surface, text inks, one categorical hue for nodes,
# and a single-hue (blue) sequential ramp for lift on edges.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
NODE_COLOR = "#2a78d6"
LIFT_RAMP = LinearSegmentedColormap.from_list(
    "lift_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def encode_transactions(transactions: list) -> pd.DataFrame:
    """One-hot encode a list of skill lists into a boolean job x skill DataFrame."""
    encoder = TransactionEncoder()
    matrix = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(matrix, columns=encoder.columns_)


def mine_frequent_itemsets(onehot: pd.DataFrame, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Run Apriori and return frequent itemsets with a `length` column."""
    itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    itemsets["length"] = itemsets["itemsets"].apply(len)
    logger.info("Apriori (min_support=%.2f) on %d transactions: %d frequent itemsets",
                min_support, len(onehot), len(itemsets))
    return itemsets


def generate_rules(itemsets: pd.DataFrame, min_confidence: float = MIN_CONFIDENCE,
                   min_lift: float = MIN_LIFT) -> pd.DataFrame:
    """Generate association rules with confidence > `min_confidence` and lift > `min_lift`, sorted by lift."""
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[(rules["confidence"] > min_confidence) & (rules["lift"] > min_lift)]
    rules = rules.sort_values(["lift", "support"], ascending=False)[RULE_COLUMNS].reset_index(drop=True)
    logger.info("Rules with confidence > %.2f and lift > %.2f: %d", min_confidence, min_lift, len(rules))
    return rules


def add_high_tier_metrics(rules: pd.DataFrame, onehot_all: pd.DataFrame, is_high: pd.Series) -> pd.DataFrame:
    """Score each rule's full itemset (antecedents + consequents) against all jobs.

    Adds:
      jobs_with_itemset  - jobs (any tier) that list every skill in the rule
      high_tier_rate     - share of those jobs in the High tier
      high_tier_lift     - high_tier_rate / overall High share; > 1 means the skill
                           combination is over-represented among High-paying jobs
    """
    out = rules.copy()
    base_rate = is_high.mean()
    counts, rates = [], []
    for ante, cons in zip(out["antecedents"], out["consequents"]):
        mask = onehot_all[list(ante | cons)].all(axis=1).to_numpy()
        counts.append(int(mask.sum()))
        rates.append(is_high.to_numpy()[mask].mean() if mask.any() else np.nan)
    out["jobs_with_itemset"] = counts
    out["high_tier_rate"] = rates
    out["high_tier_lift"] = out["high_tier_rate"] / base_rate
    return out


def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-friendly copy: itemsets as 'a, b' strings and metrics rounded to 4 places."""
    out = rules.copy()
    for col in ["antecedents", "consequents"]:
        out[col] = out[col].apply(lambda items: ", ".join(sorted(items)))
    numeric = out.select_dtypes(include=np.number).columns.drop("jobs_with_itemset", errors="ignore")
    out[numeric] = out[numeric].round(4)
    return out


def build_cooccurrence_graph(itemsets: pd.DataFrame, n_transactions: int) -> nx.Graph:
    """Build a skill co-occurrence graph from frequent 1- and 2-itemsets.

    Nodes are skills in at least one frequent pair, with `support` and `count`
    (jobs listing the skill). Edges join skills that appear together in at least
    MIN_SUPPORT of jobs, with `support`, `count` (jobs listing both), and
    `lift` = support(a, b) / (support(a) * support(b)). The edge `weight` is the lift.
    """
    singles = {next(iter(s)): sup for s, sup in
               zip(itemsets["itemsets"], itemsets["support"]) if len(s) == 1}
    graph = nx.Graph()
    for items, sup in zip(itemsets["itemsets"], itemsets["support"]):
        if len(items) != 2:
            continue
        a, b = sorted(items)
        for node in (a, b):
            graph.add_node(node, support=singles[node], count=round(singles[node] * n_transactions))
        lift = sup / (singles[a] * singles[b])
        graph.add_edge(a, b, support=sup, count=round(sup * n_transactions), lift=lift, weight=lift)
    logger.info("Co-occurrence graph: %d skills, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def plot_skill_network(graph: nx.Graph, path: Path, title: str, min_lift: float = PLOT_MIN_LIFT) -> None:
    """Draw the co-occurrence graph and save it as a PNG.

    Only edges with lift >= `min_lift` (pairs that co-occur clearly more often than chance)
    are drawn, to keep the plot readable. Skills left without an edge are omitted.
    Node area is proportional to skill frequency. Edge width and color both encode lift,
    and stronger associations are drawn on top. The Kamada-Kawai layout uses 1 / lift as
    the edge length, so strongly associated skills sit closer together.
    """
    shown = graph.edge_subgraph([(u, v) for u, v, d in graph.edges(data=True) if d["lift"] >= min_lift])
    shown = nx.Graph(shown)
    for u, v, d in shown.edges(data=True):
        d["distance"] = 1.0 / d["lift"]

    fig, ax = plt.subplots(figsize=(15, 12), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    pos = nx.kamada_kawai_layout(shown, weight="distance")
    lifts = np.array([d["lift"] for _, _, d in shown.edges(data=True)])
    norm = Normalize(vmin=min_lift, vmax=lifts.max())
    edges = sorted(shown.edges(data=True), key=lambda e: e[2]["lift"])
    nx.draw_networkx_edges(
        shown, pos, ax=ax, edgelist=[(u, v) for u, v, _ in edges],
        width=[0.8 + 5.0 * norm(d["lift"]) for _, _, d in edges],
        edge_color=[LIFT_RAMP(norm(d["lift"])) for _, _, d in edges], alpha=0.9)

    max_support = max(graph.nodes[n]["support"] for n in graph.nodes)
    area = {n: NODE_MAX_AREA * shown.nodes[n]["support"] / max_support for n in shown.nodes}
    nx.draw_networkx_nodes(shown, pos, ax=ax, nodelist=list(area), node_size=list(area.values()),
                           node_color=NODE_COLOR, edgecolors=SURFACE, linewidths=2)
    for node, (x, y) in pos.items():
        radius_pt = np.sqrt(area[node]) / 2
        ax.annotate(node, (x, y), xytext=(0, radius_pt + 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))

    sm = plt.cm.ScalarMappable(cmap=LIFT_RAMP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.01, location="right")
    cbar.set_label("Edge lift (also shown by width)", color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)

    handles = [ax.scatter([], [], s=NODE_MAX_AREA * share / max_support, color=NODE_COLOR,
                          label=f"{share:.0%} of jobs") for share in (0.05, 0.2, 0.4)]
    legend = ax.legend(handles=handles, title="Node size = skill frequency", loc="lower left",
                       frameon=False, labelspacing=3.2, handletextpad=2.2, borderpad=1.5,
                       labelcolor=TEXT_SECONDARY)
    legend.get_title().set_color(TEXT_SECONDARY)

    ax.set_title(title, loc="left", fontsize=14, color=TEXT_PRIMARY, pad=12)
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%d skills, %d edges with lift >= %.1f)",
                path, shown.number_of_nodes(), shown.number_of_edges(), min_lift)


def build_summary(n_all: int, n_high: int, itemsets_all: pd.DataFrame, itemsets_high: pd.DataFrame,
                  rules_all: pd.DataFrame, rules_high: pd.DataFrame, graph: nx.Graph) -> str:
    """Build the plain-text Stage 4 summary."""
    line = "=" * 80
    show = ["antecedents", "consequents", "support", "confidence", "lift"]
    high_show = show + ["jobs_with_itemset", "high_tier_rate", "high_tier_lift"]
    strongest = sorted(graph.edges(data=True), key=lambda e: e[2]["lift"], reverse=True)[:10]
    hubs = sorted(graph.degree, key=lambda d: d[1], reverse=True)[:10]
    top_high = format_rules(rules_high.sort_values("high_tier_lift", ascending=False))
    parts = [
        "SkillMap — Stage 4 Association Rule Mining Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Transactions: matched_skills per job (data/processed/cleaned_jobs.csv)",
        f"Parameters: min_support={MIN_SUPPORT}, confidence > {MIN_CONFIDENCE}, lift > {MIN_LIFT}",
        "",
        f"{line}\nAll jobs\n{line}",
        f"Transactions: {n_all:,}",
        f"Frequent itemsets: {len(itemsets_all)} "
        f"(by size: {itemsets_all['length'].value_counts().sort_index().to_dict()})",
        f"Rules passing filters: {len(rules_all)}",
        f"Top {TOP_N_RULES} rules by lift (outputs/04_association_rules.csv):",
        format_rules(rules_all.head(TOP_N_RULES))[show].to_string(index=False),
        "",
        f"{line}\nSkill co-occurrence graph (outputs/04_skill_network.png)\n{line}",
        f"Skills: {graph.number_of_nodes()}, edges (pairs in >= {MIN_SUPPORT:.0%} of jobs): {graph.number_of_edges()}",
        f"Plot shows edges with lift >= {PLOT_MIN_LIFT}: "
        f"{sum(d['lift'] >= PLOT_MIN_LIFT for _, _, d in graph.edges(data=True))} edges",
        "Most connected skills: " + ", ".join(f"{n} ({d})" for n, d in hubs),
        "Strongest pairs by lift:",
        *[f"  {u} + {v}: lift {d['lift']:.2f}, {d['count']:,} jobs" for u, v, d in strongest],
        "",
        f"{line}\nHigh salary tier jobs only (outputs/04_high_salary_rules.csv)\n{line}",
        f"Transactions: {n_high:,} ({n_high / n_all:.1%} of jobs)",
        f"Frequent itemsets: {len(itemsets_high)} "
        f"(by size: {itemsets_high['length'].value_counts().sort_index().to_dict()})",
        f"Rules passing filters: {len(rules_high)}",
        "high_tier_rate = share of ALL jobs with the rule's full skill set that are High tier;",
        f"high_tier_lift = high_tier_rate / {n_high / n_all:.3f} (overall High share). "
        "> 1 means the combination predicts higher pay.",
        f"Rules with high_tier_lift > 1.2: {int((rules_high['high_tier_lift'] > 1.2).sum())} "
        f"of {len(rules_high)}",
        "",
        "Top 15 High-tier rules by high_tier_lift (skill combinations most predictive of High pay):",
        top_high.head(15)[high_show].to_string(index=False),
        "",
        "Bottom 5 by high_tier_lift (common among High jobs but not predictive):",
        top_high.tail(5)[high_show].to_string(index=False),
        "",
    ]
    return "\n".join(parts)


def run_pipeline(project_root: Path = None) -> dict:
    """Run Stage 4 end to end, writing rules CSVs, the network PNG, and the summary to outputs/."""
    root = project_root or find_project_root()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_cleaned_jobs(root / "data" / "processed" / "cleaned_jobs.csv")
    logger.info("Loaded cleaned_jobs.csv: %d jobs", len(jobs))
    is_high = (jobs["salary_tier"] == HIGH_TIER).reset_index(drop=True)

    onehot_all = encode_transactions(jobs["matched_skills"].tolist())
    itemsets_all = mine_frequent_itemsets(onehot_all)
    rules_all = generate_rules(itemsets_all)
    format_rules(rules_all.head(TOP_N_RULES)).to_csv(output_dir / "04_association_rules.csv", index=False)

    graph = build_cooccurrence_graph(itemsets_all, len(onehot_all))
    plot_skill_network(graph, output_dir / "04_skill_network.png",
                       f"Skill co-occurrence network: pairs in ≥ {MIN_SUPPORT:.0%} of {len(jobs):,} job "
                       f"postings with lift ≥ {PLOT_MIN_LIFT}")

    onehot_high = encode_transactions(jobs.loc[is_high.to_numpy(), "matched_skills"].tolist())
    itemsets_high = mine_frequent_itemsets(onehot_high)
    rules_high = add_high_tier_metrics(generate_rules(itemsets_high), onehot_all, is_high)
    format_rules(rules_high).to_csv(output_dir / "04_high_salary_rules.csv", index=False)

    summary = build_summary(len(onehot_all), len(onehot_high), itemsets_all, itemsets_high,
                            rules_all, rules_high, graph)
    (output_dir / "04_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Saved rules, network plot, and summary to %s", output_dir)
    return {"rules_all": rules_all, "rules_high": rules_high, "graph": graph, "summary": summary,
            "itemsets_all": itemsets_all, "itemsets_high": itemsets_high}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline()
