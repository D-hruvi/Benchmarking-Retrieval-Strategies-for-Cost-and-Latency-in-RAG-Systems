"""
Reads results.csv (AFTER you've manually filled in the 'correctness'
column with correct / partial / wrong for every row), computes the
summary table, generates 3 charts, and writes RESULTS.md with the
ACTUAL numbers from your run -- this script pulls every number from
results.csv; nothing here is hardcoded or filled in as a placeholder.

Usage: python scripts/analyze_results.py

Run this only after results.csv has correctness filled in for every row
-- rows with an empty correctness are excluded from the accuracy
calculation (and you'll get a warning telling you how many were
skipped), so a partially-graded CSV won't silently corrupt the accuracy
numbers.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

from src import config

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def load_and_validate():
    df = pd.read_csv(config.RESULTS_CSV_PATH)
    if df.empty:
        raise SystemExit("results.csv is empty. Run scripts/run_benchmark.py first.")

    # Do the isna() check BEFORE any string conversion. astype(str) turns a
    # real NaN into the literal string "nan", which would make a second
    # isna() check downstream (e.g. in write_results_md) silently find
    # nothing -- this bit us in testing, so the ungraded count is computed
    # once, here, and threaded through explicitly rather than recomputed.
    ungraded_mask = df["correctness"].isna()
    ungraded = int(ungraded_mask.sum())
    if ungraded > 0:
        print(f"WARNING: {ungraded}/{len(df)} rows have no 'correctness' value filled "
              f"in. Those rows are EXCLUDED from accuracy calculations below. "
              f"Fill in results.csv (correct / partial / wrong) for a complete picture.")

    df["correctness"] = df["correctness"].where(~ungraded_mask, other=pd.NA)
    df.loc[~ungraded_mask, "correctness"] = (
        df.loc[~ungraded_mask, "correctness"].astype(str).str.strip().str.lower()
    )
    df.attrs["ungraded_count"] = ungraded  # threaded through to write_results_md

    invalid = df[~ungraded_mask & ~df["correctness"].isin(list(CORRECTNESS_SCORE.keys()))]
    if len(invalid) > 0:
        raise SystemExit(
            f"Found {len(invalid)} rows with a correctness value that isn't "
            f"'correct', 'partial', or 'wrong': {invalid['correctness'].unique().tolist()}. "
            f"Fix results.csv and re-run."
        )
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    graded = df[df["correctness"].isin(CORRECTNESS_SCORE.keys())].copy()
    graded["score"] = graded["correctness"].map(CORRECTNESS_SCORE)

    summary = df.groupby("variant").agg(
        avg_latency_s=("total_latency_s", "mean"),
        avg_cost_usd=("cost_usd", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        n_queries=("question_id", "count"),
    )

    accuracy = graded.groupby("variant")["score"].mean() * 100
    summary["accuracy_pct"] = accuracy

    if "cache_hit" in df.columns:
        cache_rows = df[df["variant"] == "cached"]
        if len(cache_rows) > 0:
            hit_rate = cache_rows["cache_hit"].mean() * 100
            summary.loc["cached", "cache_hit_rate_pct"] = hit_rate

    return summary.round(4)


def generate_charts(summary: pd.DataFrame, out_dir: str):
    variants = summary.index.tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, summary["avg_cost_usd"], color=["#4C72B0", "#55A868", "#C44E52"][:len(variants)])
    ax.set_ylabel("Avg cost per query (USD)")
    ax.set_title("Cost per Query by Variant")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_cost.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, summary["avg_latency_s"], color=["#4C72B0", "#55A868", "#C44E52"][:len(variants)])
    ax.set_ylabel("Avg latency per query (seconds)")
    ax.set_title("Latency per Query by Variant")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_latency.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, summary["accuracy_pct"], color=["#4C72B0", "#55A868", "#C44E52"][:len(variants)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Variant")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart_accuracy.png"), dpi=150)
    plt.close(fig)

    print(f"Charts written to {out_dir}/: chart_cost.png, chart_latency.png, chart_accuracy.png")


def pct_change(baseline, new):
    if baseline == 0:
        return float("nan")
    return (new - baseline) / baseline * 100


def write_results_md(summary: pd.DataFrame, df: pd.DataFrame, out_path: str):
    lines = ["# RESULTS", ""]
    lines.append(f"Generated from `results.csv` -- {len(df)} total (question, variant) rows across "
                 f"{df['question_id'].nunique()} questions and {df['variant'].nunique()} variants.")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Variant | Avg Latency (s) | Avg Cost/Query (USD) | Accuracy (%) | Cache Hit Rate (%) |")
    lines.append("|---|---|---|---|---|")
    for variant in summary.index:
        row = summary.loc[variant]
        hit_rate = row.get("cache_hit_rate_pct")
        hit_rate_str = f"{hit_rate:.1f}" if pd.notna(hit_rate) else "n/a"
        lines.append(
            f"| {variant} | {row['avg_latency_s']:.3f} | {row['avg_cost_usd']:.6f} | "
            f"{row['accuracy_pct']:.1f} | {hit_rate_str} |"
        )
    lines.append("")

    if "naive" in summary.index:
        naive_latency = summary.loc["naive", "avg_latency_s"]
        naive_cost = summary.loc["naive", "avg_cost_usd"]
        naive_acc = summary.loc["naive", "accuracy_pct"]

        lines.append("## Trade-offs vs Naive RAG (baseline)")
        lines.append("")
        for variant in summary.index:
            if variant == "naive":
                continue
            row = summary.loc[variant]
            lat_change = pct_change(naive_latency, row["avg_latency_s"])
            cost_change = pct_change(naive_cost, row["avg_cost_usd"])
            acc_change = row["accuracy_pct"] - naive_acc
            lat_word = "cut" if lat_change < 0 else "increased"
            cost_word = "cut" if cost_change < 0 else "increased"
            acc_word = "dropped" if acc_change < 0 else ("gained" if acc_change > 0 else "held steady at")
            lines.append(
                f"- **{variant}** {lat_word} average latency by {abs(lat_change):.1f}% "
                f"and {cost_word} average cost by {abs(cost_change):.1f}% versus naive, "
                f"with accuracy {acc_word}{'' if acc_word == 'held steady at' else f' by {abs(acc_change):.1f} percentage points'} "
                f"({naive_acc:.1f}% -> {row['accuracy_pct']:.1f}%)."
            )
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    lines.append("![Cost per variant](chart_cost.png)")
    lines.append("")
    lines.append("![Latency per variant](chart_latency.png)")
    lines.append("")
    lines.append("![Accuracy per variant](chart_accuracy.png)")
    lines.append("")

    ungraded = df.attrs.get("ungraded_count", int(df["correctness"].isna().sum()))
    if ungraded > 0:
        lines.append(f"> Note: {ungraded} row(s) were excluded from accuracy numbers above "
                      f"because their 'correctness' column was left blank in results.csv.")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"RESULTS.md written to {out_path}")


if __name__ == "__main__":
    df = load_and_validate()
    summary = build_summary(df)
    print(summary)

    out_dir = os.path.dirname(os.path.abspath(config.RESULTS_CSV_PATH))
    generate_charts(summary, out_dir)
    write_results_md(summary, df, os.path.join(out_dir, "RESULTS.md"))
