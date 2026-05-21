"""
Compute agreement metrics between GPT-4 and gpt-5.4-mini / gpt-5.4-nano on MTBench single-grading.

Reads: evals/judge_agreement/paired_scores.csv
Writes: evals/judge_agreement/results.json
        evals/judge_agreement/results.md
        evals/judge_agreement/figures/*.png

Usage:
    python analyze_agreement.py [--input paired_scores.csv] [--no-plots]
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NEW_JUDGES = {
    "mini": ("gpt-5.4-mini-2026-03-17", "mini_score"),
    "nano": ("gpt-5.4-nano-2026-03-17", "nano_score"),
}

# ── Metric helpers ────────────────────────────────────────────────────────────

def _bootstrap_ci(fn, a, b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    stats_boot = []
    n_samples = len(a)
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        try:
            stats_boot.append(fn(a[idx], b[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(ref: np.ndarray, pred: np.ndarray, n_bootstrap=1000) -> dict:
    """All agreement metrics for pred vs ref (ref = GPT-4 scores)."""
    if len(ref) < 2:
        return {}

    # Clip scores to valid 1-10 range for kappa (needs integer labels)
    ref_int = np.clip(np.round(ref).astype(int), 1, 10)
    pred_int = np.clip(np.round(pred).astype(int), 1, 10)
    all_labels = list(range(1, 11))

    pearson_r, pearson_p = stats.pearsonr(ref, pred)
    spearman_r, spearman_p = stats.spearmanr(ref, pred)
    mae = float(np.mean(np.abs(ref - pred)))
    mse_val = float(np.mean((ref - pred) ** 2))
    bias = float(np.mean(pred - ref))
    exact = float(np.mean(ref_int == pred_int))
    within1 = float(np.mean(np.abs(ref_int - pred_int) <= 1))

    try:
        kappa = float(cohen_kappa_score(ref_int, pred_int, weights="quadratic", labels=all_labels))
    except Exception:
        kappa = float("nan")

    # Bootstrap CIs
    pearson_ci = _bootstrap_ci(lambda a, b: stats.pearsonr(a, b)[0], ref, pred, n_bootstrap)
    spearman_ci = _bootstrap_ci(lambda a, b: stats.spearmanr(a, b)[0], ref, pred, n_bootstrap)
    mae_ci = _bootstrap_ci(lambda a, b: np.mean(np.abs(a - b)), ref, pred, n_bootstrap)

    return {
        "n": len(ref),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_ci_95": list(pearson_ci),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "spearman_ci_95": list(spearman_ci),
        "mae": mae,
        "mae_ci_95": list(mae_ci),
        "mse": mse_val,
        "bias": bias,
        "exact_match": exact,
        "within_1pt": within1,
        "quadratic_weighted_kappa": kappa,
    }


def compute_all_metrics(df: pd.DataFrame) -> dict:
    """Compute overall + stratified metrics for one new judge."""
    ref = df["gpt4_score"].values
    results = {}

    # Overall
    for col_suffix, col in [("mini", "mini_score"), ("nano", "nano_score")]:
        pred = df[col].values
        results[col_suffix] = {
            "overall": compute_metrics(ref, pred),
            "by_turn": {},
            "by_model": {},
            "by_category": {},
            "by_template": {},
        }
        for turn in df["turn"].unique():
            mask = df["turn"] == turn
            results[col_suffix]["by_turn"][str(turn)] = compute_metrics(ref[mask], pred[mask])
        for model in df["model"].unique():
            mask = df["model"] == model
            results[col_suffix]["by_model"][model] = compute_metrics(ref[mask], pred[mask])
        for cat in df["category"].unique():
            mask = df["category"] == cat
            results[col_suffix]["by_category"][cat] = compute_metrics(ref[mask], pred[mask])
        for tmpl in df["judge_template"].unique():
            mask = df["judge_template"] == tmpl
            results[col_suffix]["by_template"][tmpl] = compute_metrics(ref[mask], pred[mask])

    return results


def compute_gpt4_self_agreement(gpt4_path: str) -> dict:
    """Use duplicate GPT-4 runs in qwen3-4b-instruct-2507 to estimate intra-judge noise."""
    if not os.path.exists(gpt4_path):
        return {}
    import json as _json
    from collections import defaultdict
    groups = defaultdict(list)
    with open(gpt4_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            if obj["score"] != -1:
                groups[(obj["question_id"], obj["turn"])].append(obj["score"])
    pairs = [(s[0], s[1]) for s in groups.values() if len(s) >= 2]
    if not pairs:
        return {}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_pairs": len(pairs), **compute_metrics(a, b, n_bootstrap=500)}


# ── Aggregate score comparison ────────────────────────────────────────────────

def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        for judge_col, judge_label in [("gpt4_score", "gpt-4"), ("mini_score", "gpt-5.4-mini"), ("nano_score", "gpt-5.4-nano")]:
            for turn in [1, 2]:
                t = sub[sub["turn"] == turn][judge_col]
                rows.append({"model": model, "judge": judge_label, "turn": f"turn{turn}", "mean_score": t.mean()})
            rows.append({"model": model, "judge": judge_label, "turn": "overall", "mean_score": sub[judge_col].mean()})
    return pd.DataFrame(rows)


# ── Top disagreements ─────────────────────────────────────────────────────────

def top_disagreements(df: pd.DataFrame, judge_col: str, n=10) -> pd.DataFrame:
    d = df.copy()
    d["abs_diff"] = (d["gpt4_score"] - d[judge_col]).abs()
    d["signed_diff"] = d[judge_col] - d["gpt4_score"]
    return d.nlargest(n, "abs_diff")[
        ["model", "question_id", "turn", "category", "gpt4_score", judge_col, "signed_diff"]
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, figures_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    os.makedirs(figures_dir, exist_ok=True)
    colors = {"qwen3-4b-base": "#1f77b4", "qwen3-4b-instruct-2507": "#ff7f0e"}

    for judge_key, (judge_label, judge_col) in NEW_JUDGES.items():
        # Scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, model in zip(axes, df["model"].unique()):
            sub = df[df["model"] == model]
            ax.scatter(sub["gpt4_score"], sub[judge_col], alpha=0.4, s=18, color=colors.get(model, "steelblue"))
            lo, hi = df[["gpt4_score", judge_col]].min().min(), df[["gpt4_score", judge_col]].max().max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
            r, _ = stats.pearsonr(sub["gpt4_score"], sub[judge_col])
            ax.set_title(f"{model}\nr={r:.3f}", fontsize=9)
            ax.set_xlabel("GPT-4 score")
            ax.set_ylabel(f"{judge_label} score")
        fig.suptitle(f"GPT-4 vs {judge_label} (single-grading)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"scatter_{judge_key}.png"), dpi=150)
        plt.close(fig)

        # Bland-Altman
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, model in zip(axes, df["model"].unique()):
            sub = df[df["model"] == model]
            mean_score = (sub["gpt4_score"] + sub[judge_col]) / 2
            diff = sub[judge_col] - sub["gpt4_score"]
            ax.scatter(mean_score, diff, alpha=0.4, s=18, color=colors.get(model, "steelblue"))
            mean_d = diff.mean()
            std_d = diff.std()
            ax.axhline(mean_d, color="red", lw=1, label=f"bias={mean_d:.2f}")
            ax.axhline(mean_d + 1.96 * std_d, color="red", lw=0.7, ls="--")
            ax.axhline(mean_d - 1.96 * std_d, color="red", lw=0.7, ls="--")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{model}\nbias={mean_d:.2f}, ±1.96σ=[{mean_d-1.96*std_d:.2f}, {mean_d+1.96*std_d:.2f}]", fontsize=8)
            ax.set_xlabel("Mean of GPT-4 and new-judge scores")
            ax.set_ylabel("New-judge − GPT-4")
            ax.legend(fontsize=7)
        fig.suptitle(f"Bland–Altman: GPT-4 vs {judge_label}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"bland_altman_{judge_key}.png"), dpi=150)
        plt.close(fig)

    # Category heatmap (MAE)
    cats = sorted(df["category"].unique())
    mini_maes = [np.mean(np.abs(df[df["category"] == c]["gpt4_score"] - df[df["category"] == c]["mini_score"])) for c in cats]
    nano_maes = [np.mean(np.abs(df[df["category"] == c]["gpt4_score"] - df[df["category"] == c]["nano_score"])) for c in cats]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, mini_maes, w, label="gpt-5.4-mini")
    ax.bar(x + w/2, nano_maes, w, label="gpt-5.4-nano")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE vs GPT-4")
    ax.set_title("Per-category MAE: new judges vs GPT-4")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "category_mae.png"), dpi=150)
    plt.close(fig)


# ── Markdown report ───────────────────────────────────────────────────────────

def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def write_markdown(metrics: dict, agg_df: pd.DataFrame, self_agreement: dict,
                   df: pd.DataFrame, out_path: str):
    lines = []
    lines.append("# MTBench Judge-Agreement Study: GPT-4 vs GPT-5.4 Models\n")
    lines.append(f"**Models evaluated:** qwen3-4b-base, qwen3-4b-instruct-2507  ")
    lines.append(f"**Total judgment pairs:** {len(df)}  ")
    lines.append(f"**Judging mode:** single-answer grading (1–10 scale)\n")

    # GPT-4 self-agreement baseline
    if self_agreement:
        m = self_agreement
        lines.append("## GPT-4 Self-Agreement Baseline\n")
        lines.append("From the two GPT-4 judging runs on qwen3-4b-instruct-2507 "
                     f"({m.get('n', m.get('n_pairs', '?'))} duplicate pairs):\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Pearson r | {fmt(m.get('pearson_r'))} {fmt_ci(m.get('pearson_ci_95'))} |")
        lines.append(f"| Spearman ρ | {fmt(m.get('spearman_r'))} {fmt_ci(m.get('spearman_ci_95'))} |")
        lines.append(f"| MAE | {fmt(m.get('mae'))} {fmt_ci(m.get('mae_ci_95'))} |")
        lines.append(f"| Exact match | {fmt(m.get('exact_match'), 2)} |")
        lines.append(f"| Within ±1 pt | {fmt(m.get('within_1pt'), 2)} |")
        lines.append(f"| Quadratic-weighted κ | {fmt(m.get('quadratic_weighted_kappa'))} |\n")

    # Overall metrics for each new judge
    lines.append("## Overall Agreement Metrics\n")
    for judge_key, (judge_label, _) in NEW_JUDGES.items():
        m = metrics.get(judge_key, {}).get("overall", {})
        if not m:
            lines.append(f"### {judge_label}\n\n_No data available._\n")
            continue
        lines.append(f"### {judge_label}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| n | {m.get('n', '?')} |")
        lines.append(f"| Pearson r | {fmt(m.get('pearson_r'))} {fmt_ci(m.get('pearson_ci_95'))} |")
        lines.append(f"| Spearman ρ | {fmt(m.get('spearman_r'))} {fmt_ci(m.get('spearman_ci_95'))} |")
        lines.append(f"| MAE | {fmt(m.get('mae'))} {fmt_ci(m.get('mae_ci_95'))} |")
        lines.append(f"| MSE | {fmt(m.get('mse'))} |")
        lines.append(f"| Bias (new − GPT-4) | {fmt(m.get('bias'))} |")
        lines.append(f"| Exact match | {fmt(m.get('exact_match'), 2)} |")
        lines.append(f"| Within ±1 pt | {fmt(m.get('within_1pt'), 2)} |")
        lines.append(f"| Quadratic-weighted κ | {fmt(m.get('quadratic_weighted_kappa'))} |\n")

    # Aggregate scores
    lines.append("## Aggregate Scores by Judge\n")
    pivot = agg_df.pivot_table(index=["model", "judge"], columns="turn", values="mean_score")
    lines.append(pivot.to_markdown())
    lines.append("")

    # Per-turn breakdown
    lines.append("## By Turn\n")
    lines.append(_strat_table(metrics, "by_turn"))

    # Per-category breakdown
    lines.append("## By Category\n")
    lines.append(_strat_table(metrics, "by_category"))

    # Per-template breakdown
    lines.append("## By Judge Template\n")
    lines.append(_strat_table(metrics, "by_template"))

    # Per-model breakdown
    lines.append("## By Evaluated Model\n")
    lines.append(_strat_table(metrics, "by_model"))

    # Top disagreements
    lines.append("## Top 10 Largest Disagreements\n")
    for judge_key, (judge_label, judge_col) in NEW_JUDGES.items():
        lines.append(f"### {judge_label}\n")
        top = top_disagreements(df, judge_col, n=10)
        lines.append(top.to_markdown(index=False))
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report to {out_path}")


def fmt_ci(ci):
    if ci is None:
        return ""
    lo, hi = ci
    return f"[{lo:.3f}, {hi:.3f}]"


def _strat_table(metrics: dict, strat_key: str) -> str:
    rows = []
    # Collect all strata across both judges
    strata = set()
    for jk in NEW_JUDGES:
        strata |= set(metrics.get(jk, {}).get(strat_key, {}).keys())
    strata = sorted(strata)

    header = ["Stratum", "Judge", "n", "Pearson r", "Spearman ρ", "MAE", "Bias", "Exact", "Within±1", "κ"]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    for s in strata:
        for judge_key, (judge_label, _) in NEW_JUDGES.items():
            m = metrics.get(judge_key, {}).get(strat_key, {}).get(s, {})
            if not m:
                continue
            short_label = "mini" if "mini" in judge_key else "nano"
            rows.append(
                f"| {s} | {short_label} | {m.get('n','?')} "
                f"| {fmt(m.get('pearson_r'))} "
                f"| {fmt(m.get('spearman_r'))} "
                f"| {fmt(m.get('mae'))} "
                f"| {fmt(m.get('bias'))} "
                f"| {fmt(m.get('exact_match'),2)} "
                f"| {fmt(m.get('within_1pt'),2)} "
                f"| {fmt(m.get('quadratic_weighted_kappa'))} |"
            )
    return "\n".join(rows) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(SCRIPT_DIR, "paired_scores.csv"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run build_paired.py first.")
        import sys; sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # GPT-4 self-agreement baseline (from instruct-2507 duplicate runs)
    gpt4_instruct_path = os.path.join(
        SCRIPT_DIR, "../../evals/results/qwen3-4b-instruct-2507/mtbench/model_judgment/gpt-4_single.jsonl"
    )
    gpt4_instruct_path = os.path.normpath(gpt4_instruct_path)
    self_agreement = compute_gpt4_self_agreement(gpt4_instruct_path)
    if self_agreement:
        print(f"GPT-4 self-agreement baseline: pearson={self_agreement.get('pearson_r', 'N/A'):.3f}, "
              f"MAE={self_agreement.get('mae', 'N/A'):.3f} (n={self_agreement.get('n_pairs', '?')} pairs)")

    metrics = compute_all_metrics(df)
    agg_df = aggregate_scores(df)

    # Save JSON
    json_path = os.path.join(SCRIPT_DIR, "results.json")
    out = {
        "gpt4_self_agreement_baseline": self_agreement,
        "metrics": metrics,
        "aggregate_scores": agg_df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote metrics to {json_path}")

    # Print summary
    for judge_key, (judge_label, _) in NEW_JUDGES.items():
        m = metrics.get(judge_key, {}).get("overall", {})
        if m:
            print(f"\n{judge_label} overall: Pearson r={m['pearson_r']:.3f}, "
                  f"MAE={m['mae']:.3f}, within±1={m['within_1pt']:.2f}, κ={m['quadratic_weighted_kappa']:.3f}")

    # Markdown report
    md_path = os.path.join(SCRIPT_DIR, "results.md")
    write_markdown(metrics, agg_df, self_agreement, df, md_path)

    # Plots
    if not args.no_plots:
        figures_dir = os.path.join(SCRIPT_DIR, "figures")
        make_plots(df, figures_dir)
        print(f"Plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()
