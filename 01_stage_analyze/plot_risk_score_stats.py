#!/usr/bin/env python3
"""
plot_risk_score_stats.py
========================
Generate statistical visualisation plots from a file_risk_scores.json
produced by compute_file_risk_scores.py.

Plots produced (all saved to <output_dir>/):
  1. risk_score_distribution.png    — histogram + KDE + mean/median lines
  2. signal_distributions.png       — per-signal violin+box plots with stats table
  3. signal_correlation_heatmap.png — Spearman correlation matrix of all signals + risk_score
  4. top_files_bar.png              — horizontal bar chart, top-N files by risk score
  5. anti_pattern_risk_boxplot.png  — risk score distribution per anti-pattern type
  6. signals_scatter_matrix.png     — pairwise scatter grid (top 4 signals vs risk_score)

Usage:
    python plot_risk_score_stats.py <file_risk_scores.json> [--out <dir>] [--top-n 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless/server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Optional: scipy for KDE; fall back gracefully
try:
    from scipy.stats import gaussian_kde, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2563EB",     # blue
    "secondary": "#DC2626",   # red
    "accent": "#16A34A",      # green
    "warn": "#D97706",        # amber
    "neutral": "#6B7280",     # gray
    "bg": "#F9FAFB",
    "grid": "#E5E7EB",
}

AP_COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
]

SIGNAL_LABELS = {
    "anti_pattern_count":    "Anti-Pattern Count",
    "hotspot_fanin_score":   "Hotspot Fan-In Score",
    "rev_count":             "Revision Presence",
    "total_churn":           "Total Churn (lines)",
    "bug_churn_total":       "Bug Churn (lines)",
    "scc_membership_count":  "SCC Membership Count",
    "co_change_without_dep": "Co-Change w/o Dep.",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["bg"],
        "axes.grid":        True,
        "grid.color":       PALETTE["grid"],
        "grid.linewidth":   0.7,
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  9,
        "figure.dpi":       130,
    })


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _desc_stats(vals: np.ndarray) -> Dict[str, float]:
    return {
        "n":      len(vals),
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std":    float(np.std(vals)),
        "var":    float(np.var(vals)),
        "min":    float(np.min(vals)),
        "max":    float(np.max(vals)),
        "p25":    float(np.percentile(vals, 25)),
        "p75":    float(np.percentile(vals, 75)),
    }


def _annotate_stats(ax, stats: Dict[str, float], x_pos: float = 0.97, y_start: float = 0.97) -> None:
    """Add mean/median/std text box to an axes."""
    txt = (
        f"n={stats['n']}\n"
        f"mean={stats['mean']:.4f}\n"
        f"median={stats['median']:.4f}\n"
        f"std={stats['std']:.4f}\n"
        f"var={stats['var']:.4f}"
    )
    ax.text(
        x_pos, y_start, txt,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=PALETTE["grid"]),
    )


# ---------------------------------------------------------------------------
# Plot 1: Risk score distribution
# ---------------------------------------------------------------------------

def plot_risk_distribution(scores: np.ndarray, repo: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"{repo} — Risk Score Distribution", fontsize=13, fontweight="bold")

    # Histogram
    n_bins = min(40, max(10, len(scores) // 5))
    counts, bins, patches = ax.hist(
        scores, bins=n_bins, color=PALETTE["primary"], alpha=0.55,
        edgecolor="white", linewidth=0.5, label="Files"
    )

    # KDE overlay
    if HAS_SCIPY and len(scores) > 5:
        kde = gaussian_kde(scores, bw_method="scott")
        xs = np.linspace(scores.min(), scores.max(), 300)
        # Scale KDE to histogram height
        ys = kde(xs) * len(scores) * (bins[1] - bins[0])
        ax.plot(xs, ys, color=PALETTE["secondary"], linewidth=2, label="KDE")

    stats = _desc_stats(scores)
    ax.axvline(stats["mean"],   color=PALETTE["warn"],    linestyle="--", linewidth=1.5, label=f"Mean {stats['mean']:.4f}")
    ax.axvline(stats["median"], color=PALETTE["accent"],  linestyle=":",  linewidth=1.5, label=f"Median {stats['median']:.4f}")
    ax.axvline(stats["p75"],    color=PALETTE["neutral"], linestyle="-.", linewidth=1.0, label=f"P75 {stats['p75']:.4f}")

    ax.set_xlabel("Composite Risk Score")
    ax.set_ylabel("Number of Files")
    ax.legend(loc="upper right")
    _annotate_stats(ax, stats, x_pos=0.60, y_start=0.97)

    fig.tight_layout()
    out = out_dir / "risk_score_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 2: Per-signal distributions (violin + box)
# ---------------------------------------------------------------------------

def plot_signal_distributions(records: List[Dict], repo: str, out_dir: Path) -> None:
    signal_keys = [k for k in SIGNAL_LABELS if k != "rev_count"]
    n = len(signal_keys)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"{repo} — Per-Signal Distributions", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    # risk_score as first panel
    all_scores = np.array([r["risk_score"] for r in records])
    ax0 = axes_flat[0]
    ax0.set_title("Composite Risk Score", pad=6)
    vp = ax0.violinplot([all_scores], positions=[1], showmedians=True, showextrema=True)
    for pc in vp["bodies"]:
        pc.set_facecolor(PALETTE["primary"])
        pc.set_alpha(0.5)
    ax0.boxplot([all_scores], positions=[1], widths=0.15, patch_artist=True,
                boxprops=dict(facecolor=PALETTE["primary"], alpha=0.3),
                medianprops=dict(color=PALETTE["secondary"], linewidth=2),
                flierprops=dict(marker=".", markersize=3, alpha=0.4))
    st = _desc_stats(all_scores)
    _annotate_stats(ax0, st)
    ax0.set_xticks([])

    for i, sig in enumerate(signal_keys):
        ax = axes_flat[i + 1]
        vals = np.array([r["signals"].get(sig, 0) for r in records], dtype=float)
        label = SIGNAL_LABELS.get(sig, sig)
        color = list(PALETTE.values())[i % 5]

        ax.set_title(label, pad=6)
        if np.any(vals > 0):
            vp = ax.violinplot([vals], positions=[1], showmedians=True, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            ax.boxplot([vals], positions=[1], widths=0.15, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.3),
                       medianprops=dict(color=PALETTE["secondary"], linewidth=2),
                       flierprops=dict(marker=".", markersize=3, alpha=0.4))
        else:
            ax.text(0.5, 0.5, "all zero", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["neutral"])
        st = _desc_stats(vals)
        _annotate_stats(ax, st)
        ax.set_xticks([])

    # Hide unused panel
    if n + 1 < len(axes_flat):
        for ax in axes_flat[n + 1:]:
            ax.set_visible(False)

    fig.tight_layout()
    out = out_dir / "signal_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 3: Spearman correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(records: List[Dict], repo: str, out_dir: Path) -> None:
    keys = ["risk_score"] + [k for k in SIGNAL_LABELS if k != "rev_count"]
    labels = ["Risk Score"] + [SIGNAL_LABELS.get(k, k) for k in keys[1:]]

    mat = np.zeros((len(keys), len(records)))
    for j, rec in enumerate(records):
        mat[0, j] = rec["risk_score"]
        for i, k in enumerate(keys[1:], 1):
            mat[i, j] = rec["signals"].get(k, 0)

    n = len(keys)
    corr = np.zeros((n, n))
    pval = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if HAS_SCIPY and np.std(mat[i]) > 0 and np.std(mat[j]) > 0:
                r, p = spearmanr(mat[i], mat[j])
                corr[i, j] = r
                pval[i, j] = p
            elif i == j:
                corr[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle(f"{repo} — Spearman Correlation (signals × risk_score)", fontsize=13, fontweight="bold")

    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            sig = "**" if pval[i, j] < 0.01 else ("*" if pval[i, j] < 0.05 else "")
            txt = f"{val:.2f}{sig}"
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    fig.tight_layout()
    out = out_dir / "signal_correlation_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 4: Top-N files horizontal bar chart
# ---------------------------------------------------------------------------

def plot_top_files_bar(records: List[Dict], repo: str, out_dir: Path, top_n: int = 30) -> None:
    top = records[:top_n]
    scores  = np.array([r["risk_score"] for r in top])
    labels  = [Path(r["file"]).name + f"\n({Path(r['file']).parent.name})" for r in top]
    ap_tags = [", ".join(r.get("anti_patterns_seen", [])) or "—" for r in top]

    # Color by dominant anti-pattern
    all_aps = sorted({ap for r in top for ap in r.get("anti_patterns_seen", [])})
    ap_color_map = {ap: AP_COLORS[i % len(AP_COLORS)] for i, ap in enumerate(all_aps)}
    bar_colors = []
    for r in top:
        aps = r.get("anti_patterns_seen", [])
        bar_colors.append(ap_color_map[aps[0]] if aps else PALETTE["neutral"])

    fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.38)))
    fig.suptitle(f"{repo} — Top {top_n} Files by Composite Risk Score", fontsize=13, fontweight="bold")

    y_pos = np.arange(len(top))
    bars = ax.barh(y_pos, scores[::-1], color=bar_colors[::-1], edgecolor="white", height=0.7)

    # Anti-pattern label on bar
    for i, (bar, tag) in enumerate(zip(bars, ap_tags[::-1])):
        x_val = bar.get_width()
        ax.text(
            x_val + 0.003, bar.get_y() + bar.get_height() / 2,
            tag, va="center", ha="left", fontsize=7, color=PALETTE["neutral"]
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=7.5)
    ax.set_xlabel("Composite Risk Score")
    ax.set_xlim(0, scores.max() * 1.30)

    # Legend for anti-patterns
    if all_aps:
        patches = [mpatches.Patch(color=ap_color_map[ap], label=ap) for ap in all_aps]
        patches.append(mpatches.Patch(color=PALETTE["neutral"], label="(none)"))
        ax.legend(handles=patches, loc="lower right", fontsize=8, title="Anti-patterns")

    fig.tight_layout()
    out = out_dir / "top_files_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 5: Risk score distribution per anti-pattern type
# ---------------------------------------------------------------------------

def plot_antipattern_risk_boxplot(records: List[Dict], repo: str, out_dir: Path) -> None:
    # Group files by each anti-pattern they belong to (a file may appear in multiple)
    ap_groups: Dict[str, List[float]] = {}
    no_ap_scores: List[float] = []

    for r in records:
        aps = r.get("anti_patterns_seen", [])
        if not aps:
            no_ap_scores.append(r["risk_score"])
        for ap in aps:
            ap_groups.setdefault(ap, []).append(r["risk_score"])

    if not ap_groups:
        print("  Skipping anti-pattern boxplot (no anti_patterns_seen data).")
        return

    # Sort by median descending
    ap_sorted = sorted(ap_groups.items(), key=lambda kv: np.median(kv[1]), reverse=True)
    labels = [ap for ap, _ in ap_sorted]
    data   = [vals for _, vals in ap_sorted]

    if no_ap_scores:
        labels.append("(no anti-pattern)")
        data.append(no_ap_scores)

    n_groups = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_groups * 0.6 + 2)),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"{repo} — Risk Score by Anti-Pattern Type", fontsize=13, fontweight="bold")

    # Left: box plots
    ax = axes[0]
    colors = [AP_COLORS[i % len(AP_COLORS)] for i in range(len(ap_sorted))] + [PALETTE["neutral"]]
    bp = ax.boxplot(
        data, vert=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_yticks(range(1, n_groups + 1))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite Risk Score")
    ax.set_title("Risk Score Distributions per Anti-Pattern")

    # Right: stats table
    ax2 = axes[1]
    ax2.axis("off")
    col_labels = ["Anti-Pattern", "n", "Mean", "Median", "Std"]
    rows = []
    for label, vals in zip(labels, data):
        vals_arr = np.array(vals)
        rows.append([
            label,
            str(len(vals_arr)),
            f"{np.mean(vals_arr):.4f}",
            f"{np.median(vals_arr):.4f}",
            f"{np.std(vals_arr):.4f}",
        ])
    tbl = ax2.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    ax2.set_title("Statistics", pad=10)

    fig.tight_layout()
    out = out_dir / "anti_pattern_risk_boxplot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 6: Pairwise scatter — top signals vs risk_score
# ---------------------------------------------------------------------------

def plot_signals_scatter(records: List[Dict], repo: str, out_dir: Path) -> None:
    # Pick 4 most variable signals
    signal_keys = [k for k in SIGNAL_LABELS if k != "rev_count"]
    variances = {}
    for k in signal_keys:
        vals = np.array([r["signals"].get(k, 0) for r in records], dtype=float)
        variances[k] = float(np.var(vals))
    top_signals = sorted(variances, key=variances.get, reverse=True)[:4]

    risk_scores = np.array([r["risk_score"] for r in records], dtype=float)

    fig, axes = plt.subplots(1, len(top_signals), figsize=(5 * len(top_signals), 4.5), sharey=True)
    fig.suptitle(f"{repo} — Top Signals vs Risk Score (scatter)", fontsize=13, fontweight="bold")
    if len(top_signals) == 1:
        axes = [axes]

    for ax, sig in zip(axes, top_signals):
        vals = np.array([r["signals"].get(sig, 0) for r in records], dtype=float)
        label = SIGNAL_LABELS.get(sig, sig)

        ax.scatter(vals, risk_scores, alpha=0.5, s=20, color=PALETTE["primary"], edgecolors="none")

        # Trend line
        if np.std(vals) > 0:
            coeff = np.polyfit(vals, risk_scores, 1)
            xs = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(xs, np.polyval(coeff, xs), color=PALETTE["secondary"],
                    linewidth=1.5, linestyle="--", label="trend")

        ax.set_xlabel(label)
        ax.set_title(label, pad=4, fontsize=10)

    axes[0].set_ylabel("Composite Risk Score")
    fig.tight_layout()
    out = out_dir / "signals_scatter_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Summary stats JSON
# ---------------------------------------------------------------------------

def write_summary_stats(records: List[Dict], meta: Dict, out_dir: Path) -> None:
    signal_keys = list(SIGNAL_LABELS.keys())
    scores = np.array([r["risk_score"] for r in records], dtype=float)

    summary = {
        "meta": meta,
        "total_files": len(records),
        "risk_score": _desc_stats(scores),
        "signals": {},
        "anti_pattern_breakdown": {},
    }

    for k in signal_keys:
        vals = np.array([r["signals"].get(k, 0) for r in records], dtype=float)
        summary["signals"][k] = _desc_stats(vals)

    # Per anti-pattern stats
    ap_groups: Dict[str, List[float]] = {}
    for r in records:
        for ap in r.get("anti_patterns_seen", []):
            ap_groups.setdefault(ap, []).append(r["risk_score"])
    for ap, vals_list in ap_groups.items():
        summary["anti_pattern_breakdown"][ap] = _desc_stats(np.array(vals_list))

    out = out_dir / "risk_score_stats.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate statistical plots from file_risk_scores.json."
    )
    ap.add_argument("scores_json", help="Path to file_risk_scores.json")
    ap.add_argument("--out", help="Output directory for plots (default: same as scores_json)")
    ap.add_argument("--top-n", type=int, default=30, help="Top-N files in bar chart (default: 30)")
    args = ap.parse_args()

    scores_path = Path(args.scores_json).expanduser().resolve()
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found.", file=sys.stderr)
        return 1

    data = _load(scores_path)
    records = data.get("files", [])
    meta = data.get("meta", {})
    repo = meta.get("repo", scores_path.parent.name)

    if not records:
        print("No file records found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out).expanduser().resolve() if args.out else scores_path.parent / "plots" / "risk_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    print(f"Generating plots for {repo} ({len(records)} files) → {out_dir}")

    scores_arr = np.array([r["risk_score"] for r in records], dtype=float)

    plot_risk_distribution(scores_arr, repo, out_dir)
    plot_signal_distributions(records, repo, out_dir)
    plot_correlation_heatmap(records, repo, out_dir)
    plot_top_files_bar(records, repo, out_dir, top_n=args.top_n)
    plot_antipattern_risk_boxplot(records, repo, out_dir)
    plot_signals_scatter(records, repo, out_dir)
    write_summary_stats(records, meta, out_dir)

    print(f"\nDone. {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
