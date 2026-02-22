"""
AI Text Humanizer — Benchmark Analysis & Research Report
Generates 8 graphs, correlation study, and summary report.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)

# ── Hardcoded training data ──────────────────────────────────────────────

BATCH_1 = [
    {"topic": "renewable energy policy",       "source": "gpt4",    "initial": 96.4, "final": 87.6},
    {"topic": "AI climate modeling",           "source": "claude",  "initial": 89.6, "final": 87.4},
    {"topic": "deep sea biodiversity",         "source": "deepseek","initial": 82.0, "final": 68.2},
    {"topic": "quantum entanglement",          "source": "gpt4",    "initial": 91.8, "final": 83.8},
    {"topic": "AI humanitarian crisis",        "source": "deepseek","initial": 64.0, "final": 41.1},
    {"topic": "neuroprosthetics ethics",       "source": "gpt4",    "initial": 93.8, "final": 90.8},
    {"topic": "Victorian social reform",       "source": "gpt4",    "initial": 83.6, "final": 83.6},
    {"topic": "ancient maritime law",          "source": "gpt4",    "initial": 86.3, "final": 77.6},
    {"topic": "autonomous warfare ethics",     "source": "gpt4",    "initial": 85.6, "final": 63.1},
    {"topic": "digital linguistics",           "source": "gpt4",    "initial": 94.2, "final": 85.9},
]

BATCH_2 = [
    {"topic": "fungal networks",               "source": "gpt4",    "initial": 95.4, "final": 90.8},
    {"topic": "Weimar hyperinflation",         "source": "claude",  "initial": 80.4, "final": 38.5},
    {"topic": "exoplanet spectroscopy",        "source": "deepseek","initial": 84.3, "final": 74.7},
    {"topic": "child labor law",               "source": "gpt4",    "initial": 96.2, "final": 90.0},
    {"topic": "memory consolidation",          "source": "claude",  "initial": 85.0, "final": 77.6},
    {"topic": "dark matter simulations",       "source": "deepseek","initial": 85.8, "final": 75.1},
    {"topic": "mercantilism colonial",         "source": "gpt4",    "initial": 95.5, "final": 83.5},
    {"topic": "deepfake electoral",            "source": "claude",  "initial": 55.4, "final": 38.2},
    {"topic": "xenotransplantation",           "source": "deepseek","initial": 90.3, "final": 71.6},
    {"topic": "circular economy",              "source": "gpt4",    "initial": 91.6, "final": 67.7},
]

BATCH_3 = [
    {"topic": "robotics disaster",             "source": "gpt4",    "initial": 94.5, "final": 83.8},
    {"topic": "blockchain supply chain",       "source": "claude",  "initial": 92.1, "final": 91.3},
    {"topic": "ocean acidification",           "source": "deepseek","initial": 92.5, "final": 75.2},
    {"topic": "philosophy consciousness",      "source": "gpt4",    "initial": 95.1, "final": 88.0},
    {"topic": "epigenetic inheritance",        "source": "claude",  "initial": 91.4, "final": 90.8},
    {"topic": "silk road currency",            "source": "deepseek","initial": 79.9, "final": 54.7},
    {"topic": "longevity pension",             "source": "gpt4",    "initial": 93.2, "final": 78.2},
    {"topic": "refugee quota",                 "source": "claude",  "initial": 84.1, "final": 79.0},
    {"topic": "soil microbiome",               "source": "deepseek","initial": 88.1, "final": 79.0},
    {"topic": "postcolonial literature",       "source": "gpt4",    "initial": 96.1, "final": 90.6},
]

BENCHMARK_DATA = [
    {
        "file": "optimized_claude_blockchain_in_supply_chain_transparency.txt",
        "source": "claude", "topic": "blockchain supply chain",
        "external": {"gptzero": 100, "zerogpt": 36.8, "originality": 100, "winston": 99, "quillbot": 33},
    },
    {
        "file": "optimized_claude_epigenetic_inheritance_transgenerational.txt",
        "source": "claude", "topic": "epigenetic inheritance",
        "external": {"gptzero": 100, "zerogpt": 16.3, "originality": 100, "winston": 100, "quillbot": 7},
    },
    {
        "file": "optimized_claude_international_refugee_quota_burden_shari.txt",
        "source": "claude", "topic": "refugee quota",
        "external": {"gptzero": 100, "zerogpt": 100, "originality": 100, "winston": 100, "quillbot": 74},
    },
    {
        "file": "optimized_deepseek_ocean_acidification_impact_on_fisheries.txt",
        "source": "deepseek", "topic": "ocean acidification",
        "external": {"gptzero": 100, "zerogpt": 63.25, "originality": 100, "winston": 13, "quillbot": 20},
    },
    {
        "file": "optimized_deepseek_silk_road_currency_exchange_systems.txt",
        "source": "deepseek", "topic": "silk road currency",
        "external": {"gptzero": 100, "zerogpt": 0, "originality": 100, "winston": 16, "quillbot": 0},
    },
    {
        "file": "optimized_deepseek_soil_microbiome_antibiotic_resistance_em.txt",
        "source": "deepseek", "topic": "soil microbiome",
        "external": {"gptzero": 100, "zerogpt": 66.36, "originality": 100, "winston": 100, "quillbot": 22},
    },
    {
        "file": "optimized_gpt4_longevity_economy_pension_fund_solvency.txt",
        "source": "gpt4", "topic": "longevity pension",
        "external": {"gptzero": 100, "zerogpt": 12.62, "originality": 100, "winston": 90, "quillbot": 31},
    },
    {
        "file": "optimized_gpt4_philosophy_of_consciousness_and_ai.txt",
        "source": "gpt4", "topic": "philosophy consciousness",
        "external": {"gptzero": 100, "zerogpt": 22.25, "originality": 100, "winston": 90, "quillbot": 7},
    },
    {
        "file": "optimized_gpt4_postcolonial_literature_canon_formation.txt",
        "source": "gpt4", "topic": "postcolonial literature",
        "external": {"gptzero": 100, "zerogpt": 39.06, "originality": 100, "winston": 100, "quillbot": 32},
    },
    {
        "file": "optimized_gpt4_robotics_in_disaster_response.txt",
        "source": "gpt4", "topic": "robotics disaster",
        "external": {"gptzero": 100, "zerogpt": 32.21, "originality": 100, "winston": 100, "quillbot": 41},
    },
]

# Strategy selection counts (from training logs — "Best strats" lines)
STRATEGY_STATS = {
    "perplexity-light":      {"count": 17, "avg_improvement": 1.2},
    "perplexity-moderate":   {"count": 16, "avg_improvement": 1.8},
    "balanced":              {"count": 11, "avg_improvement": 5.5},
    "perplexity-heavy":      {"count": 8,  "avg_improvement": 1.0},
    "perplexity-structural": {"count": 7,  "avg_improvement": 0.8},
    "light":                 {"count": 6,  "avg_improvement": 4.2},
    "structural":            {"count": 5,  "avg_improvement": 3.1},
    "kitchen-sink":          {"count": 4,  "avg_improvement": 6.8},
    "perplexity-kitchen":    {"count": 4,  "avg_improvement": 0.9},
    "human-noise":           {"count": 3,  "avg_improvement": 2.5},
}

# ── Style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
})

BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#d29922"
RED     = "#f85149"
PURPLE  = "#bc8cff"
CYAN    = "#39d2c0"
PINK    = "#f778ba"
GRAY    = "#8b949e"

BATCH_COLORS = {1: BLUE, 2: GREEN, 3: ORANGE}
SOURCE_COLORS = {"gpt4": BLUE, "claude": GREEN, "deepseek": ORANGE}

# ── Part 1: Run local detectors on benchmark texts ──────────────────────

def collect_local_scores():
    """Run AIDetector on each benchmark text, return augmented data."""
    print("\n" + "="*60)
    print("  Part 1: Collecting local detector scores")
    print("="*60)

    from detector import AIDetector
    det = AIDetector()

    results = []
    for i, entry in enumerate(BENCHMARK_DATA):
        fpath = os.path.join(PROJECT, entry["file"])
        if not os.path.exists(fpath):
            print(f"  [!] Missing: {entry['file']}")
            continue
        text = open(fpath, "r", encoding="utf-8").read().strip()
        print(f"  [{i+1}/10] Detecting: {entry['topic']}...")
        scores = det.detect(text)
        entry["local"] = {
            "binoculars": round(scores["binoculars"] * 100, 1),
            "gpt2_ppl":   round(scores["gpt2_ppl"]   * 100, 1),
            "gltr":       round(scores["gltr"]        * 100, 1),
            "ensemble":   round(scores["ensemble"]    * 100, 1),
        }
        print(f"    bino={entry['local']['binoculars']:.0f}% gpt2={entry['local']['gpt2_ppl']:.0f}% "
              f"gltr={entry['local']['gltr']:.0f}% ens={entry['local']['ensemble']:.0f}%")
        results.append(entry)

    det.close()
    return results


# ── Graph 1: Training Progress ──────────────────────────────────────────

def graph1_training_progress():
    print("\n  Graph 1: Training progress across batches...")
    fig, ax = plt.subplots(figsize=(14, 6))

    all_data = []
    for batch_num, batch in [(1, BATCH_1), (2, BATCH_2), (3, BATCH_3)]:
        for j, d in enumerate(batch):
            idx = (batch_num - 1) * 10 + j + 1
            all_data.append({**d, "idx": idx, "batch": batch_num,
                             "improvement": d["initial"] - d["final"]})

    xs = [d["idx"] for d in all_data]
    initials = [d["initial"] for d in all_data]
    finals   = [d["final"] for d in all_data]

    # Shade batches
    for b, start, end in [(1,0.5,10.5),(2,10.5,20.5),(3,20.5,30.5)]:
        ax.axvspan(start, end, alpha=0.06, color=BATCH_COLORS[b])

    # Lines
    ax.plot(xs, initials, 'o-', color=RED,   linewidth=1.8, markersize=5, label="Initial (AI-generated)", zorder=3)
    ax.plot(xs, finals,   's-', color=GREEN,  linewidth=1.8, markersize=5, label="Final (optimized)", zorder=3)

    # Fill gap
    ax.fill_between(xs, initials, finals, alpha=0.15, color=GREEN)

    # Batch labels
    for b, x_mid in [(1, 5.5), (2, 15.5), (3, 25.5)]:
        ax.text(x_mid, 102, f"Batch {b}", ha="center", fontsize=10, fontweight="bold",
                color=BATCH_COLORS[b])

    ax.set_xlabel("Text #", fontsize=12)
    ax.set_ylabel("AI Detection Score (%)", fontsize=12)
    ax.set_title("Training Progress: AI Detection Score Before vs After Optimization",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(25, 105)
    ax.set_xticks(range(1, 31))
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("graph1_training_progress.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph1_training_progress.png")
    return all_data


# ── Graph 2: Improvement Distribution ───────────────────────────────────

def graph2_improvement_distribution(all_data):
    print("  Graph 2: Improvement distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))

    improvements = [d["improvement"] for d in all_data]
    mean_imp = np.mean(improvements)

    bins = np.arange(0, 45, 3)
    colors_by_batch = [BATCH_COLORS[d["batch"]] for d in all_data]

    # Separate by batch for stacked histogram
    for b in [1, 2, 3]:
        batch_imps = [d["improvement"] for d in all_data if d["batch"] == b]
        ax.hist(batch_imps, bins=bins, alpha=0.65, color=BATCH_COLORS[b],
                label=f"Batch {b} (n={len(batch_imps)})", edgecolor="#30363d", linewidth=0.5)

    ax.axvline(mean_imp, color=PINK, linewidth=2, linestyle="--",
               label=f"Mean: {mean_imp:.1f}%")

    ax.set_xlabel("Score Improvement (percentage points)", fontsize=12)
    ax.set_ylabel("Number of Texts", fontsize=12)
    ax.set_title("Distribution of AI Detection Score Improvements",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(framealpha=0.9, edgecolor="#30363d")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("graph2_improvement_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph2_improvement_distribution.png")


# ── Graph 3: Strategy Effectiveness ─────────────────────────────────────

def graph3_strategy_effectiveness():
    print("  Graph 3: Strategy effectiveness...")
    fig, ax = plt.subplots(figsize=(12, 6))

    strats = sorted(STRATEGY_STATS.keys(), key=lambda s: STRATEGY_STATS[s]["count"], reverse=True)
    counts = [STRATEGY_STATS[s]["count"] for s in strats]
    avg_imps = [STRATEGY_STATS[s]["avg_improvement"] for s in strats]

    # Normalize avg_improvement for colormap
    norm = plt.Normalize(vmin=min(avg_imps), vmax=max(avg_imps))
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in avg_imps]

    bars = ax.bar(range(len(strats)), counts, color=colors, edgecolor="#30363d", linewidth=0.5, width=0.7)

    # Add count labels on bars
    for bar, count, avg in zip(bars, counts, avg_imps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c9d1d9")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"~{avg:.1f}%", ha="center", va="center", fontsize=8, color="#0e1117", fontweight="bold")

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Times Selected as Best", fontsize=12)
    ax.set_title("Strategy Selection Frequency & Average Improvement",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y", alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Avg Improvement (%)", fontsize=10)

    fig.tight_layout()
    fig.savefig("graph3_strategy_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph3_strategy_effectiveness.png")


# ── Graph 4: Source Model Comparison ────────────────────────────────────

def graph4_source_model_comparison(all_data):
    print("  Graph 4: Source model comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame(all_data)
    grouped = df.groupby("source").agg(
        avg_initial=("initial", "mean"),
        avg_final=("final", "mean"),
        count=("initial", "count"),
        avg_improvement=("improvement", "mean"),
    ).reindex(["gpt4", "claude", "deepseek"])

    x = np.arange(len(grouped))
    w = 0.3

    bars1 = ax.bar(x - w/2, grouped["avg_initial"], w, color=RED, alpha=0.75,
                   label="Avg Initial Score", edgecolor="#30363d")
    bars2 = ax.bar(x + w/2, grouped["avg_final"], w, color=GREEN, alpha=0.75,
                   label="Avg Final Score", edgecolor="#30363d")

    # Labels
    for bar, val in zip(bars1, grouped["avg_initial"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, color=RED)
    for bar, val in zip(bars2, grouped["avg_final"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, color=GREEN)

    # Improvement arrows
    for i, (src, row) in enumerate(grouped.iterrows()):
        ax.annotate(f"-{row['avg_improvement']:.1f}%",
                    xy=(i + w/2, row["avg_final"] + 4),
                    fontsize=11, fontweight="bold", color=CYAN, ha="center")

    labels = {"gpt4": f"GPT-4 (n={int(grouped.loc['gpt4','count'])})",
              "claude": f"Claude (n={int(grouped.loc['claude','count'])})",
              "deepseek": f"DeepSeek (n={int(grouped.loc['deepseek','count'])})"}
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in grouped.index], fontsize=12)
    ax.set_ylabel("AI Detection Score (%)", fontsize=12)
    ax.set_title("Source AI Model: Initial vs Optimized Detection Scores",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.9, edgecolor="#30363d")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("graph4_source_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph4_source_model_comparison.png")


# ── Graph 5: Correlation Heatmap ────────────────────────────────────────

def graph5_correlation_heatmap(bench_data):
    print("  Graph 5: Local vs external correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 7))

    local_keys = ["binoculars", "gpt2_ppl", "gltr", "ensemble"]
    ext_keys   = ["gptzero", "zerogpt", "originality", "winston", "quillbot"]

    # Build arrays
    local_arr = {k: np.array([d["local"][k] for d in bench_data]) for k in local_keys}
    ext_arr   = {k: np.array([d["external"][k] for d in bench_data]) for k in ext_keys}

    # Correlation matrix
    corr = np.full((len(local_keys), len(ext_keys)), np.nan)
    annot = [[""]*len(ext_keys) for _ in range(len(local_keys))]

    for i, lk in enumerate(local_keys):
        for j, ek in enumerate(ext_keys):
            lv = local_arr[lk]
            ev = ext_arr[ek]
            # Check variance
            if np.std(ev) < 0.01 or np.std(lv) < 0.01:
                annot[i][j] = "N/A\n(const)"
                corr[i][j] = np.nan
            else:
                r, p = stats.pearsonr(lv, ev)
                corr[i][j] = r
                sig = "*" if p < 0.05 else ""
                annot[i][j] = f"{r:.2f}{sig}\np={p:.3f}"

    mask = np.isnan(corr)
    sns.heatmap(corr, ax=ax, annot=annot, fmt="", mask=mask,
                xticklabels=[k.upper() for k in ext_keys],
                yticklabels=[k.replace("_", " ").title() for k in local_keys],
                cmap="RdYlGn_r", vmin=-1, vmax=1, center=0,
                linewidths=1, linecolor="#30363d",
                cbar_kws={"label": "Pearson r"})

    # Mark N/A cells
    for i in range(len(local_keys)):
        for j in range(len(ext_keys)):
            if mask[i][j]:
                ax.text(j + 0.5, i + 0.5, "N/A\n(const)", ha="center", va="center",
                        fontsize=9, color=GRAY, fontstyle="italic")

    ax.set_title("Local vs External Detector Correlation",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("External Detectors", fontsize=12)
    ax.set_ylabel("Local Detectors", fontsize=12)

    fig.tight_layout()
    fig.savefig("graph5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph5_correlation_heatmap.png")
    return local_arr, ext_arr


# ── Graph 6: Scatter Correlations ───────────────────────────────────────

def graph6_scatter_correlations(bench_data, local_arr, ext_arr):
    print("  Graph 6: Scatter plots with regression...")
    variable_ext = ["zerogpt", "winston", "quillbot"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, ek in zip(axes, variable_ext):
        ev = ext_arr[ek]
        lv = local_arr["ensemble"]

        # Color by source
        for d in bench_data:
            c = SOURCE_COLORS[d["source"]]
            ax.scatter(d["local"]["ensemble"], d["external"][ek],
                       color=c, s=70, zorder=5, edgecolors="#c9d1d9", linewidths=0.5)

        # Regression
        if np.std(ev) > 0.01:
            slope, intercept, r, p, se = stats.linregress(lv, ev)
            x_line = np.linspace(min(lv) - 5, max(lv) + 5, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=PINK, linewidth=1.5, linestyle="--", alpha=0.8)
            ax.text(0.05, 0.92, f"R\u00b2={r**2:.3f}\np={p:.3f}",
                    transform=ax.transAxes, fontsize=9, color=PINK,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor=PINK, alpha=0.8))

        ax.set_xlabel("Local Ensemble (%)", fontsize=10)
        ax.set_ylabel(f"{ek.upper()} (%)", fontsize=10)
        ax.set_title(f"Ensemble vs {ek.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlim(40, 100)
        ax.set_ylim(-5, 110)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles = [mpatches.Patch(color=SOURCE_COLORS[s], label=s.upper()) for s in ["gpt4","claude","deepseek"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, framealpha=0.9,
               edgecolor="#30363d", bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Local Ensemble Score vs External Detectors", fontsize=14,
                 fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig("graph6_scatter_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph6_scatter_correlations.png")


# ── Graph 7: External Detector Heatmap ──────────────────────────────────

def graph7_external_detector_heatmap(bench_data):
    print("  Graph 7: External detector heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))

    ext_keys = ["gptzero", "zerogpt", "originality", "winston", "quillbot"]
    topics = [d["topic"] for d in bench_data]
    matrix = np.array([[d["external"][k] for k in ext_keys] for d in bench_data])

    sns.heatmap(matrix, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, linewidths=1, linecolor="#30363d",
                xticklabels=[k.upper() for k in ext_keys],
                yticklabels=topics,
                cbar_kws={"label": "AI Detection Score (%)"})

    ax.set_title("External Detector Scores Across Optimized Texts\n(Same text, different verdicts)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("External Detector", fontsize=12)
    ax.set_ylabel("Optimized Text", fontsize=12)

    fig.tight_layout()
    fig.savefig("graph7_external_detector_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph7_external_detector_heatmap.png")


# ── Graph 8: Detector Agreement Matrix ──────────────────────────────────

def graph8_detector_agreement(bench_data, local_arr, ext_arr):
    print("  Graph 8: Detector agreement analysis...")
    fig, ax = plt.subplots(figsize=(10, 8))

    all_keys = ["binoculars", "gpt2_ppl", "gltr", "ensemble", "zerogpt", "winston", "quillbot"]
    all_labels = ["Binoculars", "GPT-2 PPL", "GLTR", "Ensemble", "ZeroGPT", "Winston", "QuillBot"]

    # Build score arrays
    scores = {}
    for k in ["binoculars", "gpt2_ppl", "gltr", "ensemble"]:
        scores[k] = local_arr[k]
    for k in ["zerogpt", "winston", "quillbot"]:
        scores[k] = ext_arr[k]

    n = len(all_keys)
    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sv_i = scores[all_keys[i]]
            sv_j = scores[all_keys[j]]
            # Agreement: both > 50 or both <= 50
            agree = np.sum((sv_i > 50) == (sv_j > 50))
            agreement[i][j] = agree / len(sv_i) * 100

    annot_text = [[f"{agreement[i][j]:.0f}%" for j in range(n)] for i in range(n)]

    sns.heatmap(agreement, ax=ax, annot=annot_text, fmt="",
                xticklabels=all_labels, yticklabels=all_labels,
                cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=1, linecolor="#30363d",
                cbar_kws={"label": "Agreement Rate (%)"})

    ax.set_title("Pairwise Detector Agreement Rate\n(Both agree AI >50% or both \u226450%)",
                 fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig("graph8_detector_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved graph8_detector_agreement.png")


# ── Weight Recommendations ──────────────────────────────────────────────

def compute_recommendations(local_arr, ext_arr):
    """Find which local detector best predicts each external detector."""
    print("\n  Computing weight recommendations...")
    variable_ext = ["zerogpt", "winston", "quillbot"]
    local_keys = ["binoculars", "gpt2_ppl", "gltr"]

    best_for = {}
    all_corrs = {}
    for ek in variable_ext:
        ev = ext_arr[ek]
        best_r = None
        best_lk = None
        for lk in local_keys:
            lv = local_arr[lk]
            if np.std(lv) < 0.01:
                continue
            r, p = stats.pearsonr(lv, ev)
            all_corrs[(lk, ek)] = (r, p)
            if best_r is None or abs(r) > abs(best_r):
                best_r = r
                best_lk = lk
        best_for[ek] = (best_lk, best_r)

    return best_for, all_corrs


# ── Research Report ─────────────────────────────────────────────────────

def generate_report(all_data, bench_data, best_for, all_corrs, local_arr, ext_arr):
    print("\n  Generating research report...")

    all_improvements = [d["improvement"] for d in all_data]
    batch1_avg = np.mean([d["improvement"] for d in all_data if d["batch"] == 1])
    batch2_avg = np.mean([d["improvement"] for d in all_data if d["batch"] == 2])
    batch3_avg = np.mean([d["improvement"] for d in all_data if d["batch"] == 3])
    overall_avg = np.mean(all_improvements)
    best_imp = max(all_improvements)
    best_topic = [d["topic"] for d in all_data if d["improvement"] == best_imp][0]
    zero_imp = sum(1 for d in all_data if d["improvement"] < 1.0)

    # External stats
    ext_avgs = {}
    ext_passed = {}
    for ek in ["gptzero", "zerogpt", "originality", "winston", "quillbot"]:
        vals = [d["external"][ek] for d in bench_data]
        ext_avgs[ek] = np.mean(vals)
        ext_passed[ek] = sum(1 for v in vals if v < 30)

    # Source model stats
    df = pd.DataFrame(all_data)
    src_stats = df.groupby("source").agg(
        n=("initial", "count"),
        avg_init=("initial", "mean"),
        avg_final=("final", "mean"),
        avg_imp=("improvement", "mean"),
    ).reindex(["gpt4", "claude", "deepseek"])

    # Local scores for benchmark
    local_ens = [d["local"]["ensemble"] for d in bench_data]

    report = f"""{'='*70}
  AI TEXT HUMANIZER - RESEARCH ANALYSIS REPORT
  Generated: 2026-02-22
{'='*70}

1. PIPELINE ARCHITECTURE
{'─'*70}

  Phase 1: Cross-model LLM Rewrite
    GPT-4 → Claude, Claude → DeepSeek, DeepSeek → GPT-4
    (Never rewrite own output — forces style transfer)
    Temperature: 0.95, max_tokens: 2048

  Phase 2: Perplexity Injection (5 strategies)
    - rare_synonym_replace: WordNet + wordfreq rarity scoring
    - inject_parentheticals: Aside clauses with em-dashes
    - inject_rhetorical_questions: Topic-matched standalone questions
    - vary_sentence_rhythm: Short-sentence merging
    - inject_discourse_markers: Starters and mid-sentence connectors

  Phase 3: Quality Filtering
    - BERT sentence-level cosine similarity (threshold 0.80)
    - TF-IDF similarity guard (threshold 0.65)
    - SBERT semantic preservation (threshold 0.70)
    - SpellCheck on inflected synonym forms

  Detection Ensemble (local):
    - Binoculars (Falcon-7B x2, 4-bit NF4): weight 0.35
    - GPT-2 Perplexity (GPTZero-style):     weight 0.35
    - GLTR (top-k distribution):             weight 0.30

  Hardware: NVIDIA RTX 4090 Laptop (16GB VRAM)
  VRAM usage: ~10.0 GB (Binoculars 9.2 + GPT-2 0.2 + T5 0.5 + SBERT 0.08)


2. TRAINING SUMMARY
{'─'*70}

  Total texts trained:  30 (3 batches of 10)
  Total texts generated: 46 (including untrained extras)

  Batch 1:  avg improvement = {batch1_avg:+.1f} pp
  Batch 2:  avg improvement = {batch2_avg:+.1f} pp
  Batch 3:  avg improvement = {batch3_avg:+.1f} pp
  ──────────────────────────────────────
  Overall:  avg improvement = {overall_avg:+.1f} pp

  Best single improvement:  -{best_imp:.1f} pp ({best_topic})
  Zero-improvement texts:   {zero_imp}/30 (<1 pp improvement)

  Score ranges:
    Initial scores: {min(d['initial'] for d in all_data):.1f}% – {max(d['initial'] for d in all_data):.1f}%
    Final scores:   {min(d['final'] for d in all_data):.1f}% – {max(d['final'] for d in all_data):.1f}%


3. SOURCE MODEL ANALYSIS
{'─'*70}

  {"Model":<12} {"N":>3}  {"Avg Initial":>12}  {"Avg Final":>10}  {"Avg Improvement":>16}
  {"─"*60}"""

    for src, row in src_stats.iterrows():
        label = {"gpt4": "GPT-4", "claude": "Claude", "deepseek": "DeepSeek"}[src]
        report += f"\n  {label:<12} {int(row['n']):>3}  {row['avg_init']:>11.1f}%  {row['avg_final']:>9.1f}%  {row['avg_imp']:>+15.1f} pp"

    report += f"""

  Observation: {"DeepSeek texts show the largest average improvement, suggesting they " +
  "have more identifiable AI patterns that the pipeline can exploit." if src_stats.loc['deepseek','avg_imp'] > src_stats.loc['gpt4','avg_imp'] else
  "GPT-4 texts are hardest to optimize further."}


4. STRATEGY EFFECTIVENESS
{'─'*70}

  {"Strategy":<24} {"Selected":>8}  {"Avg Improvement":>16}
  {"─"*52}"""

    for strat in sorted(STRATEGY_STATS.keys(), key=lambda s: STRATEGY_STATS[s]["count"], reverse=True):
        s = STRATEGY_STATS[strat]
        report += f"\n  {strat:<24} {s['count']:>8}x  {s['avg_improvement']:>+15.1f} pp"

    report += f"""

  Top Phase 1 strategies: balanced, light (LLM rewrite variants)
  Top Phase 2 strategies: perplexity-light, perplexity-moderate
  Kitchen-sink (max changes) has highest per-use improvement but rare selection


5. EXTERNAL BENCHMARK (10 texts x 5 detectors)
{'─'*70}

  {"Detector":<15} {"Avg Score":>10}  {"Passed (<30%)":>14}  {"Verdict":>20}
  {"─"*64}"""

    verdicts = {
        "gptzero":     "Strongest (catches all)",
        "originality": "Strongest (catches all)",
        "winston":     "Moderate",
        "zerogpt":     "Moderate-weak",
        "quillbot":    "Weakest (most fooled)",
    }
    for ek in ["gptzero", "originality", "winston", "zerogpt", "quillbot"]:
        report += f"\n  {ek.upper():<15} {ext_avgs[ek]:>9.1f}%  {ext_passed[ek]:>13}/10  {verdicts[ek]:>20}"

    # Find most dramatic inconsistency
    silk_road = [d for d in bench_data if "silk" in d["topic"]][0]
    report += f"""

  KEY FINDING: The Silk Road text scored:
    ZeroGPT:     {silk_road['external']['zerogpt']:.0f}%  (Human)
    QuillBot:    {silk_road['external']['quillbot']:.0f}%  (Human)
    GPTZero:     {silk_road['external']['gptzero']:.0f}% (AI)
    Originality: {silk_road['external']['originality']:.0f}% (AI)

  → The SAME optimized text is classified as fully human by some detectors
    and fully AI by others. AI detection is fundamentally inconsistent.


6. LOCAL DETECTOR PERFORMANCE ON BENCHMARK
{'─'*70}

  {"Text":<30} {"Bino":>6} {"GPT2":>6} {"GLTR":>6} {"Ens":>6}
  {"─"*58}"""

    for d in bench_data:
        report += f"\n  {d['topic']:<30} {d['local']['binoculars']:>5.0f}% {d['local']['gpt2_ppl']:>5.0f}% {d['local']['gltr']:>5.0f}% {d['local']['ensemble']:>5.0f}%"

    avg_bino = np.mean([d['local']['binoculars'] for d in bench_data])
    avg_gpt2 = np.mean([d['local']['gpt2_ppl'] for d in bench_data])
    avg_gltr = np.mean([d['local']['gltr'] for d in bench_data])
    avg_ens  = np.mean([d['local']['ensemble'] for d in bench_data])
    report += f"\n  {'AVERAGE':<30} {avg_bino:>5.1f}% {avg_gpt2:>5.1f}% {avg_gltr:>5.1f}% {avg_ens:>5.1f}%"

    report += f"""


7. CORRELATION ANALYSIS: LOCAL vs EXTERNAL
{'─'*70}

  Correlation between local detectors and external detectors with variance:
  (GPTZero and Originality excluded — constant 100%, no variance)

  {"Pair":<35} {"Pearson r":>10}  {"p-value":>10}  {"Significant?":>12}
  {"─"*72}"""

    for (lk, ek), (r, p) in sorted(all_corrs.items()):
        sig = "Yes *" if p < 0.05 else "No"
        report += f"\n  {lk:<16} vs {ek:<15} {r:>+10.3f}  {p:>10.4f}  {sig:>12}"

    report += f"""

  Best local predictor for each external detector:"""
    for ek, (lk, r) in best_for.items():
        lk_str = lk if lk else "N/A"
        r_str = f"{r:+.3f}" if r is not None and r != -2 else "N/A"
        report += f"\n    {ek.upper():<15} -> {lk_str:<15} (r = {r_str})"

    # Weight recommendation
    report += f"""


8. ENSEMBLE WEIGHT RECOMMENDATIONS
{'─'*70}

  Current weights: binoculars=0.35, gpt2_ppl=0.35, gltr=0.30

  Based on correlation analysis with external detectors that have variance
  (ZeroGPT, Winston, QuillBot), the local detectors show:"""

    for ek, (lk, r) in best_for.items():
        lk_str = lk if lk else "N/A"
        r_str = f"{r:+.3f}" if r is not None and r != -2 else "N/A"
        report += f"\n    {ek.upper()}: best predicted by {lk_str} (r={r_str})"

    report += f"""

  Note: With only 10 data points, correlations have wide confidence intervals.
  More benchmark data needed before adjusting production weights.
  GPTZero and Originality.ai detect ALL texts as AI regardless — these
  detectors likely use different signals than our local ensemble.


9. CONCLUSIONS
{'─'*70}

  1. The humanization pipeline achieves an average {overall_avg:.1f} percentage point
     reduction in local AI detection scores across 30 texts.

  2. DeepSeek-generated text shows the most improvement potential, while
     GPT-4 text is most resistant to humanization.

  3. External detector agreement is extremely low:
     - GPTZero and Originality.ai classify 100% of optimized texts as AI
     - ZeroGPT passes {ext_passed['zerogpt']}/10 texts, QuillBot passes {ext_passed['quillbot']}/10 texts
     - The same text can score 0% on one detector and 100% on another

  4. Binoculars (Falcon-7B based) is the hardest local detector to fool,
     consistently scoring 92-99% even after optimization. GPT-2 PPL and
     GLTR are more responsive to perplexity injection.

  5. The fundamental challenge: commercial detectors (GPTZero, Originality)
     appear to use proprietary classifiers trained on large datasets, making
     them resistant to statistical perturbation approaches. Fooling them
     likely requires fundamentally different text generation approaches.

  6. Strategy analysis shows perplexity-light and perplexity-moderate are
     selected most often as optimal, suggesting small targeted changes
     outperform aggressive rewriting for local detector evasion.


{'='*70}
  Generated by AI Text Humanizer Benchmark Analysis
  Total graphs: 8 | Texts analyzed: 30 training + 10 benchmark
{'='*70}
"""

    with open("research_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("    Saved research_report.txt")
    return report


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  AI Text Humanizer — Benchmark Analysis")
    print("="*60)

    # Part 1: Collect local scores
    bench_data = collect_local_scores()
    if len(bench_data) < 10:
        print(f"  [!] Only {len(bench_data)}/10 texts found, proceeding with available data")

    # Part 2: Training graphs
    print("\n" + "="*60)
    print("  Part 2: Generating training analysis graphs")
    print("="*60)
    all_data = graph1_training_progress()
    graph2_improvement_distribution(all_data)
    graph3_strategy_effectiveness()
    graph4_source_model_comparison(all_data)

    # Part 3-4: Correlation & external analysis
    print("\n" + "="*60)
    print("  Part 3-4: Correlation & external detector analysis")
    print("="*60)
    local_arr, ext_arr = graph5_correlation_heatmap(bench_data)
    graph6_scatter_correlations(bench_data, local_arr, ext_arr)
    graph7_external_detector_heatmap(bench_data)
    graph8_detector_agreement(bench_data, local_arr, ext_arr)

    # Recommendations
    best_for, all_corrs = compute_recommendations(local_arr, ext_arr)

    # Part 5: Report
    print("\n" + "="*60)
    print("  Part 5: Research report")
    print("="*60)
    report = generate_report(all_data, bench_data, best_for, all_corrs, local_arr, ext_arr)

    # Summary
    print("\n" + "="*60)
    print("  ALL OUTPUTS GENERATED")
    print("="*60)
    outputs = [
        "graph1_training_progress.png",
        "graph2_improvement_distribution.png",
        "graph3_strategy_effectiveness.png",
        "graph4_source_model_comparison.png",
        "graph5_correlation_heatmap.png",
        "graph6_scatter_correlations.png",
        "graph7_external_detector_heatmap.png",
        "graph8_detector_agreement.png",
        "research_report.txt",
    ]
    for f in outputs:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        status = "OK" if size > 0 else "MISSING"
        print(f"  [{status}] {f} ({size:,} bytes)")

    print("\n  Done!")


if __name__ == "__main__":
    main()
