#!/usr/bin/env python3
"""
NewRAMvsSAM_34spp_Violinplot.py

Per-protein violin comparison of the new Hellinger RAM and SAM treebuilding
approaches for the 34-species OMAM20 dataset.

RAM:
    Violin = PCC distribution across all 12 layers x 20 heads
    Star   = ASTRAL consensus PCC

SAM:
    Violin = PCC distribution across all 12 layers x 12 heads
    Star   = ASTRAL consensus PCC

Accuracy is measured as percent correct clades (PCC) from RF distance:

    PCC = (1 - (RF / 2) / (N - 3)) * 100

where N = 34 species.
"""

import sys
from pathlib import Path
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scipy.stats import ttest_rel, wilcoxon

# =============================================================================
# Import RF scorer
# =============================================================================

RF_SCORER_DIR = Path("/home/rohan/PLM-Phylo-Dev/Scripts")
sys.path.append(str(RF_SCORER_DIR))

from RF_scorer import rf_distance


# =============================================================================
# Configuration
# =============================================================================

SUPERTREE = Path("/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk")

RAM_DIR = Path(
    "/home/rohan/pFM-phylo/Results/Trees/RAM_Symmetrical/34spp/Trees"
)

SAM_DIR = Path(
    "/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees"
)

OUT_PLOT_DIR = "/home/rohan/pFM-phylo/Figures/Submission_Figures"
OUT_PLOT = "Fig3c_34spp_Violins"

PROTEINS = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH",
    "DYRK1B", "GPX3", "HAUS1", "IL6", "INO80B",
    "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO",
    "SLC39A12", "TF", "TPPP2", "UBA2", "UPRT",
]

NUM_SPECIES = 34

RAM_LAYER_COUNT = 12
RAM_HEAD_COUNT = 20

SAM_LAYER_COUNT = 12
SAM_HEAD_COUNT = 12

RAM_COLOR = "dodgerblue"
SAM_COLOR = "darkorange"


# =============================================================================
# Path helpers
# =============================================================================

def ram_layerhead_path(protein, layer, head):
    domain = f"{protein}_34spp"

    return (
        RAM_DIR
        / domain
        / f"layer{layer}"
        / f"{domain}_layer{layer}_head{head}_tree.nwk"
    )


def sam_layerhead_path(protein, layer, head):
    domain = f"{protein}_34sppAligned"

    return (
        SAM_DIR
        / domain
        / f"layer{layer}"
        / f"{domain}_layer{layer}_head{head}_tree.nwk"
    )


def ram_consensus_path(protein):
    domain = f"{protein}_34spp"

    return (
        RAM_DIR
        / domain
        / "Consensus_Trees"
        / f"{domain}_RAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


def sam_consensus_path(protein):
    domain = f"{protein}_34sppAligned"

    return (
        SAM_DIR
        / domain
        / "Consensus_Trees"
        / f"{domain}_SAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


# =============================================================================
# PCC scoring
# =============================================================================

def calc_pcc(rf):
    return (1.0 - (rf / 2.0) / (NUM_SPECIES - 3)) * 100.0


def score_tree(treefile):
    if not treefile.exists():
        return np.nan

    try:
        rf = rf_distance(
            str(SUPERTREE),
            str(treefile),
        )

        return 100.0 if rf == 0 else calc_pcc(rf)

    except Exception as e:
        print(f"FAILED: {treefile}")
        print(f"  {e}")
        return np.nan


def collect_head_scores(
    protein,
    path_function,
    layer_count,
    head_count,
    label,
):
    values = []

    for layer in tqdm(
        range(1, layer_count + 1),
        desc=f"{label:12s} {protein}",
        leave=False,
    ):
        for head in range(1, head_count + 1):

            treefile = path_function(
                protein,
                layer,
                head,
            )

            if treefile.exists():

                pcc = score_tree(
                    treefile
                )

                if not np.isnan(pcc):
                    values.append(pcc)

    return values


def print_head_summary(label, values):
    if values:

        print(
            f"  {label:22s} "
            f"n={len(values):3d}  "
            f"mean={np.mean(values):7.3f}  "
            f"median={np.median(values):7.3f}  "
            f"min={np.min(values):7.3f}  "
            f"max={np.max(values):7.3f}"
        )

    else:
        print(
            f"  {label:22s} no trees found"
        )


def print_consensus_summary(label, value):
    if np.isnan(value):
        print(
            f"  {label:22s} missing"
        )
    else:
        print(
            f"  {label:22s} {value:.3f}"
        )


# =============================================================================
# Collect layer/head and consensus PCC values
# =============================================================================

ram_violin_data = []
sam_violin_data = []

ram_consensus_scores = []
sam_consensus_scores = []


print("\n============================================================")
print("34-spp comparison: Hellinger RAM vs Hellinger SAM")
print("============================================================")


for protein in PROTEINS:

    ram_values = collect_head_scores(
        protein,
        ram_layerhead_path,
        RAM_LAYER_COUNT,
        RAM_HEAD_COUNT,
        "RAM",
    )

    sam_values = collect_head_scores(
        protein,
        sam_layerhead_path,
        SAM_LAYER_COUNT,
        SAM_HEAD_COUNT,
        "SAM",
    )

    ram_violin_data.append(
        ram_values
    )

    sam_violin_data.append(
        sam_values
    )

    ram_consensus = score_tree(
        ram_consensus_path(protein)
    )

    sam_consensus = score_tree(
        sam_consensus_path(protein)
    )

    ram_consensus_scores.append(
        ram_consensus
    )

    sam_consensus_scores.append(
        sam_consensus
    )

    print(f"\n{protein}")

    print_head_summary(
        "RAM heads:",
        ram_values,
    )

    print_head_summary(
        "SAM heads:",
        sam_values,
    )

    print_consensus_summary(
        "RAM consensus:",
        ram_consensus,
    )

    print_consensus_summary(
        "SAM consensus:",
        sam_consensus,
    )


# =============================================================================
# Overall summaries
# =============================================================================

def flatten(values_by_protein):
    return [
        value
        for protein_values in values_by_protein
        for value in protein_values
    ]


all_ram = flatten(
    ram_violin_data
)

all_sam = flatten(
    sam_violin_data
)


print("\n========================================")
print("Overall layer/head PCC")
print("========================================")

print_head_summary(
    "RAM:",
    all_ram,
)

print_head_summary(
    "SAM:",
    all_sam,
)


valid_ram_cons = [
    x
    for x in ram_consensus_scores
    if not np.isnan(x)
]

valid_sam_cons = [
    x
    for x in sam_consensus_scores
    if not np.isnan(x)
]


print("\n========================================")
print("Overall consensus PCC")
print("========================================")

print_head_summary(
    "RAM consensus:",
    valid_ram_cons,
)

print_head_summary(
    "SAM consensus:",
    valid_sam_cons,
)

# =============================================================================
# Statistical significance: RAM vs SAM
# =============================================================================

def paired_method_test(label, ram_values, sam_values):
    """
    Compare RAM vs SAM using paired observations.

    Each pair represents the same protein.

    Reports:
        - number of valid protein pairs
        - RAM and SAM means
        - mean paired difference (RAM - SAM)
        - 95% CI of the mean paired difference
        - paired t-test
        - Wilcoxon signed-rank test
    """

    ram_values = np.asarray(ram_values, dtype=float)
    sam_values = np.asarray(sam_values, dtype=float)

    valid = (
        ~np.isnan(ram_values)
        & ~np.isnan(sam_values)
    )

    ram = ram_values[valid]
    sam = sam_values[valid]

    if len(ram) < 2:
        print(f"\n{label}")
        print("  Not enough paired observations.")
        return

    differences = ram - sam

    mean_difference = np.mean(differences)
    sd_difference = np.std(differences, ddof=1)
    se_difference = sd_difference / np.sqrt(len(differences))

    # 95% CI using the t distribution
    from scipy.stats import t

    tcrit = t.ppf(
        0.975,
        df=len(differences) - 1,
    )

    ci_low = (
        mean_difference
        - tcrit * se_difference
    )

    ci_high = (
        mean_difference
        + tcrit * se_difference
    )

    # Paired t-test
    t_result = ttest_rel(
        ram,
        sam,
    )

    # Wilcoxon signed-rank test
    try:
        w_result = wilcoxon(
            ram,
            sam,
            alternative="two-sided",
        )

        w_stat = w_result.statistic
        w_p = w_result.pvalue

    except ValueError:
        w_stat = np.nan
        w_p = np.nan

    n_ram_better = np.sum(
        differences > 0
    )

    n_sam_better = np.sum(
        differences < 0
    )

    n_equal = np.sum(
        differences == 0
    )

    print("\n========================================")
    print(label)
    print("========================================")

    print(
        f"Valid paired proteins: {len(ram)}"
    )

    print(
        f"RAM mean:             {np.mean(ram):.3f}"
    )

    print(
        f"SAM mean:             {np.mean(sam):.3f}"
    )

    print(
        f"Mean difference:       {mean_difference:+.3f} PCC "
        f"(RAM - SAM)"
    )

    print(
        f"95% CI:                "
        f"[{ci_low:+.3f}, {ci_high:+.3f}]"
    )

    print(
        f"RAM better:            {n_ram_better}"
    )

    print(
        f"SAM better:            {n_sam_better}"
    )

    print(
        f"Equal:                 {n_equal}"
    )

    print("\nPaired t-test")

    print(
        f"  t = {t_result.statistic:.4f}"
    )

    print(
        f"  p = {t_result.pvalue:.6g}"
    )

    print("\nWilcoxon signed-rank test")

    print(
        f"  W = {w_stat:.4f}"
    )

    print(
        f"  p = {w_p:.6g}"
    )

    if mean_difference > 0:
        direction = "RAM"
    elif mean_difference < 0:
        direction = "SAM"
    else:
        direction = "Neither method"

    if t_result.pvalue < 0.05:
        print(
            f"\nConclusion: {direction} has significantly higher "
            f"PCC by paired t-test (p < 0.05)."
        )
    else:
        print(
            f"\nConclusion: no significant difference by paired "
            f"t-test (p >= 0.05)."
        )


# =============================================================================
# Compare mean layer/head PCC per protein
# =============================================================================

ram_protein_means = [
    np.mean(values)
    if values
    else np.nan
    for values in ram_violin_data
]

sam_protein_means = [
    np.mean(values)
    if values
    else np.nan
    for values in sam_violin_data
]


paired_method_test(
    "RAM vs SAM: mean layer/head PCC per protein",
    ram_protein_means,
    sam_protein_means,
)


# =============================================================================
# Compare protein-level ASTRAL consensus PCC
# =============================================================================

paired_method_test(
    "RAM vs SAM: ASTRAL consensus PCC",
    ram_consensus_scores,
    sam_consensus_scores,
)

# =============================================================================
# Plot helpers
# =============================================================================

def draw_violins(
    ax,
    data,
    positions,
    width,
    color,
):
    nonempty_data = []
    nonempty_positions = []

    for values, position in zip(
        data,
        positions,
    ):

        if values:

            nonempty_data.append(
                values
            )

            nonempty_positions.append(
                position
            )

    if not nonempty_data:
        return

    violin = ax.violinplot(
        nonempty_data,
        positions=nonempty_positions,
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body in violin["bodies"]:

        body.set_facecolor(
            color
        )

        body.set_edgecolor(
            "black"
        )

        body.set_linewidth(
            0.7
        )

        body.set_alpha(
            0.75
        )

    violin["cmedians"].set_color(
        "black"
    )

    violin["cmedians"].set_linewidth(
        1.1
    )


def draw_consensus_stars(
    ax,
    scores,
    positions,
    color,
):
    for i, score in enumerate(scores):

        if not np.isnan(score):

            ax.plot(
                positions[i],
                score,
                marker="*",
                linestyle="",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=10,
                zorder=5,
            )


# =============================================================================
# Plot positions
# =============================================================================

x = np.arange(
    len(PROTEINS)
)

ram_positions = (
    x - 0.14
)

sam_positions = (
    x + 0.14
)

VIOLIN_WIDTH = 0.25


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12, 4.5)
)


draw_violins(
    ax,
    ram_violin_data,
    ram_positions,
    VIOLIN_WIDTH,
    RAM_COLOR,
)

draw_violins(
    ax,
    sam_violin_data,
    sam_positions,
    VIOLIN_WIDTH,
    SAM_COLOR,
)


draw_consensus_stars(
    ax,
    ram_consensus_scores,
    ram_positions,
    RAM_COLOR,
)

draw_consensus_stars(
    ax,
    sam_consensus_scores,
    sam_positions,
    SAM_COLOR,
)


# Vertical boundaries between proteins

for boundary in np.arange(
    -0.5,
    len(PROTEINS) + 0.5,
    1,
):

    ax.axvline(
        boundary,
        color="lightgray",
        linewidth=0.7,
        zorder=0,
    )


# =============================================================================
# Axes
# =============================================================================

ax.set_xlim(
    -0.5,
    len(PROTEINS) - 0.5,
)

ax.set_ylim(
    -3,
    103,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    PROTEINS,
    rotation=45,
    ha="right",
    fontsize=14,
)

ax.set_yticks(
    np.arange(
        0,
        101,
        20,
    )
)

ax.tick_params(
    axis="y",
    labelsize=14,
)

ax.set_ylabel(
    "PCC (%)",
    fontsize=16,
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5,
    zorder=0,
)

ax.set_axisbelow(
    True
)


# =============================================================================
# Legend
# =============================================================================

ram_patch = mpatches.Patch(
    facecolor=RAM_COLOR,
    edgecolor="black",
    alpha=0.75,
    label="$\it{RAM}$ heads",
)

sam_patch = mpatches.Patch(
    facecolor=SAM_COLOR,
    edgecolor="black",
    alpha=0.75,
    label="$\it{SAM}$ heads",
)

ram_star = mlines.Line2D(
    [],
    [],
    marker="*",
    linestyle="",
    color=RAM_COLOR,
    markeredgecolor="black",
    markersize=10,
    label="$\it{RAM}$ consensus",
)

sam_star = mlines.Line2D(
    [],
    [],
    marker="*",
    linestyle="",
    color=SAM_COLOR,
    markeredgecolor="black",
    markersize=10,
    label="$\it{SAM}$ consensus",
)


ax.legend(
    handles=[
        ram_patch,
        sam_patch,
        ram_star,
        sam_star,
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.19),
    ncol=4,
    frameon=False,
    fontsize=14,
)


# =============================================================================
# Save
# =============================================================================



plt.tight_layout()

for ext in ("pdf", "svg"):
    output_path = os.path.join(OUT_PLOT_DIR, f"{OUT_PLOT}.{ext}")
    plt.savefig(output_path, dpi = 500)
    print(f"Combined plot saved to: {output_path}")   
    
plt.show()

plt.close()