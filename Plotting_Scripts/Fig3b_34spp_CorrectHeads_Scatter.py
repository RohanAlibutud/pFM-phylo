#!/usr/bin/env python3
"""
Compare NEW Hellinger RAM vs SAM phylogenetic performance for the
OMAM20 34-species dataset using Percent Correct Clades (PCC).

Each point represents one OMAM20 protein.

X-axis:
    mean PCC across all RAM layer/head trees
    (both lower- and upper-triangle trees)

Y-axis:
    mean PCC across all SAM layer/head trees

New result locations:

RAM:
    /home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees

SAM:
    /home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees

PCC is calculated from RF distance to the OMAM supertree:

    PCC = [1 - (RF / 2) / (n_species - 3)] * 100

For the 34-species dataset, n_species = 34.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ============================================================
# RF scorer
# ============================================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from RF_scorer import rf_distance


# ============================================================
# Configuration
# ============================================================

OMAM20_PROTEINS = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH", "DYRK1B", "GPX3", "HAUS1",
    "IL6", "INO80B", "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO", "SLC39A12",
    "TF", "TPPP2", "UBA2", "UPRT",
]

NUM_SPECIES = 34

RAM_LAYER_COUNT = 12
RAM_HEAD_COUNT = 20

SAM_LAYER_COUNT = 12
SAM_HEAD_COUNT = 12

INCLUDE_RAM_UPPER = True

SUPERTREE = "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

RAM_ROOT = "/home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees"
SAM_ROOT = "/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees"

OUT_DIR = "/home/rohan/pFM-phylo/Plots/ESM_vs_MSAT/OMAM20_34spp"
OUT_PLOT_DIR = "/home/rohan/pFM-phylo/Figures/Submission_Figures"
OUT_PLOT = "Fig3b_34spp_CorrectHeads_Scatter"
OUT_CSV = os.path.join(
    OUT_DIR,
    "OMAM20_34spp_NewHellinger_RAM_vs_SAM_PCC.csv",
)


os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Tree paths
# ============================================================

def ram_tree_paths(protein, layer, head):
    """
    Return RAM tree paths for one layer/head.

    New RAM structure:

        RAM/34spp/Trees/
            TF_34spp/
                layer1/
                    TF_34spp_layer1_head1_tree.nwk
                    TF_34spp_layer1_head1_upper_tree.nwk
    """

    domain = f"{protein}_34spp"

    layer_dir = os.path.join(
        RAM_ROOT,
        domain,
        f"layer{layer}",
    )

    paths = [
        os.path.join(
            layer_dir,
            f"{domain}_layer{layer}_head{head}_tree.nwk",
        )
    ]

    if INCLUDE_RAM_UPPER:
        paths.append(
            os.path.join(
                layer_dir,
                f"{domain}_layer{layer}_head{head}_upper_tree.nwk",
            )
        )

    return paths


def sam_tree_paths(protein, layer, head):
    """
    Return SAM tree path for one layer/head.

    New SAM structure:

        SAM/34spp/Trees/
            TF_34sppAligned/
                layer1/
                    TF_34sppAligned_layer1_head1_tree.nwk
    """

    domain = f"{protein}_34sppAligned"

    return [
        os.path.join(
            SAM_ROOT,
            domain,
            f"layer{layer}",
            f"{domain}_layer{layer}_head{head}_tree.nwk",
        )
    ]


# ============================================================
# PCC calculation
# ============================================================

def calc_pcc(rf, num_species=NUM_SPECIES):
    return (1 - (rf / 2) / (num_species - 3)) * 100


def pcc_from_tree(tree_path):
    rf = rf_distance(
        SUPERTREE,
        tree_path,
    )

    return calc_pcc(rf)


def collect_tree_pcc(protein, layer_count, head_count, path_fn):
    values = []
    missing = []

    for layer in range(1, layer_count + 1):
        for head in range(1, head_count + 1):

            paths = path_fn(
                protein,
                layer,
                head,
            )

            for path in paths:

                if not os.path.exists(path):
                    missing.append(path)
                    continue

                values.append(
                    pcc_from_tree(path)
                )

    return values, missing


# ============================================================
# Calculate protein-level RAM and SAM PCC
# ============================================================

rows = []

ram_trees_per_head = 2 if INCLUDE_RAM_UPPER else 1

expected_ram = (
    RAM_LAYER_COUNT
    * RAM_HEAD_COUNT
    * ram_trees_per_head
)

expected_sam = (
    SAM_LAYER_COUNT
    * SAM_HEAD_COUNT
)


for protein in OMAM20_PROTEINS:

    print(f"\n=== {protein} ===")

    ram_values, ram_missing = collect_tree_pcc(
        protein,
        RAM_LAYER_COUNT,
        RAM_HEAD_COUNT,
        ram_tree_paths,
    )

    sam_values, sam_missing = collect_tree_pcc(
        protein,
        SAM_LAYER_COUNT,
        SAM_HEAD_COUNT,
        sam_tree_paths,
    )

    ram_mean = (
        np.mean(ram_values)
        if ram_values
        else np.nan
    )

    sam_mean = (
        np.mean(sam_values)
        if sam_values
        else np.nan
    )

    if ram_values:
        print(
            f"RAM: n={len(ram_values):3d}/{expected_ram}, "
            f"mean PCC={ram_mean:.3f}%"
        )
    else:
        print(
            f"RAM: n=0/{expected_ram}"
        )

    if sam_values:
        print(
            f"SAM: n={len(sam_values):3d}/{expected_sam}, "
            f"mean PCC={sam_mean:.3f}%"
        )
    else:
        print(
            f"SAM: n=0/{expected_sam}"
        )

    if ram_missing:
        print(
            f"RAM missing trees: {len(ram_missing)}"
        )

    if sam_missing:
        print(
            f"SAM missing trees: {len(sam_missing)}"
        )

    rows.append(
        {
            "protein": protein,
            "RAM_n_trees": len(ram_values),
            "SAM_n_trees": len(sam_values),
            "RAM_expected_trees": expected_ram,
            "SAM_expected_trees": expected_sam,
            "RAM_mean_PCC": ram_mean,
            "SAM_mean_PCC": sam_mean,
            "SAM_minus_RAM_PCC": sam_mean - ram_mean,
        }
    )


comparison = pd.DataFrame(
    rows
)

comparison["better_method"] = np.select(
    [
        comparison["SAM_minus_RAM_PCC"] > 0,
        comparison["SAM_minus_RAM_PCC"] < 0,
    ],
    [
        "SAM",
        "RAM",
    ],
    default="Equal",
)

comparison.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Statistics
# ============================================================

plot_df = comparison.dropna(
    subset=[
        "RAM_mean_PCC",
        "SAM_mean_PCC",
    ]
).copy()

x = plot_df[
    "RAM_mean_PCC"
].to_numpy()

y = plot_df[
    "SAM_mean_PCC"
].to_numpy()


if len(plot_df) >= 2:
    r, p = pearsonr(
        x,
        y,
    )
else:
    r, p = np.nan, np.nan


mean_ram = plot_df[
    "RAM_mean_PCC"
].mean()

mean_sam = plot_df[
    "SAM_mean_PCC"
].mean()

n_ram_better = (
    plot_df["better_method"]
    == "RAM"
).sum()

n_sam_better = (
    plot_df["better_method"]
    == "SAM"
).sum()

n_equal = (
    plot_df["better_method"]
    == "Equal"
).sum()


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("NEW HELLINGER RAM vs SAM mean PCC")
print("=" * 80)

print(
    plot_df[
        [
            "protein",
            "RAM_n_trees",
            "SAM_n_trees",
            "RAM_mean_PCC",
            "SAM_mean_PCC",
            "SAM_minus_RAM_PCC",
            "better_method",
        ]
    ].to_string(
        index=False
    )
)

print(f"\nProteins compared: {len(plot_df)}")
print(f"mean RAM PCC: {mean_ram:.3f}%")
print(f"mean SAM PCC: {mean_sam:.3f}%")
print(f"RAM better:   {n_ram_better}")
print(f"SAM better:   {n_sam_better}")
print(f"Equal:        {n_equal}")
print(f"Pearson r:    {r:.3f}")
print(f"Pearson p:    {p:.3g}")

print("\nExpected trees per complete protein:")
print(f"RAM: {expected_ram}")
print(f"SAM: {expected_sam}")


# ============================================================
# Scatterplot
# ============================================================

fig, ax = plt.subplots(
    figsize=(4.4, 4)
)

ax.scatter(
    plot_df["RAM_mean_PCC"],
    plot_df["SAM_mean_PCC"],
    s=80,
    alpha=0.85,
    c="black",
    edgecolors="black",
    linewidths=0.7,
)


# Optional protein labels
#
# for _, row in plot_df.iterrows():
#     ax.annotate(
#         row["protein"],
#         (
#             row["RAM_mean_PCC"],
#             row["SAM_mean_PCC"],
#         ),
#         xytext=(5, 5),
#         textcoords="offset points",
#         fontsize=9,
#     )


# 1:1 line

ax.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    linewidth=1.5,
    color="0.4",
)


ax.set_xlim(
    0,
    60,
)

ax.set_ylim(
    0,
    60,
)

ax.set_xlabel(
    "$\it{RAM}$ mean PCC (%)",
    fontsize=14,
)

ax.set_ylabel(
    "$\it{SAM}$ mean PCC (%)",
    fontsize=14,
)

ax.tick_params(
    axis="both",
    labelsize=12,
)

ax.grid(
    alpha=0.5
)

ax.set_axisbelow(
    True
)


# Optional statistics annotation
#
# ax.text(
#     0.03,
#     0.97,
#     f"Pearson r = {r:.3f}\n"
#     f"p = {p:.3g}\n"
#     f"n = {len(plot_df)} proteins",
#     transform=ax.transAxes,
#     ha="left",
#     va="top",
#     fontsize=10,
# )


fig.tight_layout()

for ext in ("pdf", "svg"):
    output_path = os.path.join(OUT_PLOT_DIR, f"{OUT_PLOT}.{ext}")
    fig.savefig(output_path, dpi = 500)
    print(f"Combined plot saved to: {output_path}")   

plt.show()

plt.close(
    fig
)


# ============================================================
# Output
# ============================================================

print("\nComparison CSV written to:")
print(OUT_CSV)

print("\nScatterplot written to:")
print(OUT_PLOT)