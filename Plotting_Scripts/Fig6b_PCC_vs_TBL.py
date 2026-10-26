#!/usr/bin/env python3
"""
Figure 6b: Relationship between phylogenetic accuracy (PCC) and evolutionary
rate, represented by total branch length (TBL) of the ML+MFP tree.

For each of the 20 OMAM proteins:
    - Calculate total branch length from the ML+MFP tree.
    - Calculate PCC relative to the OMAM reference phylogeny for:
        RAM
        SAM
        ML+MFP
    - Calculate Pearson correlation between ML total branch length and PCC.
    - Plot all three methods with linear regression lines.

Outputs PDF, SVG, and 500-DPI PNG.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from Bio import Phylo

# =============================================================================
# Import RF scorer
# =============================================================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from RF_scorer import rf_distance

# =============================================================================
# Configuration
# =============================================================================

SUPERTREE = "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

RAM_DIR = "/home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees"
SAM_DIR = "/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees"
ML_DIR = "/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_34sppAligned"

FIGURE_DIR = "/home/rohan/pFM-phylo/Figures/Submission_Figures"

OUTPUT_PDF = os.path.join(FIGURE_DIR, "Fig6b_PCC_vs_ML_total_branch_length.pdf")
OUTPUT_SVG = os.path.join(FIGURE_DIR, "Fig6b_PCC_vs_ML_total_branch_length.svg")
OUTPUT_PNG = os.path.join(FIGURE_DIR, "Fig6b_PCC_vs_ML_total_branch_length.png")

OUTPUT_CSV = "/home/rohan/pFM-phylo/Results/Plots/Fig6b_PCC_vs_ML_total_branch_length.csv"

DPI = 500
NUM_SPECIES = 34

PROTEINS = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH",
    "DYRK1B", "GPX3", "HAUS1", "IL6", "INO80B",
    "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO",
    "SLC39A12", "TF", "TPPP2", "UBA2", "UPRT",
]

# =============================================================================
# Figure settings
# =============================================================================

TICK_SIZE = 10
LABEL_SIZE = 14
LEGEND_SIZE = 10
STATS_SIZE = 10

AXES_WIDTH = 7.0
AXES_HEIGHT = 2.0

LEFT_MARGIN = 0.80
RIGHT_MARGIN = 0.20
BOTTOM_MARGIN = 0.60
TOP_MARGIN = 0.20

FIG_WIDTH = LEFT_MARGIN + AXES_WIDTH + RIGHT_MARGIN
FIG_HEIGHT = BOTTOM_MARGIN + AXES_HEIGHT + TOP_MARGIN

POINT_SIZE = 42
LINEWIDTH = 1.5

# =============================================================================
# Paths
# =============================================================================

def ram_tree_path(protein):
    domain = f"{protein}_34spp"
    return os.path.join(
        RAM_DIR,
        domain,
        "Consensus_Trees",
        f"{domain}_RAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


def sam_tree_path(protein):
    domain = f"{protein}_34sppAligned"
    return os.path.join(
        SAM_DIR,
        domain,
        "Consensus_Trees",
        f"{domain}_SAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


def ml_tree_path(protein):
    return os.path.join(
        ML_DIR,
        f"{protein}_34sppAligned.fas.treefile"
    )

# =============================================================================
# Validation
# =============================================================================

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def validate_inputs():
    require_file(SUPERTREE)

    missing = []

    for protein in PROTEINS:
        for method, path in (
            ("RAM", ram_tree_path(protein)),
            ("SAM", sam_tree_path(protein)),
            ("ML+MFP", ml_tree_path(protein)),
        ):
            if not os.path.isfile(path):
                missing.append(f"{protein} | {method} | {path}")

    if missing:
        raise FileNotFoundError(
            "Missing required tree files:\n" + "\n".join(missing)
        )

# =============================================================================
# PCC
# =============================================================================

def calc_pcc(rf):
    return (1.0 - (rf / 2.0) / (NUM_SPECIES - 3)) * 100.0


def tree_pcc(tree_path):
    rf = rf_distance(SUPERTREE, tree_path)
    return 100.0 if rf == 0 else calc_pcc(rf)

# =============================================================================
# ML total branch length
# =============================================================================

def total_branch_length(tree_path):
    tree = Phylo.read(tree_path, "newick")

    lengths = [
        clade.branch_length
        for clade in tree.find_clades()
        if clade is not tree.root
    ]

    if not lengths:
        raise RuntimeError(
            f"No branches found in ML tree: {tree_path}"
        )

    if any(length is None for length in lengths):
        raise RuntimeError(
            f"Missing branch length in ML tree: {tree_path}"
        )

    lengths = np.asarray(lengths, dtype=float)

    if not np.all(np.isfinite(lengths)):
        raise RuntimeError(
            f"Non-finite branch length in ML tree: {tree_path}"
        )

    if np.any(lengths < 0):
        raise RuntimeError(
            f"Negative branch length in ML tree: {tree_path}"
        )

    return float(lengths.sum())

# =============================================================================
# Build dataset
# =============================================================================

def build_dataset():
    rows = []

    print("\n=== Per-protein PCC and ML total branch length ===")

    for protein in PROTEINS:
        ram_path = ram_tree_path(protein)
        sam_path = sam_tree_path(protein)
        ml_path = ml_tree_path(protein)

        tbl = total_branch_length(ml_path)

        ram_pcc = tree_pcc(ram_path)
        sam_pcc = tree_pcc(sam_path)
        ml_pcc = tree_pcc(ml_path)

        rows.append({
            "protein": protein,
            "ml_total_branch_length": tbl,
            "ram_pcc": ram_pcc,
            "sam_pcc": sam_pcc,
            "ml_pcc": ml_pcc,
        })

        print(
            f"{protein:<10} "
            f"TBL={tbl:.6f}  "
            f"RAM={ram_pcc:.3f}  "
            f"SAM={sam_pcc:.3f}  "
            f"ML+MFP={ml_pcc:.3f}"
        )

    df = pd.DataFrame(rows)

    if len(df) != len(PROTEINS):
        raise RuntimeError(
            f"Expected {len(PROTEINS)} proteins; obtained {len(df)}"
        )

    return df

# =============================================================================
# Pearson correlation
# =============================================================================

def correlation(x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() != len(PROTEINS):
        raise RuntimeError(
            f"{label}: expected {len(PROTEINS)} finite pairs; "
            f"found {mask.sum()}"
        )

    if np.allclose(x[mask], x[mask][0]):
        raise RuntimeError(
            f"{label}: total branch lengths are constant."
        )

    if np.allclose(y[mask], y[mask][0]):
        raise RuntimeError(
            f"{label}: PCC values are constant."
        )

    r, p = pearsonr(x[mask], y[mask])

    print(f"\n=== {label} ===")
    print(f"Pearson r: {r:+.6f}")
    print(f"p-value:   {p:.6g}")
    print(f"n:         {mask.sum()}")

    return r, p

# =============================================================================
# Statistics formatting
# =============================================================================

def format_p(p):
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"

# =============================================================================
# Regression line
# =============================================================================

def add_regression(ax, x, y, color):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color=color,
        linestyle="--",
        linewidth=LINEWIDTH,
        zorder=2
    )

# =============================================================================
# Plot
# =============================================================================

def make_plot(df, ram_stats, sam_stats, ml_stats):
    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    ax = fig.add_axes([
        LEFT_MARGIN / FIG_WIDTH,
        BOTTOM_MARGIN / FIG_HEIGHT,
        AXES_WIDTH / FIG_WIDTH,
        AXES_HEIGHT / FIG_HEIGHT
    ])

    x = df["ml_total_branch_length"].to_numpy()

    # -------------------------------------------------------------------------
    # Points
    # -------------------------------------------------------------------------

    ax.scatter(
        x,
        df["ram_pcc"],
        color="dodgerblue",
        marker="o",
        s=POINT_SIZE,
        label=r"$\it{RAM}$",
        zorder=3
    )

    ax.scatter(
        x,
        df["sam_pcc"],
        color="orange",
        marker="D",
        s=POINT_SIZE,
        label=r"$\it{SAM}$",
        zorder=3
    )

    ax.scatter(
        x,
        df["ml_pcc"],
        color="green",
        marker="o",
        s=POINT_SIZE,
        label="ML+MFP",
        zorder=3
    )

    # -------------------------------------------------------------------------
    # Regression lines
    # -------------------------------------------------------------------------

    add_regression(
        ax,
        x,
        df["ram_pcc"],
        "dodgerblue"
    )

    add_regression(
        ax,
        x,
        df["sam_pcc"],
        "orange"
    )

    add_regression(
        ax,
        x,
        df["ml_pcc"],
        "green"
    )

    # -------------------------------------------------------------------------
    # Axis formatting
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "ML+MFP total branch length",
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        "PCC (%)",
        fontsize=LABEL_SIZE
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE
    )

    ax.set_axisbelow(True)

    ax.grid(
        alpha=0.5
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    ax.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower right"
    )

    # -------------------------------------------------------------------------
    # Correlation statistics
    # -------------------------------------------------------------------------

    ram_r, ram_p = ram_stats
    sam_r, sam_p = sam_stats
    ml_r, ml_p = ml_stats

    stats_text = (
        rf"$\it{{RAM}}$: $r$ = {ram_r:.2f}, $p$ = {format_p(ram_p)}"
        "\n"
        rf"$\it{{SAM}}$: $r$ = {sam_r:.2f}, $p$ = {format_p(sam_p)}"
        "\n"
        rf"ML+MFP: $r$ = {ml_r:.2f}, $p$ = {format_p(ml_p)}"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=STATS_SIZE,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            pad=3
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig.savefig(
        OUTPUT_PDF,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_SVG,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"\nPDF figure:\n{OUTPUT_PDF}")
    print(f"\nSVG figure:\n{OUTPUT_SVG}")
    print(f"\nPNG figure:\n{OUTPUT_PNG}")

# =============================================================================
# Main
# =============================================================================

def main():
    validate_inputs()

    df = build_dataset()

    ram_stats = correlation(
        df["ml_total_branch_length"],
        df["ram_pcc"],
        "RAM"
    )

    sam_stats = correlation(
        df["ml_total_branch_length"],
        df["sam_pcc"],
        "SAM"
    )

    ml_stats = correlation(
        df["ml_total_branch_length"],
        df["ml_pcc"],
        "ML+MFP"
    )

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(f"\nSaved numerical data:\n{OUTPUT_CSV}")

    make_plot(
        df,
        ram_stats,
        sam_stats,
        ml_stats
    )


if __name__ == "__main__":
    main()