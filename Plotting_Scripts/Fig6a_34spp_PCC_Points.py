#!/usr/bin/env python3
"""
Per-protein comparison of tree accuracy for 34 species:
- Hellinger RAM (esm-pFM consensus)
- Hellinger SAM (msat-pFM consensus)
- ML+MFP (IQ-TREE)

Accuracy is measured as percent correct (PC/PCC) from RF distance:

    PC = (1 - (RF / 2) / (N - 3)) * 100

where N = 34 species.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Import RF scorer
# =============================================================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from RF_scorer import rf_distance

# =============================================================================
# Configuration
# =============================================================================

SUPERTREE = Path(
    "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"
)

RAM_DIR = Path(
    "/home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees"
)

SAM_DIR = Path(
    "/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees"
)

ML_DIR = Path(
    "/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_34sppAligned"
)

FIGURE_DIR = Path("/home/rohan/pFM-phylo/Figures/Submission_Figures")
OUT_PDF = FIGURE_DIR / "Fig6a_Hellinger_RAM_SAM_MLMFP_34spp_PCC_points.pdf"
OUT_SVG = FIGURE_DIR / "Fig6a_Hellinger_RAM_SAM_MLMFP_34spp_PCC_points.svg"
DPI = 500

PROTEINS = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH",
    "DYRK1B", "GPX3", "HAUS1", "IL6", "INO80B",
    "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO",
    "SLC39A12", "TF", "TPPP2", "UBA2", "UPRT",
]

NUM_SPECIES = 34

# =============================================================================
# Path helpers
# =============================================================================

def ram_tree_path(protein):
    domain = f"{protein}_34spp"

    return (
        RAM_DIR
        / domain
        / "Consensus_Trees"
        / f"{domain}_RAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


def sam_tree_path(protein):
    domain = f"{protein}_34sppAligned"

    return (
        SAM_DIR
        / domain
        / "Consensus_Trees"
        / f"{domain}_SAM_Hellinger_ASTRAL_consensus_tree.nwk"
    )


def ml_tree_path(protein):
    return ML_DIR / f"{protein}_34sppAligned.fas.treefile"


# =============================================================================
# Scoring
# =============================================================================

def calc_pcc(rf):
    return (1.0 - (rf / 2.0) / (NUM_SPECIES - 3)) * 100.0


def score_tree(treefile, protein, method):
    if not treefile.exists():
        print(f"{protein}: {method} tree NOT FOUND")
        print(f"  Expected: {treefile}")
        return np.nan

    try:
        rf = rf_distance(str(SUPERTREE), str(treefile))
        pcc = 100.0 if rf == 0 else calc_pcc(rf)

        print(
            f"{protein}: {method:<12} "
            f"RF={rf:<4} PCC={pcc:.3f}"
        )

        return pcc

    except Exception as e:
        print(f"{protein}: {method} FAILED: {e}")
        return np.nan


# =============================================================================
# Score trees
# =============================================================================

print("\n=== 34-spp PCC calculations ===")

ram_scores = pd.Series(
    {
        protein: score_tree(
            ram_tree_path(protein),
            protein,
            "RAM"
        )
        for protein in PROTEINS
    },
    name="RAM",
)

sam_scores = pd.Series(
    {
        protein: score_tree(
            sam_tree_path(protein),
            protein,
            "SAM"
        )
        for protein in PROTEINS
    },
    name="SAM",
)

ml_scores = pd.Series(
    {
        protein: score_tree(
            ml_tree_path(protein),
            protein,
            "ML+MFP"
        )
        for protein in PROTEINS
    },
    name="ML+MFP",
)


# =============================================================================
# Summary statistics
# =============================================================================

def summarize(label, scores):
    clean = scores.dropna()

    print(
        f"{label}: "
        f"mean={clean.mean():.3f}  "
        f"median={clean.median():.3f}  "
        f"min={clean.min():.3f}  "
        f"max={clean.max():.3f}  "
        f"n={len(clean)}"
    )


print("\n=== PCC across OMAM20 proteins (34 spp) ===")

summarize("RAM (esm-pFM)", ram_scores)
summarize("SAM (msat-pFM)", sam_scores)
summarize("ML+MFP", ml_scores)


# =============================================================================
# Plot
# =============================================================================

x = np.arange(len(PROTEINS))
point_offset = 0.18

TICK_SIZE = 10
AXES_WIDTH = 7.0
AXES_HEIGHT = 2.0

LEFT_MARGIN = 0.65
RIGHT_MARGIN = 0.15
BOTTOM_MARGIN = 0.95
TOP_MARGIN = 0.55

FIG_WIDTH = LEFT_MARGIN + AXES_WIDTH + RIGHT_MARGIN
FIG_HEIGHT = BOTTOM_MARGIN + AXES_HEIGHT + TOP_MARGIN

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax = fig.add_axes([
    LEFT_MARGIN / FIG_WIDTH,
    BOTTOM_MARGIN / FIG_HEIGHT,
    AXES_WIDTH / FIG_WIDTH,
    AXES_HEIGHT / FIG_HEIGHT
])

ax.scatter(
    x - point_offset,
    ram_scores,
    color="dodgerblue",
    s=45,
    label=r"$\it{RAM}$ (esm-pFM)",
    zorder=3,
)

ax.scatter(
    x,
    sam_scores,
    color="orange",
    s=45,
    label=r"$\it{SAM}$ (msat-pFM)",
    zorder=3,
)

ax.scatter(
    x + point_offset,
    ml_scores,
    color="green",
    s=45,
    label="ML+MFP",
    zorder=3,
)

# =============================================================================
# Background grid
# =============================================================================

for boundary in np.arange(-0.5, len(PROTEINS) + 0.5, 1):
    ax.axvline(
        boundary,
        color="lightgray",
        linewidth=0.8,
        zorder=0,
    )


# =============================================================================
# Plot formatting
# =============================================================================

ax.set_xlim(-0.5, len(PROTEINS) - 0.5)
ax.set_ylim(-3, 103)

ax.set_xticks(x)

import matplotlib.transforms as transforms

label_offset = transforms.ScaledTranslation(
    5 / 72,
    5 / 72,
    ax.figure.dpi_scale_trans
)

for label in ax.get_xticklabels():
    label.set_transform(label.get_transform() + label_offset)

ax.set_xticklabels(
    PROTEINS,
    rotation=45,
    ha="right",
    fontsize=TICK_SIZE,
)

ax.tick_params(axis="y", labelsize=TICK_SIZE)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel("PCC (%)", fontsize=14)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    ncol=3,
    frameon=False,
)


FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight")
plt.savefig(OUT_SVG, dpi=DPI, bbox_inches="tight")

plt.show()
plt.close()

print(f"\nWrote PDF: {OUT_PDF}")
print(f"Wrote SVG: {OUT_SVG}")