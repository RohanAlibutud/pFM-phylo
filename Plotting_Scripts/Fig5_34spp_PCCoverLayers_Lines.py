#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr, wilcoxon

# Extend path for rf_distance
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from RF_scorer import rf_distance


# ============================================================
# Proteins
# ============================================================

OMAM20_proteins = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH", "DYRK1B", "GPX3", "HAUS1",
    "IL6", "INO80B", "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO", "SLC39A12",
    "TF", "TPPP2", "UBA2", "UPRT"
]


# ============================================================
# Parameters
# ============================================================

protein_set = "34spp"
num_species = 34

supertree = "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

OUT_DIR = "/home/rohan/pFM-phylo/Figures/Submission_Figures"
OUT_STEM = "Fig5_PC_Layers_Hellinger_RAM_vs_SAM"

out_pdf = os.path.join(OUT_DIR, f"{OUT_STEM}.pdf")
out_svg = os.path.join(OUT_DIR, f"{OUT_STEM}.svg")
out_png = os.path.join(OUT_DIR, f"{OUT_STEM}.png")

DPI = 500

ram_layer_count, ram_head_count = 12, 20
sam_layer_count, sam_head_count = 12, 12


# ============================================================
# Tree paths
# ============================================================

def ram_tree_path(protein, layer, head):
    return (
        f"/home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees/"
        f"{protein}_{protein_set}/layer{layer}/"
        f"{protein}_{protein_set}_layer{layer}_head{head}_tree.nwk"
    )


def sam_tree_path(protein, layer, head):
    return (
        f"/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees/"
        f"{protein}_{protein_set}Aligned/layer{layer}/"
        f"{protein}_{protein_set}Aligned_layer{layer}_head{head}_tree.nwk"
    )


# ============================================================
# RF -> PCC
# ============================================================

def calc_PC(RF, num_species=34):
    return round((1 - (RF / 2) / (num_species - 3)) * 100, 1)


# ============================================================
# Build per-protein PCC tensor
# Shape = protein x layer x head
# ============================================================

def build_pc_tensor(proteins, layer_count, head_count, tree_path_fn, label):
    pcs = np.full((len(proteins), layer_count, head_count), np.nan)
    found = 0
    missing = 0

    for pi, protein in enumerate(proteins):
        for layer in tqdm(
            range(1, layer_count + 1),
            desc=f"{label} - {protein}",
            leave=False
        ):
            for head in range(1, head_count + 1):
                tree_path = tree_path_fn(protein, layer, head)

                if not os.path.exists(tree_path):
                    missing += 1
                    continue

                rf = rf_distance(supertree, tree_path)
                pc = 100.0 if rf == 0 else calc_PC(rf, num_species)

                pcs[pi, layer - 1, head - 1] = pc
                found += 1

    print(f"{label}: found={found:,}, missing={missing:,}")

    return pcs


# ============================================================
# Mean PCC per layer
# ============================================================

def consensus_pc_by_layer(pc_tensor):
    """
    pc_tensor shape:
        proteins x layers x heads

    Step 1:
        Average heads within each protein/layer.

    Step 2:
        Average those protein-specific layer values across proteins.

    Returns:
        mean_pc:  mean PCC for each layer
        sem_pc:   standard error across proteins
        layer_pc: individual protein layer means
    """

    # Protein x Layer
    layer_pc = np.nanmean(pc_tensor, axis=2)

    # Mean across proteins
    mean_pc = np.nanmean(layer_pc, axis=0)

    # SEM across proteins
    n = np.sum(np.isfinite(layer_pc), axis=0)
    std_pc = np.nanstd(layer_pc, axis=0, ddof=1)
    sem_pc = std_pc / np.sqrt(n)

    return mean_pc, sem_pc, layer_pc


# ============================================================
# Plotting
# ============================================================

def plot_layer_pc(ram_mean, ram_sem, sam_mean, sam_sem):
    layers = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(6, 4))

    # RAM
    ax.plot(
        layers,
        ram_mean,
        color="dodgerblue",
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=6,
        label="RAM"
    )

    ax.fill_between(
        layers,
        ram_mean - ram_sem,
        ram_mean + ram_sem,
        color="dodgerblue",
        alpha=0.2
    )

    # SAM
    ax.plot(
        layers,
        sam_mean,
        color="orange",
        marker="D",
        linestyle="--",
        linewidth=2.5,
        markersize=6,
        label="SAM"
    )

    ax.fill_between(
        layers,
        sam_mean - sam_sem,
        sam_mean + sam_sem,
        color="orange",
        alpha=0.2
    )

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Mean PCC (%)", fontsize=14)

    ax.set_xticks(layers)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_axisbelow(True)
    ax.grid(alpha=0.5)

    leg = ax.legend(frameon=False, fontsize=12)

    for text in leg.get_texts():
        text.set_fontstyle("italic")

    fig.tight_layout()

    return fig, ax


# ============================================================
# Layer progression significance test
# ============================================================

def test_layer_trend(layer_pc, label):
    """
    layer_pc shape:
        proteins x layers

    For each protein:
        Spearman correlation between layer number and mean PCC.

    Across proteins:
        Wilcoxon signed-rank test of rho against 0.

    Positive rho = PCC tends to increase across layers.
    Negative rho = PCC tends to decrease across layers.
    """

    layers = np.arange(1, layer_pc.shape[1] + 1)
    rhos = []

    for protein_pc in layer_pc:
        mask = np.isfinite(protein_pc)

        if mask.sum() < 3:
            continue

        rho, _ = spearmanr(layers[mask], protein_pc[mask])

        if np.isfinite(rho):
            rhos.append(rho)

    rhos = np.asarray(rhos)

    if len(rhos) < 2:
        print(f"\n{label}: insufficient data for trend test")
        return np.array([]), np.nan

    statistic, pvalue = wilcoxon(rhos, alternative="two-sided")

    median_rho = np.median(rhos)
    mean_rho = np.mean(rhos)

    if median_rho > 0:
        direction = "INCREASE"
    elif median_rho < 0:
        direction = "DECREASE"
    else:
        direction = "NO DIRECTION"

    print(f"\n=== {label} layer progression test ===")
    print(f"Proteins tested: {len(rhos)}")
    print(f"Mean Spearman rho:   {mean_rho:+.3f}")
    print(f"Median Spearman rho: {median_rho:+.3f}")
    print(f"Direction: {direction}")
    print(f"Wilcoxon statistic: {statistic:.3f}")
    print(f"Wilcoxon p-value: {pvalue:.6g}")

    if pvalue < 0.05:
        print(f"Result: significant {direction.lower()} in PCC across layers")
    else:
        print("Result: no significant directional change in PCC across layers")

    return rhos, pvalue


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Build PCC tensors
    # --------------------------------------------------------

    ram_tensor = build_pc_tensor(
        proteins=OMAM20_proteins,
        layer_count=ram_layer_count,
        head_count=ram_head_count,
        tree_path_fn=ram_tree_path,
        label="Hellinger RAM",
    )

    sam_tensor = build_pc_tensor(
        proteins=OMAM20_proteins,
        layer_count=sam_layer_count,
        head_count=sam_head_count,
        tree_path_fn=sam_tree_path,
        label="Hellinger SAM",
    )

    # --------------------------------------------------------
    # Mean PCC for each layer
    # --------------------------------------------------------

    ram_mean, ram_sem, ram_layer_pc = consensus_pc_by_layer(ram_tensor)
    sam_mean, sam_sem, sam_layer_pc = consensus_pc_by_layer(sam_tensor)

    # --------------------------------------------------------
    # Test whether PCC changes systematically across layers
    # --------------------------------------------------------

    ram_rhos, ram_p = test_layer_trend(
        ram_layer_pc,
        label="Hellinger RAM"
    )

    sam_rhos, sam_p = test_layer_trend(
        sam_layer_pc,
        label="Hellinger SAM"
    )

    # --------------------------------------------------------
    # Print values
    # --------------------------------------------------------

    print("\n=== Hellinger RAM mean PCC by layer ===")

    for layer, pc, sem in zip(
        range(1, ram_layer_count + 1),
        ram_mean,
        ram_sem
    ):
        print(f"Layer {layer:2d}: {pc:.3f} +/- {sem:.3f}")

    print("\n=== Hellinger SAM mean PCC by layer ===")

    for layer, pc, sem in zip(
        range(1, sam_layer_count + 1),
        sam_mean,
        sam_sem
    ):
        print(f"Layer {layer:2d}: {pc:.3f} +/- {sem:.3f}")

    # --------------------------------------------------------
    # Plot RAM and SAM together
    # --------------------------------------------------------

    fig, ax = plot_layer_pc(
        ram_mean=ram_mean,
        ram_sem=ram_sem,
        sam_mean=sam_mean,
        sam_sem=sam_sem,
    )

    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_svg, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    
    print(f"\nSaved PDF: {out_pdf}")
    print(f"Saved SVG: {out_svg}")
    print(f"Saved PNG: {out_png}")

    plt.show()


if __name__ == "__main__":
    main()