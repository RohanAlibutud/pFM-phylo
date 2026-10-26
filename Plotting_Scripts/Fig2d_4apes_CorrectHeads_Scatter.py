#!/usr/bin/env python3
"""
Compare RAM vs SAM recovery of the accepted OMAM20 4-mammal phylogeny.

Each point represents one OMAM20 protein set.

X-axis:
    % of RAM layer/head phylogenies recovering the accepted topology

Y-axis:
    % of SAM layer/head phylogenies recovering the accepted topology

Trees are scored directly from the RAM and SAM Newick files.

Accepted 4-mammal split:

    (Homo_sapiens, Pan_troglodytes)
                  |
    (Gorilla_gorilla_gorilla, Pongo_abelii)

The topology check is root-independent: either side of the accepted
2-vs-2 bipartition may appear as a clade in the rooted Newick
representation.
"""

from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import Phylo
from scipy.stats import pearsonr


# ============================================================
# Paths
# ============================================================

RAM_DIR = Path("/home/rohan/pFM-phylo/Results/Trees/RAM/4apes/Trees")
SAM_DIR = Path("/home/rohan/pFM-phylo/Results/Trees/SAM/4apes/Trees")

OUT_DIR = Path("/home/rohan/pFM-phylo/Plots/RAM_vs_SAM/4apes")
OUT_PLOT_DIR = Path("/home/rohan/pFM-phylo/Figures/Submission_Figures")
OUT_CSV = OUT_DIR / "OMAM20_4apes_RAM_vs_SAM_correctness.csv"
OUT_HEAD_CSV = OUT_DIR / "OMAM20_4apes_RAM_vs_SAM_head_scores.csv"
OUT_PLOT = OUT_PLOT_DIR / "Fig2d_4apes_scatter"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Accepted topology
# ============================================================

EXPECTED_TAXA = {
    "Homo_sapiens",
    "Pan_troglodytes",
    "Gorilla_gorilla_gorilla",
    "Pongo_abelii",
}

HOMO_MACACA = {
    "Homo_sapiens",
    "Pan_troglodytes",
}

RATTUS_MONODELPHIS = {
    "Gorilla_gorilla_gorilla",
    "Pongo_abelii",
}


# ============================================================
# Helpers
# ============================================================

def normalize_taxon(name):
    name = str(name).strip().strip("'\"")
    name = re.sub(r"\s+", "_", name)
    return name


def normalize_domain(domain):
    """
    Normalize protein names between RAM and SAM.

    Examples:
        TF_4apes        -> TF
        TF_4apesAligned -> TF
        TF                 -> TF
    """
    domain = str(domain)

    domain = re.sub(
        r"_4apesAligned$",
        "",
        domain,
        flags=re.IGNORECASE,
    )

    domain = re.sub(
        r"_4apes$",
        "",
        domain,
        flags=re.IGNORECASE,
    )

    return domain


def get_domain_from_tree(tree_path):
    """
    Extract protein/domain name from a layer/head tree filename.

    Examples:
        TF_4apes_layer1_head1_tree.nwk
            -> TF

        TF_4apesAligned_layer1_head1_tree.nwk
            -> TF

        TF_layer1_head1_aDist_tree.nwk
            -> TF
    """
    name = tree_path.stem

    match = re.split(
        r"_layer\d+",
        name,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    domain = match[0]

    return normalize_domain(domain)


def get_layer_head(tree_path):
    name = tree_path.stem

    layer_match = re.search(
        r"layer[_-]?(\d+)",
        name,
        flags=re.IGNORECASE,
    )

    head_match = re.search(
        r"head[_-]?(\d+)",
        name,
        flags=re.IGNORECASE,
    )

    layer = int(layer_match.group(1)) if layer_match else np.nan
    head = int(head_match.group(1)) if head_match else np.nan

    return layer, head


def is_layer_head_tree(tree_path):
    """
    Keep only layer/head-specific trees.

    This excludes ASTRAL consensus trees and other Newick files
    that may also exist beneath the Trees directory.
    """
    name = tree_path.stem.lower()

    return (
        re.search(r"layer[_-]?\d+", name) is not None
        and re.search(r"head[_-]?\d+", name) is not None
    )


def score_tree(tree_path):
    """
    Return:
        True  = accepted 4-mammal topology recovered
        False = valid 4-taxon tree, but wrong/unresolved topology
        None  = tree could not be scored
    """
    try:
        tree = Phylo.read(tree_path, "newick")
    except Exception as exc:
        print(f"WARNING: Could not read {tree_path}")
        print(f"         {exc}")
        return None

    taxa = {
        normalize_taxon(t.name)
        for t in tree.get_terminals()
        if t.name is not None
    }

    if taxa != EXPECTED_TAXA:
        missing = EXPECTED_TAXA - taxa
        extra = taxa - EXPECTED_TAXA

        print(f"WARNING: Unexpected taxa in {tree_path}")

        if missing:
            print(f"         Missing: {sorted(missing)}")

        if extra:
            print(f"         Extra:   {sorted(extra)}")

        return None

    # Check every clade for either side of the accepted
    # Homo/Macaca | Rattus/Monodelphis bipartition.
    #
    # Checking both sides makes this independent of where
    # the Newick tree happened to be rooted.
    for clade in tree.find_clades(order="preorder"):
        descendants = {
            normalize_taxon(t.name)
            for t in clade.get_terminals()
            if t.name is not None
        }

        if descendants == HOMO_MACACA:
            return True

        if descendants == RATTUS_MONODELPHIS:
            return True

    return False


def scan_tree_directory(root, method):
    """
    Recursively scan a tree directory and score all layer/head trees.
    """
    if not root.exists():
        raise RuntimeError(f"Tree directory does not exist: {root}")

    tree_files = sorted(
        p for p in root.rglob("*.nwk")
        if is_layer_head_tree(p)
    )

    if not tree_files:
        raise RuntimeError(
            f"No layer/head Newick trees found under: {root}"
        )

    print(f"\n=== Scanning {method} ===")
    print(f"Directory: {root}")
    print(f"Layer/head trees found: {len(tree_files)}")

    rows = []

    for tree_path in tree_files:
        domain = get_domain_from_tree(tree_path)
        layer, head = get_layer_head(tree_path)
        correct = score_tree(tree_path)

        rows.append(
            {
                "method": method,
                "domain": domain,
                "layer": layer,
                "head": head,
                "correct": correct,
                "tree": str(tree_path),
            }
        )

    return pd.DataFrame(rows)


def summarize_correctness(df, method):
    scored = df[df["correct"].notna()].copy()

    summary = (
        df.groupby("domain")
        .agg(
            n_total=("correct", "size"),
            n_scored=("correct", lambda x: x.notna().sum()),
            n_correct=("correct", lambda x: (x == True).sum()),
        )
        .reset_index()
    )

    summary[f"{method}_pct_correct"] = (
        100.0
        * summary["n_correct"]
        / summary["n_scored"]
    )

    summary = summary.rename(
        columns={
            "n_total": f"{method}_n_total",
            "n_scored": f"{method}_n_scored",
            "n_correct": f"{method}_n_correct",
        }
    )

    return summary


# ============================================================
# Scan trees directly
# ============================================================

ram_heads = scan_tree_directory(RAM_DIR, "RAM")
sam_heads = scan_tree_directory(SAM_DIR, "SAM")

head_scores = pd.concat(
    [ram_heads, sam_heads],
    ignore_index=True,
)

head_scores.to_csv(
    OUT_HEAD_CSV,
    index=False,
)


# ============================================================
# Summarize by protein
# ============================================================

ram = summarize_correctness(
    ram_heads,
    "RAM",
)

sam = summarize_correctness(
    sam_heads,
    "SAM",
)

comparison = ram.merge(
    sam,
    on="domain",
    how="outer",
    indicator=True,
)

print("\n=== Dataset matching ===")

print(
    comparison[
        [
            "domain",
            "_merge",
        ]
    ].to_string(index=False)
)

missing = comparison[
    comparison["_merge"] != "both"
]

if not missing.empty:
    print(
        "\nWARNING: Some proteins were not found "
        "in both datasets:"
    )

    print(
        missing[
            [
                "domain",
                "_merge",
            ]
        ].to_string(index=False)
    )

comparison = comparison[
    comparison["_merge"] == "both"
].copy()

comparison = comparison.drop(
    columns="_merge"
)


# ============================================================
# Compare RAM vs SAM
# ============================================================

comparison["difference_SAM_minus_RAM"] = (
    comparison["SAM_pct_correct"]
    - comparison["RAM_pct_correct"]
)

comparison["better_method"] = np.select(
    [
        comparison["difference_SAM_minus_RAM"] > 0,
        comparison["difference_SAM_minus_RAM"] < 0,
    ],
    [
        "SAM",
        "RAM",
    ],
    default="Equal",
)

comparison = comparison.sort_values(
    "domain"
)

comparison.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Statistics
# ============================================================

x = comparison[
    "RAM_pct_correct"
].to_numpy()

y = comparison[
    "SAM_pct_correct"
].to_numpy()

valid = (
    np.isfinite(x)
    & np.isfinite(y)
)

x_valid = x[valid]
y_valid = y[valid]

if (
    len(x_valid) >= 2
    and np.std(x_valid) > 0
    and np.std(y_valid) > 0
):
    r, p = pearsonr(
        x_valid,
        y_valid,
    )
else:
    r, p = np.nan, np.nan

mean_ram = comparison[
    "RAM_pct_correct"
].mean()

mean_sam = comparison[
    "SAM_pct_correct"
].mean()

n_sam_better = (
    comparison["better_method"] == "SAM"
).sum()

n_ram_better = (
    comparison["better_method"] == "RAM"
).sum()

n_equal = (
    comparison["better_method"] == "Equal"
).sum()


# ============================================================
# Screen summary
# ============================================================

print("\n=== RAM vs SAM ===")

print(
    comparison[
        [
            "domain",
            "RAM_n_scored",
            "RAM_n_correct",
            "RAM_pct_correct",
            "SAM_n_scored",
            "SAM_n_correct",
            "SAM_pct_correct",
            "difference_SAM_minus_RAM",
            "better_method",
        ]
    ].to_string(index=False)
)

print(f"\nProteins compared: {len(comparison)}")

print(
    f"Mean RAM correctness : "
    f"{mean_ram:.2f}%"
)

print(
    f"Mean SAM correctness : "
    f"{mean_sam:.2f}%"
)

print(
    f"SAM better: "
    f"{n_sam_better}"
)

print(
    f"RAM better: "
    f"{n_ram_better}"
)

print(
    f"Equal     : "
    f"{n_equal}"
)

print(
    f"Pearson r : "
    f"{r:.3f}"
)

print(
    f"Pearson p : "
    f"{p:.3g}"
)


# ============================================================
# Per-protein tree counts
# ============================================================

print("\n=== RAM tree counts ===")

print(
    ram[
        [
            "domain",
            "RAM_n_total",
            "RAM_n_scored",
            "RAM_n_correct",
        ]
    ].to_string(index=False)
)

print("\n=== SAM tree counts ===")

print(
    sam[
        [
            "domain",
            "SAM_n_total",
            "SAM_n_scored",
            "SAM_n_correct",
        ]
    ].to_string(index=False)
)


# ============================================================
# Scatterplot
# ============================================================

fig, ax = plt.subplots(
    figsize=(4.5, 4)
)

ax.scatter(
    comparison["RAM_pct_correct"],
    comparison["SAM_pct_correct"],
    s=80,
    alpha=0.85,
    c="black",
    edgecolors="black",
    linewidths=0.7,
)

ax.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    linewidth=1.5,
    color="0.4",
)

ax.set_xlim(
    0,
    100,
)

ax.set_ylim(
    0,
    100,
)

ax.set_xlabel(
    "$\it{RAM}$ correct heads (%)",
    fontsize=18,
)

ax.set_ylabel(
    "$\it{SAM}$ correct heads (%)",
    fontsize=18,
)

ax.tick_params(
    axis="both",
    labelsize=14,
)

ax.grid(
    alpha=0.5
)

ax.set_axisbelow(
    True
)

fig.tight_layout()

# export jawns
for ext in ("pdf", "svg"):
    output_path = os.path.join(OUT_PLOT_DIR, f"{OUT_PLOT}.{ext}")
    fig.savefig(output_path, dpi = 500)
    print(f"Combined plot saved to: {output_path}")   
    
plt.show()
plt.close(fig)


# ============================================================
# Output paths
# ============================================================

print("\nHead-level scores written to:")
print(OUT_HEAD_CSV)

print("\nProtein comparison written to:")
print(OUT_CSV)

print("\nScatterplot written to:")
print(OUT_PLOT)