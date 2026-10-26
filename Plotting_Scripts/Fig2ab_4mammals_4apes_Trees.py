#!/usr/bin/env python3
"""Extract 4-mammal and 4-ape subtrees from a 34-species reference phylogeny.

Preserves phylogenetic relationships and branch lengths.
Exports publication-ready PDF and SVG figures with editable text.

Requires:
    pip install biopython matplotlib
"""

import copy
import math
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Bio import Phylo

# =============================================================================
# Input / output
# =============================================================================

REFERENCE_NWK = "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"

MAMMALS_PDF = "/home/rohan/pFM-phylo/Figures/Submission_Figures/Fig2a_4mammals_tree.pdf"
MAMMALS_SVG = "/home/rohan/pFM-phylo/Figures/Submission_Figures/Fig2a_4mammals_tree.svg"

APES_PDF = "/home/rohan/pFM-phylo/Figures/Submission_Figures/Fig2c_4apes_tree.pdf"
APES_SVG = "/home/rohan/pFM-phylo/Figures/Submission_Figures/Fig2c_4apes_tree.svg"

MAMMALS = ["Monodelphis_domestica", "Rattus_norvegicus", "Homo_sapiens", "Macaca_fascicularis"]
APES = ["Pongo_abelii", "Gorilla_gorilla_gorilla", "Homo_sapiens", "Pan_troglodytes"]

# =============================================================================
# Plot configuration
# =============================================================================

FIGSIZE = (6, 4.1)

LABEL_SIZE = 26
SCALE_SIZE = 14
BRANCH_WIDTH = 1.6

BRANCH_COLOR = "#353943"
SCALE_COLOR = "#8f939b"

ITALICIZE_TAXA = True
SHOW_SCALE = True

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

# =============================================================================
# Validate reference tree
# =============================================================================

def validate_reference(tree):
    names = [tip.name for tip in tree.get_terminals()]

    if any(not name for name in names):
        raise RuntimeError("The reference contains unnamed tips.")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate tip names in reference: {duplicates}")

    if len(names) != 34:
        raise RuntimeError(f"Expected 34 tips; found {len(names)}.")

    return set(names)


# =============================================================================
# Extract subtrees
# =============================================================================

def reorder_children(clade, rank):
    """Reorder tips without changing tree topology."""

    for child in clade.clades:
        reorder_children(child, rank)

    clade.clades.sort(key=lambda c: min(rank[t.name] for t in c.get_terminals()))


def make_subtree(reference, targets):
    subtree = copy.deepcopy(reference)

    for tip in list(subtree.get_terminals()):
        if tip.name not in targets:
            subtree.prune(tip)

    observed = [t.name for t in subtree.get_terminals()]

    if len(observed) != len(targets) or set(observed) != set(targets):
        raise RuntimeError(f"Pruning failed: expected {targets}, obtained {observed}")

    reorder_children(subtree.root, {name: i for i, name in enumerate(targets)})

    return subtree


# =============================================================================
# Validate branch lengths
# =============================================================================

def validate_lengths(reference, subtree, targets):
    lengths = [clade.branch_length for clade in subtree.find_clades() if clade is not subtree.root]

    if all(length is None for length in lengths):
        return False

    if any(length is None for length in lengths):
        raise RuntimeError(f"Partial branch lengths among {targets}.")

    if any(not math.isfinite(length) or length < 0 for length in lengths):
        raise RuntimeError(f"Invalid branch lengths among {targets}.")

    for a, b in combinations(targets, 2):
        before = reference.distance(a, b)
        after = subtree.distance(a, b)

        if not math.isclose(before, after, rel_tol=1e-8, abs_tol=1e-10):
            raise RuntimeError(f"Pruning changed the {a}/{b} distance: {before} -> {after}")

    return True


# =============================================================================
# Plot subtree
# =============================================================================

def plot_subtree(tree, title, has_lengths, output_pdf, output_svg):
    tips = tree.get_terminals()
    n = len(tips)

    x = {tree.root: 0.0}
    y = {tip: float(n - i - 1) for i, tip in enumerate(tips)}

    def assign_x(parent):
        for child in parent.clades:
            edge = float(child.branch_length) if has_lengths else 1.0
            x[child] = x[parent] + edge
            assign_x(child)

    def assign_y(node):
        if node.is_terminal():
            return y[node]

        y[node] = sum(assign_y(child) for child in node.clades) / len(node.clades)
        return y[node]

    assign_x(tree.root)
    assign_y(tree.root)

    max_x = max(x[tip] for tip in tips)

    if max_x <= 0:
        raise RuntimeError(f"{title}: all root-to-tip distances are zero.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw branches
    for parent in tree.find_clades(order="preorder"):
        if not parent.clades:
            continue

        children = parent.clades

        ax.plot(
            [x[parent], x[parent]],
            [min(y[c] for c in children), max(y[c] for c in children)],
            color=BRANCH_COLOR,
            linewidth=BRANCH_WIDTH,
            solid_capstyle="butt"
        )

        for child in children:
            ax.plot(
                [x[parent], x[child]],
                [y[child], y[child]],
                color=BRANCH_COLOR,
                linewidth=BRANCH_WIDTH,
                solid_capstyle="butt"
            )

    # Species labels
    for tip in tips:
        ax.text(
            x[tip] + 0.034 * max_x,
            y[tip],
            tip.name.replace("_", " "),
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            fontstyle="italic" if ITALICIZE_TAXA else "normal",
            color="black"
        )

    # Branch-length scale
    if has_lengths and SHOW_SCALE:
        baseline = -0.48

        ax.plot(
            [0, max_x],
            [baseline, baseline],
            color=SCALE_COLOR,
            linewidth=1.1
        )

        locator = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10])

        for tick in locator.tick_values(0, max_x):
            if tick <= 0 or tick >= max_x * (1 - 1e-8):
                continue

            ax.plot(
                [tick, tick],
                [baseline, baseline - 0.065],
                color=SCALE_COLOR,
                linewidth=1.1
            )

            ax.text(
                tick,
                baseline - 0.105,
                f"{tick:g}",
                ha="center",
                va="top",
                fontsize=SCALE_SIZE,
                color=SCALE_COLOR
            )

    # Axes
    ax.set_xlim(-0.035 * max_x, 2.15 * max_x)
    ax.set_ylim(-0.95 if has_lengths and SHOW_SCALE else -0.40, n - 1 + 0.38)
    ax.axis("off")

    fig.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.05)

    # Save PDF
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.12)
    print(f"Saved: {output_pdf}")

    # Save SVG
    os.makedirs(os.path.dirname(output_svg), exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight", pad_inches=0.12)
    print(f"Saved: {output_svg}")
    

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.isfile(REFERENCE_NWK):
        raise RuntimeError(f"Reference Newick not found: {REFERENCE_NWK}")

    reference = Phylo.read(REFERENCE_NWK, "newick")
    available = validate_reference(reference) 

    # Validate requested species
    for targets in (MAMMALS, APES):
        if len(targets) != len(set(targets)):
            raise RuntimeError(f"Duplicate requested species: {targets}")

        missing = sorted(set(targets) - available)
        if missing:
            raise RuntimeError(f"Species missing from reference: {missing}")

    # ==================== 4 MAMMALS ====================

    mammals_tree = make_subtree(reference, MAMMALS)
    mammals_lengths = validate_lengths(reference, mammals_tree, MAMMALS)

    print("\n4-mammal reference subtree:")
    print("  " + ", ".join(t.name for t in mammals_tree.get_terminals()))

    plot_subtree(
        mammals_tree,
        "4mammals",
        mammals_lengths,
        MAMMALS_PDF,
        MAMMALS_SVG
    )

    # ==================== 4 APES ====================

    apes_tree = make_subtree(reference, APES)
    apes_lengths = validate_lengths(reference, apes_tree, APES)

    print("\n4-ape reference subtree:")
    print("  " + ", ".join(t.name for t in apes_tree.get_terminals()))

    plot_subtree(
        apes_tree,
        "4apes",
        apes_lengths,
        APES_PDF,
        APES_SVG
    )


if __name__ == "__main__":
    main()