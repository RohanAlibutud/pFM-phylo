#!/usr/bin/env python3
"""
Three compact OMAM tanglegrams: reference vs RAM, SAM, and ML+MFP consensus.
Each comparison tree is annotated with ASTRAL local posterior probabilities.

ASTRAL scores the EXISTING consensus topology against the 20 constituent
protein trees (-q -t 3); no consensus tree is reconstructed. Tip rotations
minimize connector crossings without altering topology. All connectors are
neutral gray; no singleton-error identification is performed.

PDF and SVG figures are saved to FIGURE_DIR. Branch-support tables, input-tree
manifests and ASTRAL logs are saved to OUT_DIR.
"""

import copy
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo

# =============================================================================
# INPUTS / OUTPUTS
# =============================================================================

REFERENCE_TREE = "/home/rohan/pFM-phylo/Data/Trees/OMAM_supertree_pruned_34spp.nwk"
RAM_TREE = "/home/rohan/pFM-phylo/Results/Trees/Consensus/RAM_20protein_ASTRAL_consensus_of_consensus_ROOTED.nwk"
SAM_TREE = "/home/rohan/pFM-phylo/Results/Trees/Consensus/SAM_20protein_ASTRAL_consensus_of_consensus_ROOTED.nwk"
ML_TREE = "/home/rohan/pFM-phylo/Results/Trees/Consensus/ML_MFP_20protein_consensus_of_consensus.nwk"

RAM_ROOT = "/home/rohan/pFM-phylo/Results/Trees/RAM/34spp/Trees"
SAM_ROOT = "/home/rohan/pFM-phylo/Results/Trees/SAM/34spp/Trees"
ML_ROOT = "/home/rohan/ESM/Results/OMAM20/ML_MFP/OMAM20_34sppAligned"

ASTRAL_JAR = "/home/rohan/Tools/Astral.5.7.8/Astral/astral.5.7.8.jar"

# TSVs, manifests, ASTRAL logs, etc.
OUT_DIR = "/home/rohan/pFM-phylo/Results/Plots/FigS2_Tanglegrams"

# PDF/SVG figures only.
FIGURE_DIR = "/home/rohan/pFM-phylo/Figures/Submission_Figures"

PROTEINS = ("AIMP2", "APOBEC2", "AR", "CASQ1", "COCH", "DYRK1B", "GPX3", "HAUS1", "IL6", "INO80B",
            "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO", "SLC39A12", "TF", "TPPP2", "UBA2", "UPRT")

METHODS = (("RAM", RAM_TREE), ("SAM", SAM_TREE), ("ML+MFP", ML_TREE))
OUTGROUP = "Monodelphis_domestica"

# =============================================================================
# FIGURE SETTINGS
# =============================================================================

FIGSIZE = (12, 8)
DPI = 300

LEFT_ROOT_X, LEFT_TIP_X = 0.025, 0.18
RIGHT_TIP_X, RIGHT_ROOT_X = 0.685, 1

TREE_LINEWIDTH = 1.15
CONNECTOR_LINEWIDTH, CONNECTOR_ALPHA = 0.75, 0.50
CONNECTOR_GAP_POINTS = 4.0

LABEL_FONTSIZE = 12
SUPPORT_FONTSIZE = 9
TITLE_FONTSIZE = 14
SUPPORT_OFFSET_POINTS = 2.0

MAX_EXACT_CHILDREN = 16
SHOW_PLOTS = False

# =============================================================================
# FILE DISCOVERY / VALIDATION
# =============================================================================

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def read_tree(path):
    require_file(path)
    tree = Phylo.read(path, "newick")
    names = [tip.name for tip in tree.get_terminals()]

    if any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError(f"Missing or duplicated taxon names: {path}")

    return tree


def get_taxa(tree):
    return {tip.name for tip in tree.get_terminals()}


def check_taxa(reference, comparison, name):
    a, b = get_taxa(reference), get_taxa(comparison)

    if a != b:
        raise RuntimeError(
            f"Taxon mismatch for {name}: "
            f"missing={sorted(a-b)}, extra={sorted(b-a)}"
        )


def root_tree(tree):
    matches = [
        tip for tip in tree.get_terminals()
        if tip.name == OUTGROUP
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one outgroup '{OUTGROUP}', "
            f"found {len(matches)}"
        )

    tree.root_with_outgroup(matches[0])


def protein_candidates(method, protein):
    if method == "RAM":
        return [
            os.path.join(
                RAM_ROOT,
                f"{protein}_34spp",
                "Consensus_Trees",
                f"{protein}_34spp_RAM_Hellinger_ASTRAL_consensus_tree.nwk"
            )
        ]

    if method == "SAM":
        return [
            os.path.join(
                SAM_ROOT,
                f"{protein}_{suffix}",
                "Consensus_Trees",
                f"{protein}_{suffix}_SAM_Hellinger_ASTRAL_consensus_tree.nwk"
            )
            for suffix in ("34spp", "34sppAligned")
        ]

    return [
        os.path.join(ML_ROOT, f"{protein}_34sppAligned.fas.treefile"),
        os.path.join(ML_ROOT, protein, f"{protein}.treefile"),
        os.path.join(ML_ROOT, protein, f"{protein}_34sppAligned.fas.treefile"),
        os.path.join(ML_ROOT, f"{protein}.treefile"),
        os.path.join(
            ML_ROOT,
            f"{protein}_34sppAligned",
            f"{protein}_34sppAligned.fas.treefile"
        )
    ]


def collect_protein_trees(method, reference):
    paths, trees, missing = {}, {}, []

    for protein in PROTEINS:
        choices = protein_candidates(method, protein)
        match = next(
            (path for path in choices if os.path.isfile(path)),
            None
        )

        if match is None:
            missing.append(
                f"{protein}: " + " OR ".join(choices)
            )
            continue

        tree = read_tree(match)
        check_taxa(reference, tree, f"{method} / {protein}")

        paths[protein] = match
        trees[protein] = tree

    if missing:
        raise FileNotFoundError(
            f"Missing {method} constituent protein trees:\n"
            + "\n".join(missing)
        )

    if len(trees) != len(PROTEINS):
        raise RuntimeError(
            f"Expected {len(PROTEINS)} {method} protein trees, "
            f"found {len(trees)}"
        )

    print(
        f"{method}: loaded {len(trees)} "
        f"constituent protein trees"
    )

    return paths, trees

# =============================================================================
# UNROOTED BIPARTITIONS AND ASTRAL SUPPORT
# =============================================================================

def canonical_split(side, all_taxa):
    side = frozenset(side)
    other = frozenset(all_taxa - side)

    if len(side) != len(other):
        return side if len(side) < len(other) else other

    return min(
        (side, other),
        key=lambda x: tuple(sorted(x))
    )


def clade_split(clade, all_taxa):
    descendants = {
        tip.name for tip in clade.get_terminals()
    }

    if (
        len(descendants) < 2
        or len(all_taxa - descendants) < 2
    ):
        return None

    return canonical_split(
        descendants,
        all_taxa
    )


def nontrivial_bipartitions(tree):
    taxa = get_taxa(tree)

    return {
        split
        for clade in tree.find_clades(order="postorder")
        if (
            split := clade_split(clade, taxa)
        ) is not None
    }


def clean_internal_labels(tree):
    tree = copy.deepcopy(tree)

    for node in tree.get_nonterminals():
        node.name = None
        node.confidence = None
        node.comment = None

    return tree


def score_with_astral(method, consensus, protein_trees, target_dir):
    """Score the unmodified consensus topology: ASTRAL -q -t 3 yields LPP labels."""

    require_file(ASTRAL_JAR)
    os.makedirs(target_dir, exist_ok=True)

    tag = method.replace("+", "_").replace(" ", "_")

    input_file = os.path.join(
        target_dir,
        f"{tag}_20protein_input_trees.nwk"
    )

    target_file = os.path.join(
        target_dir,
        f"{tag}_fixed_topology.nwk"
    )

    scored_file = os.path.join(
        target_dir,
        f"{tag}_ASTRAL_scored_LPP.nwk"
    )

    log_file = os.path.join(
        target_dir,
        f"{tag}_ASTRAL_scoring.log"
    )

    Phylo.write(
        [
            clean_internal_labels(protein_trees[p])
            for p in PROTEINS
        ],
        input_file,
        "newick"
    )

    Phylo.write(
        clean_internal_labels(consensus),
        target_file,
        "newick"
    )

    cmd = [
        "java",
        "-jar",
        ASTRAL_JAR,
        "-q",
        target_file,
        "-i",
        input_file,
        "-t",
        "3",
        "-o",
        scored_file
    ]

    print(
        f"  ASTRAL scoring: {method} "
        f"(20 proteins; fixed consensus topology)"
    )

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ASTRAL scoring failed for {method} "
            f"(exit {result.returncode}). "
            f"See {log_file}"
        )

    scored = read_tree(scored_file)

    check_taxa(
        consensus,
        scored,
        f"{method} ASTRAL-scored tree"
    )

    expected = nontrivial_bipartitions(consensus)
    observed = nontrivial_bipartitions(scored)

    if expected != observed:
        raise RuntimeError(
            f"ASTRAL changed the {method} target topology: "
            f"missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}"
        )

    values = {}
    taxa = get_taxa(scored)

    for clade in scored.get_nonterminals():
        split = clade_split(clade, taxa)

        if split is None:
            continue

        value = clade.confidence

        if value is None and clade.name is not None:
            try:
                value = float(clade.name)
            except ValueError:
                pass

        if (
            value is None
            or not np.isfinite(value)
            or not (0 <= value <= 1)
        ):
            raise RuntimeError(
                f"Invalid or missing ASTRAL LPP for "
                f"{method}: {sorted(split)} "
                f"(value={value!r})"
            )

        if (
            split in values
            and not np.isclose(
                values[split],
                value,
                atol=1e-5
            )
        ):
            raise RuntimeError(
                f"Conflicting ASTRAL labels on the same "
                f"{method} branch: {sorted(split)}"
            )

        values[split] = float(value)

    if set(values) != expected:
        raise RuntimeError(
            f"Missing ASTRAL LPP labels for "
            f"{len(expected - set(values))} "
            f"{method} internal branches. "
            f"See {scored_file}"
        )

    return values


def save_branch_table(
    method,
    consensus,
    reference,
    astral_values,
    output_path
):
    taxa = get_taxa(consensus)
    reference_splits = nontrivial_bipartitions(reference)

    rows = []
    seen = set()

    for clade in consensus.get_nonterminals(
        order="preorder"
    ):
        split = clade_split(clade, taxa)

        if split is None or split in seen:
            continue

        seen.add(split)

        rows.append(
            dict(
                method=method,
                branch=len(rows) + 1,
                bipartition_side=";".join(
                    sorted(split)
                ),
                opposite_side=";".join(
                    sorted(taxa - set(split))
                ),
                reference_match=split in reference_splits,
                astral_lpp=astral_values[split]
            )
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(
        f"  ASTRAL branch table: {output_path} "
        f"({len(rows)} internal splits)"
    )

# =============================================================================
# EXACT CROSSING MINIMIZATION
# =============================================================================

def rotate_to_reference(clade, rank):
    for child in clade.clades:
        rotate_to_reference(child, rank)

    clade.clades.sort(
        key=lambda child: np.mean(
            [
                rank[t.name]
                for t in child.get_terminals()
            ]
        )
    )


def child_order_cost(order, costs):
    return sum(
        costs[a][b]
        for pos, a in enumerate(order)
        for b in order[pos+1:]
    )


def optimize_child_order(groups):
    n = len(groups)

    if n < 2:
        return list(range(n))

    baseline = sorted(
        range(n),
        key=lambda i: (
            np.mean(groups[i]),
            min(groups[i])
        )
    )

    ranked = [
        groups[i]
        for i in baseline
    ]

    costs = [
        [
            sum(
                a > b
                for a in ranked[i]
                for b in ranked[j]
            )
            if i != j
            else 0
            for j in range(n)
        ]
        for i in range(n)
    ]

    if n <= MAX_EXACT_CHILDREN:
        best = [
            float("inf")
        ] * (1 << n)

        last = [
            -1
        ] * (1 << n)

        best[0] = 0

        for mask in range(1, 1 << n):
            for j in range(n-1, -1, -1):
                if not mask & (1 << j):
                    continue

                prev = mask ^ (1 << j)

                score = (
                    best[prev]
                    + sum(
                        costs[i][j]
                        for i in range(n)
                        if prev & (1 << i)
                    )
                )

                if score < best[mask]:
                    best[mask] = score
                    last[mask] = j

        order = []
        mask = (1 << n) - 1

        while mask:
            j = last[mask]
            order.append(j)
            mask ^= 1 << j

        order.reverse()

    else:
        print(
            f"  {n}-way polytomy: locally optimizing "
            f"(exact limit {MAX_EXACT_CHILDREN})."
        )

        order = list(range(n))

        while True:
            lowest = child_order_cost(
                order,
                costs
            )

            improved = None

            for i in range(n):
                remaining = (
                    order[:i]
                    + order[i+1:]
                )

                for j in range(n):
                    if i != j:
                        candidate = (
                            remaining[:j]
                            + [order[i]]
                            + remaining[j:]
                        )

                        score = child_order_cost(
                            candidate,
                            costs
                        )

                        if score < lowest:
                            lowest = score
                            improved = candidate

            if improved is None:
                break

            order = improved

    return [
        baseline[i]
        for i in order
    ]


def minimize_connector_crossings(clade, rank):
    for child in clade.clades:
        minimize_connector_crossings(
            child,
            rank
        )

    children = list(clade.clades)

    if len(children) >= 2:
        groups = [
            [
                rank[t.name]
                for t in child.get_terminals()
            ]
            for child in children
        ]

        clade.clades = [
            children[i]
            for i in optimize_child_order(groups)
        ]


def count_crossings(
    reference_order,
    comparison_order
):
    rank = {
        taxon: i
        for i, taxon in enumerate(
            reference_order
        )
    }

    indices = [
        rank[taxon]
        for taxon in comparison_order
    ]

    return sum(
        indices[i] > indices[j]
        for i in range(len(indices))
        for j in range(
            i+1,
            len(indices)
        )
    )

# =============================================================================
# TREE GEOMETRY / DRAWING
# =============================================================================

def topological_depths(tree):
    depths = {
        tree.root: 0
    }

    def recurse(node):
        for child in node.clades:
            depths[child] = depths[node] + 1
            recurse(child)

    recurse(tree.root)

    return depths


def y_coordinates(tree):
    terminals = tree.get_terminals()

    coordinates = {
        tip: float(
            len(terminals) - 1 - i
        )
        for i, tip in enumerate(terminals)
    }

    def recurse(node):
        if node.is_terminal():
            return coordinates[node]

        children = [
            recurse(child)
            for child in node.clades
        ]

        coordinates[node] = (
            min(children)
            + max(children)
        ) / 2

        return coordinates[node]

    recurse(tree.root)

    return coordinates


def x_coordinates(tree, side):
    depths = topological_depths(tree)
    max_depth = max(depths.values()) or 1

    if side == "left":
        return {
            node:
            LEFT_TIP_X
            if node.is_terminal()
            else (
                LEFT_ROOT_X
                + (
                    LEFT_TIP_X
                    - LEFT_ROOT_X
                )
                * depth
                / max_depth
            )
            for node, depth in depths.items()
        }

    if side == "right":
        return {
            node:
            RIGHT_TIP_X
            if node.is_terminal()
            else (
                RIGHT_ROOT_X
                - (
                    RIGHT_ROOT_X
                    - RIGHT_TIP_X
                )
                * depth
                / max_depth
            )
            for node, depth in depths.items()
        }

    raise ValueError(
        "side must be 'left' or 'right'"
    )


def draw_tree(ax, tree, side):
    x = x_coordinates(tree, side)
    y = y_coordinates(tree)

    def recurse(parent):
        if parent.clades:
            ax.plot(
                [x[parent], x[parent]],
                [
                    min(
                        y[c]
                        for c in parent.clades
                    ),
                    max(
                        y[c]
                        for c in parent.clades
                    )
                ],
                color="black",
                lw=TREE_LINEWIDTH,
                zorder=3
            )

        for child in parent.clades:
            ax.plot(
                [x[parent], x[child]],
                [y[child], y[child]],
                color="black",
                lw=TREE_LINEWIDTH,
                zorder=3
            )

            recurse(child)

    recurse(tree.root)

    positions = {}
    labels = {}

    for tip in tree.get_terminals():
        positions[tip.name] = (
            x[tip],
            y[tip]
        )

        left = side == "left"

        labels[tip.name] = ax.text(
            x[tip] + (
                0.006
                if left
                else -0.006
            ),
            y[tip],
            tip.name.replace("_", " "),
            ha=(
                "left"
                if left
                else "right"
            ),
            va="center",
            fontsize=LABEL_FONTSIZE,
            fontstyle="italic",
            zorder=5
        )

    return positions, labels, x, y


def draw_connectors(
    fig,
    ax,
    taxa,
    left_pos,
    right_pos,
    left_labels,
    right_labels
):
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()

    gap_pixels = (
        CONNECTOR_GAP_POINTS
        * fig.dpi
        / 72
    )

    for taxon in taxa:
        a = left_labels[
            taxon
        ].get_window_extent(
            renderer=renderer
        )

        b = right_labels[
            taxon
        ].get_window_extent(
            renderer=renderer
        )

        left_edge = inverse.transform(
            [
                a.x1 + gap_pixels,
                a.y0
            ]
        )[0]

        right_edge = inverse.transform(
            [
                b.x0 - gap_pixels,
                b.y0
            ]
        )[0]

        if left_edge >= right_edge:
            print(
                f"  Warning: insufficient label gap "
                f"for connector to {taxon}; "
                f"omitting connector"
            )
            continue

        ax.plot(
            [left_edge, right_edge],
            [
                left_pos[taxon][1],
                right_pos[taxon][1]
            ],
            color="gray",
            lw=CONNECTOR_LINEWIDTH,
            alpha=CONNECTOR_ALPHA,
            zorder=1
        )


def draw_branch_support(
    ax,
    tree,
    x,
    y,
    supports
):
    taxa = get_taxa(tree)
    seen = set()

    for parent in tree.get_nonterminals(
        order="preorder"
    ):
        for child in parent.clades:
            split = clade_split(
                child,
                taxa
            )

            if (
                split is None
                or split in seen
            ):
                continue

            seen.add(split)

            if split not in supports:
                raise RuntimeError(
                    f"No ASTRAL support for "
                    f"branch {sorted(split)}"
                )

            ax.annotate(
                f"{supports[split]:.2f}",
                xy=(
                    (
                        x[parent]
                        + x[child]
                    ) / 2,
                    y[child]
                ),
                xytext=(
                    0,
                    SUPPORT_OFFSET_POINTS
                ),
                textcoords="offset points",
                fontsize=SUPPORT_FONTSIZE,
                ha="center",
                va="bottom",
                zorder=6,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.92,
                    pad=0.3
                )
            )

    if seen != set(supports):
        raise RuntimeError(
            f"Annotated {len(seen)} internal "
            f"branches, expected {len(supports)}"
        )


def save_tanglegram(
    reference,
    comparison,
    method,
    supports,
    output_stem
):
    ref_order = [
        tip.name
        for tip in reference.get_terminals()
    ]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.025,
        top=0.985
    )

    left_pos, left_labels, _, _ = draw_tree(
        ax,
        reference,
        "left"
    )

    right_pos, right_labels, x, y = draw_tree(
        ax,
        comparison,
        "right"
    )

    max_y = len(ref_order) - 1

    ax.set_xlim(
        -0.008,
        1.008
    )

    ax.set_ylim(
        -0.55,
        max_y + 2.7
    )

    ax.text(
        (
            LEFT_ROOT_X
            + LEFT_TIP_X
        ) / 2,
        max_y + 1,
        "OMAM reference",
        ha="center",
        va="bottom",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold"
    )

    ax.text(
        (
            RIGHT_TIP_X
            + RIGHT_ROOT_X
        ) / 2,
        max_y + 1,
        rf"$\mathbfit{{{method}}}$ 20-protein phylogeny",
        ha="center",
        va="bottom",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        linespacing=1.1
    )

    ax.axis("off")

    draw_branch_support(
        ax,
        comparison,
        x,
        y,
        supports
    )

    draw_connectors(
        fig,
        ax,
        ref_order,
        left_pos,
        right_pos,
        left_labels,
        right_labels
    )

    pdf_path = output_stem + ".pdf"
    svg_path = output_stem + ".svg"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.07
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.07
    )

    if (
        SHOW_PLOTS
        and "agg"
        not in plt.get_backend().lower()
    ):
        plt.show()

    plt.close(fig)

    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved SVG: {svg_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    for path in (
        REFERENCE_TREE,
        RAM_TREE,
        SAM_TREE,
        ML_TREE,
        ASTRAL_JAR
    ):
        require_file(path)

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    reference_original = read_tree(
        REFERENCE_TREE
    )

    for method, consensus_path in METHODS:
        print(
            f"\n{'='*78}\n"
            f"{method}: {consensus_path}\n"
            f"{'='*78}"
        )

        reference = copy.deepcopy(
            reference_original
        )

        comparison = read_tree(
            consensus_path
        )

        check_taxa(
            reference,
            comparison,
            method
        )

        root_tree(reference)
        root_tree(comparison)

        lookup_method = (
            "ML"
            if method == "ML+MFP"
            else method
        )

        paths, constituent = collect_protein_trees(
            lookup_method,
            reference
        )

        astral_stats = score_with_astral(
            method,
            comparison,
            constituent,
            os.path.join(
                OUT_DIR,
                "ASTRAL_scoring"
            )
        )

        reference.ladderize()

        ref_order = [
            tip.name
            for tip in reference.get_terminals()
        ]

        rank = {
            taxon: i
            for i, taxon in enumerate(ref_order)
        }

        original = count_crossings(
            ref_order,
            [
                t.name
                for t in comparison.get_terminals()
            ]
        )

        heuristic = copy.deepcopy(
            comparison
        )

        rotate_to_reference(
            heuristic.root,
            rank
        )

        heuristic_count = count_crossings(
            ref_order,
            [
                t.name
                for t in heuristic.get_terminals()
            ]
        )

        original_splits = nontrivial_bipartitions(
            comparison
        )

        minimize_connector_crossings(
            comparison.root,
            rank
        )

        optimized = count_crossings(
            ref_order,
            [
                t.name
                for t in comparison.get_terminals()
            ]
        )

        if (
            nontrivial_bipartitions(
                comparison
            ) != original_splits
            or optimized > heuristic_count
        ):
            raise RuntimeError(
                f"Invalid crossing optimization "
                f"for {method}"
            )

        print(
            f"  Connector crossings: "
            f"original={original}, "
            f"mean-rank={heuristic_count}, "
            f"optimized={optimized}"
        )

        prefix = method.replace(
            "+",
            "_"
        )

        # Non-figure outputs remain in original OUT_DIR.
        pd.DataFrame(
            [
                {
                    "protein": protein,
                    "tree_path": paths[protein]
                }
                for protein in PROTEINS
            ]
        ).to_csv(
            os.path.join(
                OUT_DIR,
                f"{prefix}_20protein_tree_sources.tsv"
            ),
            sep="\t",
            index=False
        )

        save_branch_table(
            method,
            comparison,
            reference,
            astral_stats,
            os.path.join(
                OUT_DIR,
                f"{prefix}_20protein_ASTRAL_branch_support.tsv"
            )
        )

        # Only PDF/SVG figures go to Submission_Figures.
        save_tanglegram(
            reference,
            comparison,
            method,
            astral_stats,
            os.path.join(
                FIGURE_DIR,
                f"Fig4_OMAM_vs_{prefix}_20protein"
            )
        )

    print(
        "\nDone.\n"
        f"PDF/SVG figures saved to:\n{FIGURE_DIR}\n\n"
        f"ASTRAL scoring files, TSVs and manifests saved to:\n{OUT_DIR}"
    )


if __name__ == "__main__":
    main()