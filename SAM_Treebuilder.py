#!/usr/bin/env python3

"""
SAM_Treebuilder.py

Species Attention Matrix (SAM) phylogenetic treebuilding using row-wise
Hellinger distance on MSA Transformer column-attention matrices.

For each layer/head:
    1. Extract the MSA Transformer's column attentions.
    2. For each alignment column, use the species x species attention matrix.
    3. For each species pair A/B, calculate Hellinger distance between
       species row A and species row B.
    4. Calculate the root-mean-square (RMS) of those distances across
       all alignment columns.
    5. Use the resulting pairwise species distances to build an NJ tree.
    6. Combine all layer/head trees with ASTRAL.

Unlike RAM Hellinger treebuilding, no pairwise realignment or pruning is
required because all sequences already share the same MSA coordinate system.
"""

import argparse
import gc
import os
import subprocess
import sys
import time

import numpy as np
import torch
from Bio import Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from esm import pretrained
from tqdm import tqdm


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Build SAM phylogenies using row-wise Hellinger distance on MSA Transformer column attentions.")
    parser.add_argument("--msa", required=True, help="Path to the aligned input MSA in FASTA format.")
    parser.add_argument("--out", required=True, help="Root output folder.")
    parser.add_argument("--domain", required=True, help="Protein/domain label used in output filenames.")
    parser.add_argument("--astral", required=True, help="Path to the ASTRAL .jar file.")
    return parser.parse_args()


# =============================================================================
# UTILITIES
# =============================================================================

def create_directory(path):
    os.makedirs(path, exist_ok=True)


def load_msa(msa_file):
    species_tags = []
    msa_data = []
    lengths = set()

    for record in SeqIO.parse(msa_file, "fasta"):
        species_id = record.id
        sequence = str(record.seq).upper()
        species_tags.append(species_id)
        msa_data.append((species_id, sequence))
        lengths.add(len(sequence))

    if len(species_tags) < 3:
        raise ValueError("At least three sequences are required for treebuilding.")
    if len(lengths) != 1:
        raise ValueError(f"Input is not a valid MSA: found sequence lengths {sorted(lengths)}.")

    return species_tags, msa_data, lengths.pop()


# =============================================================================
# MODEL / COLUMN ATTENTION
# =============================================================================

def init_model():
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print("Loading MSA Transformer...")

    model, alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
    model.to(device)
    model.eval()

    print(f"Model device: {next(model.parameters()).device}")

    return model, alphabet, device


def extract_column_attentions(model, alphabet, device, msa_data, aln_length):
    """
    Return column attentions as:
        (layers, heads, alignment_columns, species, species)

    The first sequence-position token is BOS and is removed. No species axis
    is pruned.
    """
    batch_converter = alphabet.get_batch_converter()
    _, _, msa_batch_tokens = batch_converter(msa_data)
    msa_batch_tokens = msa_batch_tokens.to(device)

    with torch.no_grad():
        results = model(msa_batch_tokens, need_head_weights=True)

    col_attns = results["col_attentions"].detach()

    if col_attns.ndim != 6:
        raise RuntimeError(f"Expected 6D column-attention tensor, got shape {tuple(col_attns.shape)}.")

    # Existing ESM MSA Transformer output:
    # (batch, layers, heads, columns+1, species, species)
    if col_attns.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1, got shape {tuple(col_attns.shape)}.")

    layer_count = int(col_attns.shape[1])
    head_count = int(col_attns.shape[2])

    print(f"Layers: {layer_count}")
    print(f"Heads per layer: {head_count}")

    col_attns = col_attns[0, :, :, 1:, :, :]

    if col_attns.shape[2] != aln_length:
        raise RuntimeError(f"Expected {aln_length} alignment columns after BOS removal, got {col_attns.shape[2]}.")

    col_attns = col_attns.to("cpu").to(torch.float32).numpy()

    del results
    torch.cuda.empty_cache()
    gc.collect()

    return col_attns


# =============================================================================
# HELLINGER SAM DISTANCE
# =============================================================================

def normalize_species_rows(col_attns):
    """
    Normalize the final species-attention axis so every species row is a
    probability distribution. Column-attention rows should already sum to 1;
    this mainly protects against small numerical drift.
    """
    row_sums = col_attns.sum(axis=-1, keepdims=True)

    if np.any(~np.isfinite(row_sums)):
        raise FloatingPointError("Non-finite column-attention row sum encountered.")
    if np.any(row_sums <= 0.0):
        raise FloatingPointError("Column-attention row with sum <= 0 encountered.")

    return col_attns / row_sums


def calculate_distance_matrices(col_attns, species_tags):
    """
    For every species pair and every layer/head:

        H_z(A,B) = H(row_A, row_B)

    for each alignment column z, followed by:

        d(A,B) = sqrt(mean_z(H_z(A,B)^2))

    Returns:
        (layers, heads, species, species)
    """
    layer_count, head_count, aln_length, num_species_A, num_species_B = col_attns.shape
    num_species = len(species_tags)

    if num_species_A != num_species or num_species_B != num_species:
        raise RuntimeError(
            f"Column-attention species dimensions {num_species_A}x{num_species_B} do not match {num_species} MSA sequences.")

    col_attns = normalize_species_rows(col_attns)
    sqrt_attns = np.sqrt(col_attns)
    distance_matrices = np.zeros((layer_count, head_count, num_species, num_species), dtype=np.float64)

    print(f"\nCalculating row-wise Hellinger distances with RMS aggregation across {aln_length} alignment columns...")

    for i, sp_A in enumerate(species_tags):
        for j in tqdm(range(i + 1, num_species), desc=f"{sp_A} vs others", leave=False):
            row_A = sqrt_attns[:, :, :, i, :]
            row_B = sqrt_attns[:, :, :, j, :]

            column_hellinger = np.sqrt(0.5 * np.sum((row_A - row_B) ** 2, axis=-1))
            pairwise_distance = np.sqrt(np.mean(column_hellinger ** 2, axis=2))

            distance_matrices[:, :, i, j] = pairwise_distance
            distance_matrices[:, :, j, i] = pairwise_distance

    return distance_matrices


# =============================================================================
# PHYLOGENY
# =============================================================================

def build_layer_head_trees(domain, species_tags, distance_matrices, distance_folder, tree_folder):
    layer_count, head_count, _, _ = distance_matrices.shape
    tree_paths = []

    print("\nBuilding layer/head neighbor-joining trees...")

    for layer in tqdm(range(layer_count), desc="Layers"):
        layer_number = layer + 1
        distance_layer_folder = os.path.join(distance_folder, f"layer{layer_number}")
        tree_layer_folder = os.path.join(tree_folder, f"layer{layer_number}")
        create_directory(distance_layer_folder)
        create_directory(tree_layer_folder)

        for head in range(head_count):
            head_number = head + 1
            species_matrix = distance_matrices[layer, head]

            matrix_path = os.path.join(
                distance_layer_folder, f"{domain}_layer{layer_number}_head{head_number}_distance_matrix.csv")
            tree_path = os.path.join(
                tree_layer_folder, f"{domain}_layer{layer_number}_head{head_number}_tree.nwk")

            np.savetxt(matrix_path, species_matrix, delimiter=",", fmt="%.10f")

            lower_triangle = [species_matrix[i, : i + 1].tolist() for i in range(len(species_tags))]
            dm = DistanceMatrix(names=species_tags, matrix=lower_triangle)
            tree = DistanceTreeConstructor().nj(dm)
            Phylo.write(tree, tree_path, "newick")
            tree_paths.append(tree_path)

    return tree_paths


def combine_newicks(tree_paths, output_newick):
    with open(output_newick, "w") as out_f:
        for tree_path in tree_paths:
            with open(tree_path, "r") as in_f:
                tree = in_f.read().strip()
            if tree:
                out_f.write(tree + "\n")


def call_astral(astral_path, input_newick, output_newick):
    command = ["java", "-jar", astral_path, "-i", input_newick, "-o", output_newick, "-t", "1"]
    print("\nCalling ASTRAL:")
    print(" ".join(command))
    subprocess.run(command, check=True)


# =============================================================================
# MAIN
# =============================================================================

def SAM_treebuild(args):
    start_time = time.time()

    distance_folder = os.path.join(args.out, "Distance_Matrices", args.domain)
    tree_folder = os.path.join(args.out, "Trees", args.domain)
    consensus_folder = os.path.join(tree_folder, "Consensus_Trees")
    create_directory(distance_folder)
    create_directory(tree_folder)
    create_directory(consensus_folder)

    species_tags, msa_data, aln_length = load_msa(args.msa)

    print("\nInput MSA:")
    print(f"  Taxa: {len(species_tags)}")
    print(f"  Alignment columns: {aln_length}")

    model, alphabet, device = init_model()

    col_attns = extract_column_attentions(
        model, alphabet, device, msa_data, aln_length)

    print(f"Column-attention tensor: {col_attns.shape}")

    distance_matrices = calculate_distance_matrices(col_attns, species_tags)

    distance_npz = os.path.join(distance_folder, f"{args.domain}_SAM_Hellinger_distance_matrices.npz")
    np.savez_compressed(distance_npz, distance_matrices=distance_matrices, species=np.asarray(species_tags))
    print(f"\nSaved full distance tensor:\n{distance_npz}")

    del col_attns
    gc.collect()

    tree_paths = build_layer_head_trees(
        domain=args.domain, species_tags=species_tags, distance_matrices=distance_matrices,
        distance_folder=distance_folder, tree_folder=tree_folder)

    concatenated_newick = os.path.join(
        consensus_folder, f"{args.domain}_SAM_Hellinger_concatenated_trees.nwk")
    astral_tree = os.path.join(
        consensus_folder, f"{args.domain}_SAM_Hellinger_ASTRAL_consensus_tree.nwk")

    combine_newicks(tree_paths, concatenated_newick)
    print(f"\nCombined {len(tree_paths)} layer/head trees into:\n{concatenated_newick}")

    call_astral(args.astral, concatenated_newick, astral_tree)

    runtime = time.time() - start_time
    print("\nSAM treebuilding complete.")
    print(f"ASTRAL consensus tree:\n{astral_tree}")
    print(f"Total runtime: {runtime / 60:.2f} minutes")


if __name__ == "__main__":
    args = parse_arguments()
    SAM_treebuild(args)