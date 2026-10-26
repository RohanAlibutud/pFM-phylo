#!/usr/bin/env python3

"""
RAM_Treebuilder.py

Residue Attention Matrix (RAM) phylogenetic treebuilding using
pairwise-alignment-guided, row-wise Hellinger distance.

For each protein sequence:
    1. Run the native, unaligned sequence through ESM once.
    2. Extract one residue attention matrix (RAM) per layer/head.

For each unordered species pair:
    3. Globally align the two original amino-acid sequences ONCE.
    4. Build one alignment map containing only positions where both
       sequences have a residue.
    5. Prune BOTH rows and columns of the two native RAMs using the
       corresponding residue indices.
    6. Renormalize every pruned RAM row so that it sums to 1.
    7. Compute one Hellinger distance per corresponding row.
    8. Calculate the root-mean-square (RMS) of those row-wise Hellinger
       distances to obtain one species-pair distance for that layer/head.
    9. Recalculate the reverse B->A distance independently using the SAME
       alignment map; do not copy A->B into B->A.

For each layer/head:
   10. Assemble the full species distance matrix without enforcing symmetry.
   11. Test that every matrix is exactly symmetric. Save the symmetry test
       to a log file and raise RuntimeError if any asymmetry is detected.
   12. Infer one NJ tree from the lower triangle only.

Finally:
   13. Concatenate one tree from every layer/head.
   14. Call ASTRAL to infer the protein-level consensus tree.
"""

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from Bio import Phylo, SeqIO
from Bio.Align import PairwiseAligner
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from tqdm import tqdm


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Build RAM phylogenies using alignment-guided row-wise Hellinger distance.")
    parser.add_argument("--model", required=True, help="Path to the saved ESM model object.")
    parser.add_argument("--tokenizer", required=True, help="Path to the saved ESM alphabet/tokenizer pickle.")
    parser.add_argument("--input", required=True, help="Path to the UNALIGNED input FASTA.")
    parser.add_argument("--out", required=True, help="Root output folder.")
    parser.add_argument("--domain", required=True, help="Protein/domain label used in output filenames.")
    parser.add_argument("--astral", required=True, help="Path to the ASTRAL .jar file.")
    parser.add_argument("--save_attn", action="store_true", help="Save native unaligned attention matrices as compressed .npz files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU worker threads for pairwise Hellinger calculations. Default: 1.")
    parser.add_argument("--match_score", type=float, default=1.0, help="Pairwise-alignment match score. Default: 1.0")
    parser.add_argument("--mismatch_score", type=float, default=-1.0, help="Pairwise-alignment mismatch score. Default: -1.0")
    parser.add_argument("--gap_open_score", type=float, default=-1.0, help="Pairwise-alignment gap-open score. Default: -1.0")
    parser.add_argument("--gap_extend_score", type=float, default=-1.0, help="Pairwise-alignment gap-extension score. Default: -1.0")
    return parser.parse_args()


# =============================================================================
# FILE / DIRECTORY UTILITIES
# =============================================================================

def create_directory(path):
    os.makedirs(path, exist_ok=True)


def load_fasta(fasta_file):
    species_tags = []
    sequences = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        species_id = record.id
        sequence = str(record.seq).upper()
        if species_id in sequences:
            raise ValueError(f"Duplicate FASTA identifier: {species_id}")
        if "-" in sequence:
            raise ValueError(f"Input sequence {species_id} contains '-' characters. RAM_Treebuilder.py expects native unaligned sequences.")
        species_tags.append(species_id)
        sequences[species_id] = sequence

    if len(species_tags) < 3:
        raise ValueError("At least three sequences are required for treebuilding.")
    return species_tags, sequences


# =============================================================================
# MODEL / ATTENTION METHODS
# =============================================================================

def load_model(model_path, tokenizer_path):
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {device}")
    model = torch.load(model_path, weights_only=False)
    with open(tokenizer_path, "rb") as f:
        alphabet = pickle.load(f)

    model.to(device)
    model.eval()
    layer_count = len(model.layers)
    head_count = model.layers[0].self_attn.num_heads
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Layers: {layer_count}")
    print(f"Heads per layer: {head_count}")
    return model, alphabet, layer_count, head_count


def extract_native_attn(domain, model, alphabet, protein_sequence, sequence_identifier, layer_count, head_count, save_attn=False, attn_outdir=None):
    """Run one native unaligned sequence through ESM and return RAMs as (layers, heads, L, L)."""
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([(sequence_identifier, protein_sequence)])
    if batch_tokens.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1, got {tuple(batch_tokens.shape)}")

    device = next(model.parameters()).device
    batch_tokens = batch_tokens.to(device)
    L = len(protein_sequence)
    T = int(batch_tokens.shape[-1])

    with torch.no_grad():
        output = model(batch_tokens, need_head_weights=True)
        attn = output["attentions"].detach()

    if attn.ndim != 5:
        raise RuntimeError(f"Expected 5D attention tensor, got shape {tuple(attn.shape)}")

    if attn.shape[0] == layer_count:
        if attn.shape[1] != 1 or attn.shape[2] != head_count:
            raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")
        attn = attn[:, 0, :, :, :]
    else:
        if attn.shape[0] != 1 or attn.shape[1] != layer_count or attn.shape[2] != head_count:
            raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")
        attn = attn[0, :, :, :, :]

    start = 1
    end = start + L
    if end > T:
        raise RuntimeError(f"Cannot slice residue tokens: sequence length={L}, token length={T}")
    attn = attn[:, :, start:end, start:end]

    expected_shape = (layer_count, head_count, L, L)
    if tuple(attn.shape) != expected_shape:
        raise RuntimeError(f"Unexpected post-slice attention shape: {tuple(attn.shape)}; expected {expected_shape}")

    attn_np = attn.to("cpu").to(torch.float16).numpy()

    if save_attn:
        if attn_outdir is None:
            raise ValueError("save_attn=True but attn_outdir is None")
        create_directory(attn_outdir)
        out_npz = os.path.join(attn_outdir, f"{domain}_{sequence_identifier}_native_RAMs.npz")
        payload = {}
        for layer in range(layer_count):
            for head in range(head_count):
                payload[f"layer{layer + 1:02d}_head{head + 1:02d}"] = attn_np[layer, head]
        np.savez_compressed(out_npz, **payload)

    return attn_np


def extract_all_native_attentions(domain, species_tags, sequences, model, alphabet, layer_count, head_count, save_attn, attn_outdir):
    """Infer native RAMs once per species and retain them for all pairwise comparisons."""
    attention_cache = {}
    print("\nExtracting native unaligned RAMs...")
    for species_id in tqdm(species_tags, desc="ESM inference"):
        attention_cache[species_id] = extract_native_attn(
            domain=domain, model=model, alphabet=alphabet, protein_sequence=sequences[species_id],
            sequence_identifier=species_id, layer_count=layer_count, head_count=head_count,
            save_attn=save_attn, attn_outdir=attn_outdir)
    return attention_cache


# =============================================================================
# PAIRWISE ALIGNMENT / ALIGNMENT MAP
# =============================================================================

def make_pairwise_aligner(match_score, mismatch_score, gap_open_score, gap_extend_score):
    """Construct a global pairwise aligner used only to establish residue correspondence."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_open_score
    aligner.extend_gap_score = gap_extend_score
    return aligner


def build_alignment_map(seq_A, seq_B, aligner):
    """Align A and B once and return native residue indices for positions occupied in both sequences."""
    alignments = aligner.align(seq_A, seq_B)
    alignment = next(iter(alignments), None)
    if alignment is None:
        raise RuntimeError("Pairwise alignment returned no solutions.")

    blocks_A, blocks_B = alignment.aligned
    idx_A = []
    idx_B = []

    for (start_A, end_A), (start_B, end_B) in zip(blocks_A, blocks_B):
        block_len_A = int(end_A - start_A)
        block_len_B = int(end_B - start_B)
        if block_len_A != block_len_B:
            raise RuntimeError("Aligned residue blocks have different lengths, which should not occur for paired alignment blocks.")
        idx_A.extend(range(int(start_A), int(end_A)))
        idx_B.extend(range(int(start_B), int(end_B)))

    idx_A = np.asarray(idx_A, dtype=np.int64)
    idx_B = np.asarray(idx_B, dtype=np.int64)
    if len(idx_A) == 0:
        raise RuntimeError("No mutually aligned residue positions were found.")
    if len(idx_A) != len(idx_B):
        raise RuntimeError(f"Alignment-map length mismatch: {len(idx_A)} vs {len(idx_B)}")
    return idx_A, idx_B


# =============================================================================
# HELLINGER RAM DISTANCE
# =============================================================================

def prune_and_renormalize_ram(ram, indices):
    """Prune both RAM dimensions to retained residues and renormalize each retained row."""
    pruned = ram[np.ix_(indices, indices)].astype(np.float32, copy=False)
    row_sums = pruned.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)):
        raise FloatingPointError("Non-finite RAM row sum encountered.")
    if np.any(row_sums <= 0.0):
        bad_rows = np.where(row_sums[:, 0] <= 0.0)[0]
        raise FloatingPointError(f"Encountered {len(bad_rows)} pruned RAM row(s) with sum <= 0.")
    return pruned / row_sums


def rms_rowwise_hellinger(ram_A, ram_B, idx_A, idx_B):
    """Prune, row-normalize, calculate corresponding-row Hellinger distances, and return their RMS."""
    norm_A = prune_and_renormalize_ram(ram_A, idx_A)
    norm_B = prune_and_renormalize_ram(ram_B, idx_B)
    if norm_A.shape != norm_B.shape:
        raise RuntimeError(f"Pruned RAM shape mismatch: {norm_A.shape} vs {norm_B.shape}")

    rowwise_hellinger = np.sqrt(0.5 * np.sum((np.sqrt(norm_A) - np.sqrt(norm_B)) ** 2, axis=1))
    if np.any(~np.isfinite(rowwise_hellinger)):
        raise FloatingPointError("Non-finite Hellinger distance encountered.")
    return float(np.sqrt(np.mean(rowwise_hellinger ** 2)))


def calculate_species_pair_distances(attn_A, attn_B, idx_A, idx_B, layer_count, head_count):
    """Calculate the complete layer x head distance array for one direction of one species pair."""
    pairwise_distances = np.zeros((layer_count, head_count), dtype=np.float64)
    for layer in range(layer_count):
        for head in range(head_count):
            pairwise_distances[layer, head] = rms_rowwise_hellinger(
                ram_A=attn_A[layer, head], ram_B=attn_B[layer, head], idx_A=idx_A, idx_B=idx_B)
    return pairwise_distances


def calculate_bidirectional_species_pair_distances(attn_A, attn_B, idx_A, idx_B, layer_count, head_count):
    """Calculate A->B and B->A independently from the same single alignment map."""
    d_AB = calculate_species_pair_distances(attn_A, attn_B, idx_A, idx_B, layer_count, head_count)
    d_BA = calculate_species_pair_distances(attn_B, attn_A, idx_B, idx_A, layer_count, head_count)
    return d_AB, d_BA


def calculate_all_distance_matrices(species_tags, sequences, attention_cache, layer_count, head_count, aligner, workers=1):
    """
    Build a full species distance matrix per layer/head without imposing symmetry.

    Each unordered species pair A/B is aligned exactly once. That single alignment
    map is then used to calculate A->B and B->A independently. Symmetry is tested
    only after all matrices have been populated.
    """
    if workers < 1:
        raise ValueError("--workers must be >= 1.")

    num_species = len(species_tags)
    distance_matrices = np.zeros((layer_count, head_count, num_species, num_species), dtype=np.float64)
    pair_jobs = []

    print("\nBuilding one alignment map per unordered species pair...")
    for i, sp_A in enumerate(species_tags):
        seq_A = sequences[sp_A]
        for j in range(i + 1, num_species):
            sp_B = species_tags[j]
            idx_A, idx_B = build_alignment_map(seq_A, sequences[sp_B], aligner)
            pair_jobs.append((i, j, idx_A, idx_B))

    print(f"Built {len(pair_jobs)} pairwise alignment maps.")
    print(f"\nCalculating A->B and B->A independently from each single alignment with {workers} CPU worker{'s' if workers != 1 else ''}...")

    if workers == 1:
        for i, j, idx_A, idx_B in tqdm(pair_jobs, desc="Species pairs"):
            sp_A = species_tags[i]
            sp_B = species_tags[j]
            d_AB, d_BA = calculate_bidirectional_species_pair_distances(
                attention_cache[sp_A], attention_cache[sp_B], idx_A, idx_B, layer_count, head_count)
            distance_matrices[:, :, i, j] = d_AB
            distance_matrices[:, :, j, i] = d_BA
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_pair = {}
            for i, j, idx_A, idx_B in pair_jobs:
                sp_A = species_tags[i]
                sp_B = species_tags[j]
                future = executor.submit(
                    calculate_bidirectional_species_pair_distances,
                    attention_cache[sp_A], attention_cache[sp_B], idx_A, idx_B, layer_count, head_count)
                future_to_pair[future] = (i, j)

            for future in tqdm(as_completed(future_to_pair), total=len(future_to_pair), desc="Species pairs"):
                i, j = future_to_pair[future]
                d_AB, d_BA = future.result()
                distance_matrices[:, :, i, j] = d_AB
                distance_matrices[:, :, j, i] = d_BA

    gc.collect()
    return distance_matrices


def validate_distance_matrix_symmetry(distance_matrices, log_path):
    """Require exact symmetry before treebuilding; write diagnostics to screen and log file."""
    transposed = np.swapaxes(distance_matrices, -1, -2)
    abs_diff = np.abs(distance_matrices - transposed)
    exact_by_head = np.all(distance_matrices == transposed, axis=(-2, -1))
    total = exact_by_head.size
    max_difference = float(np.max(abs_diff))
    mean_difference = float(np.mean(abs_diff))

    lines = [
        "=" * 80,
        "RAM DISTANCE MATRIX SYMMETRY TEST",
        "=" * 80,
        f"Exact symmetric layer/heads: {int(np.sum(exact_by_head))}/{total}",
        f"Maximum |d_ij - d_ji|: {max_difference:.12g}",
        f"Mean absolute matrix difference: {mean_difference:.12g}",
        f"OVERALL EXACTLY SYMMETRIC: {bool(np.all(exact_by_head))}",
    ]

    if not np.all(exact_by_head):
        lines.append("")
        lines.append("Asymmetric layer/head combinations:")
        for layer, head in np.argwhere(~exact_by_head):
            head_diff = float(np.max(abs_diff[layer, head]))
            lines.append(f"  layer {layer + 1:2d}, head {head + 1:2d}: max difference = {head_diff:.12g}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(log_path, "w") as f:
        f.write(text + "\n")

    if not np.all(exact_by_head):
        raise RuntimeError(
            f"RAM distance matrices are asymmetric. Treebuilding aborted. See symmetry log: {log_path}"
        )

    print(f"Symmetry confirmed. Results saved to:\n{log_path}")


# =============================================================================
# PHYLOGENY METHODS
# =============================================================================

def build_layer_head_trees(domain, species_tags, distance_matrices, distance_folder, tree_folder):
    """Build one NJ tree per layer/head from the lower triangle of the verified-symmetric matrix."""
    layer_count, head_count, _, _ = distance_matrices.shape
    tree_paths = []

    print("\nBuilding lower-triangle NJ trees...")
    for layer in tqdm(range(layer_count), desc="Layers"):
        layer_number = layer + 1
        distance_layer_folder = os.path.join(distance_folder, f"layer{layer_number}")
        tree_layer_folder = os.path.join(tree_folder, f"layer{layer_number}")
        create_directory(distance_layer_folder)
        create_directory(tree_layer_folder)

        for head in range(head_count):
            head_number = head + 1
            species_matrix = distance_matrices[layer, head]
            matrix_path = os.path.join(distance_layer_folder, f"{domain}_layer{layer_number}_head{head_number}_distance_matrix.csv")
            tree_path = os.path.join(tree_layer_folder, f"{domain}_layer{layer_number}_head{head_number}_tree.nwk")
            np.savetxt(matrix_path, species_matrix, delimiter=",", fmt="%.10f")

            lower_triangle = [species_matrix[i, :i + 1].tolist() for i in range(len(species_tags))]
            dm = DistanceMatrix(names=species_tags, matrix=lower_triangle)
            tree = DistanceTreeConstructor().nj(dm)
            Phylo.write(tree, tree_path, "newick")
            tree_paths.append(tree_path)

    return tree_paths


def combine_newicks(tree_paths, output_newick):
    """Concatenate exactly the layer/head trees generated during this run."""
    with open(output_newick, "w") as out_f:
        for tree_path in tree_paths:
            with open(tree_path, "r") as in_f:
                tree_string = in_f.read().strip()
            if tree_string:
                out_f.write(tree_string + "\n")


def call_astral(astral_path, input_newick, output_newick):
    """Infer a consensus/species tree from all layer/head NJ trees using ASTRAL."""
    command = ["java", "-jar", astral_path, "-i", input_newick, "-o", output_newick, "-t", "1"]
    print("\nCalling ASTRAL:")
    print(" ".join(command))
    subprocess.run(command, check=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def RAM_treebuild(args):
    start_time = time.time()

    attention_folder = os.path.join(args.out, "Attention_Matrices", args.domain)
    distance_folder = os.path.join(args.out, "Distance_Matrices", args.domain)
    tree_folder = os.path.join(args.out, "Trees", args.domain)
    consensus_folder = os.path.join(tree_folder, "Consensus_Trees")

    if args.save_attn:
        create_directory(attention_folder)
    create_directory(distance_folder)
    create_directory(tree_folder)
    create_directory(consensus_folder)

    species_tags, sequences = load_fasta(args.input)
    print("\nInput taxa:")
    for species_id in species_tags:
        print(f"  {species_id}: {len(sequences[species_id])} aa")

    model, alphabet, layer_count, head_count = load_model(args.model, args.tokenizer)
    attention_cache = extract_all_native_attentions(
        domain=args.domain, species_tags=species_tags, sequences=sequences, model=model,
        alphabet=alphabet, layer_count=layer_count, head_count=head_count, save_attn=args.save_attn,
        attn_outdir=attention_folder)

    aligner = make_pairwise_aligner(
        match_score=args.match_score, mismatch_score=args.mismatch_score,
        gap_open_score=args.gap_open_score, gap_extend_score=args.gap_extend_score)

    distance_matrices = calculate_all_distance_matrices(
        species_tags=species_tags, sequences=sequences, attention_cache=attention_cache,
        layer_count=layer_count, head_count=head_count, aligner=aligner, workers=args.workers)

    symmetry_log = os.path.join(distance_folder, f"{args.domain}_RAM_Hellinger_symmetry.log")
    validate_distance_matrix_symmetry(distance_matrices, symmetry_log)

    distance_npz = os.path.join(distance_folder, f"{args.domain}_RAM_Hellinger_distance_matrices.npz")
    np.savez_compressed(distance_npz, distance_matrices=distance_matrices, species=np.asarray(species_tags))
    print(f"\nSaved full verified-symmetric distance tensor:\n{distance_npz}")

    del attention_cache
    gc.collect()

    tree_paths = build_layer_head_trees(
        domain=args.domain, species_tags=species_tags, distance_matrices=distance_matrices,
        distance_folder=distance_folder, tree_folder=tree_folder)

    concatenated_newick = os.path.join(consensus_folder, f"{args.domain}_RAM_Hellinger_concatenated_trees.nwk")
    astral_tree = os.path.join(consensus_folder, f"{args.domain}_RAM_Hellinger_ASTRAL_consensus_tree.nwk")
    combine_newicks(tree_paths=tree_paths, output_newick=concatenated_newick)
    print(f"\nCombined {len(tree_paths)} layer-head trees into:\n{concatenated_newick}")
    call_astral(astral_path=args.astral, input_newick=concatenated_newick, output_newick=astral_tree)

    runtime = time.time() - start_time
    print("\nRAM treebuilding complete.")
    print(f"ASTRAL consensus tree:\n{astral_tree}")
    print(f"Symmetry log:\n{symmetry_log}")
    print(f"Total runtime: {runtime / 60:.2f} minutes")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_arguments()
    RAM_treebuild(args)
