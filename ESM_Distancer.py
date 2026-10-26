import torch
import os
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

def _as_tensor(x, device=None, dtype=torch.float32):
    """Accept numpy array or torch tensor; return torch tensor on desired device."""
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t.to(dtype)


def _gapless_indices(seq_A, seq_B, seq_len, device):
    """Indices where neither aligned sequence has a gap; clipped to attention matrix bounds."""
    mask = torch.tensor([a != "-" and b != "-" for a, b in zip(seq_A, seq_B)], device=device)
    idx = torch.where(mask)[0]
    return idx[idx < seq_len]


def p_distance(seq_A, seq_B, matrix_A=None, matrix_B=None) -> float:
    """
    Pairwise p-distance between two *aligned* sequences.
    p = (# mismatches) / (# compared sites), where compared sites exclude gaps in either sequence.

    Accepts matrix_A/matrix_B only for API compatibility with other distance functions.
    """
    if len(seq_A) != len(seq_B):
        raise ValueError(f"Aligned sequences must be same length. Got {len(seq_A)} vs {len(seq_B)}")

    compared = 0
    mismatches = 0

    for a, b in zip(seq_A, seq_B):
        # ignore sites with gaps in either sequence
        if a == "-" or b == "-":
            continue

        compared += 1
        if a != b:
            mismatches += 1

    if compared == 0:
        return 1.0  # no comparable sites -> max distance

    return mismatches / compared

def pearson_distance(seq_A, seq_B, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> float:
    """
    Compute distance between two matrices based on Pearson correlation.

    Parameters:
    - matrix_a: torch.Tensor - The first matrix.
    - matrix_b: torch.Tensor - The second matrix.

    Returns:
    - distance: float - The distance score between the two matrices.
    """

    # Accept numpy arrays or torch tensors
    if not torch.is_tensor(matrix_a):
        matrix_a = torch.as_tensor(matrix_a)
    if not torch.is_tensor(matrix_b):
        matrix_b = torch.as_tensor(matrix_b)

    # Flatten matrices and calculate correlation on GPU
    flattened_a = matrix_a.flatten()
    flattened_b = matrix_b.flatten()
    
    # bugcheck to see if this step is goofed or not
    if torch.isnan(flattened_a).any() or torch.isnan(flattened_b).any():
        raise ValueError("NaN values detected in input matrices.")

    correlation = torch.corrcoef(torch.stack((flattened_a, flattened_b)))[0, 1].abs()
    
    # Calculate distance as 1 minus the absolute value of correlation
    distance = 1 - correlation.item()
    return distance

def gapless_distance(seq_A, seq_B, matrix_A, matrix_B):
    
    #-------------------------[REMOVE GAP POSITIONS]---------------------------
    # Identify gap positions where either sequence has a gap
    gap_positions = [i for i, (res_A, res_B) in enumerate(zip(seq_A, seq_B)) if res_A == '-' or res_B == '-']

    # Flatten matrices before applying the gap mask
    flattened_A = matrix_A.flatten().to(torch.float64)
    flattened_B = matrix_B.flatten().to(torch.float64)

    # Create a mask to exclude gap positions
    mask = torch.tensor([i not in gap_positions for i in range(len(flattened_A))], dtype=torch.bool)
    filtered_matrix_A = flattened_A[mask]
    filtered_matrix_B = flattened_B[mask]

    # Compute mean-centered vectors
    mean_A = filtered_matrix_A.mean()
    mean_B = filtered_matrix_B.mean()
    centered_A = filtered_matrix_A - mean_A
    centered_B = filtered_matrix_B - mean_B
    #--------------------------------------------------------------------------
    
    #-------------------------[CALCULATE PEARSON'S R]--------------------------
    numerator = torch.dot(centered_A, centered_B)
    denominator = torch.sqrt(torch.dot(centered_A, centered_A) * torch.dot(centered_B, centered_B))
    
    # Check for near-zero denominator to avoid NaN
    if denominator < 1e-10:
        print("Low denominator in correlation calculation; setting distance to 1.")
        return 1.0  # Maximum distance if correlation is undefined
    
    # Calculate correlation and distance
    correlation = numerator / denominator
    distance = 1 - correlation.abs().item()  # Use absolute value to ensure positive similarity
    #--------------------------------------------------------------------------

    return distance

def new_gapless_distance(seq_A, seq_B, matrix_A, matrix_B):
    """
    Computes correlation-based distance while ensuring matrices remain aligned.
    - Uses a sequence-derived mask to filter attention matrices.
    - Also, normalizes based on remaining sequence length
    """
    device = matrix_A.device
    
    #-------------------------[REMOVE GAP POSITIONS]--------------------------#
    non_gap_mask = torch.tensor(
        [a != '-' and b != '-' for a, b in zip(seq_A, seq_B)], device=device
    )
    non_gap_indices = torch.where(non_gap_mask)[0]
    
    # If there are no valid positions, return max distance
    valid_positions = len(non_gap_indices)
    if valid_positions == 0:
        return 1.0  # No valid positions = maximum distance

    # Apply the mask to both rows and columns of the matrices
    filtered_A = matrix_A[non_gap_indices][:, non_gap_indices]
    filtered_B = matrix_B[non_gap_indices][:, non_gap_indices]

    # Flatten the filtered matrices
    filtered_A = filtered_A.flatten()
    filtered_B = filtered_B.flatten()

    # Compute mean-centered vectors
    mean_A = filtered_A.mean()
    mean_B = filtered_B.mean()
    centered_A = filtered_A - mean_A
    centered_B = filtered_B - mean_B
    #-------------------------------------------------------------------------#
    
    #-------------------------[CALCULATE PEARSON'S R]-------------------------#
    numerator = torch.dot(centered_A, centered_B)
    denominator = torch.sqrt(torch.dot(centered_A, centered_A) * torch.dot(centered_B, centered_B))

    if denominator < 1e-10:
        return 1.0  # Avoid divide-by-zero

    correlation = numerator / denominator

    # Normalize correlation by valid positions to prevent bias from gaps
    adjusted_correlation = correlation * (valid_positions / len(seq_A))
    #-------------------------------------------------------------------------#
    
    return 1 - adjusted_correlation.abs().item()

def psa_gapless_distance(seq_A, seq_B, matrix_A, matrix_B):
    """
    Computes correlation-based distance while ensuring matrices remain aligned.
    - Uses a sequence-derived mask to filter attention matrices.
    - Normalizes based on remaining sequence length.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU timing requires CUDA.")

    device = torch.device("cuda")

    if not torch.is_tensor(matrix_A):
        matrix_A = torch.as_tensor(matrix_A, dtype=torch.float32, device=device)
    else:
        matrix_A = matrix_A.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(matrix_B):
        matrix_B = torch.as_tensor(matrix_B, dtype=torch.float32, device=device)
    else:
        matrix_B = matrix_B.to(device=device, dtype=torch.float32)

    # Create mask for non-gap positions
    non_gap_mask = torch.tensor(
        [a != '-' and b != '-' for a, b in zip(seq_A, seq_B)], device=device
    )
    non_gap_indices = torch.where(non_gap_mask)[0]

    # Ensure valid indices within matrix bounds
    seq_len = matrix_A.shape[0]  # Sequence length should be first dimension (no batch)
    non_gap_indices = non_gap_indices[non_gap_indices < seq_len]

    if len(non_gap_indices) == 0:
        return 1.0  # No valid positions = maximum distance

    # Correct indexing: Apply non-gap filtering only to the last two dimensions
    filtered_A = matrix_A[non_gap_indices][:, non_gap_indices]  # No batch dim, just seq positions
    filtered_B = matrix_B[non_gap_indices][:, non_gap_indices]

    # convert to tensors
    filtered_A = torch.tensor(filtered_A, dtype=torch.float32, device=device)
    filtered_B = torch.tensor(filtered_B, dtype=torch.float32, device=device)

    # Flatten the filtered matrices
    filtered_A = filtered_A.flatten()
    filtered_B = filtered_B.flatten()

    # Compute mean-centered vectors
    mean_A = filtered_A.mean()
    mean_B = filtered_B.mean()
    centered_A = filtered_A - mean_A
    centered_B = filtered_B - mean_B

    # Compute Pearson correlation
    numerator = torch.dot(centered_A, centered_B)
    denominator = torch.sqrt(torch.dot(centered_A, centered_A) * torch.dot(centered_B, centered_B))

    if denominator < 1e-10:
        return 1.0  # Avoid divide-by-zero

    correlation = numerator / denominator

    return 1 - correlation.abs().item()

def psa_normalized_distance(seq_A, seq_B, matrix_A, matrix_B):
    """
    Computes correlation-based distance while ensuring matrices remain aligned.
    - Uses a sequence-derived mask to filter attention matrices.
    - Normalizes based on remaining sequence length.
    """
    device = matrix_A.device

    # Create mask for non-gap positions
    non_gap_mask = torch.tensor(
        [a != '-' and b != '-' for a, b in zip(seq_A, seq_B)], device=device
    )
    non_gap_indices = torch.where(non_gap_mask)[0]

    # Ensure valid indices within matrix bounds
    seq_len = matrix_A.shape[0]  # Sequence length should be first dimension (no batch)
    non_gap_indices = non_gap_indices[non_gap_indices < seq_len]

    if len(non_gap_indices) == 0:
        return 1.0  # No valid positions = maximum distance

    # Correct indexing: Apply non-gap filtering only to the last two dimensions
    filtered_A = matrix_A[non_gap_indices][:, non_gap_indices]  # No batch dim, just seq positions
    filtered_B = matrix_B[non_gap_indices][:, non_gap_indices]

    # convert to tensors
    filtered_A = torch.tensor(filtered_A, dtype=torch.float32, device=device)
    filtered_B = torch.tensor(filtered_B, dtype=torch.float32, device=device)

    # Flatten the filtered matrices
    filtered_A = filtered_A.flatten()
    filtered_B = filtered_B.flatten()

    # Compute mean-centered vectors
    mean_A = filtered_A.mean()
    mean_B = filtered_B.mean()
    centered_A = filtered_A - mean_A
    centered_B = filtered_B - mean_B

    # Compute Pearson correlation
    numerator = torch.dot(centered_A, centered_B)
    denominator = torch.sqrt(torch.dot(centered_A, centered_A) * torch.dot(centered_B, centered_B))

    if denominator < 1e-10:
        return 1.0  # Avoid divide-by-zero

    correlation = numerator / denominator
    distance = (1 - correlation.abs()) / seq_len

    return distance




def euclidean_distance(seq_A, seq_B, matrix_A, matrix_B, gapless=True) -> float:
    """
    Euclidean (L2) distance between two attention matrices after flattening.
    If gapless=True, removes rows/cols corresponding to gapped alignment positions.
    """
    # choose device from matrix_A if possible
    device = matrix_A.device if torch.is_tensor(matrix_A) else (matrix_B.device if torch.is_tensor(matrix_B) else None)

    A = _as_tensor(matrix_A, device=device, dtype=torch.float32)
    B = _as_tensor(matrix_B, device=device, dtype=torch.float32)

    if A.shape != B.shape:
        raise ValueError(f"Matrix shapes differ: {tuple(A.shape)} vs {tuple(B.shape)}")

    if gapless and (seq_A is not None) and (seq_B is not None):
        seq_len = A.shape[0]
        idx = _gapless_indices(seq_A, seq_B, seq_len, A.device)
        if len(idx) == 0:
            return 1.0
        A = A[idx][:, idx]
        B = B[idx][:, idx]

    diff = (A - B).flatten()
    return torch.linalg.vector_norm(diff, ord=2).item()

def cosine_distance(seq_A, seq_B, matrix_A, matrix_B) -> float:
    """
    Cosine distance between flattened attention matrices: 1 - cos_sim.
    Uses gapless masking (orthologous positions only).
    """
    device = matrix_A.device if torch.is_tensor(matrix_A) else None
    A = _as_tensor(matrix_A, device=device)
    B = _as_tensor(matrix_B, device=device)

    if A.shape != B.shape:
        raise ValueError(f"Matrix shapes differ: {tuple(A.shape)} vs {tuple(B.shape)}")

    idx = _gapless_indices(seq_A, seq_B, A.shape[0], A.device)
    if len(idx) == 0:
        return 1.0

    va = A[idx][:, idx].flatten()
    vb = B[idx][:, idx].flatten()

    denom = (torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb))
    if denom < 1e-12:
        return 1.0

    cos_sim = torch.dot(va, vb) / denom
    return (1.0 - cos_sim).item()

def frobenius_distance(seq_A, seq_B, matrix_A, matrix_B, gapless=True) -> float:
    """
    Frobenius norm distance: ||A - B||_F.
    If gapless=True, removes rows/cols corresponding to gapped alignment positions.
    """
    device = matrix_A.device if torch.is_tensor(matrix_A) else (matrix_B.device if torch.is_tensor(matrix_B) else None)

    A = _as_tensor(matrix_A, device=device, dtype=torch.float32)
    B = _as_tensor(matrix_B, device=device, dtype=torch.float32)

    if A.shape != B.shape:
        raise ValueError(f"Matrix shapes differ: {tuple(A.shape)} vs {tuple(B.shape)}")

    if gapless and (seq_A is not None) and (seq_B is not None):
        seq_len = A.shape[0]
        idx = _gapless_indices(seq_A, seq_B, seq_len, A.device)
        if len(idx) == 0:
            return 1.0
        A = A[idx][:, idx]
        B = B[idx][:, idx]

    return torch.linalg.matrix_norm(A - B, ord="fro").item()