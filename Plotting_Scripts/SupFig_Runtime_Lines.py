#!/usr/bin/env python3

import csv
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

SYSTEM_NAME = "Korolev"

N_REPLICATES = 10

# Current RAM treebuilder supports CPU workers for the Hellinger calculations.
# Leave at 1 to time the default pipeline behavior.
RAM_WORKERS = 1

# Keep the complete tree outputs from every timing replicate?
# False is recommended because 20 proteins x 10 reps x 2 methods is large.
KEEP_TREE_OUTPUTS = False


# =============================================================================
# OMAM20 proteins
# =============================================================================

PROTEINS = [
    "AIMP2", "APOBEC2", "AR", "CASQ1", "COCH",
    "DYRK1B", "GPX3", "HAUS1", "IL6", "INO80B",
    "KRT23", "NEUROG1", "OPN1SW", "PPOX", "RHO",
    "SLC39A12", "TF", "TPPP2", "UBA2", "UPRT",
]


# =============================================================================
# pFM-phylo paths
# =============================================================================

BASE = "/home/rohan/pFM-phylo"

RAM_SCRIPT = f"{BASE}/Scripts/RAM_Treebuilder.py"
SAM_SCRIPT = f"{BASE}/Scripts/SAM_Treebuilder.py"

RAM_INPUT_DIR = f"{BASE}/Data/Alignments/OMAM20/OMAM20_34spp"
SAM_INPUT_DIR = f"{BASE}/Data/Alignments/OMAM20/OMAM20_34sppAligned"

MODEL = f"{BASE}/Models/ESM2_35M_FULLMODULE.pt"
TOKENIZER = f"{BASE}/Models/ESM2_alphabet.pkl"

ASTRAL = "/home/rohan/Tools/Astral.5.7.8/Astral/astral.5.7.8.jar"


# =============================================================================
# Timing output
# =============================================================================

TIMING_DIR = f"{BASE}/Results/GPU_Timing"

RAM_TIMING_TSV = (
    f"{TIMING_DIR}/{SYSTEM_NAME}_RAM_Hellinger_treebuild_timing.tsv"
)

SAM_TIMING_TSV = (
    f"{TIMING_DIR}/{SYSTEM_NAME}_SAM_Hellinger_treebuild_timing.tsv"
)

# Existing ML+MFP timings: DO NOT rerun IQ-TREE
ML_TIMING_TSV = (
    "/home/rohan/PLM-Phylo-Dev/Results/GPU_Timing/"
    "Korolev_IQTREE_MFP_treebuild_timing.tsv"
)

LOG_DIR = f"{TIMING_DIR}/treebuilding_logs"

RAM_LOG_DIR = f"{LOG_DIR}/{SYSTEM_NAME}_RAM_Hellinger"
SAM_LOG_DIR = f"{LOG_DIR}/{SYSTEM_NAME}_SAM_Hellinger"

# Temporary treebuilding outputs used only during timing
TIMING_TREE_DIR = f"{TIMING_DIR}/timing_tree_outputs"

RAM_TIMING_TREE_DIR = f"{TIMING_TREE_DIR}/RAM"
SAM_TIMING_TREE_DIR = f"{TIMING_TREE_DIR}/SAM"


# =============================================================================
# Final figure
# =============================================================================

FIGURE_DIR = f"{BASE}/Figures/Submission_Figures"
OUTPUT_PDF = f"{FIGURE_DIR}/Fig6b_{SYSTEM_NAME}_Hellinger_RAM_SAM_MFP_timing.pdf"
OUTPUT_SVG = f"{FIGURE_DIR}/Fig6b_{SYSTEM_NAME}_Hellinger_RAM_SAM_MFP_timing.svg"
DPI = 500

OUTPUT_CSV = (
    f"{TIMING_DIR}/{SYSTEM_NAME}_Hellinger_RAM_SAM_MFP_timing_summary.csv"
)


# =============================================================================
# TSV columns
# =============================================================================

TIMING_COLUMNS = [
    "system",
    "method",
    "threads",
    "alignment",
    "alignment_length",
    "seq_count",
    "replicate",
    "n_replicates",
    "status",
    "exit_code",
    "time_seconds",
]


# =============================================================================
# FASTA utilities
# =============================================================================

def count_sequences(fasta_path):
    with open(fasta_path) as handle:
        return sum(line.startswith(">") for line in handle)


def first_sequence_length(fasta_path, ungapped=True):
    seq_parts = []
    in_first_record = False

    with open(fasta_path) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if in_first_record:
                    break

                in_first_record = True
                continue

            if in_first_record:
                seq_parts.append(line)

    if not seq_parts:
        return None

    sequence = "".join(seq_parts)

    if ungapped:
        sequence = sequence.replace("-", "")

    return len(sequence)


# =============================================================================
# Timing TSV utilities
# =============================================================================

def create_timing_tsv(tsv_path):
    if os.path.exists(tsv_path):
        return

    with open(tsv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TIMING_COLUMNS,
            delimiter="\t",
        )

        writer.writeheader()


def append_timing_row(tsv_path, row):
    with open(tsv_path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TIMING_COLUMNS,
            delimiter="\t",
        )

        writer.writerow(row)


def load_completed_runs(tsv_path):
    """
    Return successful (alignment, replicate) combinations already present.

    This makes the pipeline resumable if it is interrupted.
    """

    completed = set()

    if not os.path.exists(tsv_path):
        return completed

    try:
        df = pd.read_csv(tsv_path, sep="\t")

    except Exception:
        return completed

    if len(df) == 0:
        return completed

    df = df[df["status"] == "success"]

    for _, row in df.iterrows():
        completed.add(
            (
                str(row["alignment"]),
                int(row["replicate"]),
            )
        )

    return completed


# =============================================================================
# Validation
# =============================================================================

def validate_paths():
    required_files = [
        RAM_SCRIPT,
        SAM_SCRIPT,
        MODEL,
        TOKENIZER,
        ASTRAL,
        ML_TIMING_TSV,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    required_dirs = [
        RAM_INPUT_DIR,
        SAM_INPUT_DIR,
    ]

    for path in required_dirs:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing required directory:\n{path}"
            )

    for protein in PROTEINS:
        ram_fasta = (
            f"{RAM_INPUT_DIR}/{protein}_34spp.fas"
        )

        sam_fasta = (
            f"{SAM_INPUT_DIR}/{protein}_34sppAligned.fas"
        )

        if not os.path.exists(ram_fasta):
            raise FileNotFoundError(
                f"Missing RAM input:\n{ram_fasta}"
            )

        if not os.path.exists(sam_fasta):
            raise FileNotFoundError(
                f"Missing SAM input:\n{sam_fasta}"
            )


# =============================================================================
# Command builders
# =============================================================================

def build_ram_command(fasta_path, domain, out_dir):
    return [
        sys.executable,
        RAM_SCRIPT,

        "--model",
        MODEL,

        "--tokenizer",
        TOKENIZER,

        "--input",
        fasta_path,

        "--out",
        out_dir,

        "--domain",
        domain,

        "--astral",
        ASTRAL,

        "--workers",
        str(RAM_WORKERS),
    ]


def build_sam_command(fasta_path, domain, out_dir):
    return [
        sys.executable,
        SAM_SCRIPT,

        "--msa",
        fasta_path,

        "--out",
        out_dir,

        "--domain",
        domain,

        "--astral",
        ASTRAL,
    ]


# =============================================================================
# Run one timed treebuilding replicate
# =============================================================================

def run_timed_treebuild(
    method,
    alignment,
    alignment_length,
    seq_count,
    replicate,
    command,
    run_output_dir,
    log_file,
    timing_tsv,
    threads,
):
    # -------------------------------------------------------------------------
    # Remove any incomplete/stale output from an earlier failed run
    # -------------------------------------------------------------------------

    if os.path.exists(run_output_dir):
        shutil.rmtree(run_output_dir)

    os.makedirs(run_output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Run treebuilder
    # -------------------------------------------------------------------------

    start = time.perf_counter()

    try:
        with open(log_file, "w") as log_handle:
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        exit_code = completed.returncode

    except Exception as error:
        elapsed = time.perf_counter() - start

        with open(log_file, "a") as log_handle:
            log_handle.write(
                f"\nTIMING WRAPPER ERROR:\n{error}\n"
            )

        exit_code = -1

    else:
        elapsed = time.perf_counter() - start

    status = (
        "success"
        if exit_code == 0
        else "failed"
    )

    # -------------------------------------------------------------------------
    # Save timing
    # -------------------------------------------------------------------------

    row = {
        "system": SYSTEM_NAME,
        "method": method,
        "threads": threads,
        "alignment": alignment,
        "alignment_length": alignment_length,
        "seq_count": seq_count,
        "replicate": replicate,
        "n_replicates": N_REPLICATES,
        "status": status,
        "exit_code": exit_code,
        "time_seconds": f"{elapsed:.6f}",
    }

    append_timing_row(
        timing_tsv,
        row,
    )

    # -------------------------------------------------------------------------
    # Delete temporary tree output AFTER timing has stopped
    # -------------------------------------------------------------------------

    if not KEEP_TREE_OUTPUTS:
        if os.path.exists(run_output_dir):
            shutil.rmtree(run_output_dir)

    return status, elapsed


# =============================================================================
# RAM timing
# =============================================================================

def time_ram():
    print()
    print("============================================================")
    print("RAM HELLINGER TIMING")
    print("============================================================")

    create_timing_tsv(
        RAM_TIMING_TSV
    )

    completed_runs = load_completed_runs(
        RAM_TIMING_TSV
    )

    total_runs = (
        len(PROTEINS)
        * N_REPLICATES
    )

    with tqdm(
        total=total_runs,
        desc="RAM timing",
        unit="run",
    ) as pbar:

        for protein in PROTEINS:
            domain = f"{protein}_34spp"

            fasta_path = (
                f"{RAM_INPUT_DIR}/{domain}.fas"
            )

            alignment = os.path.basename(
                fasta_path
            )

            seq_count = count_sequences(
                fasta_path
            )

            protein_length = first_sequence_length(
                fasta_path,
                ungapped=True,
            )

            for replicate in range(
                1,
                N_REPLICATES + 1,
            ):
                key = (
                    alignment,
                    replicate,
                )

                if key in completed_runs:
                    tqdm.write(
                        f"RAM | {alignment} | "
                        f"rep {replicate} | already complete"
                    )

                    pbar.update(1)
                    continue

                run_name = (
                    f"{domain}_rep{replicate:02d}"
                )

                run_output_dir = (
                    f"{RAM_TIMING_TREE_DIR}/{run_name}"
                )

                log_file = (
                    f"{RAM_LOG_DIR}/"
                    f"{run_name}.treebuild.log"
                )

                command = build_ram_command(
                    fasta_path,
                    domain,
                    run_output_dir,
                )

                status, elapsed = run_timed_treebuild(
                    method="RAM Hellinger",
                    alignment=alignment,
                    alignment_length=protein_length,
                    seq_count=seq_count,
                    replicate=replicate,
                    command=command,
                    run_output_dir=run_output_dir,
                    log_file=log_file,
                    timing_tsv=RAM_TIMING_TSV,
                    threads=RAM_WORKERS,
                )

                tqdm.write(
                    f"RAM | {alignment} | "
                    f"rep {replicate}/{N_REPLICATES} | "
                    f"{status} | "
                    f"{elapsed:.3f} s"
                )

                pbar.update(1)


# =============================================================================
# SAM timing
# =============================================================================

def time_sam():
    print()
    print("============================================================")
    print("SAM HELLINGER TIMING")
    print("============================================================")

    create_timing_tsv(
        SAM_TIMING_TSV
    )

    completed_runs = load_completed_runs(
        SAM_TIMING_TSV
    )

    total_runs = (
        len(PROTEINS)
        * N_REPLICATES
    )

    with tqdm(
        total=total_runs,
        desc="SAM timing",
        unit="run",
    ) as pbar:

        for protein in PROTEINS:
            domain = (
                f"{protein}_34sppAligned"
            )

            fasta_path = (
                f"{SAM_INPUT_DIR}/{domain}.fas"
            )

            alignment = os.path.basename(
                fasta_path
            )

            seq_count = count_sequences(
                fasta_path
            )

            protein_length = first_sequence_length(
                fasta_path,
                ungapped=True,
            )

            for replicate in range(
                1,
                N_REPLICATES + 1,
            ):
                key = (
                    alignment,
                    replicate,
                )

                if key in completed_runs:
                    tqdm.write(
                        f"SAM | {alignment} | "
                        f"rep {replicate} | already complete"
                    )

                    pbar.update(1)
                    continue

                run_name = (
                    f"{domain}_rep{replicate:02d}"
                )

                run_output_dir = (
                    f"{SAM_TIMING_TREE_DIR}/{run_name}"
                )

                log_file = (
                    f"{SAM_LOG_DIR}/"
                    f"{run_name}.treebuild.log"
                )

                command = build_sam_command(
                    fasta_path,
                    domain,
                    run_output_dir,
                )

                status, elapsed = run_timed_treebuild(
                    method="SAM Hellinger",
                    alignment=alignment,
                    alignment_length=protein_length,
                    seq_count=seq_count,
                    replicate=replicate,
                    command=command,
                    run_output_dir=run_output_dir,
                    log_file=log_file,
                    timing_tsv=SAM_TIMING_TSV,
                    threads=1,
                )

                tqdm.write(
                    f"SAM | {alignment} | "
                    f"rep {replicate}/{N_REPLICATES} | "
                    f"{status} | "
                    f"{elapsed:.3f} s"
                )

                pbar.update(1)


# =============================================================================
# Summarize timing data
# =============================================================================

def load_and_summarize_timings():
    ram = pd.read_csv(
        RAM_TIMING_TSV,
        sep="\t",
    )

    sam = pd.read_csv(
        SAM_TIMING_TSV,
        sep="\t",
    )

    ml = pd.read_csv(
        ML_TIMING_TSV,
        sep="\t",
    )

    # -------------------------------------------------------------------------
    # Successful runs only
    # -------------------------------------------------------------------------

    ram = ram[
        ram["status"] == "success"
    ].copy()

    sam = sam[
        sam["status"] == "success"
    ].copy()

    ml = ml[
        ml["status"] == "success"
    ].copy()

    # -------------------------------------------------------------------------
    # Numeric columns
    # -------------------------------------------------------------------------

    for df in (
        ram,
        sam,
        ml,
    ):
        df["alignment_length"] = pd.to_numeric(
            df["alignment_length"]
        )

        df["time_seconds"] = pd.to_numeric(
            df["time_seconds"]
        )

    # -------------------------------------------------------------------------
    # Normalize protein names
    # -------------------------------------------------------------------------

    for df in (
        ram,
        sam,
        ml,
    ):
        df["protein"] = (
            df["alignment"]
            .str.replace(
                r"\.(fas|fasta|fa|faa)$",
                "",
                regex=True,
            )
        )

    # AIMP2_34sppAligned -> AIMP2_34spp
    sam["protein"] = sam[
        "protein"
    ].str.replace(
        r"Aligned$",
        "",
        regex=True,
    )

    ml["protein"] = ml[
        "protein"
    ].str.replace(
        r"Aligned$",
        "",
        regex=True,
    )

    # -------------------------------------------------------------------------
    # Average runtime across replicates
    # -------------------------------------------------------------------------

    ram_summary = (
        ram
        .groupby(
            "protein",
            as_index=False,
        )
        .agg(
            protein_length=(
                "alignment_length",
                "first",
            ),
            ram_runtime=(
                "time_seconds",
                "mean",
            ),
            ram_sd=(
                "time_seconds",
                "std",
            ),
            ram_n=(
                "time_seconds",
                "count",
            ),
        )
    )

    sam_summary = (
        sam
        .groupby(
            "protein",
            as_index=False,
        )
        .agg(
            sam_runtime=(
                "time_seconds",
                "mean",
            ),
            sam_sd=(
                "time_seconds",
                "std",
            ),
            sam_n=(
                "time_seconds",
                "count",
            ),
        )
    )

    ml_summary = (
        ml
        .groupby(
            "protein",
            as_index=False,
        )
        .agg(
            ml_runtime=(
                "time_seconds",
                "mean",
            ),
            ml_sd=(
                "time_seconds",
                "std",
            ),
            ml_n=(
                "time_seconds",
                "count",
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Match proteins present in all three
    # -------------------------------------------------------------------------

    comparison = pd.merge(
        ram_summary,
        sam_summary,
        on="protein",
        how="inner",
    )

    comparison = pd.merge(
        comparison,
        ml_summary,
        on="protein",
        how="inner",
    )

    comparison = comparison.sort_values(
        "protein_length"
    )

    return (
        ram_summary,
        sam_summary,
        ml_summary,
        comparison,
    )


# =============================================================================
# Plot
# =============================================================================

def make_plot():
    (
        ram_summary,
        sam_summary,
        ml_summary,
        comparison,
    ) = load_and_summarize_timings()

    print()
    print("============================================================")
    print("TIMING SUMMARY")
    print("============================================================")

    print(
        f"RAM proteins:     "
        f"{len(ram_summary)}"
    )

    print(
        f"SAM proteins:     "
        f"{len(sam_summary)}"
    )

    print(
        f"ML+MFP proteins:  "
        f"{len(ml_summary)}"
    )

    print(
        f"Shared proteins:  "
        f"{len(comparison)}"
    )

    if len(comparison) == 0:
        raise ValueError(
            "No proteins are shared across "
            "RAM, SAM, and ML+MFP timing datasets."
        )

    # -------------------------------------------------------------------------
    # Save numerical summary
    # -------------------------------------------------------------------------

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print data
    # -------------------------------------------------------------------------

    print()
    print(
        comparison[
            [
                "protein",
                "protein_length",
                "ram_runtime",
                "sam_runtime",
                "ml_runtime",
                "ram_n",
                "sam_n",
                "ml_n",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    TICK_SIZE = 10
    AXES_WIDTH = 7.0
    AXES_HEIGHT = 2.0
    
    LEFT_MARGIN = 0.95
    RIGHT_MARGIN = 0.25
    BOTTOM_MARGIN = 0.65
    TOP_MARGIN = 0.20
    
    FIG_WIDTH = LEFT_MARGIN + AXES_WIDTH + RIGHT_MARGIN
    FIG_HEIGHT = BOTTOM_MARGIN + AXES_HEIGHT + TOP_MARGIN
    
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.add_axes([
        LEFT_MARGIN / FIG_WIDTH,
        BOTTOM_MARGIN / FIG_HEIGHT,
        AXES_WIDTH / FIG_WIDTH,
        AXES_HEIGHT / FIG_HEIGHT
    ])

    ax.plot(
        comparison["protein_length"],
        comparison["ram_runtime"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="dodgerblue",
        label=r"$\it{RAM}$",
    )

    ax.plot(
        comparison["protein_length"],
        comparison["sam_runtime"],
        marker="D",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        color="orange",
        label="$\it{SAM}$",
    )

    ax.plot(
        comparison["protein_length"],
        comparison["ml_runtime"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="green",
        label="ML+MFP",
    )

    ax.set_xlabel(
        "Protein Length (aa)",
        fontsize=14,
    )

    ax.set_ylabel(
        "mean runtime (s)",
        fontsize=14,
    )

    ax.tick_params(
    axis="both",
    labelsize=TICK_SIZE,
    )

    ax.set_axisbelow(True)
    ax.grid(
        alpha=0.5
    )

    ax.legend(
        frameon=False,
    )


    plt.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    plt.savefig(OUTPUT_SVG, dpi=DPI, bbox_inches="tight")

    plt.show()
    plt.close()

    print()
    print(
        f"Timing summary:\n"
        f"{OUTPUT_CSV}"
    )

    print(f"\nPDF figure:\n{OUTPUT_PDF}")
    print(f"\nSVG figure:\n{OUTPUT_SVG}")


# =============================================================================
# Main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_paths()

    # -------------------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------------------

    os.makedirs(
        TIMING_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RAM_LOG_DIR,
        exist_ok=True,
    )

    os.makedirs(
        SAM_LOG_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RAM_TIMING_TREE_DIR,
        exist_ok=True,
    )

    os.makedirs(
        SAM_TIMING_TREE_DIR,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Configuration summary
    # -------------------------------------------------------------------------

    print(
        "============================================================"
    )

    print(
        "HELLINGER RAM / SAM TREEBUILDING TIMING PIPELINE"
    )

    print(
        "============================================================"
    )

    print(
        f"System: {SYSTEM_NAME}"
    )

    print(
        f"Proteins: {len(PROTEINS)}"
    )

    print(
        f"Replicates: {N_REPLICATES}"
    )

    print(
        f"RAM workers: {RAM_WORKERS}"
    )

    print(
        f"Keep timing trees: "
        f"{KEEP_TREE_OUTPUTS}"
    )

    print()
    print(
        "RAM timing TSV:"
    )

    print(
        RAM_TIMING_TSV
    )

    print()
    print(
        "SAM timing TSV:"
    )

    print(
        SAM_TIMING_TSV
    )

    print()
    print(
        "Existing ML+MFP timing TSV:"
    )

    print(
        ML_TIMING_TSV
    )

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    time_ram()

    time_sam()

    # -------------------------------------------------------------------------
    # Final three-method figure
    # -------------------------------------------------------------------------

    make_plot()


if __name__ == "__main__":
    main()