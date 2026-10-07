#!/usr/bin/env python3
"""
Compare an independently generated Python reference matrix against a
Galeri/Tpetra Matrix Market matrix.

Validation separation
---------------------
The reference generator (galeri_validate.py) must never read the matrix being
validated.  This program is the first stage allowed to read both matrices.

Typical use:

    ./matrix_compare.py \\
        Laplace3D_10x10x10.python.mtx.gz \\
        Laplace3D_10x10x10.mtx.gz \\
        --atol 0 --rtol 0

Both plain .mtx and gzip-compressed .mtx.gz inputs are supported.

The common output prefix is inferred automatically.  In this example the
visualizations are written as:

    Laplace3D_10x10x10.python.spy.png
    Laplace3D_10x10x10.tpetra.spy.png
    Laplace3D_10x10x10.difference.spy.png
    Laplace3D_10x10x10.offset-histogram.png

Spy output and the diagonal-offset histogram are enabled by default.  Use
--no-spy or --no-histogram to disable either one.  --spy-prefix remains
available to override the derived plot prefix/path.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread


def _mmread_once(path: Path, **kwargs):
    """Read one Matrix Market object, opening .gz files explicitly."""

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return mmread(stream, **kwargs)
    return mmread(path, **kwargs)


def _mmread(path: Path):
    """Read Matrix Market data without depending on SciPy's changing default."""

    try:
        return _mmread_once(path, spmatrix=True)
    except TypeError:
        # Compatibility with older SciPy releases that lack spmatrix=.
        # Reopen compressed inputs so the fallback always starts at byte zero.
        return _mmread_once(path)


def load_sparse_matrix(path: str | Path) -> sp.csr_matrix:
    """Read and canonicalize a Matrix Market matrix."""

    path = Path(path)
    obj = _mmread(path)

    if sp.issparse(obj):
        A = obj.tocsr()
    else:
        A = sp.csr_matrix(np.asarray(obj))

    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    return A


def _strip_expected_suffix(name: str, suffix: str) -> str:
    """Strip an expected Matrix Market suffix, with optional gzip compression."""

    if name.endswith(".gz"):
        name = name[:-len(".gz")]
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return Path(name).stem


def derive_output_prefix(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> Path:
    """Infer the common problem prefix from the two Matrix Market inputs."""

    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)

    ref_base = _strip_expected_suffix(reference_path.name, ".python.mtx")
    cand_base = _strip_expected_suffix(candidate_path.name, ".mtx")

    if ref_base == cand_base:
        return reference_path.parent / ref_base

    common = os.path.commonprefix([ref_base, cand_base]).rstrip("._-")
    if common:
        return reference_path.parent / common

    # Rare fallback for unrelated names.  Keeping both names is clearer than
    # silently choosing one input's stem.
    return reference_path.parent / f"{ref_base}.vs.{cand_base}"


def same_pattern(A: sp.csr_matrix, B: sp.csr_matrix) -> bool:
    """True iff canonical CSR matrices have exactly the same nonzero positions."""

    return (
        A.shape == B.shape
        and np.array_equal(A.indptr, B.indptr)
        and np.array_equal(A.indices, B.indices)
    )


def pattern_differences(
    reference: sp.csr_matrix,
    candidate: sp.csr_matrix,
):
    """Return coordinates present only in each matrix."""

    ref_pattern = reference.copy()
    ref_pattern.data = np.ones(ref_pattern.nnz, dtype=np.int8)

    cand_pattern = candidate.copy()
    cand_pattern.data = np.ones(candidate.nnz, dtype=np.int8)

    delta = (ref_pattern - cand_pattern).tocoo()
    delta.sum_duplicates()
    delta.eliminate_zeros()

    only_reference = []
    only_candidate = []

    for row, col, value in zip(delta.row, delta.col, delta.data):
        if value > 0:
            only_reference.append((int(row), int(col)))
        elif value < 0:
            only_candidate.append((int(row), int(col)))

    return only_reference, only_candidate


def format_scalar(value) -> str:
    if np.iscomplexobj(value):
        return f"{value.real:.17g}{value.imag:+.17g}j"
    return f"{value:.17g}"


def compare_values(
    reference: sp.csr_matrix,
    candidate: sp.csr_matrix,
    *,
    atol: float,
    rtol: float,
):
    """Compare values after an exact sparsity-pattern match."""

    diff = candidate.data - reference.data
    abs_diff = np.abs(diff)

    tolerance = atol + rtol * np.abs(reference.data)
    bad = abs_diff > tolerance

    max_abs_error = float(abs_diff.max()) if abs_diff.size else 0.0

    ref_norm = float(np.linalg.norm(reference.data))
    diff_norm = float(np.linalg.norm(diff))

    if ref_norm == 0.0:
        relative_frobenius = 0.0 if diff_norm == 0.0 else float("inf")
    else:
        relative_frobenius = diff_norm / ref_norm

    return {
        "bad": bad,
        "num_bad": int(np.count_nonzero(bad)),
        "max_abs_error": max_abs_error,
        "frobenius_error": diff_norm,
        "relative_frobenius_error": relative_frobenius,
        "exact_values": np.array_equal(reference.data, candidate.data),
    }


def csr_entry_coordinates(A: sp.csr_matrix, data_indices: np.ndarray):
    """Map positions in CSR .data to (row, column)."""

    rows = np.searchsorted(A.indptr, data_indices, side="right") - 1
    cols = A.indices[data_indices]
    return rows, cols


def write_spy_plots(
    reference: sp.csr_matrix,
    candidate: sp.csr_matrix,
    prefix: str | Path,
) -> list[Path]:
    """Write Python, Tpetra, and (when possible) difference spy plots."""

    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    output = Path(str(prefix) + ".python.spy.png")
    fig = plt.figure()
    plt.spy(reference, markersize=1)
    plt.title("Python reference")
    plt.xlabel("column")
    plt.ylabel("row")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    outputs.append(output)

    output = Path(str(prefix) + ".tpetra.spy.png")
    fig = plt.figure()
    plt.spy(candidate, markersize=1)
    plt.title("Tpetra / Galeri")
    plt.xlabel("column")
    plt.ylabel("row")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    outputs.append(output)

    if reference.shape == candidate.shape:
        delta = (candidate - reference).tocsr()
        delta.eliminate_zeros()

        output = Path(str(prefix) + ".difference.spy.png")
        fig = plt.figure()
        plt.spy(delta, markersize=1)
        plt.title("Nonzero difference locations")
        plt.xlabel("column")
        plt.ylabel("row")
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)
        outputs.append(output)

    return outputs


def diagonal_offset_counts(
    A: sp.csr_matrix,
    *,
    chunk_rows: int = 100_000,
) -> dict[int, int]:
    """
    Count nonzeros by diagonal offset (column - row) with bounded memory.

    A direct COO conversion of a multi-million-row matrix materializes a full
    row-index array.  Chunking keeps this useful for the solver-scale matrices
    for which the histogram is most valuable.
    """

    counts: dict[int, int] = {}
    nrows = A.shape[0]

    for row0 in range(0, nrows, chunk_rows):
        row1 = min(row0 + chunk_rows, nrows)
        start = int(A.indptr[row0])
        stop = int(A.indptr[row1])

        if start == stop:
            continue

        row_nnz = np.diff(A.indptr[row0 : row1 + 1])
        rows = np.repeat(
            np.arange(row0, row1, dtype=np.int64),
            row_nnz,
        )
        cols = A.indices[start:stop].astype(np.int64, copy=False)
        offsets = cols - rows

        unique, local_counts = np.unique(offsets, return_counts=True)
        for offset, count in zip(unique, local_counts):
            key = int(offset)
            counts[key] = counts.get(key, 0) + int(count)

    return dict(sorted(counts.items()))


def print_offset_summary(
    reference_counts: dict[int, int],
    candidate_counts: dict[int, int],
    *,
    max_rows: int = 32,
) -> None:
    all_offsets = sorted(set(reference_counts) | set(candidate_counts))

    print()
    print(f"Diagonal offsets            : {len(all_offsets)} unique")

    if len(all_offsets) > max_rows:
        print(
            f"Offset table                 : omitted ({len(all_offsets)} entries; "
            "see histogram)"
        )
        return

    print()
    print("Diagonal offset counts (column - row):")
    print(f"{'offset':>14} {'python':>14} {'tpetra':>14}")
    for offset in all_offsets:
        print(
            f"{offset:14d} "
            f"{reference_counts.get(offset, 0):14d} "
            f"{candidate_counts.get(offset, 0):14d}"
        )


def write_offset_histogram(
    reference_counts: dict[int, int],
    candidate_counts: dict[int, int],
    prefix: str | Path,
    *,
    max_bars: int = 128,
) -> Path:
    """Write a categorical histogram of diagonal-offset populations."""

    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output = Path(str(prefix) + ".offset-histogram.png")

    all_offsets = sorted(set(reference_counts) | set(candidate_counts))

    if len(all_offsets) > max_bars:
        # Preserve the most populated offsets in the plot; the textual
        # comparison remains exact regardless of this visualization limit.
        ranked = sorted(
            all_offsets,
            key=lambda offset: max(
                reference_counts.get(offset, 0),
                candidate_counts.get(offset, 0),
            ),
            reverse=True,
        )[:max_bars]
        offsets = sorted(ranked)
        title = f"Diagonal offsets (top {max_bars} of {len(all_offsets)})"
    else:
        offsets = all_offsets
        title = "Diagonal offset histogram"

    x = np.arange(len(offsets), dtype=np.float64)
    ref = np.asarray([reference_counts.get(v, 0) for v in offsets])
    cand = np.asarray([candidate_counts.get(v, 0) for v in offsets])

    width = 0.42
    figure_width = min(24.0, max(8.0, 0.55 * max(1, len(offsets))))
    fig = plt.figure(figsize=(figure_width, 6.0))
    plt.bar(x - width / 2.0, ref, width=width, label="Python")
    plt.bar(x + width / 2.0, cand, width=width, label="Tpetra")
    plt.xticks(x, [str(v) for v in offsets], rotation=90 if len(offsets) > 12 else 0)
    plt.xlabel("column - row")
    plt.ylabel("nonzeros")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    return output


def compare(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    atol: float,
    rtol: float,
    max_report: int,
    spy: bool,
    histogram: bool,
    plot_prefix: str | Path,
) -> bool:
    reference = load_sparse_matrix(reference_path)
    candidate = load_sparse_matrix(candidate_path)

    print(f"Reference : {reference_path}")
    print(f"Candidate : {candidate_path}")
    print(f"Plot prefix: {plot_prefix}")
    print()
    print(f"Reference shape : {reference.shape}")
    print(f"Candidate shape : {candidate.shape}")
    print(f"Reference nnz   : {reference.nnz}")
    print(f"Candidate nnz   : {candidate.nnz}")

    reference_offsets = None
    candidate_offsets = None

    if histogram:
        reference_offsets = diagonal_offset_counts(reference)
        candidate_offsets = diagonal_offset_counts(candidate)
        print_offset_summary(reference_offsets, candidate_offsets)
        histogram_path = write_offset_histogram(
            reference_offsets,
            candidate_offsets,
            plot_prefix,
        )
        print()
        print(f"Offset histogram            : {histogram_path}")

    if spy:
        spy_paths = write_spy_plots(reference, candidate, plot_prefix)
        print("Spy plots                   : " + ", ".join(str(p) for p in spy_paths))

    if reference.shape != candidate.shape:
        print()
        print("FAIL: matrix shapes differ")
        return False

    pattern_equal = same_pattern(reference, candidate)

    print()
    print(f"Sparsity pattern identical : {'YES' if pattern_equal else 'NO'}")

    if not pattern_equal:
        only_reference, only_candidate = pattern_differences(reference, candidate)

        print(f"Entries only in reference  : {len(only_reference)}")
        print(f"Entries only in candidate  : {len(only_candidate)}")

        if only_reference:
            print()
            print("First entries present only in Python reference:")
            for row, col in only_reference[:max_report]:
                print(f"  ({row}, {col})")

        if only_candidate:
            print()
            print("First entries present only in Tpetra/Galeri:")
            for row, col in only_candidate[:max_report]:
                print(f"  ({row}, {col})")

        print()
        print("FAIL: sparsity patterns differ")
        return False

    result = compare_values(
        reference,
        candidate,
        atol=atol,
        rtol=rtol,
    )

    print(f"Values exactly identical   : {'YES' if result['exact_values'] else 'NO'}")
    print(f"Absolute tolerance         : {atol:.17g}")
    print(f"Relative tolerance         : {rtol:.17g}")
    print(f"Entries outside tolerance  : {result['num_bad']}")
    print(f"Max absolute error         : {result['max_abs_error']:.17g}")
    print(f"Frobenius error            : {result['frobenius_error']:.17g}")
    print(
        "Relative Frobenius error   : "
        f"{result['relative_frobenius_error']:.17g}"
    )

    if result["num_bad"]:
        bad_indices = np.flatnonzero(result["bad"])
        rows, cols = csr_entry_coordinates(reference, bad_indices)

        print()
        print("First value mismatches:")

        for data_index, row, col in zip(
            bad_indices[:max_report],
            rows[:max_report],
            cols[:max_report],
        ):
            ref_value = reference.data[data_index]
            cand_value = candidate.data[data_index]
            abs_error = abs(cand_value - ref_value)
            allowed = atol + rtol * abs(ref_value)

            print(
                f"  ({int(row)}, {int(col)}): "
                f"python={format_scalar(ref_value)}, "
                f"tpetra={format_scalar(cand_value)}, "
                f"abs_error={abs_error:.17g}, "
                f"allowed={allowed:.17g}"
            )

        print()
        print("FAIL: matrix values differ")
        return False

    print()
    print("PASS: matrices are identical within the requested tolerance")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an independently generated Python sparse matrix against "
            "a Galeri/Tpetra Matrix Market matrix."
        )
    )

    parser.add_argument(
        "reference",
        help="independently generated <prefix>.python.mtx[.gz] reference matrix",
    )
    parser.add_argument(
        "candidate",
        help="Galeri/Tpetra <prefix>.mtx[.gz] matrix being validated",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-14,
        help="absolute value-comparison tolerance (default: 1e-14)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-12,
        help="relative value-comparison tolerance (default: 1e-12)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="maximum number of mismatches to print (default: 20)",
    )

    spy_group = parser.add_mutually_exclusive_group()
    spy_group.add_argument(
        "--spy",
        dest="spy",
        action="store_true",
        default=True,
        help="write spy plots (default)",
    )
    spy_group.add_argument(
        "--no-spy",
        dest="spy",
        action="store_false",
        help="disable spy plots",
    )

    parser.add_argument(
        "--spy-prefix",
        default=None,
        help=(
            "override the automatically derived plot prefix/path; also used "
            "for the histogram filename"
        ),
    )

    histogram_group = parser.add_mutually_exclusive_group()
    histogram_group.add_argument(
        "--histogram",
        dest="histogram",
        action="store_true",
        default=True,
        help="write diagonal-offset histogram (default)",
    )
    histogram_group.add_argument(
        "--no-histogram",
        dest="histogram",
        action="store_false",
        help="disable diagonal-offset histogram",
    )

    args = parser.parse_args()

    if args.atol < 0.0:
        parser.error("--atol must be nonnegative")
    if args.rtol < 0.0:
        parser.error("--rtol must be nonnegative")
    if args.max_report < 1:
        parser.error("--max-report must be at least 1")

    return args


def main() -> int:
    args = parse_args()

    derived_prefix = derive_output_prefix(args.reference, args.candidate)
    plot_prefix = Path(args.spy_prefix) if args.spy_prefix else derived_prefix

    try:
        passed = compare(
            args.reference,
            args.candidate,
            atol=args.atol,
            rtol=args.rtol,
            max_report=args.max_report,
            spy=args.spy,
            histogram=args.histogram,
            plot_prefix=plot_prefix,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
