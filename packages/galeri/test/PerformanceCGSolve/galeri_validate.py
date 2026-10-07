#!/usr/bin/env python3
"""
Independently reconstruct selected Galeri matrix families from logical labels.

IMPORTANT VALIDATION PROPERTY
-----------------------------
This program never reads the Galeri/Tpetra sparse matrix being validated.
Its only Galeri-produced input is the logical-label Matrix Market file written
by GaleriHelper-validation.hpp:

    [ gid, ix, iy, iz, component ]

The sparse reference matrix is reconstructed independently from the matrix type,
logical geometry, and explicitly supplied operator parameters.

Typical use:

    python3 galeri_validate.py \
        --logical Laplace3D_10x10x10.logical.mtx.gz \
        --matrix-type Laplace3D

By default this writes:

    Laplace3D_10x10x10.python.mtx.gz

Both plain .mtx and gzip-compressed .mtx.gz logical inputs are supported.
The derived output preserves the input's compression mode.

Use --output only to override that derived name.

Optional Galeri-style parameters may be supplied independently:

    --param stretchx=2.0 --param stretchy=1.0 --param stretchz=0.5

No option exists for reading the Galeri matrix itself.  Comparison against the
Tpetra/Galeri .mtx[.gz] should be performed as a separate validation step.
"""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mmwrite


Coord = Tuple[int, int, int, int]  # ix, iy, iz, component
Offset = Tuple[int, int, int]


@dataclass(frozen=True)
class LogicalLayout:
    """Logical coordinate-to-GID mapping exported by the C++ helper."""

    gid_to_coord: Mapping[int, Coord]
    coord_to_gid: Mapping[Coord, int]
    num_dofs: int
    nx: int
    ny: int
    nz: int
    num_components: int


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


def _mmwrite(path: Path, obj) -> None:
    """Write Matrix Market data, gzip-compressing paths ending in .gz."""

    if path.suffix == ".gz":
        with gzip.open(path, "wb") as stream:
            mmwrite(stream, obj)
        return
    mmwrite(path, obj)


def _as_dense_array(path: Path) -> np.ndarray:
    obj = _mmread(path)
    if sp.issparse(obj):
        obj = obj.toarray()
    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_logical_layout(path: str | Path) -> LogicalLayout:
    """
    Read ONLY the logical coordinate metadata.

    Expected columns:
        gid ix iy iz component

    The helper currently writes these through Tpetra's dense Matrix Market
    writer, so the numeric storage type may be floating point.  Values must
    nevertheless be exact integers.
    """

    path = Path(path)
    labels = _as_dense_array(path)

    if labels.shape[1] != 5:
        raise ValueError(
            f"{path}: expected an N x 5 logical-label matrix "
            f"[gid ix iy iz component], got shape {labels.shape}"
        )

    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{path}: logical labels contain non-integer values")

    labels = rounded.astype(np.int64)

    gids = labels[:, 0]
    if len(np.unique(gids)) != len(gids):
        raise ValueError(f"{path}: duplicate GIDs in logical labels")

    # GaleriHelper-validation.hpp deliberately constructs a zero-based,
    # contiguous map.  Assert that invariant rather than quietly depending on it.
    expected = np.arange(len(gids), dtype=np.int64)
    if not np.array_equal(np.sort(gids), expected):
        raise ValueError(
            f"{path}: expected zero-based contiguous GIDs 0..{len(gids)-1}"
        )

    gid_to_coord: Dict[int, Coord] = {}
    coord_to_gid: Dict[Coord, int] = {}

    for gid, ix, iy, iz, component in labels:
        coord = (int(ix), int(iy), int(iz), int(component))
        gid_i = int(gid)

        if coord in coord_to_gid:
            raise ValueError(
                f"{path}: logical coordinate {coord} maps to multiple GIDs"
            )

        gid_to_coord[gid_i] = coord
        coord_to_gid[coord] = gid_i

    if np.any(labels[:, 1:] < 0):
        raise ValueError(f"{path}: negative logical coordinate/component")

    nx = int(labels[:, 1].max()) + 1
    ny = int(labels[:, 2].max()) + 1
    nz = int(labels[:, 3].max()) + 1
    num_components = int(labels[:, 4].max()) + 1

    # Check that the metadata describes a complete Cartesian grid for every
    # component, rather than accepting a damaged coordinate file.
    expected_dofs = nx * ny * nz * num_components
    if expected_dofs != len(gids):
        raise ValueError(
            f"{path}: logical labels are not a complete Cartesian product: "
            f"nx*ny*nz*ncomp={expected_dofs}, rows={len(gids)}"
        )

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                for component in range(num_components):
                    coord = (ix, iy, iz, component)
                    if coord not in coord_to_gid:
                        raise ValueError(f"{path}: missing logical coordinate {coord}")

    return LogicalLayout(
        gid_to_coord=gid_to_coord,
        coord_to_gid=coord_to_gid,
        num_dofs=len(gids),
        nx=nx,
        ny=ny,
        nz=nz,
        num_components=num_components,
    )


def _require_scalar(layout: LogicalLayout, matrix_type: str) -> None:
    if layout.num_components != 1:
        raise ValueError(
            f"{matrix_type} is a scalar matrix but logical metadata contains "
            f"{layout.num_components} components per node"
        )


def _require_dimensions(
    layout: LogicalLayout,
    matrix_type: str,
    dimension: int,
) -> None:
    if dimension == 1:
        ok = layout.ny == 1 and layout.nz == 1
    elif dimension == 2:
        ok = layout.nz == 1
    elif dimension == 3:
        ok = True
    else:
        raise AssertionError("unsupported dimension")

    if not ok:
        raise ValueError(
            f"{matrix_type} expects a {dimension}D logical grid, got "
            f"({layout.nx}, {layout.ny}, {layout.nz})"
        )


def _assemble_scalar_stencil(
    layout: LogicalLayout,
    stencil: Mapping[Offset, float],
) -> sp.csr_matrix:
    """
    Assemble by logical-neighbor lookup.

    Crucially, this does NOT derive a neighboring GID arithmetically.  It asks
    coord_to_gid which GID represents the requested logical coordinate.  Thus
    the reference assembly remains valid even if the C++ numbering convention
    changes while the geometry stays the same.

    Off-grid stencil points are simply omitted.  This matches the default
    Galeri behavior for these stencil problems: Dirichlet boundaries with
    keepBCs=false truncate out-of-domain off-diagonals while retaining the
    nominal diagonal coefficient.
    """

    _require_scalar(layout, "scalar stencil")

    rows = []
    cols = []
    data = []

    for row_gid in range(layout.num_dofs):
        ix, iy, iz, component = layout.gid_to_coord[row_gid]
        assert component == 0

        for (dx, dy, dz), value in stencil.items():
            target = (ix + dx, iy + dy, iz + dz, 0)
            col_gid = layout.coord_to_gid.get(target)
            if col_gid is None:
                continue

            rows.append(row_gid)
            cols.append(col_gid)
            data.append(float(value))

    A = sp.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(layout.num_dofs, layout.num_dofs),
    ).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A


def _get_float(params: Mapping[str, float], name: str, default: float) -> float:
    value = params.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parameter {name!r} must be numeric, got {value!r}") from exc


def _reject_unknown_params(
    params: Mapping[str, float],
    allowed: Iterable[str],
    matrix_type: str,
) -> None:
    allowed = set(allowed)
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported parameter(s) for {matrix_type}: {', '.join(unknown)}"
        )


def build_reference_matrix(
    layout: LogicalLayout,
    matrix_type: str,
    params: Mapping[str, float] | None = None,
) -> sp.csr_matrix:
    """Independently construct a supported Galeri-equivalent matrix."""

    params = {} if params is None else dict(params)

    if matrix_type == "Identity":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 1)
        _reject_unknown_params(params, {"a"}, matrix_type)
        a = _get_float(params, "a", 1.0)
        return _assemble_scalar_stencil(layout, {(0, 0, 0): a})

    if matrix_type == "Laplace1D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 1)
        _reject_unknown_params(params, set(), matrix_type)
        stencil = {
            (-1, 0, 0): -1.0,
            (0, 0, 0): 2.0,
            (+1, 0, 0): -1.0,
        }
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "Laplace2D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 2)
        _reject_unknown_params(params, {"stretchx", "stretchy"}, matrix_type)
        sx = _get_float(params, "stretchx", 1.0)
        sy = _get_float(params, "stretchy", 1.0)
        if sx == 0.0 or sy == 0.0:
            raise ValueError("stretchx and stretchy must be nonzero")
        wx = -1.0 / (sx * sx)
        wy = -1.0 / (sy * sy)
        stencil = {
            (-1, 0, 0): wx,
            (+1, 0, 0): wx,
            (0, -1, 0): wy,
            (0, +1, 0): wy,
            (0, 0, 0): -2.0 * wx - 2.0 * wy,
        }
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "Laplace3D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 3)
        _reject_unknown_params(
            params, {"stretchx", "stretchy", "stretchz"}, matrix_type
        )
        sx = _get_float(params, "stretchx", 1.0)
        sy = _get_float(params, "stretchy", 1.0)
        sz = _get_float(params, "stretchz", 1.0)
        if sx == 0.0 or sy == 0.0 or sz == 0.0:
            raise ValueError("stretchx/stretchy/stretchz must be nonzero")
        wx = -1.0 / (sx * sx)
        wy = -1.0 / (sy * sy)
        wz = -1.0 / (sz * sz)
        stencil = {
            (-1, 0, 0): wx,
            (+1, 0, 0): wx,
            (0, -1, 0): wy,
            (0, +1, 0): wy,
            (0, 0, -1): wz,
            (0, 0, +1): wz,
            (0, 0, 0): -2.0 * wx - 2.0 * wy - 2.0 * wz,
        }
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "Star2D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 2)
        names = {"a", "b", "c", "d", "e", "z1", "z2", "z3", "z4"}
        _reject_unknown_params(params, names, matrix_type)
        values = {
            "a": _get_float(params, "a", 8.0),
            "b": _get_float(params, "b", -1.0),
            "c": _get_float(params, "c", -1.0),
            "d": _get_float(params, "d", -1.0),
            "e": _get_float(params, "e", -1.0),
            "z1": _get_float(params, "z1", -1.0),
            "z2": _get_float(params, "z2", -1.0),
            "z3": _get_float(params, "z3", -1.0),
            "z4": _get_float(params, "z4", -1.0),
        }
        stencil = {
            (0, 0, 0): values["a"],
            (-1, 0, 0): values["b"],
            (+1, 0, 0): values["c"],
            (0, -1, 0): values["d"],
            (0, +1, 0): values["e"],
            (-1, -1, 0): values["z1"],
            (+1, -1, 0): values["z2"],
            (-1, +1, 0): values["z3"],
            (+1, +1, 0): values["z4"],
        }
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "BigStar2D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 2)
        names = {
            "a", "b", "c", "d", "e", "z1", "z2", "z3", "z4",
            "bb", "cc", "dd", "ee",
        }
        _reject_unknown_params(params, names, matrix_type)
        defaults = {
            "a": 20.0,
            "b": -8.0,
            "c": -8.0,
            "d": -8.0,
            "e": -8.0,
            "z1": 2.0,
            "z2": 2.0,
            "z3": 2.0,
            "z4": 2.0,
            "bb": 1.0,
            "cc": 1.0,
            "dd": 1.0,
            "ee": 1.0,
        }
        v = {name: _get_float(params, name, default) for name, default in defaults.items()}
        stencil = {
            (0, 0, 0): v["a"],
            (-1, 0, 0): v["b"],
            (+1, 0, 0): v["c"],
            (0, -1, 0): v["d"],
            (0, +1, 0): v["e"],
            (-1, -1, 0): v["z1"],
            (+1, -1, 0): v["z2"],
            (-1, +1, 0): v["z3"],
            (+1, +1, 0): v["z4"],
            (-2, 0, 0): v["bb"],
            (+2, 0, 0): v["cc"],
            (0, -2, 0): v["dd"],
            (0, +2, 0): v["ee"],
        }
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "Brick3D":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 3)
        _reject_unknown_params(params, set(), matrix_type)
        stencil: Dict[Offset, float] = {}
        for dz in (-1, 0, +1):
            for dy in (-1, 0, +1):
                for dx in (-1, 0, +1):
                    if dx == 0 and dy == 0 and dz == 0:
                        stencil[(dx, dy, dz)] = 26.0
                    else:
                        stencil[(dx, dy, dz)] = -1.0
        return _assemble_scalar_stencil(layout, stencil)

    if matrix_type == "Scalar3D_27Pt":
        _require_scalar(layout, matrix_type)
        _require_dimensions(layout, matrix_type, 3)

        names = {
            f"S{ix}{iy}{iz}"
            for iz in (1, 2, 3)
            for iy in (1, 2, 3)
            for ix in (1, 2, 3)
        }
        _reject_unknown_params(params, names, matrix_type)

        stencil: Dict[Offset, float] = {}
        for iz in (1, 2, 3):
            for iy in (1, 2, 3):
                for ix in (1, 2, 3):
                    name = f"S{ix}{iy}{iz}"
                    default = 26.0 if name == "S222" else -1.0
                    stencil[(ix - 2, iy - 2, iz - 2)] = _get_float(
                        params, name, default
                    )
        return _assemble_scalar_stencil(layout, stencil)

    unsupported = {
        "AnisotropicDiffusion",
        "Recirc2D",
        "HexFEM_LapStiff",
        "HexFEM_Mass",
        "Elasticity2D",
        "Elasticity3D",
    }
    if matrix_type in unsupported:
        raise NotImplementedError(
            f"{matrix_type} is intentionally not implemented in this first "
            "independent reference generator.  It requires an independent "
            "PDE/finite-element/operator assembly, not merely a stencil guessed "
            "from the Galeri matrix."
        )

    raise ValueError(f"unsupported matrix type: {matrix_type}")


def _parse_param(text: str) -> Tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("parameters must have the form NAME=VALUE")
    name, value = text.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name:
        raise argparse.ArgumentTypeError("parameter name may not be empty")
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"parameter {name!r} must have a numeric value"
        ) from exc
    return name, number


def default_output_path(logical_path: str | Path) -> Path:
    """Derive <prefix>.python.mtx[.gz] from <prefix>.logical.mtx[.gz]."""

    logical_path = Path(logical_path)
    compressed = logical_path.suffix == ".gz"
    name = logical_path.name[:-len(".gz")] if compressed else logical_path.name
    suffix = ".logical.mtx"

    if name.endswith(suffix):
        prefix = name[:-len(suffix)]
    else:
        # Sensible fallback for manually named logical files.
        prefix = Path(name).stem

    output_name = prefix + ".python.mtx"
    if compressed:
        output_name += ".gz"
    return logical_path.with_name(output_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently reconstruct a Galeri-equivalent sparse matrix from "
            "logical coordinate metadata only."
        )
    )
    parser.add_argument(
        "--logical",
        required=True,
        help="<prefix>.logical.mtx[.gz] written by the C++ validation helper",
    )
    parser.add_argument(
        "--matrix-type",
        required=True,
        choices=[
            "Identity",
            "Laplace1D",
            "Laplace2D",
            "Laplace3D",
            "Star2D",
            "BigStar2D",
            "Brick3D",
            "Scalar3D_27Pt",
            "AnisotropicDiffusion",
            "Recirc2D",
            "HexFEM_LapStiff",
            "HexFEM_Mass",
            "Elasticity2D",
            "Elasticity3D",
        ],
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "independent Galeri-style operator parameter; may be repeated, "
            "for example --param stretchx=2 --param stretchy=1"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "override output Matrix Market path; .gz suffix enables gzip; "
            "default preserves the logical input compression mode"
        ),
    )

    args = parser.parse_args()

    params: Dict[str, float] = {}
    for item in args.param:
        name, value = _parse_param(item)
        if name in params:
            parser.error(f"parameter {name!r} specified more than once")
        params[name] = value

    layout = load_logical_layout(args.logical)
    A = build_reference_matrix(layout, args.matrix_type, params)

    output = Path(args.output) if args.output is not None else default_output_path(args.logical)
    output.parent.mkdir(parents=True, exist_ok=True)
    _mmwrite(output, A)

    print(f"matrix type : {args.matrix_type}")
    print(f"logical grid: {layout.nx} x {layout.ny} x {layout.nz}")
    print(f"components  : {layout.num_components}")
    print(f"shape       : {A.shape[0]} x {A.shape[1]}")
    print(f"nnz         : {A.nnz}")
    print(f"wrote       : {output}")


if __name__ == "__main__":
    main()


