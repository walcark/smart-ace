"""Voxelise a cloud shape into a SMART-G grid embedded in a periodic domain.

A cloud :class:`~typer_cli.geometry.shapes.Shape` is turned into a set of cell
*edges* ``(xgrid, ygrid, zgrid)`` in km plus the indices of the voxels that
contain cloud, while keeping the number of cloud voxels minimal. The
:class:`~typer_cli.geometry.core.Geometry` class wraps that result.

This module depends on numpy/pydantic only (no SMART-G); the conversion to
SMART-G objects lives in the atmosphere module.

Placement convention: the cloud is centred horizontally on ``(0, 0)`` and spans
``z in [zmin, zmin + dz]`` vertically. The domain is a :class:`Box` too.
"""

import numpy as np
from numpy.typing import NDArray

from .shapes import Box, Domain, Shape, _semi_axes, shape_bbox


def _resolution(
    cloud_shape: Shape, n: int | tuple[int, int, int]
) -> tuple[int, int, int]:
    """Normalise `n` to a triplet (a box is always a single voxel)."""
    if isinstance(cloud_shape, Box):
        return (1, 1, 1)
    if isinstance(n, int):
        return (n, n, n)
    return n


def _axis_interval(
    cloud_shape: Shape, domain: Domain
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Per-axis (lo, hi) intervals [km] of the cloud once embedded in `domain`.

    Output shape is as followed: (xmin, xmax), (ymin, ymax), (zmin, zmax)

    The cloud is centred horizontally on (0, 0). The vertical span is
    ``[zmin, zmin + dz]``. Infinite extents (e.g. ``InfXBox``) are clipped to
    the domain (a cloud that fills a whole period is, with periodic boundaries,
    effectively infinite). A *finite* extent sticking out of the domain is an
    error.
    """
    cb = shape_bbox(cloud_shape)
    cloud = (
        (-cb.dx / 2, cb.dx / 2),
        (-cb.dy / 2, cb.dy / 2),
        (cb.zmin, cb.zmin + cb.dz),
    )
    dom = (
        (-domain.dx / 2, domain.dx / 2),
        (-domain.dy / 2, domain.dy / 2),
        (domain.zmin, domain.zmin + domain.dz),
    )

    out: list[tuple[float, float]] = []
    tol = 1e-9
    for name, (c_lo, c_hi), (d_lo, d_hi) in zip("xyz", cloud, dom):
        if not np.isfinite(
            [c_lo, c_hi]
        ).all():  # infinite extent -> fill domain
            out.append((d_lo, d_hi))
            continue
        if c_lo < d_lo - tol or c_hi > d_hi + tol:
            raise ValueError(
                f"Cloud extends beyond the domain along {name}: "
                f"cloud [{c_lo:g}, {c_hi:g}] vs domain [{d_lo:g}, {d_hi:g}]."
            )
        out.append((max(c_lo, d_lo), min(c_hi, d_hi)))
    return out[0], out[1], out[2]


def _axis_grid(
    interval: tuple[float, float], domain: tuple[float, float], n: int
) -> tuple[NDArray[np.float64], int]:
    """Build the cell edges along one axis up to the `domain` bounds.

    The layout is an optional border cell, `n` cells over `interval`, then an
    optional border cell.

    Returns the edges and the number of leading border cells (0 or 1), used to
    offset the cloud cell indices.
    """
    c_lo, c_hi = interval
    d_lo, d_hi = domain
    tol = 1e-9
    inner = np.linspace(c_lo, c_hi, n + 1)
    left = [d_lo] if c_lo > d_lo + tol else []
    right = [d_hi] if c_hi < d_hi - tol else []
    edges = np.concatenate([np.asarray(left), inner, np.asarray(right)])
    return edges, len(left)


def _clamped_sq(
    grid: NDArray[np.float64], center: float, semi: float
) -> NDArray[np.float64]:
    """Squared scaled distance from `center` to each voxel interval of `grid`.

    For every cell ``[grid[i], grid[i+1]]`` returns ``(d / semi) ** 2`` where
    ``d`` is the distance from `center` to the closest point of the cell
    (0 when the centre falls inside the cell).
    """
    lo = (grid[:-1] - center) / semi
    hi = (grid[1:] - center) / semi
    closest = np.maximum(lo, np.minimum(0.0, hi))
    return closest**2


def grid_and_indexes(
    cloud_shape: Shape,
    domain: Domain,
    n: int | tuple[int, int, int] = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """Build the ideal 3D grid over `domain` and the cloud voxel indices.

    The grid spans the whole (periodic) domain: the cloud is resolved with
    ``n`` cells per axis, surrounded by single large border cells filling the
    rest of the domain. This keeps the number of cloud voxels minimal (the
    efficient ``[0, 3, 4, 7]`` layout of the IPRT examples).

    Parameters
    ----------
    cloud_shape:
        The cloud shape to voxelise.
    domain:
        The full periodic domain (a :class:`Box`) the cloud is embedded in. It
        sets the grid extent and clips any infinite cloud extent (``InfXBox``).
    n:
        Number of cells resolving the cloud along each axis, scalar or
        ``(nx, ny, nz)``. Ignored for a box, always a single voxel (uniform
        optical properties).

    Returns
    -------
    (xgrid, ygrid, zgrid, cell_indices):
        ``*grid`` are 1-D arrays of cell edges [km]. ``cell_indices`` is an
        ``(N, 3)`` int32 array of the cloud voxels in the **1-based IPRT
        convention** expected by SMART-G.

    A voxel is flagged as cloud as soon as it *overlaps* the shape
    (conservative: no cloud leaks out of the grid).
    """
    ix, iy, iz = _axis_interval(cloud_shape, domain)
    dx_dom = (-domain.dx / 2, domain.dx / 2)
    dy_dom = (-domain.dy / 2, domain.dy / 2)
    dz_dom = (domain.zmin, domain.zmin + domain.dz)

    # A box has uniform properties: the minimal resolution is a single voxel.
    nx, ny, nz = _resolution(cloud_shape, n)

    # Grid along each axis and cloud offset for indexing
    xgrid, ox = _axis_grid(ix, dx_dom, nx)
    ygrid, oy = _axis_grid(iy, dy_dom, ny)
    zgrid, oz = _axis_grid(iz, dz_dom, nz)

    # Compute inner cloud indexes without offset
    if isinstance(cloud_shape, Box):
        # Every inner cell is cloud; here that is the single (0, 0, 0) voxel.
        inner_idx = np.array([[0, 0, 0]], dtype=np.int64)
    else:
        # Voxel/ellipsoid overlap on the inner (cloud-resolving) edges: scale
        # each axis so the ellipsoid becomes the unit sphere (AABB stays AABB),
        # then keep voxels whose closest point to the centre is within unit
        # distance. Per-axis squared clamped distances combine by broadcasting.
        horizontal, vertical = _semi_axes(cloud_shape)
        zc = iz[0] + vertical  # ellipsoid centre (x=y=0 by convention)
        inner_x = np.linspace(ix[0], ix[1], nx + 1)
        inner_y = np.linspace(iy[0], iy[1], ny + 1)
        inner_z = np.linspace(iz[0], iz[1], nz + 1)
        qx2 = _clamped_sq(inner_x, 0.0, horizontal)
        qy2 = _clamped_sq(inner_y, 0.0, horizontal)
        qz2 = _clamped_sq(inner_z, zc, vertical)
        inside = (
            qx2[:, None, None] + qy2[None, :, None] + qz2[None, None, :]
        ) <= 1.0
        inner_idx = np.argwhere(inside)

    # Shift inner (0-based) indices by the leading border cells, then to IPRT
    # 1-based.
    cell_indices = (inner_idx + np.array([ox, oy, oz]) + 1).astype(np.int32)
    return xgrid, ygrid, zgrid, cell_indices
