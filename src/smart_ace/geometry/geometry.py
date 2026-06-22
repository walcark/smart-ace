"""The Geometry: a cloud voxelisation inside a periodic domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._hash import stable_hash
from .shapes import Box, Domain, Shape, shape_bbox
from .voxelize import _resolution, grid_and_indexes

Float64 = NDArray[np.float64]
Int32 = NDArray[np.int32]


@dataclass(frozen=True, eq=False)
class Geometry:
    """Self-contained cloud geometry voxelised inside a periodic domain.

    Holds the SMART-G grid edges and the cloud voxel indices, together with the
    inputs that produced them so the geometry can be hashed/cached and rebuilt.
    """

    shape: Shape
    domain: Domain
    n: tuple[int, int, int]
    xgrid: Float64
    ygrid: Float64
    zgrid: Float64
    cell_indices: Int32

    @classmethod
    def build(
        cls,
        cloud_shape: Shape,
        domain: Domain,
        n: int | tuple[int, int, int] = 1,
    ) -> "Geometry":
        """Voxelise `cloud_shape` inside `domain` at resolution `n`."""
        xg, yg, zg, ci = grid_and_indexes(cloud_shape, domain, n)
        # Store the normalised resolution (a box collapses to a single voxel)
        # so identical boxes hash the same regardless of the requested value.
        triplet = _resolution(cloud_shape, n)
        return cls(cloud_shape, domain, triplet, xg, yg, zg, ci)

    @property
    def cloud_bbox(self) -> Box:
        """Axis-aligned bounding box of the cloud shape."""
        return shape_bbox(self.shape)

    @property
    def n_cloud_voxels(self) -> int:
        """Number of voxels flagged as cloud."""
        return int(len(self.cell_indices))

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Total number of cells of the full domain grid (nx, ny, nz)."""
        return (self.xgrid.size - 1, self.ygrid.size - 1, self.zgrid.size - 1)

    def mask(self) -> NDArray[np.bool_]:
        """Boolean (nx, ny, nz) cloud mask over the full domain grid."""
        m = np.zeros(self.grid_shape, dtype=bool)
        idx = self.cell_indices - 1  # IPRT 1-based -> 0-based
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return m

    def key(self) -> str:
        """Stable short hash of the geometry inputs (for the DB cache)."""
        return stable_hash(
            {
                "shape": self.shape.model_dump(),
                "domain": self.domain.model_dump(),
                "n": list(self.n),
            }
        )

    def show(self, ax=None) -> None:
        """Render the geometry in its domain (see :func:`plot_geometry`)."""
        from .plot import plot_geometry

        return plot_geometry(self, ax=ax)
