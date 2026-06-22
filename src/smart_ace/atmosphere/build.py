"""Assemble the SMART-G objects (Grid3D -> Cloud3D -> Atm3D -> AtmAFGL).

This is the **only** place that imports SMART-G; the imports are done lazily
inside the functions so the rest of the package stays usable without it. The
cloud grid and voxel indices come ready-made from a
:class:`~typer_cli.geometry.grid.Geometry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .params import AtmoParams, CloudOptics

if TYPE_CHECKING:
    from ..geometry import Geometry


def build_grid3d(geometry: "Geometry", toa_alt: float) -> Any:
    """Wrap the geometry edges in a periodic ``Grid3D`` up to ``toa_alt``."""
    from smartg.libATM3D import Grid3D

    return Grid3D(
        geometry.xgrid.astype(np.float32),
        geometry.ygrid.astype(np.float32),
        geometry.zgrid.astype(np.float32),
        periodic=True,
        vert_extend_limit=toa_alt,
    )


def build_cloud3d(
    geometry: "Geometry", grid3: Any, optics: CloudOptics
) -> Any:
    """Build the ``Cloud3D`` from the geometry voxel indices and optics."""
    from smartg.libATM3D import Cloud3D

    n = geometry.n_cloud_voxels
    return Cloud3D(
        optics.species,
        w_ref=optics.wl_ref,
        ext_ref=np.full(n, optics.kext, dtype=np.float64),
        reff=np.full(n, optics.reff, dtype=np.float64),
        xyz_grids=[grid3.xGRID, grid3.yGRID, grid3.zGRID],
        cell_indices=geometry.cell_indices,
    )


def build_atm3d(
    geometry: "Geometry", optics: CloudOptics, atmo: AtmoParams
) -> Any:
    """Combine the 1D atmosphere and the 3D cloud into a SMART-G ``Atm3D``."""
    from smartg.libATM3D import Atm3D

    grid3 = build_grid3d(geometry, atmo.toa_alt)
    cloud3d = build_cloud3d(geometry, grid3, optics)
    return Atm3D(
        atmo.profile,
        grid3,
        wls=np.atleast_1d(np.float32(atmo.wl)),
        cloud_3d=cloud3d,
        O3=None,
        H2O=None,
        NO2=True,
        P0=atmo.surface_pressure,
    )


def build_atmafgl(
    geometry: "Geometry", optics: CloudOptics, atmo: AtmoParams
) -> Any:
    """Build the calculated 3D ``AtmAFGL`` profile ready for a SMART-G run."""
    from smartg.libATM3D import AtmAFGL

    atm3 = build_atm3d(geometry, optics, atmo)

    grid_opt = atm3.get_grid()
    prof_ray = atm3.get_glob_molecular_sca()
    prof_abs = atm3.get_glob_molecular_abs()
    ext_aer = atm3.get_glob_aer_ext()
    ssa_aer = np.ones_like(ext_aer)
    prof_phases = atm3.get_glob_aer_phase(NBTHETA=atmo.nth)
    cells = atm3.get_cells_info()

    atm = AtmAFGL(
        "ATM3D",
        US=False,
        grid=grid_opt,
        prof_ray=prof_ray,
        prof_abs=prof_abs,
        prof_aer=(ext_aer, ssa_aer),
        prof_phases=prof_phases,
        cells=cells,
    )
    return atm.calc(atm3.wls, NBTHETA=atmo.nth, truncation=optics.trunc)
