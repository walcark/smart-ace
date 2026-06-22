"""3D rendering of a cloud :class:`~typer_cli.geometry.grid.Geometry`.

Each cloud voxel is drawn at its true physical position/size, with the full
domain shown as a dashed wireframe so the cloud is seen in its real space.
"""

import numpy as np

from .geometry import Geometry


def _draw_voxels(ax, geometry: Geometry) -> None:
    """Draw the cloud voxels at their true physical position/size."""
    xg, yg, zg = geometry.xgrid, geometry.ygrid, geometry.zgrid
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    ax.voxels(
        X,
        Y,
        Z,
        geometry.mask(),
        facecolor="#4c9be8cc",
        edgecolor="#1f4e79",
        linewidth=0.3,
    )
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")


def _draw_box(ax, xr, yr, zr, **kw) -> None:
    """Wireframe of the [xr] x [yr] x [zr] box (its 12 edges)."""
    (x0, x1), (y0, y1), (z0, z1) = xr, yr, zr
    for a, b in [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)),
        ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
    ]:
        ax.plot(*zip(a, b), **kw)


def plot_geometry(geometry: Geometry, ax=None):
    """Render the cloud voxels placed in the full domain (real final space).

    If `ax` is None a new 3D figure is created; the axes are returned so the
    caller can compose subplots or call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")

    _draw_voxels(ax, geometry)

    dom = geometry.domain
    dxr = (-dom.dx / 2, dom.dx / 2)
    dyr = (-dom.dy / 2, dom.dy / 2)
    dzr = (dom.zmin, dom.zmin + dom.dz)
    _draw_box(ax, dxr, dyr, dzr, color="0.45", lw=0.8, ls="--")

    ax.set_xlim(*dxr)
    ax.set_ylim(*dyr)
    ax.set_zlim(*dzr)
    ax.set_box_aspect((dom.dx, dom.dy, dom.dz))
    ax.set_title(
        f"{geometry.shape.shape}  |  {geometry.n_cloud_voxels} voxels\n"
        f"domain {dom.dx:g}x{dom.dy:g}x{dom.dz:g} km",
        fontsize=9,
    )
    return ax
