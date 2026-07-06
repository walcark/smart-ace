"""Observable: a physical quantity sampled by SMART-G over a sensor layout.

An :class:`Observable` bundles an :class:`Atmosphere` with a sensor ``layout``
and a viewing geometry. It knows how to place the sensors on the 3D grid,
launch a SMART-G run and pack the output into a :class:`Result` — a 1D profile
for a :class:`Transect` layout, a 2D image for a :class:`Map`.

SMART-G is imported lazily (inside the functions) so importing this module
stays light and does not require a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from smart_ace.atmosphere import Atmosphere
from smart_ace.atmosphere.build import build_grid3d

from .layout import Layout, Transect, positions

if TYPE_CHECKING:
    from smartg.smartg import Sensor

    from smart_ace import Geometry

SensorType = Literal["sat", "ground"]


def build_sensors(
    geo: "Geometry",
    sensor_type: SensorType,
    layout: Layout,
    THDEG: float | None = None,
    PHDEG: float | None = None,
) -> tuple[list["Sensor"], tuple[int, ...]]:
    """Build the list of SMART-G sensors and return the sampling grid shape.

    Sensor positions come from :func:`positions` (cell-centred, so they never
    fall on a voxel face). Each sensor gets the ``ICELL`` of the voxel it sits
    in, computed against the *with-boundary* grid ``grid3.xGRID/yGRID/zGRID`` —
    the only grid whose cell numbering matches the SMART-G engine.
    """
    from smartg.libATM3D import locate_3Dregular_cells
    from smartg.smartg import Sensor

    xx, yy = positions(layout)
    shape = xx.shape
    xx, yy = xx.flatten(), yy.flatten()

    if sensor_type == "sat":
        zz = np.full(xx.size, geo.zgrid.max() - 1.0)
    elif sensor_type == "ground":
        zz = np.full(xx.size, 0.0)
    else:
        raise ValueError(f"Wrong sensor type: {sensor_type}")

    grid3 = build_grid3d(geo, toa_alt=geo.zgrid.max() + 1.0)

    icells = locate_3Dregular_cells(
        grid3.xGRID,
        grid3.yGRID,
        grid3.zGRID,  # bien la grille AVEC bords
        xx,
        yy,
        zz,
    )

    sensors = [
        Sensor(
            POSX=float(px),
            POSY=float(py),
            POSZ=float(zz[0]),
            THDEG=THDEG,
            PHDEG=PHDEG,
            ICELL=int(ic),
            LOC="ATMOS",
        )
        for px, py, ic in zip(xx, yy, icells)
    ]

    return sensors, shape


@dataclass
class Result:
    """Reshaped output of a SMART-G run over a sensor layout.

    Parameters
    ----------
    quantity:
        Name of the sampled SMART-G quantity (e.g. ``"I_up (TOA)"``).
    values:
        The radiances, reshaped to match the layout: 1-D ``(n,)`` for a
        transect, 2-D ``(ny, nx)`` for a map.
    coords:
        Sampling coordinates [km]. ``(s,)`` for a transect (position along its
        axis), ``(x, y)`` for a map (the 1-D tick coordinates of each axis).
    is_map:
        ``True`` for a 2-D map, ``False`` for a 1-D transect.
    axis:
        For a transect, the axis it runs along (``"x"`` or ``"y"``); ``None``
        for a map.
    """

    quantity: str
    values: np.ndarray
    coords: tuple[np.ndarray, ...]
    is_map: bool
    axis: str | None = None

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the result (line for a transect, image for a map)."""
        from .plot import plot_result

        return plot_result(self, ax=ax, **kwargs)


@dataclass
class Observable:
    """An atmosphere sampled by a sensor layout, ready to be run with SMART-G.

    Parameters
    ----------
    atmosphere:
        The :class:`Atmosphere` (cloud geometry + optics + 1D background).
    layout:
        The horizontal sensor arrangement (:class:`Map` or :class:`Transect`).
    sensor_type:
        ``"sat"`` (near TOA, looking down) or ``"ground"`` (at the surface).
    THDEG, PHDEG:
        Viewing zenith / azimuth of the sensors [deg]. Default nadir satellite.
    quantity:
        The SMART-G output quantity to extract from the run.
    """

    atmosphere: Atmosphere
    layout: Layout
    sensor_type: SensorType = "sat"
    THDEG: float = 180.0
    PHDEG: float = 0.0
    quantity: str = "I_up (TOA)"

    def run(
        self,
        S: Any,
        *,
        surf: Any,
        le: Any,
        NBPHOTONS: float,
    ) -> Result:
        """Place the sensors, run SMART-G and pack the output into a Result.

        Parameters
        ----------
        S:
            A configured :class:`smartg.smartg.Smartg` instance.
        surf:
            Surface object passed to ``S.run`` (e.g. a ``LambSurface``).
        le:
            Local-estimate direction, e.g. ``{"th_deg": 0., "phi_deg": 0.}``.
        NBPHOTONS:
            Number of photons for the run.
        """
        geo = self.atmosphere.geometry
        wl = self.atmosphere.atmo.wl

        sensors, shape = build_sensors(
            geo, self.sensor_type, self.layout, self.THDEG, self.PHDEG
        )

        mlut = S.run(
            wl=wl,
            atm=self.atmosphere.build(),
            surf=surf,
            sensor=sensors,
            le=le,
            NBPHOTONS=NBPHOTONS,
        )

        arr = np.asarray(mlut[self.quantity][:, 0, 0], dtype=float)
        xx, yy = positions(self.layout)

        if isinstance(self.layout, Transect):
            coord = (xx if self.layout.axis == "x" else yy).flatten()
            return Result(
                self.quantity,
                arr,
                (coord,),
                is_map=False,
                axis=self.layout.axis,
            )

        values = arr.reshape(shape)
        return Result(self.quantity, values, (xx[0, :], yy[:, 0]), is_map=True)
