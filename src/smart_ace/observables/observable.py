"""Observable: a physical quantity sampled by SMART-G over a sensor layout.

An :class:`Observable` bundles an :class:`Atmosphere` with a sensor ``layout``
and a viewing geometry. It knows how to place the sensors on the 3D grid,
launch a SMART-G run and pack the output into a :class:`Result` — a 1D profile
for a :class:`Transect` layout, a 2D image for a :class:`Map`.

SMART-G is imported lazily (inside the functions) so importing this module
stays light and does not require a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from smart_ace.atmosphere import Atmosphere
from smart_ace.atmosphere.build import build_grid3d

from .layout import Layout, Transect, positions
from .sensor import SensorParams

if TYPE_CHECKING:
    from smartg.smartg import Sensor

    from smart_ace import Geometry


def build_sensors(
    geo: "Geometry",
    sensor: SensorParams,
    layout: Layout,
) -> tuple[list["Sensor"], tuple[int, ...]]:
    """Build the list of SMART-G sensors and return the sampling grid shape.

    Horizontal positions come from :func:`positions` (cell-centred, so they
    never fall on a voxel face); the vertical position, viewing direction and
    ``TYPE``/``FOV`` come from ``sensor`` (:class:`SensorParams`). Each sensor
    gets the ``ICELL`` of the voxel it sits in, computed against the
    *with-boundary* grid ``grid3.xGRID/yGRID/zGRID`` — the only grid whose cell
    numbering matches the SMART-G engine.
    """
    from smartg.libATM3D import locate_3Dregular_cells
    from smartg.smartg import Sensor

    xx, yy = positions(layout)
    shape = xx.shape
    xx, yy = xx.flatten(), yy.flatten()

    z = sensor.zpos(geo)
    zz = np.full(xx.size, z)

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
            POSZ=z,
            THDEG=sensor.theta,
            PHDEG=sensor.phdeg,
            TYPE=sensor.TYPE,
            FOV=sensor.FOV,
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
    """A fully declarative SMART-G measurement, ready to be run.

    An ``Observable`` holds *everything* needed to compute itself: the
    atmosphere, the sensor layout and viewing geometry, and the run parameters
    (surface, local estimate, photon budget). ``run`` therefore takes only the
    SMART-G engine.

    Parameters
    ----------
    atmosphere:
        The :class:`Atmosphere` (cloud geometry + optics + 1D background).
    layout:
        The horizontal sensor arrangement (:class:`Map` or :class:`Transect`).
    sensor:
        The :class:`SensorParams` (measured quantity, location, direction).
    surf:
        Surface object passed to ``S.run`` (e.g. a ``LambSurface``).
    le:
        Local-estimate (solar) direction, ``{"th_deg": 0., "phi_deg": 0.}``.
    NBPHOTONS:
        Number of photons for the run.
    direct:
        Include the directly transmitted (unscattered) beam. SMART-G drops it
        by default; it is essential for a ground flux (which is dominated by
        the direct solar beam) and harmless for a nadir radiance (no direct
        sun-to-sensor path), so the default here is ``True``.
    """

    atmosphere: Atmosphere
    layout: Layout
    sensor: SensorParams = field(default_factory=SensorParams)
    surf: Any = None
    le: Any = None
    NBPHOTONS: float = 1e7
    direct: bool = True

    def run(self, S: Any) -> Result:
        """Place the sensors, run SMART-G and pack the output into a Result.

        Parameters
        ----------
        S:
            A configured :class:`smartg.smartg.Smartg` instance.
        """
        geo = self.atmosphere.geometry
        wl = self.atmosphere.atmo.wl
        quantity = self.sensor.output_key

        sensors, shape = build_sensors(geo, self.sensor, self.layout)

        mlut = S.run(
            wl=wl,
            atm=self.atmosphere.build(),
            surf=self.surf,
            sensor=sensors,
            le=self.le,
            NBPHOTONS=self.NBPHOTONS,
            DIRECT=self.direct,
        )

        arr = np.asarray(mlut[quantity][:, 0, 0], dtype=float)
        xx, yy = positions(self.layout)

        if isinstance(self.layout, Transect):
            coord = (xx if self.layout.axis == "x" else yy).flatten()
            return Result(
                quantity,
                arr,
                (coord,),
                is_map=False,
                axis=self.layout.axis,
            )

        values = arr.reshape(shape)
        return Result(quantity, values, (xx[0, :], yy[:, 0]), is_map=True)


@dataclass
class Results:
    """Name-addressable collection of :class:`Result` objects.

    Returned by :meth:`Study.run`. Deliberately minimal: it only maps a name
    to its :class:`Result` and delegates everything else (plotting in
    particular) to :class:`Result` itself.
    """

    results: dict[str, Result] = field(default_factory=dict)

    def get(self, name: str) -> Result:
        """Return the :class:`Result` registered under ``name``."""
        if name not in self.results:
            raise KeyError(
                f"No result named {name!r}. Available: {list(self.results)}"
            )
        return self.results[name]


@dataclass
class Study:
    """An explicit, named registry of observables run together.

    Add observables under a name, then :meth:`run` computes each one (one
    SMART-G run per observable, reusing :meth:`Observable.run`) and returns a
    :class:`Results` mapping. Being an ordinary object rather than a global
    registry, a ``Study`` is notebook-safe and reproducible: what runs is
    exactly what was added to *this* instance.
    """

    _observables: dict[str, Observable] = field(default_factory=dict)

    def add_observable(self, name: str, observable: Observable) -> None:
        """Register ``observable`` under ``name`` (must be unique)."""
        if name in self._observables:
            raise ValueError(f"An observable named {name!r} already exists.")
        self._observables[name] = observable

    def run(self, S: Any) -> Results:
        """Run every registered observable and collect the results by name."""
        return Results(
            {name: obs.run(S) for name, obs in self._observables.items()}
        )
