"""Sensor parameters: what an observable measures and where.

:class:`SensorParams` is the declarative description of a SMART-G sensor,
independent of *where* on the horizontal grid it sits (that is the job of the
:data:`~smart_ace.observables.layout.Layout`). Two semantic axes drive
everything else:

* ``quantity`` — ``"radiance"`` (a pencil beam, SMART-G ``TYPE=0``) or
  ``"flux"`` (a cosine-weighted hemisphere, ``TYPE=1`` with ``FOV=90``).
* ``loc`` — ``"toa"`` (near the top of the atmosphere, looking down) or
  ``"0+"`` (just above the surface, looking up).

The SMART-G ``TYPE``/``FOV``, the vertical position and the default viewing
direction are all derived from these two, so the caller only states intent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from smart_ace import Geometry

Quantity = Literal["radiance", "flux"]
Loc = Literal["toa", "0+"]


class SensorParams(BaseModel):
    """Declarative sensor: the measured quantity and where it sits.

    Parameters
    ----------
    quantity : {'radiance', 'flux'}, default 'radiance'
        ``'radiance'`` -> pencil beam (``TYPE=0``, ``FOV=0``).
        ``'flux'`` -> cosine-weighted hemisphere (``TYPE=1``, ``FOV=90``).
    loc : {'toa', '0+'}, default 'toa'
        ``'toa'`` -> near the top of the atmosphere, looks down by default.
        ``'0+'`` -> just above the surface, looks up by default.
    thdeg : float, optional
        Viewing zenith angle [deg]. If ``None``, use the ``loc`` default
        (180 for ``'toa'``, 0 for ``'0+'``). For a flux sensor it tilts the
        collector's normal.
    phdeg : float, default 0.0
        Viewing azimuth angle [deg].
    """

    quantity: Quantity = "radiance"
    loc: Loc = "toa"
    thdeg: float | None = None
    phdeg: float = 0.0

    @property
    def TYPE(self) -> int:
        """SMART-G sensor type: 0 for radiance, 1 for (planar) flux."""
        return 0 if self.quantity == "radiance" else 1

    @property
    def FOV(self) -> float:
        """Field of view [deg]: 0 for radiance, 90 (hemisphere) for flux."""
        return 0.0 if self.quantity == "radiance" else 90.0

    @property
    def theta(self) -> float:
        """Resolved viewing zenith [deg] (``thdeg`` or the ``loc`` default)."""
        default = 180.0 if self.loc == "toa" else 0.0
        return default if self.thdeg is None else self.thdeg

    @property
    def output_key(self) -> str:
        """MLUT key to read.

        In backward mode with a solar local estimate the useful signal is
        always tallied at the up-TOA level, whatever ``loc`` is, so the key is
        constant.
        """
        return "I_up (TOA)"

    def zpos(self, geo: "Geometry") -> float:
        """Vertical position [km]: just below TOA for toa, 0 for ``'0+'``."""
        return float(geo.zgrid.max() - 1.0) if self.loc == "toa" else 0.0
