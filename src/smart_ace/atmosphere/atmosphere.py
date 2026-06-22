"""The Atmosphere: a cloud Geometry dressed with optics and a 1D background."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .._hash import stable_hash
from ..geometry import Geometry
from .params import AtmoParams, CloudOptics


@dataclass
class Atmosphere:
    """A cloud :class:`Geometry` plus its optical and atmospheric parameters.

    Holds everything needed to assemble the SMART-G ``AtmAFGL`` profile, but
    builds it lazily (``build()`` is the only SMART-G-requiring call).
    """

    geometry: Geometry
    cloud: CloudOptics
    atmo: AtmoParams

    def key(self) -> str:
        """Stable short hash of the atmosphere (geometry + optics + profile)."""
        return stable_hash(
            {
                "geometry": self.geometry.key(),
                "cloud": asdict(self.cloud),
                "atmo": asdict(self.atmo),
            }
        )

    def build(self) -> Any:
        """Assemble and return the calculated SMART-G ``AtmAFGL`` (needs smartg)."""
        from .build import build_atmafgl

        return build_atmafgl(self.geometry, self.cloud, self.atmo)

    def show(self, ax=None):
        """Render the underlying cloud geometry in its domain."""
        return self.geometry.show(ax=ax)
