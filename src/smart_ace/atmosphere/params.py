"""Parameter dataclasses describing the cloud optics and background atmosphere.

They carry no SMART-G dependency so they can be built and inspected anywhere;
the actual SMART-G objects are assembled lazily in :mod:`atmosphere.build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CloudOptics:
    """Optical properties of the (uniform) cloud filling the cloud voxels.

    Parameters
    ----------
    kext:
        Extinction coefficient at ``wl_ref`` in km^-1.
    reff:
        Droplet effective radius in micrometres.
    wl_ref:
        Reference wavelength for the extinction, in nanometres.
    species:
        SMART-G cloud species key (``"wc"`` = water cloud).
    trunc:
        Phase-function truncation (smartg ``GT_trunc``); ``None`` = no truncation.
    """

    kext: float
    reff: float
    wl_ref: float = 550.0
    species: str = "wc"
    trunc: Any = None


@dataclass
class AtmoParams:
    """1D background atmosphere and single-wavelength simulation settings.

    Parameters
    ----------
    wl:
        Working wavelength in nanometres.
    profile:
        AFGL atmospheric profile name (e.g. ``"afglus"``, ``"afglt"``).
    surface_pressure:
        Surface pressure in hPa; ``None`` keeps the profile's own value.
    toa_alt:
        Top-of-atmosphere altitude in km; the vertical grid is extended up to it.
    nth:
        Number of angles used to discretise the phase functions.
    """

    wl: float
    profile: str = "afglus"
    surface_pressure: float | None = None
    toa_alt: float = 120.0
    nth: int = 7200
