"""Atmosphere: turn a cloud Geometry into a SMART-G AtmAFGL profile.

This is the SMART-G boundary of the library: importing this package does *not*
require SMART-G (it is imported lazily by ``Atmosphere.build``), but building
the profile does.
"""

from .atmosphere import Atmosphere
from .params import AtmoParams, CloudOptics

__all__ = ["Atmosphere", "CloudOptics", "AtmoParams"]
