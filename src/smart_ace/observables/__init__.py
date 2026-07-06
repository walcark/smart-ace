"""Observables: sample a SMART-G run over a sensor layout and plot the result.

An :class:`Observable` ties an :class:`~smart_ace.atmosphere.Atmosphere` to a
sensor :data:`Layout` (:class:`Map` or :class:`Transect`), runs SMART-G and
returns a :class:`Result` that a plot helper renders as a line or an image.

Importing this package stays light: SMART-G and matplotlib are imported lazily,
only when actually running or plotting.
"""

from .layout import Layout, Map, Transect, positions
from .observable import Observable, Result, build_sensors
from .plot import plot_result

__all__ = [
    "Layout",
    "Map",
    "Transect",
    "positions",
    "Observable",
    "Result",
    "build_sensors",
    "plot_result",
]
