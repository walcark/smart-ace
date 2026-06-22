"""Cloud-adjacency simulation library: geometry → atmosphere → observables → run.

The headline classes are re-exported here for convenience. Importing this
package stays light (matplotlib and SMART-G are imported lazily, only when
plotting or building/running).
"""

from .geometry import (
    Box,
    Domain,
    Geometry,
    InfXBox,
    Shape,
    Sphere,
    Spheroid,
)

__all__ = [
    "Box",
    "InfXBox",
    "Sphere",
    "Spheroid",
    "Shape",
    "Domain",
    "Geometry",
]
