"""Cloud geometry: shapes, voxelisation into a periodic domain, and rendering.

Depends on numpy/pydantic/matplotlib only (no SMART-G).
"""

from .geometry import Geometry
from .plot import plot_geometry
from .shapes import Box, Domain, InfXBox, Shape, Sphere, Spheroid, shape_bbox
from .voxelize import grid_and_indexes

__all__ = [
    "Box",
    "InfXBox",
    "Sphere",
    "Spheroid",
    "Shape",
    "Domain",
    "Geometry",
    "shape_bbox",
    "grid_and_indexes",
    "plot_geometry",
]
