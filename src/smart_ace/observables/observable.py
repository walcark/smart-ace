"""Observable object - describes the physical quantity sampled by Smart-G."""

from typing import Literal

from smart_ace import Geometry
from smartg.smartg import Sensor

from smartg.libATM3D import locate_3Dregular_cells

import numpy as np

from .layout import Layout, Map, Transect, positions
from smart_ace.atmosphere.build import build_grid3d


def build_sensors(
    geo: Geometry, 
    sensor_type: str, 
    layout: Layout, 
    THDEG: float | None = None, 
    PHDEG: float | None = None,
) -> tuple[list[Sensor], tuple[int, ...]]:
    """Build an arrow of sensors, and return the original shape."""
    
    xx, yy = positions(layout)
    shape = xx.shape
    xx, yy = xx.flatten(), yy.flatten()

    if sensor_type == "sat":
        zz = np.full(xx.size, geo.zgrid.max()-1.0)
    elif sensor_type == "ground":
        zz = np.full(xx.size, 0.0)
    else:
        raise ValueError(f"Wrong sensor type: {sensor_type}")

    grid3 = build_grid3d(geo, toa_alt=geo.zgrid.max()+1.0)

    icells = locate_3Dregular_cells(
          grid3.xGRID, grid3.yGRID, grid3.zGRID,   # bien la grille AVEC bords
          xx, yy, zz,
    )

    sensors = [
        Sensor(
            POSX=float(px), 
            POSY=float(py), 
            POSZ=zz[0],
            THDEG=THDEG, 
            PHDEG=PHDEG, 
            ICELL=int(ic),
            LOC="ATMOS"
        )
        for px, py, ic in zip(xx, yy, icells)
    ]
    
    return (sensors, shape)
