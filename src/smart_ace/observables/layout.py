"""Horizontal arrangement (layout) of the sensors of an observable.

The modes are a discriminated union on ``mode``:

* ``map``      — a 2D grid of sensors (an image).
* ``transect`` — a 1D line of sensors along x or y (a profile).
"""

from typing import Annotated, Literal

import numpy as np

from pydantic import BaseModel, Field


class Map(BaseModel):
    """2D sampling grid for sensors objects in Smart-G.

    Parameters
    ----------
    res : float, default = 0.5
        Space between sensors (on both axes) [km].
    n : int, default = 10
        Number of sensors per axis.
    """

    mode: Literal["map"] = "map"
    res: float = Field(default=0.5, gt=0, description="Sensor spacing [km]")
    n: int = Field(default=10, gt=0, description="Number of sensors")


class Transect(BaseModel):
    """1D sampling transect for sensors objects in Smart-G.

    Parameters
    ----------
    axis : Literal['x', 'y'], default = 'x'
        Axis along which the transect is performed.
    res : float, default = 0.5
        Space between sensors (on ``axis``) [km]
    n : int, default = 10
        Number of sensors per axis.
    """

    mode: Literal["transect"] = "transect"
    axis: Literal["x", "y"] = "x"
    res: float = Field(default=0.5, gt=0, description="Sensor spacing [km]")
    n: int = Field(default=10, gt=0, description="Number of sensors per axis")


Layout = Annotated[Map | Transect, Field(discriminator="mode")]


def positions(layout: Layout) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) positions of the sensors."""
    
    res = layout.res
    n = layout.n
    
    half = n*res/2
    x: np.ndarray = - half + res/2 + np.arange(n) * res
    y: np.ndarray = - half + res/2 + np.arange(n) * res
    
    if isinstance(layout, Transect):
        if layout.axis == "x":
            y = np.zeros((1))
        if layout.axis == "y":
            x = np.zeros((1))

    xx, yy = np.meshgrid(x, y)
    
    return xx, yy
