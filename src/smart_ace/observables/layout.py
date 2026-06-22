"""Horizontal arrangement (layout) of the sensors of an observable.

The modes are a discriminated union on ``mode``:

* ``map``      — a 2D grid of sensors (an image).
* ``transect`` — a 1D line of sensors along x or y (a profile).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Map(BaseModel):
    """2D sampling grid for sensors objects in Smart-G.

    Parameters
    ----------
    res : float, default = 0.5
        Space between sensors (on both axes) [km].
    """

    mode: Literal["map"] = "map"
    res: float = Field(default=0.5, gt=0, description="Sensor spacing [km]")


class Transect(BaseModel):
    """1D sampling transect for sensors objects in Smart-G.

    Parameters
    ----------
    axis : Literal['x', 'y'], default = 'x'
        Axis along which the transect is performed.
    res : float, default = 0.5
        Space between sensors (on ``axis``) [km]
    """

    mode: Literal["transect"] = "transect"
    axis: Literal["x", "y"] = "x"
    res: float = Field(default=0.5, gt=0, description="Sensor spacing [km]")


Layout = Annotated[Map | Transect, Field(discriminator="mode")]
