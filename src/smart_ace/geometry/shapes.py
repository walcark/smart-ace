"""Define all the shapes allowed for the cloud.

Shapes are stored in a discriminated type union called `Shape`. They are
used to model cloud shapes, and a special shape called `Box` is used both
to represent an ideal cloud shape and to represent the study domain and
cloud bounding-box.
"""

from math import inf
from typing import Annotated, Any, ClassVar, Literal, assert_never, override

from pydantic import BaseModel, Field, model_validator


class Box(BaseModel):
    """A box with finite extents along all axes."""

    shape: Literal["box"] = "box"
    dx: float = Field(
        default=10.0, gt=0, description="Cloud size along the x-axis [km]"
    )
    dy: float = Field(
        default=10.0, gt=0, description="Cloud size along the y-axis [km]"
    )
    dz: float = Field(default=1.0, gt=0, description="Cloud thickness [km]")
    zmin: float = Field(
        default=1.0, ge=0, description="Bottom elevation of the box [km]"
    )


class InfXBox(Box):
    """A box infinitely extended along the x-axis."""

    shape: Literal["infxbox"] = "infxbox"  # type: ignore[assignment]
    DX_FIXED: ClassVar[float] = inf

    @override
    def model_post_init(self, context: Any, /) -> None:
        self.dx = self.DX_FIXED
        return super().model_post_init(context)


class Sphere(BaseModel):
    """A sphere defined by its radius and bottom elevation."""

    shape: Literal["sphere"] = "sphere"
    radius: float = Field(default=10.0, gt=0, description="Sphere radius [km]")
    zmin: float = Field(
        default=1.0, ge=0, description="Bottom elevation of the sphere [km]"
    )


class Spheroid(BaseModel):
    """An ellipsoid of revolution defined by two semi-axis lengths.

    The `shape` mode sets the orientation of the axis of revolution:

    * `prolate` (egg-shaped): the `major` axis is the axis of revolution and is
      vertical, so the horizontal extent uses `minor`.
    * `oblate` (flattened): the `minor` axis is the axis of revolution and is
      vertical, so the horizontal extent uses `major`.
    """

    shape: Literal["prolate", "oblate"]
    major: float = Field(
        default=10.0, gt=0, description="Larger semi-axis [km]"
    )
    minor: float = Field(
        default=5.0, gt=0, description="Smaller semi-axis [km]"
    )
    zmin: float = Field(
        default=1.0, ge=0, description="Bottom elevation of the spheroid [km]"
    )

    @model_validator(mode="after")
    def _check_axes(self) -> "Spheroid":
        if self.minor > self.major:
            raise ValueError("`minor` must be <= `major`")
        return self


Shape = Annotated[
    Box | InfXBox | Sphere | Spheroid,
    Field(discriminator="shape"),
]

# A domain is just an axis-aligned box; alias for readable signatures.
Domain = Box


def _semi_axes(shape: Sphere | Spheroid) -> tuple[float, float]:
    """Return the (horizontal, vertical) semi-axes [km] of an ellipsoid.

    The axis of revolution is vertical: for a prolate spheroid it is the major
    axis, for an oblate one the minor axis. A sphere is the isotropic case.
    """
    if isinstance(shape, Sphere):
        return shape.radius, shape.radius
    if shape.shape == "prolate":
        return shape.minor, shape.major
    return shape.major, shape.minor  # oblate


def shape_bbox(shape: Shape) -> Box:
    """Axis-aligned bounding box of a shape, whatever its kind.

    The bounding box of a shape is itself a :class:`Box` (a shape), so this
    maps any shape onto the box geometry that encloses it.
    """
    # `Box` also covers `InfXBox` (a Box with an infinite x-extent). Normalise
    # to a plain Box so callers never have to deal with the subclass.
    if isinstance(shape, Box):
        return Box(dx=shape.dx, dy=shape.dy, dz=shape.dz, zmin=shape.zmin)

    if isinstance(shape, (Sphere, Spheroid)):
        horizontal, vertical = _semi_axes(shape)
        return Box(
            dx=2 * horizontal,
            dy=2 * horizontal,
            dz=2 * vertical,
            zmin=shape.zmin,
        )

    assert_never(shape)
