"""Gallery of the supported cloud shapes, each voxelised inside its full domain.

Run with::

    pixi run python scripts/demo_cloud_shapes.py
"""

import matplotlib.pyplot as plt

from smart_ace import Box, Geometry, InfXBox, Sphere, Spheroid
from smart_ace.geometry import plot_geometry

#: (title, cloud shape, domain, resolution) for each panel.
EXAMPLES = [
    (
        "Box cube 1km (IPRT [0,3,4,7])",
        Box(dx=1, dy=1, dz=1, zmin=2),
        Box(dx=7, dy=7, dz=5, zmin=0),
        1,
    ),
    ("Sphere r=4, n=8", Sphere(radius=4, zmin=2), Box(dx=14, dy=14, dz=12, zmin=0), 8),
    ("Sphere r=4, n=16", Sphere(radius=4, zmin=2), Box(dx=14, dy=14, dz=12, zmin=0), 16),
    (
        "Spheroid prolate (tall)",
        Spheroid(shape="prolate", major=5, minor=2, zmin=1),
        Box(dx=14, dy=14, dz=14, zmin=0),
        10,
    ),
    (
        "Spheroid oblate (flat)",
        Spheroid(shape="oblate", major=5, minor=2, zmin=1),
        Box(dx=16, dy=16, dz=10, zmin=0),
        10,
    ),
    ("InfXBox (fills x)", InfXBox(dy=2, dz=1, zmin=3), Box(dx=8, dy=8, dz=10, zmin=0), 1),
]


def main() -> None:
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Cloud geometry in the full domain", fontsize=13)
    for i, (_, shape, domain, n) in enumerate(EXAMPLES, start=1):
        geo = Geometry.build(shape, domain, n)
        plot_geometry(geo, ax=fig.add_subplot(2, 3, i, projection="3d"))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
