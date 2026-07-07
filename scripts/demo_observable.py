"""Sample a cloud with SMART-G through an Observable (map or transect).

Builds a cloud geometry, wraps it in an :class:`Atmosphere`, then runs an
:class:`Observable` (satellite or ground, map or transect) and plots the result.
Swap ``layout`` between :class:`Map` and :class:`Transect` to switch between a
2D image and a 1D profile.
"""

import matplotlib.pyplot as plt
from smartg.smartg import Albedo_cst, LambSurface, Smartg

from smart_ace.atmosphere import AtmoParams, Atmosphere, CloudOptics
from smart_ace.geometry import Domain, Geometry, Spheroid
from smart_ace.observables import (  # noqa: F401
    Map,
    Observable,
    SensorParams,
    Transect,
)


def main() -> None:
    wl = 560.0

    geo = Geometry.build(
        cloud_shape=Spheroid(shape="oblate", major=4, minor=1),
        domain=Domain(dx=30, dy=30, dz=100, zmin=0),
        n=(10, 10, 5),
    )

    atm = Atmosphere(
        geo,
        CloudOptics(kext=10.0, reff=10.0),
        AtmoParams(wl=wl, nth=2000),
    )

    obs = Observable(
        atmosphere=atm,
        layout=Map(res=0.5, n=20),  # ou Transect(res=0.5, axis="x", n=40)
        sensor=SensorParams(quantity="radiance", loc="toa"),
        surf=LambSurface(ALB=Albedo_cst(0.2)),
        le={"th_deg": 0.0, "phi_deg": 0.0},  # direction solaire (local estimate)
        NBPHOTONS=1e8,
    )

    S = Smartg(opt3D=True, back=True, double=True, bias=True)

    result = obs.run(S)  # tout est déjà dans l'Observable

    result.plot()
    plt.show()


if __name__ == "__main__":
    main()
