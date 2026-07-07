"""Evolution of a satellite transect with the surface atmospheric pressure.

Runs the *same* satellite transect across a cloud for several surface pressures
and overlays the resulting radiance profiles. Each pressure is a separate,
fully declarative :class:`Observable` (only ``surface_pressure`` changes); they
are gathered in a :class:`Study`, run in one call, then addressed by name to be
overlaid on a single axes.
"""

import matplotlib.pyplot as plt
from smartg.smartg import Albedo_cst, LambSurface, Smartg

from smart_ace.atmosphere import AtmoParams, Atmosphere, CloudOptics
from smart_ace.geometry import Domain, Geometry, Box
from smart_ace.observables import Observable, SensorParams, Study, Transect


def main() -> None:
    wl = 560.0
    pressures = [500.0, 1000.0, 3000.0]

    geo = Geometry.build(
        cloud_shape=Box(dx=5, dy=5),
        domain=Domain(dx=30, dy=30, dz=100, zmin=0),
    )
    opt = CloudOptics(kext=10.0, reff=10.0)
    layout = Transect(res=0.5, axis="x", n=40)

    # One Observable per pressure, gathered in a Study and named by pressure.
    study = Study()
    for p0 in pressures:
        atm = Atmosphere(
            geo,
            opt,
            AtmoParams(wl=wl, nth=2000, surface_pressure=p0),
        )
        study.add_observable(
            f"{p0:.0f} hPa",
            Observable(
                atmosphere=atm,
                layout=layout,
                sensor=SensorParams(quantity="radiance", loc="toa", thdeg=180.0, phdeg=0.0),
                surf=LambSurface(ALB=Albedo_cst(0.0)),
                le={"th_deg": 0.0, "phi_deg": 0.0},
                NBPHOTONS=1e8,
            ),
        )

    # The Smartg kernel is compiled once and reused for every observable.
    S = Smartg(opt3D=True, back=True, double=True, bias=True)
    results = study.run(S)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for p0 in pressures:
        name = f"{p0:.0f} hPa"
        results.get(name).plot(ax=ax, label=name)

    ax.set_title("Satellite transect vs surface pressure")
    ax.legend(title="Surface pressure")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
