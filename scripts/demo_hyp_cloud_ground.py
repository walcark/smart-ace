"""Impact of the 'cloud at ground level' hypothesis on nadir TOA radiance."""

import matplotlib.pyplot as plt
from smartg.smartg import Albedo_cst, LambSurface, Smartg
from smartg.truncation import GT_trunc

from smart_ace.atmosphere import AtmoParams, Atmosphere, CloudOptics
from smart_ace.geometry import Box, Domain, Geometry, Spheroid
from smart_ace.observables import Observable, SensorParams, Study, Transect


def main() -> None:
    wl = 860.0

    dz = 0.1
    kext = 100.0
    base_altitudes = [0.0, 0.5, 1.0, 2.0]

    domain = Domain(dx=30, dy=30, dz=100, zmin=0)
    trunc = GT_trunc(
        trunc_frac=0.435,
        theta_tol=20,
        integral_method="lobatto",
        lobatto_optimization=True,
    )
    opt = CloudOptics(kext=kext, reff=5.0, trunc=trunc)
    layout = Transect(res=0.5, axis="x", n=40)  # x in [-10, 10] km

    # One Observable per cloud altitude, gathered in a Study.
    study = Study()
    for zmin in base_altitudes:
        geo = Geometry.build(
            cloud_shape=Spheroid(shape="prolate", major=4.0, minor=dz, zmin=zmin),
            domain=domain,
            n=(10, 10, 5)
        )
        atm = Atmosphere(geo, opt, AtmoParams(wl=wl, nth=2000))
        study.add_observable(
            f"z = {zmin:.1f} km",
            Observable(
                atmosphere=atm,
                layout=layout,
                sensor=SensorParams(quantity="radiance", loc="toa"),
                surf=LambSurface(ALB=Albedo_cst(0.2)),
                le={"th_deg": 0.0, "phi_deg": 0.0},  # zenith sun
                NBPHOTONS=1e8,
            ),
        )

    # The Smartg kernel is compiled once and reused for every observable.
    S = Smartg(opt3D=True, back=True, double=True, bias=True)
    results = study.run(S)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for zmin in base_altitudes:
        name = f"z = {zmin:.1f} km"
        results.get(name).plot(ax=ax, label=name)

    ax.set_title(f"Nadir TOA radiance vs cloud base altitude (tau = {kext*dz})")
    ax.legend(title="Cloud base")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
