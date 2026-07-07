"""Impact of a cloud on the downwelling flux at ground level.

Three cases over a 30x30 km domain are compared through the downwelling flux
at the surface (a hemispheric, FOV=90 planar-flux sensor, ``quantity="flux"``,
``loc="0+"``), sampled by a ground transect of 100 sensors from x = -10 to
+10 km:

* no cloud (a transparent Box, i.e. clear sky),
* a 1x1 km cloud,
* a 5x5 km cloud.

The wider the cloud, the broader and deeper the flux deficit it casts on the
ground beneath it.

Note: "no cloud" is approximated by an optically transparent Box (``kext``≈0);
proper clear-sky support (no cloud voxel at all) is still a library TODO.
"""

import matplotlib.pyplot as plt
from smartg.smartg import Albedo_cst, LambSurface, Smartg
from smartg.truncation import GT_trunc

from smart_ace.atmosphere import AtmoParams, Atmosphere, CloudOptics
from smart_ace.geometry import Box, Domain, Geometry
from smart_ace.observables import Observable, SensorParams, Study, Transect


def main() -> None:
    wl = 560.0

    domain = Domain(dx=30, dy=30, dz=100, zmin=0)
    layout = Transect(res=0.2, axis="x", n=100)  # 100 sensors, x in [-10, 10] km

    # Optically thick cloud slab at [1.0, 1.5] km (tau = 20 * 0.5 = 10);
    # the "no cloud" baseline uses a transparent Box (kext ~ 0 ~ clear sky).
    trunc = GT_trunc(
        trunc_frac=0.435,
        theta_tol=20,
        integral_method="lobatto",
        lobatto_optimization=True,
    )
    thick = CloudOptics(kext=2.0, reff=30.0, trunc=trunc)
    clear = CloudOptics(kext=1e-8, reff=30.0, trunc=trunc)

    cases = [
        ("no cloud", Box(dx=1.0, dy=1.0, dz=0.5, zmin=1.0), clear),
        ("1x1 km", Box(dx=1.0, dy=1.0, dz=0.5, zmin=1.0), thick),
        ("5x5 km", Box(dx=5.0, dy=5.0, dz=0.5, zmin=1.0), thick),
    ]

    # One Observable per case, gathered in a Study.
    study = Study()
    for name, cloud, opt in cases:
        geo = Geometry.build(cloud_shape=cloud, domain=domain)
        atm = Atmosphere(geo, opt, AtmoParams(wl=wl, nth=2000))
        study.add_observable(
            name,
            Observable(
                atmosphere=atm,
                layout=layout,
                sensor=SensorParams(quantity="flux", loc="0+"),
                surf=LambSurface(ALB=Albedo_cst(0.2)),
                le={"th_deg": 0.0, "phi_deg": 0.0},  # zenith sun
                NBPHOTONS=1e8,
            ),
        )

    # The Smartg kernel is compiled once and reused for every observable.
    S = Smartg(opt3D=True, back=True, double=True, bias=True)
    results = study.run(S)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, _, _ in cases:
        results.get(name).plot(ax=ax, label=name)

    ax.set_title("Downwelling ground flux vs cloud size")
    ax.set_ylabel("Downwelling flux at ground")
    ax.legend(title="Cloud")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
