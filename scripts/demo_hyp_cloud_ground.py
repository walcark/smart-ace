"""Impact of the 'cloud at ground level' hypothesis on nadir TOA radiance.

A single optically thick Box cloud (optical thickness 5) is placed at three
altitudes — [0, 0.1], [1.0, 1.1] and [2.0, 2.1] km — with everything else kept
identical. For each case a nadir satellite radiance transect is drawn across
the cloud, so the three curves isolate how much the *assumed cloud altitude
alone* changes the top-of-atmosphere signal (i.e. the error made when a cloud
is wrongly assumed to sit on the ground).
"""

import matplotlib.pyplot as plt
from smartg.smartg import Albedo_cst, LambSurface, Smartg

from smart_ace.atmosphere import AtmoParams, Atmosphere, CloudOptics
from smart_ace.geometry import Box, Domain, Geometry
from smart_ace.observables import Observable, SensorParams, Study, Transect


def main() -> None:
    wl = 560.0

    # Optical thickness tau = kext * dz = 50 km^-1 * 0.1 km = 5.
    dz = 0.1
    kext = 50.0
    base_altitudes = [0.0, 1.0, 2.0]  # km, cloud bottom (zmin)

    domain = Domain(dx=30, dy=30, dz=100, zmin=0)
    opt = CloudOptics(kext=kext, reff=10.0)
    layout = Transect(res=0.5, axis="x", n=40)  # x in [-10, 10] km

    # One Observable per cloud altitude, gathered in a Study.
    study = Study()
    for zmin in base_altitudes:
        geo = Geometry.build(
            cloud_shape=Box(dx=4.0, dy=4.0, dz=dz, zmin=zmin),
            domain=domain,
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

    ax.set_title("Nadir TOA radiance vs cloud base altitude (tau = 5)")
    ax.legend(title="Cloud base")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
