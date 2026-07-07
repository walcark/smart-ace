"""Impact of the cloud extinction on the downwelling flux at ground level.

Same setup as ``demo_cloud_impact_on_ground_flux`` but with a single 5x5 km
Box cloud whose extinction coefficient ``kext`` is swept over
``[0, 0.2, 1, 2, 20]`` km^-1 (i.e. optical thickness ``tau = kext * 0.5`` from
0 to 10). The downwelling flux (FOV=90 hemisphere) is sampled by a ground
transect of 100 sensors from x = -10 to +10 km: the thicker the cloud, the
deeper the flux deficit beneath it.

The geometry is identical across cases (only the optics change), so it is built
once and reused for every ``kext``.
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
    layout = Transect(res=0.2001, axis="x", n=100)  # 100 sensors, x in [-10, 10] km

    # Same 5x5 km cloud slab at [1.0, 1.5] km for every case; only kext varies
    # (kext=0 is the clear-sky reference).
    cloud = Box(dx=5.0, dy=5.0, dz=0.5, zmin=1.0)
    geo = Geometry.build(cloud_shape=cloud, domain=domain)

    kexts = [1e-8, 0.2, 1.0, 2.0, 5.0, 10.0]  # km^-1
    trunc = GT_trunc(
        trunc_frac=0.435,
        theta_tol=20,
        integral_method="lobatto",
        lobatto_optimization=True,
    )

    # One Observable per kext, gathered in a Study.
    study = Study()
    for kext in kexts:
        atm = Atmosphere(
            geo,
            CloudOptics(kext=kext, reff=10.0, trunc=trunc),
            AtmoParams(wl=wl, nth=2000),
        )
        study.add_observable(
            f"kext = {kext:g}",
            Observable(
                atmosphere=atm,
                layout=layout,
                sensor=SensorParams(quantity="flux", loc="0+"),
                surf=LambSurface(ALB=Albedo_cst(0.2)),
                le={"th_deg": 60.0, "phi_deg": 0.0},  # zenith sun
                NBPHOTONS=1e8, 
                direct=True
            ),
        )

    # The Smartg kernel is compiled once and reused for every observable.
    S = Smartg(opt3D=True, back=True, double=True, bias=True)
    results = study.run(S)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for kext in kexts:
        name = f"kext = {kext:g}"
        results.get(name).plot(ax=ax, label=name)

    ax.set_title("Downwelling ground flux under a 5x5 km cloud vs kext")
    ax.set_ylabel("Downwelling flux at ground")
    ax.legend(title="kext [km$^{-1}$]")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
