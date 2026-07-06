"""Create an atmosphere with a cloud for Smart-G.

Then, a simulation is launched in order to start implementation of a
Smart-G run.
"""

from smart_ace.atmosphere import AtmoParams, CloudOptics, Atmosphere
from smart_ace.geometry import Geometry, Box, Domain, Spheroid
from smart_ace.observables.observable import build_sensors
from smart_ace.observables.layout import Map, Transect
from smartg.smartg import Smartg, LambSurface, Albedo_cst
from smartg.smartg import Smartg
import matplotlib.pyplot as plt
import numpy as np

def main():
    wl = 560.0

    geo: Geometry = Geometry.build(
        cloud_shape=Spheroid(shape="oblate", major=4, minor=1),
        domain=Domain(dx=30, dy=30, dz=100, zmin=0),
        n=(10, 10, 5)
    )

    opt: CloudOptics = CloudOptics(kext=10.0, reff=10.0)
    atmp: AtmoParams = AtmoParams(wl=wl, nth=2000)

    atm: Atmosphere = Atmosphere(geo, opt, atmp)

    S = Smartg(opt3D=True, back=True, double=True, bias=True)
 
    #layout = Transect(res=0.5, axis="x", n=20)
    layout = Map(res=0.5, n=20)
    sensors, shape = build_sensors(geo=geo, sensor_type="sat", layout=layout, THDEG=180.0, PHDEG=0.0)

    result = S.run(
        wl=wl,
        atm=atm.build(),
        surf=LambSurface(ALB=Albedo_cst(0.2)),
        sensor=sensors,
        le={"th_deg": 0.0, "phi_deg": 0.0},  # direction solaire (local estimate)
        NBPHOTONS=1E8,
    )

    result.describe() 
    arr = result["I_up (TOA)"][:, 0, 0]
    print(arr)

    if isinstance(layout, Transect):
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.plot(np.arange(arr.size), np.reshape(arr, shape).flatten())
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.imshow(np.reshape(arr, shape))
        plt.show()



if __name__ == "__main__":
    main()
