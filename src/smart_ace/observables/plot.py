"""Plot a :class:`Result`: a line for a transect, an image for a map.

Matplotlib is imported lazily so importing this module stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .observable import Result


def plot_result(result: "Result", ax: Any = None, **kwargs: Any) -> Any:
    """Plot a run result on ``ax`` (created if ``None``); return the axes.

    * transect -> ``ax.plot`` of the radiance along the transect axis.
    * map -> ``ax.imshow`` with a colorbar, spatial ``extent`` [km].

    Extra ``kwargs`` are forwarded to the underlying matplotlib call, so a
    transect can be given a ``label`` to overlay several results on one axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.8))

    if result.is_map:
        x, y = result.coords
        extent = [
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
        ]
        im = ax.imshow(
            result.values,
            origin="lower",
            extent=extent,
            aspect="equal",
            **kwargs,
        )
        ax.figure.colorbar(im, ax=ax, label=result.quantity)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
    else:
        (s,) = result.coords
        ax.plot(s, result.values, **kwargs)
        ax.set_xlabel(f"{result.axis} [km]")
        ax.set_ylabel(result.quantity)

    return ax
