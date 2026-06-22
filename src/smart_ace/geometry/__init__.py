"""Parameter models for SMART-ACE.

``GROUPS`` is the single registry the CLI consumes: it maps each result-dict
key to its Pydantic model. Adding a parameter group = adding a model here; the
CLI derives its options and instantiation generically (see ``smart_ace.cli``).
"""

from pydantic import BaseModel

from smart_ace.params.cloud import CloudParams
from smart_ace.params.grid import GridParams
from smart_ace.params.simu import SimuParams

# Result-dict key -> its model. Order also defines CLI option order.
GROUPS: dict[str, type[BaseModel]] = {
    "cloud": CloudParams,
    "grid": GridParams,
    "simu": SimuParams,
}

__all__ = ["GROUPS", "CloudParams", "GridParams", "SimuParams"]
