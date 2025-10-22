from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcdc.object_.surface import Surface

####

import numpy as np
import sympy

from numpy import float64
from numpy.typing import NDArray
from operator import attrgetter
from types import NoneType
from typing import Annotated, Iterable
from sympy.logic.boolalg import Boolean

####

from mcdc.constant import (
    BOOL_AND,
    BOOL_NOT,
    BOOL_OR,
    FILL_LATTICE,
    FILL_MATERIAL,
    FILL_NONE,
    FILL_UNIVERSE,
    PI,
)
from mcdc.object_.base import ObjectNonSingleton
from mcdc.object_.material import MaterialBase
from mcdc.object_.simulation import simulation
from mcdc.object_.tally import TallyCSBin
from mcdc.object_.universe import Universe, Lattice
from mcdc.print_ import print_error


# ======================================================================================
# Compressed Sensing Bin
# ======================================================================================


class CSBin(ObjectNonSingleton):
    label: str = "csbin"
    non_numba: list[str] = []

    center: Annotated[NDArray[float64], (3,)]
    size: Annotated[NDArray[float64], (3,)]
    tallies: list[TallyCSBin]

    def __init__(
        self,
        center: Iterable[float] = (0.0, 0.0, 0.0),
        size: float | Iterable[float] = 1.0,
        name: str = "",
    ):
        super().__init__()
        self.name = name if name else f"{self.label}_{self.numba_ID}"

        self.center = np.array(center, dtype=float)
        self.size = np.array(size if not np.isscalar(size) else [size] * 3, dtype=float)
        self.tallies = []
