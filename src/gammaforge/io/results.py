"""Results contract (GRAND_PLAN.md §3.6).

One slice shape for every engine. A tabulated engine fills a `PhasespaceSlice` directly;
a Monte-Carlo engine histograms its samples into the same shape — so the predecessor's
Sampled-vs-Binned duck typing has nothing left to be ambiguous about.

**Axes are an enum with canonical CGS units.** `Axis.ENERGY` is in erg, always; the
predecessor's unit-baked-into-the-name smell (``"E_eV"``) is gone, and display conversion
belongs to the plotting and serialization layers alone.

`Results` deliberately carries **no back-reference to the configuration that produced it**
(P9): no ``Results.cfg``, no derived properties duplicating beam or laser fields. Whoever
needs a parameter reads it from the parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

__all__ = [
    "Axis",
    "PhasespaceSlice",
    "PhotonMacroparticles",
    "Results",
    "ALLOWED_AXIS_GROUPINGS",
]


class Axis(Enum):
    """A slice axis, its serialization key, and the canonical CGS unit its values carry.

    The value is the ``(key, unit)`` pair rather than the unit alone for a load-bearing
    reason: ``X`` and ``Y`` share the unit ``cm``, as do ``THETA_X`` and ``THETA_Y`` with
    ``rad``, and `Enum` silently collapses members that share a value into **aliases** —
    which would make ``Axis.Y is Axis.X`` true and quietly destroy every two-dimensional
    slice. Distinct keys keep the six members six.
    """

    ENERGY = ("energy", "erg")
    TIME = ("time", "s")
    X = ("x", "cm")
    Y = ("y", "cm")
    THETA_X = ("theta_x", "rad")
    THETA_Y = ("theta_y", "rad")

    def __init__(self, key: str, unit: str) -> None:
        self._key = key
        self._unit = unit

    @property
    def key(self) -> str:
        """Stable lowercase identifier for serialization (HDF5 dataset names, YAML keys)."""
        return self._key

    @property
    def unit(self) -> str:
        return self._unit

    @classmethod
    def from_key(cls, key: str) -> "Axis":
        for axis in cls:
            if axis.key == key:
                return axis
        raise ValueError(f"no Axis with key {key!r}")


#: The closed set of axis groupings a slice may have, carried over from the predecessor's
#: results contract. Validated rather than documented: an engine that invents a grouping
#: has made an error the GUI cannot render, and it should surface here, not there.
ALLOWED_AXIS_GROUPINGS: frozenset[frozenset[Axis]] = frozenset(
    frozenset(group)
    for group in [
        (),
        (Axis.ENERGY,),
        (Axis.TIME,),
        (Axis.X, Axis.Y),
        (Axis.THETA_X, Axis.THETA_Y),
        (Axis.ENERGY, Axis.THETA_X),
        (Axis.ENERGY, Axis.THETA_Y),
        (Axis.ENERGY, Axis.THETA_X, Axis.THETA_Y),
        (Axis.ENERGY, Axis.X, Axis.Y),
    ]
)


@dataclass(frozen=True)
class PhasespaceSlice:
    """A photon density over a named set of axes.

    ``axes`` maps each `Axis` to its bin-centre values; ``distr`` holds the density, with
    one dimension per axis in the iteration order of ``axes``. An empty ``axes`` means a
    0D total yield, whose ``distr`` is a scalar-shaped array.

    Densities are **per unit of each axis** (e.g. photons/erg for a spectrum), so
    integrating with the axis spacing gives the yield. Storing densities rather than
    per-bin counts is what makes two engines with different binning directly comparable.
    """

    axes: Mapping[Axis, np.ndarray]
    distr: np.ndarray

    def __post_init__(self) -> None:
        axes = {axis: np.asarray(values, dtype=float) for axis, values in self.axes.items()}
        grouping = frozenset(axes)
        if len(grouping) != len(axes):
            raise ValueError("PhasespaceSlice: repeated axis")
        if grouping not in ALLOWED_AXIS_GROUPINGS:
            raise ValueError(
                f"PhasespaceSlice: {sorted(a.name for a in grouping)} is not an allowed "
                f"axis grouping (GRAND_PLAN.md §3.6)"
            )
        distr = np.asarray(self.distr, dtype=float)
        expected = tuple(values.size for values in axes.values())
        if distr.shape != expected:
            raise ValueError(
                f"PhasespaceSlice: distr shape {distr.shape} does not match axis sizes {expected}"
            )
        object.__setattr__(self, "axes", MappingProxyType(axes))
        object.__setattr__(self, "distr", distr)

    @property
    def axis_order(self) -> tuple[Axis, ...]:
        return tuple(self.axes)

    def integrate(self) -> float:
        """Total yield: the density integrated over every axis.

        Uses the trapezoid rule over the bin centres, which is what makes
        ``integral(spectrum) == integral(angular spectrum) == total_yield`` an exact
        identity between slices of the same run rather than a tolerance (§7).

        Every axis needs at least two samples: a one-sample axis carries no width, so its
        integral is not defined and silently assuming one would corrupt the identity
        above. A 0D slice (no axes) integrates to its own scalar.
        """
        result = self.distr
        for axis in reversed(self.axis_order):
            values = self.axes[axis]
            if values.size < 2:
                raise ValueError(
                    f"PhasespaceSlice.integrate: axis {axis.name} has {values.size} sample(s); "
                    f"at least 2 are needed for a bin width to exist"
                )
            result = np.trapezoid(result, values, axis=-1)
        return float(result)

    def scaled(self, factor: float) -> "PhasespaceSlice":
        """The same slice with its density scaled — how a charge edit rescales results.

        Every output is exactly linear in ``N_e`` (§3.5), so a charge-only change is a
        pure display operation on existing results, never an engine run (§5).
        """
        return PhasespaceSlice(axes=dict(self.axes), distr=self.distr * factor)


@dataclass(frozen=True)
class PhotonMacroparticles:
    """Per-photon arrays from a Monte-Carlo engine. CGS.

    Typed separately from the final electrons: the predecessor's kascade already kept the
    two populations apart (``ph_*`` arrays vs ``eps_f``/``thx_f``/...), and the rebuild
    keeps that distinction in the type system rather than in a naming convention.
    """

    energy: np.ndarray  # erg
    theta_x: np.ndarray  # rad
    theta_y: np.ndarray  # rad
    x: np.ndarray  # cm
    y: np.ndarray  # cm
    z: np.ndarray  # cm
    t: np.ndarray  # s, emission time
    weight: np.ndarray  # photons per macroparticle
    order: np.ndarray | None = None  # scattering order / parent index, engine-dependent

    @property
    def n_macroparticles(self) -> int:
        return int(np.asarray(self.energy).shape[0])


@dataclass(frozen=True)
class Results:
    """What every `Engine.run()` returns.

    ``photon_slices`` is keyed by `OutputKind` — engines omit what they do not support,
    and the GUI renders whatever came back rather than asking what to expect.
    ``electrons``/``photons`` are produced only by MC engines.

    **Open question (§10.8):** whether ``electrons`` should keep reusing `Bunch` or get a
    symmetric ``ElectronMacroparticles`` type is deliberately unresolved — `Bunch` has no
    per-particle emission time, and a second MC engine in development elsewhere should
    settle the right shape before this is locked in.
    """

    photon_slices: Mapping[object, PhasespaceSlice]
    electrons: object | None = None  # Bunch | None — see the §10.8 note above
    photons: PhotonMacroparticles | None = None
    model_specific: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "photon_slices", MappingProxyType(dict(self.photon_slices)))

    def scaled(self, factor: float) -> "Results":
        """Photon slices and macroparticle weights rescaled for a charge-only edit."""
        photons = (
            replace(self.photons, weight=np.asarray(self.photons.weight) * factor)
            if self.photons is not None
            else None
        )
        return Results(
            photon_slices={kind: slice_.scaled(factor) for kind, slice_ in self.photon_slices.items()},
            electrons=self.electrons,
            photons=photons,
            model_specific=dict(self.model_specific),
        )
