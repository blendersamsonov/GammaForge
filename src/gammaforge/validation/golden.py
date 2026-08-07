"""Golden reference snapshots: what they are, where they live, how they compare.

A golden is one committed HDF5 file holding the `Results` some *other* implementation
produced for one scenario — today only the predecessor repo's models, via
`make_references`. New-vs-golden comparison is the default validation leg; regenerating
a golden is a deliberate act on the author's machine (§7).

**Goldens are transitional.** They encode whatever the old repo did, bugs included, so a
disagreement is a question to investigate rather than an automatic failure of the new
code. The real anchors are the closed-form identities and delta (§7, §12); these files
exist so that a rewrite cannot silently change an answer nobody was watching.

Format: the ordinary `gammaforge.io.formats.hdf5` results file — one serialization path
for results, not a private second one — plus a ``provenance`` group of string attributes
naming what produced it, and a ``scalars`` group for the model's own reported numbers
(`Results.model_specific`), which are frequently the sharpest comparison available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from ..io.formats.hdf5 import load_results, save_results
from ..io.results import Results
from ..io.target import OutputKind
from .metrics import Deviation, compare_slices, relative_error

__all__ = [
    "REFERENCE_DIR",
    "Provenance",
    "Golden",
    "golden_path",
    "save_golden",
    "load_golden",
    "available_goldens",
    "compare_to_golden",
    "SliceComparison",
    "GoldenComparison",
]

#: Where committed snapshots live. Inside the package, so they travel with an install and
#: are reachable without knowing where the source tree is.
REFERENCE_DIR = Path(__file__).resolve().parent / "references" / "data"


@dataclass(frozen=True)
class Provenance:
    """Where a golden came from. Every field is written as an HDF5 string attribute.

    ``source_commit`` is the generating repo's HEAD and ``source_dirty`` records whether
    its tree was clean: a golden generated from an uncommitted state is still useful, but
    it is not reproducible, and saying so beats discovering it later.
    """

    model: str
    scenario: str
    source_repo: str
    source_commit: str
    source_dirty: bool
    generated_utc: str
    note: str = ""

    def as_attrs(self) -> dict[str, str]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "source_repo": self.source_repo,
            "source_commit": self.source_commit,
            "source_dirty": "true" if self.source_dirty else "false",
            "generated_utc": self.generated_utc,
            "note": self.note,
        }

    @classmethod
    def from_attrs(cls, attrs) -> "Provenance":
        text = {key: _text(attrs[key]) for key in attrs}
        return cls(
            model=text["model"],
            scenario=text["scenario"],
            source_repo=text["source_repo"],
            source_commit=text["source_commit"],
            source_dirty=text["source_dirty"] == "true",
            generated_utc=text["generated_utc"],
            note=text.get("note", ""),
        )


@dataclass(frozen=True)
class Golden:
    """A loaded snapshot: the results, the numbers the model reported, and its origin."""

    results: Results
    scalars: dict[str, float]
    provenance: Provenance


def golden_path(scenario: str, model: str, directory: Path | None = None) -> Path:
    return (directory or REFERENCE_DIR) / scenario / f"{model}.h5"


def save_golden(
    results: Results,
    provenance: Provenance,
    scalars: dict[str, float] | None = None,
    directory: Path | None = None,
) -> Path:
    """Write one snapshot, creating its scenario directory."""
    path = golden_path(provenance.scenario, provenance.model, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, path)
    with h5py.File(path, "a") as handle:
        group = handle.create_group("provenance")
        for key, value in provenance.as_attrs().items():
            group.attrs[key] = value
        handle.create_group("scalars").attrs["json"] = json.dumps(scalars or {}, sort_keys=True)
    return path


def load_golden(scenario: str, model: str, directory: Path | None = None) -> Golden:
    path = golden_path(scenario, model, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"no golden for scenario {scenario!r}, model {model!r} at {path} — generate it "
            f"with `python -m gammaforge.validation.make_references`"
        )
    results = load_results(path, kind_from_name=OutputKind)
    with h5py.File(path, "r") as handle:
        provenance = Provenance.from_attrs(handle["provenance"].attrs)
        scalars = json.loads(_text(handle["scalars"].attrs["json"]))
    return Golden(results=results, scalars=scalars, provenance=provenance)


def available_goldens(directory: Path | None = None) -> list[tuple[str, str]]:
    """Every committed ``(scenario, model)`` pair, sorted — what a run can compare against."""
    root = directory or REFERENCE_DIR
    if not root.exists():
        return []
    return sorted(
        (path.parent.name, path.stem)
        for path in root.glob("*/*.h5")
    )


@dataclass(frozen=True)
class SliceComparison:
    kind: OutputKind
    deviation: Deviation
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.deviation.worst() <= self.tolerance


@dataclass(frozen=True)
class GoldenComparison:
    """One engine's results against one golden, output by output and scalar by scalar."""

    scenario: str
    model: str
    slices: tuple[SliceComparison, ...]
    scalars: dict[str, float]  # name -> relative error
    scalar_tolerance: float
    missing: tuple[OutputKind, ...]

    @property
    def passed(self) -> bool:
        return (
            all(comparison.passed for comparison in self.slices)
            and all(abs(error) <= self.scalar_tolerance for error in self.scalars.values())
        )

    def summary(self) -> str:
        lines = [f"{self.model} vs golden [{self.scenario}]: "
                 f"{'PASS' if self.passed else 'FAIL'}"]
        for comparison in self.slices:
            deviation = comparison.deviation
            lines.append(
                f"  {comparison.kind.value:<28} yield {deviation.yield_error:+.3e}  "
                f"L1 {deviation.weighted_l1:.3e}  max {deviation.max_window:.3e}  "
                f"(tol {comparison.tolerance:.1e}) "
                f"{'ok' if comparison.passed else 'FAIL'}"
            )
        for name, error in sorted(self.scalars.items()):
            ok = abs(error) <= self.scalar_tolerance
            lines.append(f"  scalar {name:<21} {error:+.3e} {'ok' if ok else 'FAIL'}")
        if self.missing:
            lines.append(f"  not produced by this engine: "
                         f"{', '.join(kind.value for kind in self.missing)}")
        return "\n".join(lines)


def compare_to_golden(
    results: Results,
    golden: Golden,
    *,
    tolerance: float = 2e-2,
    scalar_tolerance: float = 1e-6,
    scalars: dict[str, float] | None = None,
    tolerances: dict[OutputKind, float] | None = None,
) -> GoldenComparison:
    """Compare fresh ``results`` (and optionally derived ``scalars``) against a snapshot.

    Outputs the golden has and ``results`` does not are **reported, not failed**: engines
    legitimately omit what they cannot produce (§4.1), and a spectrum-only engine should
    not fail for lacking an angular distribution. The reverse — an output the golden lacks
    — is simply not comparable and is skipped.

    Two tolerances, because the two comparisons are different in kind: slice tolerances
    absorb Monte-Carlo noise and genuine method differences (default 2%), while ``scalars``
    are closed-form quantities both implementations compute from the same inputs, so they
    are held to near-machine agreement (default 1e-6).
    """
    comparisons = []
    missing = []
    for kind, reference in golden.results.photon_slices.items():
        if kind not in results.photon_slices:
            missing.append(kind)
            continue
        comparisons.append(
            SliceComparison(
                kind=kind,
                deviation=compare_slices(results.photon_slices[kind], reference),
                tolerance=(tolerances or {}).get(kind, tolerance),
            )
        )
    scalar_errors = {
        name: relative_error(value, golden.scalars[name])
        for name, value in (scalars or {}).items()
        if name in golden.scalars
    }
    return GoldenComparison(
        scenario=golden.provenance.scenario,
        model=golden.provenance.model,
        slices=tuple(comparisons),
        scalars=scalar_errors,
        scalar_tolerance=scalar_tolerance,
        missing=tuple(missing),
    )


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
