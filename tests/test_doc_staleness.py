"""Doc-staleness guard (GRAND_PLAN.md §12, C2).

Checks that backticked tokens in the "current-state" docs actually resolve — either as
a real file somewhere in the repo, or as a real symbol in the installed `gammaforge`
package (or a Python builtin). Scope is every decision file under
`docs/decisions/{proposed,implemented,rejected}/` and every derivation file under
`docs/derivations/{derived,validated,verified,rejected}/` — not either tree's
`archived/`, which describes code that's since moved or gone, and not either tree's
`README.md`/`INDEX.md`, which are navigation/meta prose in the same category as
`GRAND_PLAN.md`/`PROGRESS.md`. See `docs/decisions/` for why an after-the-fact decision
log is the one place "every backtick resolves" is true by construction (originally RES002;
see its entry for the superseding id) — derivations extend the same reasoning: `derived`/
`validated`/`verified` all describe a claim about real, present code or a real, present
formula, not a future promise (RES057).

This is a heuristic, not a full parser: a token that doesn't clearly look like a file
path or a Python identifier is skipped rather than guessed at — a false positive here
(flagging something that's actually fine) is worse than a missed check, since it trains
people to ignore the guard.
"""

from __future__ import annotations

import builtins
import dataclasses
import enum
import importlib
import pkgutil
import re
from pathlib import Path

import gammaforge

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DECISION_LIFECYCLES = ("proposed", "implemented", "rejected")
CHECKED_DERIVATION_STATUSES = ("derived", "validated", "verified", "rejected")


def _checked_docs() -> list[Path]:
    decisions_root = REPO_ROOT / "docs" / "decisions"
    derivations_root = REPO_ROOT / "docs" / "derivations"
    docs = []
    for lifecycle in CHECKED_DECISION_LIFECYCLES:
        docs.extend(sorted((decisions_root / lifecycle).rglob("*.md")))
    for status in CHECKED_DERIVATION_STATUSES:
        docs.extend(sorted((derivations_root / status).rglob("*.md")))
    return docs

BACKTICK_RE = re.compile(r"`([^`\n]+)`")
FILE_LIKE_RE = re.compile(r"^[\w./-]*\.[A-Za-z0-9]{1,15}$")
GAMMAFORGE_DOTTED_RE = re.compile(r"^gammaforge(\.[A-Za-z_][A-Za-z0-9_]*)+$")
CLASS_ATTR_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*\.[a-zA-Z_][A-Za-z0-9_]*$")
BARE_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

_IGNORED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv"}


def _iter_repo_files():
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and not _IGNORED_DIR_NAMES & set(path.parts):
            yield path


def _file_exists_by_basename(basename: str) -> bool:
    return any(p.name == basename for p in _iter_repo_files())


def _repo_extensions() -> set[str]:
    """Suffixes that actually occur in this repo.

    Derived rather than hardcoded, so the file-like rule stays self-maintaining. It exists
    to stop `np.cov` being read as "a file named cov": a dotted token only counts as a path
    when its suffix is one this repo really uses.
    """
    return {p.suffix.lower() for p in _iter_repo_files() if p.suffix}


def _iter_gammaforge_modules():
    yield gammaforge
    for info in pkgutil.walk_packages(gammaforge.__path__, prefix="gammaforge."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:
            # Optional-dependency modules (cupy/numba) may not be installed here;
            # validating references into them isn't this guard's job.
            continue


def _package_symbol_index():
    names: set[str] = set()
    classes: dict[str, type] = {}
    for module in _iter_gammaforge_modules():
        for attr_name, value in vars(module).items():
            if attr_name.startswith("_"):
                continue
            names.add(attr_name)
            if isinstance(value, type):
                classes[attr_name] = value
                # Class-level names -- enum members (`WIDTH`), declared tables (`UNITS`,
                # `LIGHT_TIME_FIELDS`) -- are not module-level, so a doc naming one bare
                # would read as stale however current it is.
                names.update(n for n in vars(value) if not n.startswith("_"))
                if issubclass(value, enum.Enum):
                    names.update(member.name for member in value)
    return names, classes


def _has_member(cls: type, attr: str) -> bool:
    """Whether ``attr`` is a real member of ``cls``, including annotation-only fields.

    ``hasattr`` alone is not enough: a dataclass field declared without a default (most of
    them) exists only as an annotation, so `Bunch.weight` would look stale while being
    perfectly current.
    """
    if hasattr(cls, attr):
        return True
    if dataclasses.is_dataclass(cls) and any(f.name == attr for f in dataclasses.fields(cls)):
        return True
    return any(attr in getattr(base, "__annotations__", {}) for base in cls.__mro__)


def _classify_and_check(token: str, names: set[str], classes: dict[str, type]) -> bool | None:
    """True/False if the token was checkable; None if its shape was skipped."""
    token = token.strip()
    if token.endswith("()"):
        token = token[:-2]
    if not token:
        return None

    if token.startswith("."):
        # A bare extension (`.ele`, `.h5`) names a format, not a file in this repo.
        return None

    # Several shapes are genuinely ambiguous — `FieldKind.CHOICE` and `GRAND_PLAN.md` both
    # read as "a name, a dot, a suffix". So every applicable interpretation is tried and
    # *any* of them resolving is enough; a token is only reported stale when at least one
    # interpretation applied and none of them found anything. Guessing an order instead
    # produces false positives, which train people to ignore the guard.
    applicable = False
    if GAMMAFORGE_DOTTED_RE.match(token):
        applicable = True
        obj = gammaforge
        for part in token.split(".")[1:]:
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            return True

    if CLASS_ATTR_RE.match(token):
        applicable = True
        cls_name, attr = token.split(".", 1)
        cls = classes.get(cls_name)
        if cls is not None and _has_member(cls, attr):
            return True

    if FILE_LIKE_RE.match(token) and Path(token).suffix.lower() in _repo_extensions():
        applicable = True
        if _file_exists_by_basename(Path(token).name):
            return True

    if BARE_NAME_RE.match(token):
        applicable = True
        if token in names or hasattr(builtins, token):
            return True

    return False if applicable else None


def test_decisions_doc_backticks_resolve():
    names, classes = _package_symbol_index()
    failures = []
    for doc in _checked_docs():
        text = doc.read_text()
        for token in BACKTICK_RE.findall(text):
            if _classify_and_check(token, names, classes) is False:
                failures.append(f"{doc.relative_to(REPO_ROOT)}: `{token}`")
    assert not failures, "Stale doc references:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# The guard's own behaviour
# ---------------------------------------------------------------------------
# A doc-staleness guard fails silently when it gets *weaker*: broadening a shape rule to
# stop a false positive can easily stop it detecting anything at all, and nothing would
# say so. These cases pin both directions -- what must resolve, and what must still be
# caught. The "stale" column is deliberately made of names this project rejected on
# purpose (`GRAND_PLAN.md` P8/P10, §3.2), so it doubles as a check that they stay gone.
RESOLVES = [
    "FieldKind.CHOICE",       # enum member via its class
    "Axis.THETA_Y",
    "WIDTH",                  # enum member named bare
    "Bunch.weight",           # annotation-only dataclass field
    "Axis.unit",              # property
    "GaussianParaxialLaser",  # module-level class
    "gammaforge.io.bunch",     # dotted module path
    "GRAND_PLAN.md",          # repo file whose shape also looks like Class.attr
    "laser.py",
    "UNITS",                    # class-level declared table
    "LIGHT_TIME_FIELDS",
    "Bunch.get",                # method
]
STALE = [
    "FieldKind.TENSOR",
    "Axis.MOMENTUM",
    "Bunch.n_electrons",   # removed in the rebuild (§3.2)
    "BeamFittedParams",    # rejected (P8)
    "ModelCapabilities",   # rejected (P10)
    "NoConvention",        # rejected (§2.1)
    "gammaforge.io.nowhere",
    "nonexistent_module.py",
]
SKIPPED = [
    ".ele",             # a bare format extension, not a file in this repo
    "a0 profile",       # prose
    "1/n",              # a formula
    "np.cov",           # third-party dotted name -- not this package's to validate
    "dataclasses.replace",
]


def test_guard_resolves_current_references():
    names, classes = _package_symbol_index()
    unresolved = [token for token in RESOLVES if _classify_and_check(token, names, classes) is not True]
    assert not unresolved, f"guard fails to resolve current references: {unresolved}"


def test_guard_still_detects_stale_references():
    names, classes = _package_symbol_index()
    missed = [token for token in STALE if _classify_and_check(token, names, classes) is not False]
    assert not missed, f"guard no longer detects stale references: {missed}"


def test_guard_skips_shapes_it_cannot_judge():
    names, classes = _package_symbol_index()
    judged = [token for token in SKIPPED if _classify_and_check(token, names, classes) is not None]
    assert not judged, f"guard guessed at tokens it should skip: {judged}"
