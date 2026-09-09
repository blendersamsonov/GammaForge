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
import os
from dataclasses import dataclass
from pathlib import Path

import gammaforge
import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

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

_IGNORED_DIR_NAMES = {
    ".git", ".claude", ".venv", "venv", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "__pycache__", "build", "dist", "graphify-out",
    ".agents", ".codex", ".openscience",
}


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


@dataclass
class CheckoutIndex:
    """In-memory index of checkout files and installed symbols built once per check run."""

    repo_root: Path
    repo_files: set[str]
    basenames: dict[str, list[str]]
    extensions: set[str]
    names: set[str]
    classes: dict[str, type]

    @classmethod
    def build(cls, repo_root: Path = REPO_ROOT) -> CheckoutIndex:
        repo_files: set[str] = set()
        basenames: dict[str, list[str]] = {}
        extensions: set[str] = set()

        for root_str, dirs, files in os.walk(repo_root, topdown=True):
            root_path = Path(root_str)
            # Exclude nested worktrees, dot-dirs, virtualenvs and generated graphs
            # before traversal, not only after expensive recursion.
            dirs[:] = [
                d for d in dirs
                if d not in _IGNORED_DIR_NAMES
                and not d.startswith(".")
                and not (root_path / d / ".git").exists()
            ]
            rel_root = root_path.relative_to(repo_root)
            for f in files:
                rel_path = (rel_root / f).as_posix() if str(rel_root) != "." else f
                repo_files.add(rel_path)
                basenames.setdefault(f, []).append(rel_path)
                suffix = Path(f).suffix.lower()
                if suffix:
                    extensions.add(suffix)

        if repo_root == REPO_ROOT:
            names, classes = _package_symbol_index()
        else:
            names, classes = set(), {}

        return cls(
            repo_root=repo_root,
            repo_files=repo_files,
            basenames=basenames,
            extensions=extensions,
            names=names,
            classes=classes,
        )

    def file_resolves(self, token: str) -> bool:
        if "/" in token:
            # Qualified path: must match a real path in the checkout.
            # Either exact relative path from repo_root, or suffix path (e.g. io/laser.py or xigma/schema.py).
            if token in self.repo_files:
                return True
            token_suffix = f"/{token}"
            return any(p.endswith(token_suffix) for p in self.repo_files)
        else:
            # Bare filename shorthand: must exist as a file in the checkout.
            return token in self.basenames


_DEFAULT_INDEX: CheckoutIndex | None = None


def get_default_index(repo_root: Path = REPO_ROOT) -> CheckoutIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None or _DEFAULT_INDEX.repo_root != repo_root:
        _DEFAULT_INDEX = CheckoutIndex.build(repo_root)
    return _DEFAULT_INDEX


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


def _classify_and_check(
    token: str,
    index: CheckoutIndex | None = None,
    names: set[str] | None = None,
    classes: dict[str, type] | None = None,
) -> bool | None:
    """True/False if the token was checkable; None if its shape was skipped."""
    if index is None:
        index = get_default_index()
    if names is None:
        names = index.names
    if classes is None:
        classes = index.classes

    token = token.strip()
    if token.endswith("()"):
        token = token[:-2]
    if not token:
        return None

    if token.startswith("."):
        # A bare extension (`.ele`, `.h5`) names a format, not a file in this repo.
        return None

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

    if FILE_LIKE_RE.match(token) and Path(token).suffix.lower() in index.extensions:
        applicable = True
        if index.file_resolves(token):
            return True

    if BARE_NAME_RE.match(token):
        applicable = True
        if token in names or hasattr(builtins, token):
            return True

    return False if applicable else None


def test_decisions_doc_backticks_resolve():
    index = get_default_index()
    failures = []
    for doc in _checked_docs():
        text = doc.read_text()
        for token in BACKTICK_RE.findall(text):
            if _classify_and_check(token, index) is False:
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
    "laser.py",               # bare filename shorthand
    "src/gammaforge/io/laser.py", # exact relative path
    "io/laser.py",            # suffix relative path
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
    "src/does/not/exist/laser.py", # qualified path with nonexistent prefix
    "DECISIONS.md",               # historical root file that does not exist in checkout
]
SKIPPED = [
    ".ele",             # a bare format extension, not a file in this repo
    "a0 profile",       # prose
    "1/n",              # a formula
    "np.cov",           # third-party dotted name -- not this package's to validate
    "dataclasses.replace",
]


def test_guard_resolves_current_references():
    index = get_default_index()
    unresolved = [token for token in RESOLVES if _classify_and_check(token, index) is not True]
    assert not unresolved, f"guard fails to resolve current references: {unresolved}"


def test_guard_still_detects_stale_references():
    index = get_default_index()
    missed = [token for token in STALE if _classify_and_check(token, index) is not False]
    assert not missed, f"guard no longer detects stale references: {missed}"


def test_guard_skips_shapes_it_cannot_judge():
    index = get_default_index()
    judged = [token for token in SKIPPED if _classify_and_check(token, index) is not None]
    assert not judged, f"guard guessed at tokens it should skip: {judged}"


def test_nested_worktree_and_generated_files_do_not_satisfy_missing_paths(tmp_path):
    """A10: nested worktrees, virtualenvs and generated files must not satisfy missing paths."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "real_file.py").write_text("# real")

    # Nested worktree (e.g. .claude/worktrees/...)
    (tmp_path / ".claude" / "worktrees" / "nested").mkdir(parents=True)
    (tmp_path / ".claude" / "worktrees" / "nested" / "DECISIONS.md").write_text("# fake")
    (tmp_path / ".claude" / "worktrees" / "nested" / "fake_file.py").write_text("# fake")

    # Generated graph directory (e.g. graphify-out/)
    (tmp_path / "graphify-out").mkdir(parents=True)
    (tmp_path / "graphify-out" / "graph.json").write_text("{}")

    # Virtualenv (e.g. .venv/)
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "venv_file.py").write_text("# venv")

    index = CheckoutIndex.build(tmp_path)

    # Legitimate references in current checkout resolve:
    assert index.file_resolves("real_file.py")
    assert index.file_resolves("src/pkg/real_file.py")
    assert index.file_resolves("pkg/real_file.py")

    # Missing files in current checkout DO NOT resolve even if present in nested worktrees/generated dirs:
    assert not index.file_resolves("DECISIONS.md")
    assert not index.file_resolves("fake_file.py")
    assert not index.file_resolves("src/does/not/exist/real_file.py")
    assert not index.file_resolves("graph.json")
    assert not index.file_resolves("venv_file.py")


def test_checkout_indexing_occurs_once(monkeypatch):
    """A10: indexing occurs once per check run rather than repeating on every token."""
    import sys
    m = sys.modules[__name__]
    monkeypatch.setattr(m, "_DEFAULT_INDEX", None)
    build_calls = []
    original_build = m.CheckoutIndex.build

    def spy_build(repo_root=REPO_ROOT):
        build_calls.append(repo_root)
        return original_build(repo_root)

    monkeypatch.setattr(m.CheckoutIndex, "build", spy_build)

    idx1 = m.get_default_index()
    idx2 = m.get_default_index()
    assert idx1 is idx2
    assert len(build_calls) == 1

