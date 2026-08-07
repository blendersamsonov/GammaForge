"""Doc-staleness guard (GRAND_PLAN.md §12, C2).

Checks that backticked tokens in the "current-state" docs actually resolve — either as
a real file somewhere in the repo, or as a real symbol in the installed `gammaforge`
package (or a Python builtin). Scope is deliberately limited to `DECISIONS.md` for now;
see that file's own D002 entry for why `GRAND_PLAN.md` and `PROGRESS.md` are excluded.

This is a heuristic, not a full parser: a token that doesn't clearly look like a file
path or a Python identifier is skipped rather than guessed at — a false positive here
(flagging something that's actually fine) is worse than a missed check, since it trains
people to ignore the guard.
"""

from __future__ import annotations

import builtins
import importlib
import pkgutil
import re
from pathlib import Path

import gammaforge

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DOCS = [REPO_ROOT / "DECISIONS.md"]

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
    return names, classes


def _classify_and_check(token: str, names: set[str], classes: dict[str, type]) -> bool | None:
    """True/False if the token was checkable; None if its shape was skipped."""
    token = token.strip()
    if token.endswith("()"):
        token = token[:-2]
    if not token:
        return None

    if FILE_LIKE_RE.match(token):
        return _file_exists_by_basename(Path(token).name)

    if GAMMAFORGE_DOTTED_RE.match(token):
        obj = gammaforge
        for part in token.split(".")[1:]:
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)
        return True

    if CLASS_ATTR_RE.match(token):
        cls_name, attr = token.split(".", 1)
        cls = classes.get(cls_name)
        return cls is not None and hasattr(cls, attr)

    if BARE_NAME_RE.match(token):
        return token in names or hasattr(builtins, token)

    return None


def test_decisions_doc_backticks_resolve():
    names, classes = _package_symbol_index()
    failures = []
    for doc in CHECKED_DOCS:
        text = doc.read_text()
        for token in BACKTICK_RE.findall(text):
            if _classify_and_check(token, names, classes) is False:
                failures.append(f"{doc.relative_to(REPO_ROOT)}: `{token}`")
    assert not failures, "Stale doc references:\n" + "\n".join(failures)
