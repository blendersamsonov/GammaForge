"""Derivation-format guard (docs/derivations/README.md).

Checks every derivation file's filename, header block, and heading set against the
confidence-pipeline skeleton `docs/derivations/README.md` defines — stricter as a
derivation's status climbs from `derived` to `verified` — and cross-checks the tree
against `docs/derivations/INDEX.md`. Exact structural checks, not judgment calls, same
spirit as `test_decision_format.py`.

Two roots are checked: `docs/derivations/` (a real project's actual derivations — empty
in this template repo itself) and `examples/derivations/` (this repo's own worked
examples, which double as this test's fixture data).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

STATUSES = ("derived", "validated", "verified", "rejected", "archived")

FILENAME_RE = re.compile(r"^(DER\d{3,})-[a-z0-9-]+\.md$")
TITLE_LINE_RE = re.compile(r"^# (DER\d{3,}) — (.+)$")
STATUS_LINE_RE = re.compile(r"^Status: (derived|validated|verified|rejected — .+)$")
ARCHIVED_LINE_RE = re.compile(r"^Archived: (\d{4}-\d{2}-\d{2})$")
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Cumulative: verified requires everything validated requires, plus its own addition.
REQUIRED_BY_STATUS = {
    "derived": {"Setup"},
    "validated": {"Setup", "Result"},
    "verified": {"Setup", "Result", "Verification"},
    "rejected": {"Setup"},
}


@dataclass
class DerivationFile:
    path: Path
    status_folder: str
    id_: str
    text: str


def _discover(root: Path) -> list[DerivationFile]:
    found = []
    for status in STATUSES:
        folder = root / status
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            m = FILENAME_RE.match(path.name)
            if not m:
                continue
            found.append(
                DerivationFile(path=path, status_folder=status, id_=m.group(1), text=path.read_text())
            )
    return found


def _header_errors(df: DerivationFile) -> list[str]:
    errors = []
    lines = df.text.splitlines()

    title_match = TITLE_LINE_RE.match(lines[0]) if lines else None
    if not title_match:
        errors.append("first line must be '# DERNNN — <title>'")
    elif title_match.group(1) != df.id_:
        errors.append(f"title id {title_match.group(1)} != filename id {df.id_}")

    status_line = next((l for l in lines[:8] if l.startswith("Status:")), None)
    if status_line is None or not STATUS_LINE_RE.match(status_line):
        errors.append("missing or malformed 'Status:' line")
    else:
        status_value = STATUS_LINE_RE.match(status_line).group(1)
        if df.status_folder == "derived" and status_value != "derived":
            errors.append(f"file under derived/ but Status: {status_value}")
        if df.status_folder == "validated" and status_value != "validated":
            errors.append(f"file under validated/ but Status: {status_value}")
        if df.status_folder == "verified" and status_value != "verified":
            errors.append(f"file under verified/ but Status: {status_value}")
        if df.status_folder == "rejected" and not status_value.startswith("rejected — "):
            errors.append(f"file under rejected/ but Status: {status_value}")

    archived_line = next((l for l in lines[:8] if l.startswith("Archived:")), None)
    if df.status_folder == "archived" and (archived_line is None or not ARCHIVED_LINE_RE.match(archived_line)):
        errors.append("archived/ file missing a well-formed 'Archived: YYYY-MM-DD' line")
    if df.status_folder != "archived" and archived_line is not None:
        errors.append("non-archived file has an 'Archived:' line")

    return errors


def _effective_status(df: DerivationFile) -> str:
    """The status whose skeleton applies -- for archived/, whatever Status: says."""
    if df.status_folder != "archived":
        return df.status_folder
    for status in ("verified", "validated", "derived"):
        if f"Status: {status}" in df.text:
            return status
    return "rejected"


def _heading_errors(df: DerivationFile) -> list[str]:
    headings = set(HEADING_RE.findall(df.text))
    effective = _effective_status(df)
    required = REQUIRED_BY_STATUS[effective]
    missing = required - headings
    if missing:
        return [f"{effective} derivation missing required section(s): {sorted(missing)}"]
    return []


def _id_errors(derivations: list[DerivationFile]) -> list[str]:
    seen: dict[str, Path] = {}
    errors = []
    for df in derivations:
        if df.id_ in seen:
            errors.append(f"duplicate id {df.id_}: {seen[df.id_]} and {df.path}")
        else:
            seen[df.id_] = df.path
    return errors


def _index_errors(derivations: list[DerivationFile], index_path: Path) -> list[str]:
    if not index_path.is_file():
        return [f"{index_path} does not exist"]
    index_text = index_path.read_text()
    index_rows = {}
    for line in index_text.splitlines():
        m = re.match(r"^\|\s*(DER\d{3,})\s*\|.*\|\s*(\S+)\s*\|\s*(\S+)\s*\|$", line)
        if m:
            index_rows[m.group(1)] = {"status": m.group(2), "path": m.group(3)}

    errors = []
    on_disk_ids = {df.id_ for df in derivations}
    for df in derivations:
        row = index_rows.get(df.id_)
        if row is None:
            errors.append(f"{df.id_} exists on disk but has no INDEX.md row")
            continue
        rel_path = str(df.path.relative_to(index_path.parent))
        if row["path"] not in (rel_path, df.path.name):
            errors.append(f"{df.id_}: INDEX.md path '{row['path']}' doesn't match on-disk path '{rel_path}'")
    for id_ in index_rows:
        if id_ not in on_disk_ids:
            errors.append(f"{id_} has an INDEX.md row but no file on disk")
    return errors


def _check_tree(root: Path, index_path: Path | None) -> None:
    derivations = _discover(root)
    if not derivations:
        pytest.skip(f"no derivation files under {root}")

    failures = []
    for df in derivations:
        rel = df.path.relative_to(root)
        for err in _header_errors(df):
            failures.append(f"{rel}: {err}")
        for err in _heading_errors(df):
            failures.append(f"{rel}: {err}")
    for err in _id_errors(derivations):
        failures.append(err)
    if index_path is not None:
        for err in _index_errors(derivations, index_path):
            failures.append(f"INDEX.md: {err}")

    assert not failures, "Derivation format violations:\n" + "\n".join(failures)


def test_docs_derivations_tree_is_well_formed():
    _check_tree(REPO_ROOT / "docs" / "derivations", REPO_ROOT / "docs" / "derivations" / "INDEX.md")


def test_examples_derivations_tree_is_well_formed():
    _check_tree(REPO_ROOT / "examples" / "derivations", index_path=None)
