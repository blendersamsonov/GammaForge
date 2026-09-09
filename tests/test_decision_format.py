"""Decision-format guard (docs/decisions/README.md).

Checks every decision file's filename, header block, and heading set against the
lifecycle/class skeleton `docs/decisions/README.md` defines, and cross-checks the tree
against `docs/decisions/INDEX.md`. These are exact structural checks, not judgment calls
-- unlike a doc-staleness guard that has to decide whether a prose reference "counts,"
every rule here is mechanical, so this doesn't need a heuristic's self-testing scaffolding.

Two roots are checked: `docs/decisions/` (a real project's actual decisions -- empty in
this template repo itself) and `examples/decisions/` (this repo's own worked examples,
which double as this test's fixture data). A project that copies this test but not
`examples/` will simply have nothing there to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

REPO_ROOT = Path(__file__).resolve().parents[1]

LIFECYCLES = ("proposed", "implemented", "rejected", "archived")
CLASSES = ("feature", "bug-fix", "simplification", "architecture", "process", "testing")

FILENAME_RE = re.compile(r"^(RES\d{3,})-[a-z0-9-]+\.md$")
TITLE_LINE_RE = re.compile(r"^# (RES\d{3,}) — (.+)$")
STATUS_LINE_RE = re.compile(r"^Status: (proposed|implemented|rejected — .+)$")
CLASS_LINE_RE = re.compile(r"^Class: ([a-z-]+)$")
ARCHIVED_LINE_RE = re.compile(r"^Archived: (\d{4}-\d{2}-\d{2})$")
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Sections that only make sense once something has actually shipped.
IMPLEMENTED_ONLY = {"Decision", "Consequences", "Rationale"}
# Sections that only make sense while something is still a proposal.
PROPOSAL_ONLY = {"Proposal", "Acceptance criteria", "Risks"}
REQUIRED_ALWAYS = {"Problem", "Alternatives considered"}


@dataclass
class DecisionFile:
    path: Path
    lifecycle: str
    class_: str
    id_: str
    text: str


def _discover(root: Path) -> tuple[list[DecisionFile], list[str]]:
    found = []
    errors = []
    for path in sorted(root.rglob("*.md")):
        if path.name in ("README.md", "INDEX.md"):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) != 3 or rel.parts[0] not in LIFECYCLES or rel.parts[1] not in CLASSES:
            errors.append(f"{rel}: file not in valid '<lifecycle>/<class>/' directory")
            continue
        m = FILENAME_RE.match(path.name)
        if not m:
            errors.append(f"{rel}: malformed filename (expected 'RES<NNN>-<slug>.md')")
            continue
        found.append(
            DecisionFile(
                path=path,
                lifecycle=rel.parts[0],
                class_=rel.parts[1],
                id_=m.group(1),
                text=path.read_text(),
            )
        )
    return found, errors


def _header_errors(df: DecisionFile) -> list[str]:
    errors = []
    lines = df.text.splitlines()

    title_match = TITLE_LINE_RE.match(lines[0]) if lines else None
    if not title_match:
        errors.append("first line must be '# RESNNN — <title>'")
    elif title_match.group(1) != df.id_:
        errors.append(f"title id {title_match.group(1)} != filename id {df.id_}")

    status_line = next((l for l in lines[:8] if l.startswith("Status:")), None)
    if status_line is None or not STATUS_LINE_RE.match(status_line):
        errors.append("missing or malformed 'Status:' line")
    else:
        status_value = STATUS_LINE_RE.match(status_line).group(1)
        if df.lifecycle == "proposed" and status_value != "proposed":
            errors.append(f"file under proposed/ but Status: {status_value}")
        if df.lifecycle == "implemented" and status_value != "implemented":
            errors.append(f"file under implemented/ but Status: {status_value}")
        if df.lifecycle == "rejected" and not status_value.startswith("rejected — "):
            errors.append(f"file under rejected/ but Status: {status_value}")
        if df.lifecycle == "archived" and status_value == "proposed":
            errors.append("archived/ file has Status: proposed -- only implemented or rejected decisions are archived")

    class_line = next((l for l in lines[:8] if l.startswith("Class:")), None)
    if class_line is None or not CLASS_LINE_RE.match(class_line):
        errors.append("missing or malformed 'Class:' line")
    elif CLASS_LINE_RE.match(class_line).group(1) != df.class_:
        errors.append(f"Class: line ({CLASS_LINE_RE.match(class_line).group(1)}) != folder ({df.class_})")

    archived_line = next((l for l in lines[:8] if l.startswith("Archived:")), None)
    if df.lifecycle == "archived" and (archived_line is None or not ARCHIVED_LINE_RE.match(archived_line)):
        errors.append("archived/ file missing a well-formed 'Archived: YYYY-MM-DD' line")
    if df.lifecycle != "archived" and archived_line is not None:
        errors.append("non-archived file has an 'Archived:' line")

    return errors


def _effective_lifecycle(df: DecisionFile) -> str:
    """The lifecycle whose heading rules apply -- for archived/, that's whatever the
    Status: line says it was before archiving, not the literal folder name."""
    if df.lifecycle != "archived":
        return df.lifecycle
    if "Status: implemented" in df.text:
        return "implemented"
    return "rejected"


def _heading_errors(df: DecisionFile) -> list[str]:
    errors = []
    headings = set(HEADING_RE.findall(df.text))

    missing_required = REQUIRED_ALWAYS - headings
    if missing_required:
        errors.append(f"missing required section(s): {sorted(missing_required)}")

    effective = _effective_lifecycle(df)
    if effective == "implemented":
        stray = headings & PROPOSAL_ONLY
        if stray:
            errors.append(f"implemented decision has proposal-era section(s): {sorted(stray)}")
        if "Decision" not in headings:
            errors.append("implemented decision missing '## Decision'")
    else:  # proposed or rejected
        stray = headings & IMPLEMENTED_ONLY
        if stray:
            errors.append(f"{effective} decision has implemented-only section(s): {sorted(stray)}")
        if "Proposal" not in headings:
            errors.append(f"{effective} decision missing '## Proposal'")

    return errors


def _id_errors(decisions: list[DecisionFile]) -> list[str]:
    seen: dict[str, Path] = {}
    errors = []
    for df in decisions:
        if df.id_ in seen:
            errors.append(f"duplicate id {df.id_}: {seen[df.id_]} and {df.path}")
        else:
            seen[df.id_] = df.path
    return errors


def _index_errors(decisions: list[DecisionFile], index_path: Path) -> list[str]:
    if not index_path.is_file():
        return [f"{index_path} does not exist"]
    index_text = index_path.read_text()
    index_rows = {}
    for line in index_text.splitlines():
        m = re.match(r"^\|\s*(RES\d{3,})\s*\|.*\|\s*([a-z-]+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|$", line)
        if m:
            index_rows[m.group(1)] = {"class": m.group(2), "status": m.group(3), "path": m.group(4)}

    errors = []
    on_disk_ids = {df.id_ for df in decisions}
    for df in decisions:
        row = index_rows.get(df.id_)
        if row is None:
            errors.append(f"{df.id_} exists on disk but has no INDEX.md row")
            continue
        if row["class"] != df.class_:
            errors.append(f"{df.id_}: INDEX.md class '{row['class']}' != on-disk class '{df.class_}'")
        rel_path = str(df.path.relative_to(index_path.parent))
        if row["path"] not in (rel_path, df.path.name):
            errors.append(f"{df.id_}: INDEX.md path '{row['path']}' doesn't match on-disk path '{rel_path}'")

        # Status cross-check
        row_status = row["status"]
        if df.lifecycle == "archived":
            eff = _effective_lifecycle(df)
            if not (row_status == eff or (eff == "rejected" and row_status.startswith("rejected"))):
                errors.append(f"{df.id_}: INDEX.md status '{row_status}' != archived effective status '{eff}'")
            if not row["path"].startswith("archived/"):
                errors.append(f"{df.id_}: INDEX.md path '{row['path']}' for archived decision must start with 'archived/'")
        else:
            if not (row_status == df.lifecycle or (df.lifecycle == "rejected" and row_status.startswith("rejected"))):
                errors.append(f"{df.id_}: INDEX.md status '{row_status}' != on-disk lifecycle '{df.lifecycle}'")

    for id_ in index_rows:
        if id_ not in on_disk_ids:
            errors.append(f"{id_} has an INDEX.md row but no file on disk")
    return errors


def _check_tree(root: Path, index_path: Path | None) -> None:
    decisions, disc_errors = _discover(root)
    if not decisions and not disc_errors:
        pytest.skip(f"no decision files under {root}")

    failures = list(disc_errors)
    for df in decisions:
        rel = df.path.relative_to(root)
        for err in _header_errors(df):
            failures.append(f"{rel}: {err}")
        for err in _heading_errors(df):
            failures.append(f"{rel}: {err}")
    for err in _id_errors(decisions):
        failures.append(err)
    if index_path is not None:
        for err in _index_errors(decisions, index_path):
            failures.append(f"INDEX.md: {err}")

    assert not failures, "Decision format violations:\n" + "\n".join(failures)


def test_docs_decisions_tree_is_well_formed():
    _check_tree(REPO_ROOT / "docs" / "decisions", REPO_ROOT / "docs" / "decisions" / "INDEX.md")


# ---------------------------------------------------------------------------
# Structural guard tests (A21)
# ---------------------------------------------------------------------------

def _write_valid_decision(path: Path, id_: str, title: str, lifecycle: str, class_: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {id_} — {title}",
        "",
        f"Status: {status}",
        f"Class: {class_}",
    ]
    if lifecycle == "archived":
        body.append("Archived: 2026-09-06")
    body.extend([
        "",
        "## Problem",
        "Problem text.",
        "",
        "## Alternatives considered",
        "Alternative text.",
    ])
    if lifecycle == "proposed":
        body.extend([
            "",
            "## Proposal",
            "Proposal text.",
        ])
    else:
        body.extend([
            "",
            "## Decision",
            "Decision text.",
            "",
            "## Rationale",
            "Rationale text.",
            "",
            "## Consequences",
            "Consequences text.",
        ])
    path.write_text("\n".join(body) + "\n")


def test_decision_format_flags_malformed_filename(tmp_path):
    """A21: files with malformed filenames under decisions tree must fail rather than skip."""
    bad_file = tmp_path / "implemented" / "architecture" / "bad_name.md"
    _write_valid_decision(bad_file, "RES999", "Bad filename", "implemented", "architecture", "implemented")
    with pytest.raises(AssertionError, match="malformed filename"):
        _check_tree(tmp_path, None)


def test_decision_format_flags_unexpected_directory(tmp_path):
    """A21: files in non-lifecycle or non-class directories must fail."""
    bad_dir_file = tmp_path / "somewhere_else" / "RES999-test.md"
    _write_valid_decision(bad_dir_file, "RES999", "Bad dir", "implemented", "architecture", "implemented")
    with pytest.raises(AssertionError, match="not in valid '<lifecycle>/<class>/' directory"):
        _check_tree(tmp_path, None)


def test_decision_format_flags_duplicate_id(tmp_path):
    """A21: duplicate decision ids must be flagged."""
    f1 = tmp_path / "implemented" / "architecture" / "RES999-first.md"
    f2 = tmp_path / "proposed" / "feature" / "RES999-second.md"
    _write_valid_decision(f1, "RES999", "First", "implemented", "architecture", "implemented")
    _write_valid_decision(f2, "RES999", "Second", "proposed", "feature", "proposed")
    with pytest.raises(AssertionError, match="duplicate id RES999"):
        _check_tree(tmp_path, None)


def test_decision_format_flags_inconsistent_index_status(tmp_path):
    """A21: INDEX.md status inconsistent with disk must fail."""
    f = tmp_path / "implemented" / "architecture" / "RES999-test.md"
    _write_valid_decision(f, "RES999", "Test", "implemented", "architecture", "implemented")
    index = tmp_path / "INDEX.md"
    # Row declares proposed status when disk is implemented
    index.write_text(
        "| id | title | class | status | path |\n"
        "|----|-------|-------|--------|------|\n"
        "| RES999 | Test | architecture | proposed | implemented/architecture/RES999-test.md |\n"
    )
    with pytest.raises(AssertionError, match="INDEX.md status 'proposed' != on-disk lifecycle 'implemented'"):
        _check_tree(tmp_path, index)


def test_decision_format_accepts_archived_decision_with_historical_status(tmp_path):
    """A21: archived decisions retain historical status and must pass when index matches."""
    f = tmp_path / "archived" / "architecture" / "RES999-old.md"
    _write_valid_decision(f, "RES999", "Old", "archived", "architecture", "implemented")
    index = tmp_path / "INDEX.md"
    index.write_text(
        "| id | title | class | status | path |\n"
        "|----|-------|-------|--------|------|\n"
        "| RES999 | Old | architecture | implemented | archived/architecture/RES999-old.md |\n"
    )
    _check_tree(tmp_path, index)

