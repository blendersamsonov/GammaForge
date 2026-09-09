"""Pytest configuration and tiered test execution controls."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run heavy Tier 3 tests (Monte Carlo validations, 16x refinement, SymPy verifications)",
    )
    parser.addoption(
        "--tier",
        action="store",
        default=None,
        choices=["tier0", "tier1", "tier2", "tier3", "fast", "heavy", "all"],
        help="Filter tests by execution tier: tier0, tier1, tier2, tier3, fast, heavy, all",
    )


def _get_item_tier(item: pytest.Item) -> str | None:
    """Return the item's tier, giving function-level markers precedence over module markers."""
    for marker in item.own_markers:
        if marker.name in ("tier0", "tier1", "tier2", "tier3"):
            return marker.name
    for marker in item.iter_markers():
        if marker.name in ("tier0", "tier1", "tier2", "tier3"):
            return marker.name
    return None


def _is_heavy(item: pytest.Item) -> bool:
    if item.get_closest_marker("heavy"):
        return True
    return _get_item_tier(item) == "tier3"


def _matches_tier(item: pytest.Item, tier: str) -> bool:
    if tier == "all":
        return True
    if tier == "fast":
        return (not _is_heavy(item)) and (
            _get_item_tier(item) in ("tier0", "tier1") or bool(item.get_closest_marker("fast"))
        )
    if tier == "heavy":
        return _is_heavy(item)
    return _get_item_tier(item) == tier


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tier = config.getoption("--tier")
    run_heavy = config.getoption("--run-heavy")
    markexpr = config.getoption("-m", default="")

    if tier:
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            if _matches_tier(item, tier):
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
        return

    # If the user explicitly requested a marker expression involving heavy, tier3, or symbolic,
    # let standard pytest marker filtering handle it.
    if any(m in markexpr for m in ("heavy", "tier3", "symbolic")):
        return

    # By default, unless --run-heavy is passed, deselect heavy / tier3 tests so everyday
    # pytest invocations complete quickly instead of 11+ minutes.
    # If the user explicitly targeted heavy tests/files directly (i.e. every collected test
    # is heavy), do not deselect them so targeted execution works seamlessly.
    if not run_heavy:
        selected = [item for item in items if not _is_heavy(item)]
        deselected = [item for item in items if _is_heavy(item)]
        if selected:
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)
