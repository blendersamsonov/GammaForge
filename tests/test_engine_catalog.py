"""The engine catalog is the one public engine boundary (RES095).

These are architectural-contract checks, not a re-run of the deleted Kascade coverage: each
one guards a boundary a frontend could silently break — enumeration, role visibility, the
runner seam, and the failure modes for names the catalog does not have.
"""

from dataclasses import replace

import numpy as np
import pytest
import yaml

pytestmark = [pytest.mark.tier0, pytest.mark.fast]

from gammaforge.engines.base import Engine
from gammaforge.engines.catalog import (
    ENGINES,
    EngineRole,
    engine_names,
    engine_schemas,
    get_engine,
    selectable_engines,
)
from gammaforge.engines.runner import LocalRunner
from gammaforge.gui.defaults import DEFAULTS_VERSION, GuiDefaultsStore
from gammaforge.gui.state import InputState
from gammaforge.io.calculation import CalculationRequest
from gammaforge.io.formats.calculation import request_from_dict, request_to_dict
from gammaforge.io.interaction import SamplingSpec
from gammaforge.io.results import PhasespaceSlice, Results
from gammaforge.io.schema import Parameters
from gammaforge.io.target import OutputKind, OutputRequest
from gammaforge.validation import scenarios

#: An engine that is no longer shipped. Any name the catalog cannot resolve must fail
#: loudly rather than being silently ignored or, worse, half-applied.
RETIRED = "kascade"

#: The classification this change settled (RES095). Asserting the intended mapping rather
#: than only the catalog's self-consistency is what makes a role regression fail instead of
#: quietly redefining itself: promoting the validation reference to a user-facing choice
#: would otherwise still satisfy a test derived from the same table it mutated.
INTENDED_ROLES = {
    "xigma": EngineRole.CALCULATION,
    "analytical": EngineRole.ESTIMATE,
    "delta": EngineRole.REFERENCE,
}


def _zero_yield() -> PhasespaceSlice:
    return PhasespaceSlice(axes={}, distr=np.array(0.0))


def _fast_request(engine_params=None) -> CalculationRequest:
    scenario = scenarios.BASELINE
    return CalculationRequest(
        beam=scenario.beam,
        laser=scenario.laser,
        target=replace(scenario.target, outputs=(OutputRequest(OutputKind.TOTAL_YIELD),)),
        sampling=SamplingSpec(n_particles=64, seed=0, prefilter=0.0),
        engine_params=engine_params or {},
    )


def test_catalog_resolves_every_engine_by_stable_name_and_satisfies_the_protocol():
    """One enumeration, and every entry in it is a real `Engine` keyed by its own name."""
    assert tuple(ENGINES) == engine_names()

    for name, spec in ENGINES.items():
        engine = get_engine(name)
        assert engine.name == name == spec.name
        assert isinstance(engine, Engine)
        assert isinstance(engine.schema, Parameters)
        assert engine.supported_outputs and engine.recompute_costs

    # A caller that hands the mapping around cannot mutate the catalog through it.
    with pytest.raises(TypeError):
        ENGINES["injected"] = ENGINES["xigma"]  # type: ignore[index]


def test_roles_match_the_intended_classification():
    """xigma is the calculation, analytical the overlay, delta a validation-only reference."""
    assert {name: spec.role for name, spec in ENGINES.items()} == INTENDED_ROLES


def test_role_is_declarative_data_and_keeps_reference_engines_out_of_the_gui():
    """A frontend filters by role; it never needs to know which name a role refers to."""
    selectable = selectable_engines()
    assert set(selectable) == {
        name for name, role in INTENDED_ROLES.items() if role is EngineRole.CALCULATION
    }
    assert "xigma" in selectable, "the intended user-facing calculation engine must be offered"

    for name, role in INTENDED_ROLES.items():
        if role is EngineRole.CALCULATION:
            continue
        assert name not in selectable, f"{name} ({role.name}) must not be an engine choice"
        # ...but it is still resolvable by name, so validation and scripts can use it.
        assert get_engine(name).name == name

    # Schemas cover every entry, so a request recorded against a reference engine replays.
    assert set(engine_schemas()) == set(ENGINES)


def test_local_runner_defaults_to_the_catalog_and_still_accepts_an_injected_mapping():
    assert set(LocalRunner().engines) == set(selectable_engines())

    class _Stub:
        name = "stub"
        schema = get_engine("xigma").schema
        supported_outputs = (OutputKind.TOTAL_YIELD,)
        recompute_costs = {}

        def run(self, interaction, params):
            return Results(photon_slices={OutputKind.TOTAL_YIELD: _zero_yield()})

    runner = LocalRunner(engines={"stub": _Stub()})
    assert set(runner.engines) == {"stub"}
    results = runner.calculate(_fast_request({"stub": _Stub.schema}))
    assert set(results) == {"analytical", "stub"}, "the estimate overlay stays present"
    assert runner.errors == {}


def test_script_facing_flow_selects_and_runs_an_engine_without_importing_its_module():
    """The whole public path: enumerate, take a schema, build a request, calculate.

    Every step here goes through the catalog or the runner. Importing an engine
    implementation module is not needed to do any of it, which is the point of the boundary.
    """
    name = engine_names((EngineRole.CALCULATION,))[0]
    schema = engine_schemas()[name].with_values(
        backend="numpy",
        n_steps=32,
        n_bins_gamma=8,
        n_bins_theta_x=8,
        n_bins_theta_y=8,
        n_bins_a0_shape=8,
        n_bins_ahat=4,
    )

    results = LocalRunner().calculate(_fast_request({name: schema}))

    assert results[name].photon_slices[OutputKind.TOTAL_YIELD].integrate() > 0.0
    assert "analytical" in results, "RES058 estimate/overlay behavior is unchanged"


def test_unknown_engine_names_fail_clearly_at_every_entry_point():
    with pytest.raises(ValueError, match=RETIRED):
        get_engine(RETIRED)

    request = _fast_request({RETIRED: get_engine("xigma").schema})
    with pytest.raises(ValueError, match=RETIRED):
        LocalRunner().calculate(request)
    with pytest.raises(ValueError, match="unknown calculation engine"):
        LocalRunner().calculate_one(RETIRED, _fast_request())

    # Serialization resolves schemas through the catalog, so a recorded retired engine is
    # still an explicit error rather than a silently dropped parameter block.
    document = request_to_dict(_fast_request({RETIRED: get_engine("xigma").schema}))
    assert RETIRED in document["engine_params"]
    with pytest.raises(ValueError, match="missing engine schemas"):
        request_from_dict(document)


def test_request_round_trips_without_the_caller_supplying_any_schema():
    """`load_request`-style restoration needs no engine knowledge from the caller."""
    request = _fast_request({"xigma": get_engine("xigma").schema.with_values(n_steps=32)})
    assert request_from_dict(request_to_dict(request)) == request


def test_gui_state_derives_forms_and_output_gating_from_the_catalog():
    state = InputState(selectable_engines())

    # One parameter group per catalogued choice, and no group for a retired engine.
    for name in selectable_engines():
        assert f"engine:{name}" in state.groups
    assert f"engine:{RETIRED}" not in state.groups

    selected = state.selected_engine
    assert selected in selectable_engines()
    for kind in OutputKind:
        assert state.supports(kind) is (kind in selectable_engines()[selected].supported_outputs)

    # An unknown name is rejected, including one the catalog used to have; clearing the
    # selection outright stays legal, and reselecting a real engine restores output gating.
    assert state.set_selected_engine(RETIRED) is False
    assert state.selected_engine == selected
    assert state.set_selected_engine(None) is True
    assert state.supports(OutputKind.TOTAL_YIELD) is False
    assert state.set_selected_engine(selected) is True
    assert state.supports(OutputKind.TOTAL_YIELD) is True


def test_saved_gui_defaults_naming_a_removed_engine_are_ignored(tmp_path):
    """An existing user config with a retired engine tab must not break workspace startup."""
    path = tmp_path / "gui-defaults.yaml"
    path.write_text(
        yaml.safe_dump({
            "version": DEFAULTS_VERSION,
            "engines": {RETIRED: {"n_time": 81, "quantum": "thomson"}, "xigma": {"n_steps": 48}},
        }),
        encoding="utf-8",
    )

    state = InputState(selectable_engines())
    GuiDefaultsStore(path).apply(state)

    assert f"engine:{RETIRED}" not in state.groups
    assert state.groups["engine:xigma"].values["n_steps"] == 48
