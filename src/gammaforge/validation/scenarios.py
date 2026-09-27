"""The shared scenario bank every validation leg runs against (GRAND_PLAN.md §7).

A `Scenario` is a **physics** statement — beam, laser, target, sampling — and nothing
else. Engine numeric knobs are deliberately absent: they live in each engine's own
`Parameters` schema (P5); engine-specific knobs have no place on the model-agnostic
scenario dataclass. `runners.run_engine` takes engine parameters as its own argument.

The established operating point is gamma0 = 2000, 10 nC, and 20 J at 1030 nm. It is
written here in CGS with explicit units rather than as converted literals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..io.bunch import GaussianElectronBeam
from ..io.interaction import InteractionParameters, SamplingSpec, build_interaction
from ..io.laser import GaussianParaxialLaser, LaserField
from ..io.target import OutputKind, OutputRequest, Target
from ..io.units import MEC2_CGS, Quantity

__all__ = [
    "Scenario",
    "BASELINE",
    "LOW_A0",
    "NEAR_A0_MAX",
    "SCENARIOS",
    "by_name",
    "build",
    "DEFAULT_SAMPLING",
]

#: Sampling every scenario shares unless it says otherwise. The fixed seed makes the
#: scenario bank reproducible across validation legs.
DEFAULT_SAMPLING = SamplingSpec(n_particles=100_000, seed=20260721, prefilter=1e-3)

#: What every scenario asks for. Modest resolutions: the bank is run often and in full,
#: and a validation leg compares shapes, not pixels.
_DEFAULT_OUTPUTS = (
    OutputRequest(OutputKind.TOTAL_YIELD),
    OutputRequest(OutputKind.SPECTRUM, resolution=(256,)),
    OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, resolution=(64, 64)),
    OutputRequest(OutputKind.COLLIMATED_SPECTRUM, resolution=(64, 16, 16)),
)


@dataclass(frozen=True)
class Scenario:
    """One named physical configuration: what collides with what, and what to look at.

    ``laser`` is typed as `LaserField`, not `GaussianParaxialLaser` (P15) — a scenario
    built on a future non-Gaussian pulse is a valid member of this bank, and the harness
    must not be the place that assumes otherwise.
    """

    name: str
    beam: GaussianElectronBeam
    laser: LaserField
    target: Target
    sampling: SamplingSpec = DEFAULT_SAMPLING


def _baseline() -> Scenario:
    gamma0 = 2000.0
    # The operating point quotes energy spread relative to *gamma*; this schema stores it
    # relative to *kinetic* energy (§3.2). The two differ by gamma0 / (gamma0 - 1), which
    # is a 0.05% effect here — converted rather than assumed equal, since assuming it is
    # exactly the kind of silent approximation that obscures validation failures.
    sigma_gamma_over_gamma = 0.005
    kinetic_energy = (gamma0 - 1.0) * MEC2_CGS
    rel_energy_spread = sigma_gamma_over_gamma * gamma0 * MEC2_CGS / kinetic_energy

    # Normalized emittances, as an accelerator quotes them; geometric is what the beam
    # stores. The y plane is deliberately 100x smaller — a flat beam, so any code path
    # that silently assumes a round one shows up as a failing validation leg.
    norm_emit_x = Quantity(1e-4, "cm * rad")
    norm_emit_y = Quantity(1e-6, "cm * rad")

    beam = GaussianElectronBeam(
        bunch_charge=Quantity(10.0, "nC"),
        kinetic_energy=Quantity(kinetic_energy, "erg"),
        rel_energy_spread=rel_energy_spread,
        sigma_x=Quantity(10.0, "um"),
        sigma_y=Quantity(10.0, "um"),
        emit_x=norm_emit_x / gamma0,
        emit_y=norm_emit_y / gamma0,
        sigma_z=Quantity(10.0, "ps"),  # a length, quoted as a light-transit time (§2.1)
    )
    laser = GaussianParaxialLaser(
        pulse_energy=Quantity(20.0, "J"),
        wavelength=Quantity(1030.0, "nm"),
        sigma_x=Quantity(10.0, "um"),
        sigma_y=Quantity(10.0, "um"),
        duration=Quantity(30.0, "ps"),
    )
    target = Target(
        theta_x_col=Quantity(1.0, "mrad"),
        theta_y_col=Quantity(1.0, "mrad"),
        outputs=_DEFAULT_OUTPUTS,
    )
    return Scenario(name="baseline", beam=beam, laser=laser, target=target)


BASELINE = _baseline()


def _scaled_pulse_energy(name: str, pulse_energy: Quantity) -> Scenario:
    """The baseline with a different pulse energy — the one lever that moves a0 alone.

    ``a0`` scales as the square root of the pulse energy, and nothing else about the
    collision changes, which is what makes these three scenarios a regime scan rather than
    three unrelated configurations.
    """
    return replace(BASELINE, name=name, laser=replace(BASELINE.laser, pulse_energy=pulse_energy))


#: Deep in the linear regime.
LOW_A0 = _scaled_pulse_energy("low_a0", Quantity(2.0, "J"))
#: Near the top of xigma's documented validity range (§2.3).
NEAR_A0_MAX = _scaled_pulse_energy("near_a0_max", Quantity(100.0, "J"))

#: The bank. Iterate this; do not hardcode individual scenario names in a runner.
SCENARIOS: tuple[Scenario, ...] = (BASELINE, LOW_A0, NEAR_A0_MAX)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"no scenario named {name!r} (have {[s.name for s in SCENARIOS]})")


def build(scenario: Scenario, sampling: SamplingSpec | None = None) -> InteractionParameters:
    """Sample the scenario into the `InteractionParameters` an engine consumes.

    ``sampling`` overrides the scenario's own spec — which is how the seed-determinism and
    prefilter-invariance legs (§7) vary one knob while holding the physics fixed.
    """
    return build_interaction(
        beam=scenario.beam,
        laser=scenario.laser,
        target=scenario.target,
        sampling=sampling or scenario.sampling,
    )
