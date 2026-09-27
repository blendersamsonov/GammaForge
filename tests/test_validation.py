"""Validation metrics and proof that invariance checks detect scientific defects."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.tier2]

from gammaforge.engines.base import RecomputeCost
from gammaforge.io.results import Axis, PhasespaceSlice, Results
from gammaforge.io.schema import DIMENSIONLESS, FieldKind, FieldSpec, Parameters
from gammaforge.io.target import OutputKind, compton_edge_energy
from gammaforge.io.units import Quantity
from gammaforge.validation import invariance, metrics, scenarios


CHUNK = FieldSpec(
    key="chunk",
    label="Chunk size",
    kind=FieldKind.SCALAR,
    unit=DIMENSIONLESS,
    default=4096,
    value_range=(1, 1e9),
    integer=True,
)
DEVICE = FieldSpec(
    key="device",
    label="Backend",
    kind=FieldKind.CHOICE,
    unit=DIMENSIONLESS,
    default="numpy",
    choices=("numpy", "cupy", "numba"),
)


class _InvariantEngine:
    name = "stub"
    schema = Parameters.from_specs((CHUNK, DEVICE))
    supported_outputs = (OutputKind.TOTAL_YIELD, OutputKind.SPECTRUM)
    recompute_costs = {
        "chunk": RecomputeCost.FULL_RERUN,
        "device": RecomputeCost.FULL_RERUN,
    }

    def _spectrum(self, interaction, offset=0.0):
        from gammaforge.io.laser import fit_gaussian_paraxial

        photon = fit_gaussian_paraxial(interaction.laser).photon_energy()
        edge = compton_edge_energy(interaction.beam, photon)
        energy = np.linspace(0.0, 1.2 * edge, 128)
        density = np.exp(
            -0.5 * ((energy - 0.6 * edge * (1.0 + offset)) / (0.2 * edge)) ** 2
        )
        return PhasespaceSlice(
            {Axis.ENERGY: energy}, density * interaction.N_e
        )

    def run(self, interaction, params):
        spectrum = self._spectrum(interaction)
        return Results(
            {
                OutputKind.SPECTRUM: spectrum,
                OutputKind.TOTAL_YIELD: PhasespaceSlice(
                    {}, np.asarray(spectrum.integrate())
                ),
            }
        )


class _NumericsLeakEngine(_InvariantEngine):
    name = "leaky"

    def __init__(self, field):
        self.field = field
        self._seen = {}

    def run(self, interaction, params):
        offset = 1e-3 * (
            1 + self._seen.setdefault(str(params[self.field]), len(self._seen))
        )
        spectrum = self._spectrum(interaction, offset)
        return Results({OutputKind.SPECTRUM: spectrum})


class _ParticleCountEngine(_InvariantEngine):
    name = "counter"

    def run(self, interaction, params):
        spectrum = self._spectrum(interaction).scaled(interaction.bunch.n_particles)
        return Results({OutputKind.SPECTRUM: spectrum})


@pytest.fixture(scope="module")
def small_scenario():
    return replace(
        scenarios.BASELINE,
        name="baseline",
        sampling=replace(scenarios.BASELINE.sampling, n_particles=2000),
    )


@pytest.fixture(scope="module")
def prefilter_scenario():
    beam = replace(
        scenarios.BASELINE.beam,
        sigma_x=Quantity(1.0, "mm"),
        sigma_y=Quantity(1.0, "mm"),
    )
    laser = replace(scenarios.BASELINE.laser, duration=Quantity(30.0, "fs"))
    return replace(
        scenarios.BASELINE,
        name="prefilter_probe",
        beam=beam,
        laser=laser,
        sampling=replace(scenarios.BASELINE.sampling, n_particles=2000),
    )


def _spectrum(values, low=0.0, high=1e-5):
    energy = np.linspace(low, high, len(values))
    return PhasespaceSlice({Axis.ENERGY: energy}, np.asarray(values, dtype=float))


def test_comparison_detects_uniform_scaling_and_localized_defects():
    values = np.exp(-np.linspace(-2, 2, 256) ** 2)
    reference = _spectrum(values)
    scaled = metrics.compare_slices(reference.scaled(1.05), reference)
    assert scaled.weighted_l1 == pytest.approx(0.05, rel=1e-6)
    assert scaled.yield_error == pytest.approx(0.05, rel=1e-6)

    broken = values.copy()
    broken[8:12] *= 4.0
    localized = metrics.compare_slices(_spectrum(broken), reference)
    assert localized.max_window > 20 * localized.weighted_l1


def test_comparison_resamples_without_inventing_out_of_support_flux():
    fine = np.linspace(0.0, 1e-5, 257)
    coarse = np.linspace(0.0, 1e-5, 65)
    shape = lambda x: np.exp(-((x - 5e-6) / 2e-6) ** 2)
    deviation = metrics.compare_slices(
        PhasespaceSlice({Axis.ENERGY: fine}, shape(fine)),
        PhasespaceSlice({Axis.ENERGY: coarse}, shape(coarse)),
    )
    assert deviation.weighted_l1 < 1e-3

    values = metrics.resample_to([0.0, 5.0, 10.0], [4.0, 6.0], [1.0, 1.0])
    np.testing.assert_allclose(values, [0.0, 1.0, 0.0])


def test_comparison_detects_flux_beyond_a_reference_edge():
    energy = np.linspace(0.0, 1e-5, 256)
    reference = PhasespaceSlice(
        {Axis.ENERGY: energy}, np.where(energy <= 5e-6, 1.0, 0.0)
    )
    leaked = reference.distr.copy()
    leaked[200:205] = 0.05
    deviation = metrics.compare_slices(
        PhasespaceSlice({Axis.ENERGY: energy}, leaked), reference
    )

    assert deviation.yield_error > 0.0
    assert deviation.max_window > 0.1


def test_invariance_harness_catches_numerics_and_prefilter_dependence(
    small_scenario, prefilter_scenario
):
    assert invariance.check_chunk_invariance(
        _InvariantEngine(), small_scenario, "chunk", [1024, 4096]
    ).passed
    assert not invariance.check_chunk_invariance(
        _NumericsLeakEngine("chunk"),
        small_scenario,
        "chunk",
        [1024, 4096, 8192],
    ).passed
    assert not invariance.check_backend_agreement(
        _NumericsLeakEngine("device"),
        small_scenario,
        "device",
        ["numpy", "cupy", "numba"],
    ).passed
    assert not invariance.check_prefilter_invariance(
        _ParticleCountEngine(), prefilter_scenario
    ).passed
