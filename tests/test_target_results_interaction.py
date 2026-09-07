"""Target auto-ranging, the results contract, and interaction assembly.

GRAND_PLAN.md §3.4, §3.5, §3.6 — Phase 1 exit criteria.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gammaforge.io.bunch import GaussianElectronBeam, sample_gaussian_bunch
from gammaforge.io.interaction import (
    PREFILTER_OFF,
    InteractionParameters,
    SamplingSpec,
    build_interaction,
)
from gammaforge.io.laser import GaussianParaxialLaser
from gammaforge.io.plotting import project_slice
from gammaforge.io.results import (
    ALLOWED_AXIS_GROUPINGS,
    Axis,
    PhasespaceSlice,
    PhotonMacroparticles,
    Results,
)
from gammaforge.io.target import (
    RANGE_HEADROOM,
    OutputKind,
    OutputRequest,
    Target,
    auto_ranges,
    compton_edge_energy,
    slice_axis_values,
    slice_axis_widths,
)
from gammaforge.io.units import C_CGS, EV_CGS, Quantity as Q


def make_beam(**overrides) -> GaussianElectronBeam:
    defaults = dict(
        bunch_charge=Q(100, "pC"), kinetic_energy=Q(100, "MeV"),
        rel_energy_spread=0.01, sigma_x=Q(20, "um"), sigma_y=Q(30, "um"),
        emit_x=Q(1e-7, "cm * rad"), emit_y=Q(2e-7, "cm * rad"), sigma_z=Q(100, "um"),
    )
    return GaussianElectronBeam(**{**defaults, **overrides})


def make_laser(**overrides) -> GaussianParaxialLaser:
    defaults = dict(pulse_energy=Q(1, "J"), wavelength=Q(800, "nm"),
                    sigma_x=Q(10, "um"), sigma_y=Q(10, "um"), duration=Q(30, "fs"))
    return GaussianParaxialLaser(**{**defaults, **overrides})


def make_target(*outputs) -> Target:
    return Target(theta_x_col=Q(1, "mrad"), theta_y_col=Q(2, "mrad"), outputs=tuple(outputs))


# ---------------------------------------------------------------------------
# Axis and slices (§3.6)
# ---------------------------------------------------------------------------
def test_all_six_axes_are_distinct_members():
    # X/Y and THETA_X/THETA_Y share a unit; if the enum value were the unit alone, Enum
    # would collapse them into aliases and every 2D slice would silently break.
    assert len(set(Axis)) == 6
    assert Axis.X is not Axis.Y
    assert Axis.THETA_X is not Axis.THETA_Y


def test_axis_keys_round_trip():
    for axis in Axis:
        assert Axis.from_key(axis.key) is axis


def test_slice_validates_shape_against_its_axes():
    with pytest.raises(ValueError, match="does not match axis sizes"):
        PhasespaceSlice(axes={Axis.ENERGY: np.arange(5.0)}, distr=np.ones(4))


def test_slice_rejects_an_axis_grouping_outside_the_contract():
    with pytest.raises(ValueError, match="not an allowed axis grouping"):
        PhasespaceSlice(axes={Axis.ENERGY: np.arange(3.0), Axis.TIME: np.arange(3.0)},
                        distr=np.ones((3, 3)))


def test_every_declared_grouping_is_constructible():
    for grouping in ALLOWED_AXIS_GROUPINGS:
        axes = {axis: np.linspace(0.0, 1.0, 4) for axis in grouping}
        shape = tuple(4 for _ in grouping)
        assert PhasespaceSlice(axes=axes, distr=np.ones(shape)).distr.shape == shape


def test_zero_dimensional_slice_is_the_total_yield():
    total = PhasespaceSlice(axes={}, distr=np.asarray(42.0))
    assert total.integrate() == pytest.approx(42.0)


def test_integrating_a_uniform_density_gives_its_span():
    energy = np.linspace(1.0, 3.0, 101)
    assert PhasespaceSlice(axes={Axis.ENERGY: energy}, distr=np.full(101, 2.0)).integrate() == pytest.approx(4.0)


def test_integration_rejects_an_axis_with_no_width():
    single = PhasespaceSlice(axes={Axis.ENERGY: np.asarray([1.0])}, distr=np.ones(1))
    with pytest.raises(ValueError, match="at least 2"):
        single.integrate()


def test_histogram_measure_preserves_edge_bin_mass_and_allows_one_bin_axes():
    centres = np.array([0.5, 1.5])
    histogram = PhasespaceSlice(
        {Axis.ENERGY: centres}, np.ones(2), widths={Axis.ENERGY: np.ones(2)}
    )
    assert histogram.integrate() == pytest.approx(2.0)
    one_bin = PhasespaceSlice(
        {Axis.ENERGY: np.array([2.0])}, np.array([3.0]), widths={Axis.ENERGY: np.array([0.25])}
    )
    assert one_bin.integrate() == pytest.approx(0.75)


def test_nonuniform_histogram_measure_and_projection_preserve_mass():
    axes = {
        Axis.ENERGY: np.array([0.1, 0.7]),
        Axis.THETA_X: np.array([-0.4, 0.3]),
        Axis.THETA_Y: np.array([-0.5, 0.2]),
    }
    widths = {
        Axis.ENERGY: np.array([0.2, 0.8]),
        Axis.THETA_X: np.array([0.3, 0.7]),
        Axis.THETA_Y: np.array([0.4, 0.6]),
    }
    density = np.arange(1.0, 9.0).reshape(2, 2, 2)
    source = PhasespaceSlice(axes, density, widths=widths)
    projected = project_slice(source, (Axis.ENERGY,))
    expected = np.sum(density * widths[Axis.THETA_X][None, :, None]
                      * widths[Axis.THETA_Y][None, None, :], axis=(1, 2))
    np.testing.assert_allclose(projected.distr, expected)
    np.testing.assert_array_equal(projected.widths[Axis.ENERGY], widths[Axis.ENERGY])
    assert projected.integrate() == pytest.approx(source.integrate())


def test_smooth_nonconstant_slice_retains_trapezoidal_quadrature():
    energy = np.array([0.0, 0.5, 2.0])
    density = energy**2 + 1.0
    smooth = PhasespaceSlice({Axis.ENERGY: energy}, density)
    assert smooth.integrate() == pytest.approx(np.trapezoid(density, energy))


def test_one_bin_histogram_axes_project_without_inventing_a_point_measure():
    source = PhasespaceSlice(
        {
            Axis.ENERGY: np.array([1.0, 3.0]),
            Axis.THETA_X: np.array([0.0]),
            Axis.THETA_Y: np.array([0.0]),
        },
        np.array([[[2.0]], [[4.0]]]),
        widths={
            Axis.ENERGY: np.array([1.0, 2.0]),
            Axis.THETA_X: np.array([0.5]),
            Axis.THETA_Y: np.array([0.25]),
        },
    )
    assert project_slice(source, (Axis.ENERGY,)).integrate() == pytest.approx(source.integrate())


@pytest.mark.parametrize(
    "axes,widths,match",
    [
        ({Axis.ENERGY: np.array([0.0, 0.0])}, None, "strictly increasing"),
        ({Axis.ENERGY: np.array([0.0, np.inf])}, None, "finite"),
        ({Axis.ENERGY: np.array([0.0, 1.0])}, {Axis.ENERGY: np.array([1.0, 0.0])}, "positive"),
        ({Axis.ENERGY: np.array([0.0, 1.0])}, {Axis.ENERGY: np.array([1.0])}, "match"),
    ],
)
def test_slice_rejects_invalid_coordinates_and_measures(axes, widths, match):
    with pytest.raises(ValueError, match=match):
        PhasespaceSlice(axes, np.ones(2), widths=widths)


def test_integrating_a_3d_slice_agrees_with_its_energy_marginal():
    # The §7 identity "integral of the angular spectrum == integral of the spectrum",
    # exercised on the two groupings the contract actually allows.
    energy = np.linspace(1.0, 4.0, 33)
    theta_x = np.linspace(-1.0, 1.0, 17)
    theta_y = np.linspace(-1.0, 1.0, 21)
    density = (
        np.exp(-energy[:, None, None])
        * np.exp(-theta_x[None, :, None] ** 2)
        * np.exp(-theta_y[None, None, :] ** 2)
    )
    spectral_angular = PhasespaceSlice(
        axes={Axis.ENERGY: energy, Axis.THETA_X: theta_x, Axis.THETA_Y: theta_y}, distr=density
    )
    marginal = np.trapezoid(np.trapezoid(density, theta_y, axis=2), theta_x, axis=1)
    spectrum = PhasespaceSlice(axes={Axis.ENERGY: energy}, distr=marginal)
    assert spectral_angular.integrate() == pytest.approx(spectrum.integrate(), rel=1e-12)


def test_one_dimensional_angular_marginals_are_outside_the_contract():
    # The nine allowed groupings are a closed set; a bare (theta_x,) slice is not one of
    # them, and an engine that produces one has made an error the GUI cannot render.
    with pytest.raises(ValueError, match="not an allowed axis grouping"):
        PhasespaceSlice(axes={Axis.THETA_X: np.arange(4.0)}, distr=np.ones(4))


def test_results_rescale_linearly_for_a_charge_edit():
    slices = {
        OutputKind.TOTAL_YIELD: PhasespaceSlice(axes={}, distr=np.asarray(10.0)),
        OutputKind.SPECTRUM: PhasespaceSlice(axes={Axis.ENERGY: np.arange(4.0)}, distr=np.ones(4)),
    }
    results = Results(photon_slices=slices)
    doubled = results.scaled(2.0)
    assert doubled.photon_slices[OutputKind.TOTAL_YIELD].integrate() == pytest.approx(20.0)
    # The original is untouched — rescaling is a pure function on existing results (§5).
    assert results.photon_slices[OutputKind.TOTAL_YIELD].integrate() == pytest.approx(10.0)


def test_results_slices_are_not_mutable_through_the_mapping():
    results = Results(photon_slices={})
    with pytest.raises(TypeError):
        results.photon_slices["x"] = None  # type: ignore[index]


def test_photon_macroparticles_report_their_count():
    n = 5
    photons = PhotonMacroparticles(
        energy=np.ones(n), theta_x=np.zeros(n), theta_y=np.zeros(n),
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n), t=np.zeros(n), weight=np.ones(n),
    )
    assert photons.n_macroparticles == n


# ---------------------------------------------------------------------------
# Target (§3.4)
# ---------------------------------------------------------------------------
def test_output_request_checks_its_resolution_against_its_axes():
    with pytest.raises(ValueError, match="needs 3 resolution"):
        OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (32,))
    with pytest.raises(ValueError, match="not a slice"):
        OutputRequest(OutputKind.MACROPARTICLE_DUMP, (32,))
    assert OutputRequest(OutputKind.MACROPARTICLE_DUMP).resolution == ()


def test_output_request_snapshots_nested_ranges_and_resolution():
    resolution = [4, 5]
    ranges = {Axis.X: [0.0, 1.0]}
    request = OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, resolution, ranges)
    resolution[0] = 99
    ranges[Axis.X][0] = -99.0
    assert request.resolution == (4, 5)
    assert request.manual_ranges[Axis.X] == (0.0, 1.0)
    with pytest.raises(TypeError):
        request.manual_ranges[Axis.X] = (0.0, 2.0)  # type: ignore[index]


def test_target_snapshots_an_output_sequence():
    outputs = [OutputRequest(OutputKind.TOTAL_YIELD)]
    target = Target(theta_x_col=Q(1, "mrad"), theta_y_col=Q(2, "mrad"), outputs=outputs)
    outputs.append(OutputRequest(OutputKind.SPECTRUM, (4,)))
    assert target.outputs == (OutputRequest(OutputKind.TOTAL_YIELD),)


def test_manual_ranges_are_an_advanced_option_for_the_spatial_output_only():
    OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (8, 8), {Axis.X: (-1.0, 1.0)})
    with pytest.raises(ValueError, match="advanced override"):
        OutputRequest(OutputKind.SPECTRUM, (8,), {Axis.ENERGY: (0.0, 1.0)})


def test_target_rejects_a_duplicated_output_kind():
    with pytest.raises(ValueError, match="only once"):
        make_target(OutputRequest(OutputKind.SPECTRUM, (8,)), OutputRequest(OutputKind.SPECTRUM, (16,)))


def test_target_rejects_a_non_positive_collimation_window():
    with pytest.raises(ValueError, match="half-angles"):
        Target(theta_x_col=Q(0.0, "mrad"), theta_y_col=Q(1, "mrad"))


def test_interaction_rejects_nonpositive_or_nonfinite_electron_count():
    bunch = sample_gaussian_bunch(make_beam(), 2, seed=1)
    kwargs = dict(beam=make_beam(), laser=make_laser(), bunch=bunch,
                  target=make_target(), sampling=SamplingSpec(2, 1, 0.0))
    for value in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="N_e"):
            InteractionParameters(N_e=value, **kwargs)


# -- auto-ranging ------------------------------------------------------------
def test_compton_edge_approaches_four_gamma_squared_in_the_thomson_limit():
    beam, laser = make_beam(), make_laser()
    edge = compton_edge_energy(beam, laser.photon_energy())
    thomson = 4.0 * beam.gamma0() ** 2 * laser.photon_energy()
    # At 100 MeV against 1.55 eV the recoil parameter 2*gamma*E_L/mc^2 is ~2.4e-3, so the
    # edge sits that far below the Thomson value — small, but not zero.
    assert edge == pytest.approx(thomson, rel=5e-3)
    assert edge < thomson  # recoil only ever lowers it


def test_recoil_lowers_the_edge_at_high_energy():
    low = make_beam(kinetic_energy=Q(100, "MeV"))
    high = make_beam(kinetic_energy=Q(100, "GeV"))
    photon = make_laser().photon_energy()
    assert compton_edge_energy(high, photon) / (4 * high.gamma0() ** 2 * photon) < 0.5
    assert compton_edge_energy(low, photon) / (4 * low.gamma0() ** 2 * photon) > 0.99


def test_energy_range_covers_the_compton_edge_with_headroom():
    beam, laser = make_beam(), make_laser()
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (64,)))
    low, high = auto_ranges(target, beam, laser)[OutputKind.SPECTRUM][Axis.ENERGY]
    assert low == 0.0
    assert high == pytest.approx(RANGE_HEADROOM * compton_edge_energy(beam, laser.photon_energy()))
    assert high > compton_edge_energy(beam, laser.photon_energy())


def test_angular_range_scales_with_the_radiation_cone():
    laser = make_laser()
    target = make_target(OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, (16, 16)))
    ranges = auto_ranges(target, make_beam(), laser)[OutputKind.ANGULAR_DISTRIBUTION]
    low, high = ranges[Axis.THETA_X]
    assert low == pytest.approx(-high)
    assert high > 1.0 / make_beam().gamma0()
    # A hotter beam radiates into a narrower cone.
    hotter = make_beam(kinetic_energy=Q(400, "MeV"))
    narrower = auto_ranges(target, hotter, laser)[OutputKind.ANGULAR_DISTRIBUTION][Axis.THETA_X]
    assert narrower[1] < high


def test_angular_range_widens_with_beam_divergence():
    laser = make_laser()
    target = make_target(OutputRequest(OutputKind.ANGULAR_DISTRIBUTION, (16, 16)))
    tight = auto_ranges(target, make_beam(), laser)[OutputKind.ANGULAR_DISTRIBUTION][Axis.THETA_X][1]
    divergent = auto_ranges(target, make_beam(emit_x=Q(1e-4, "cm * rad")), laser)[OutputKind.ANGULAR_DISTRIBUTION][Axis.THETA_X][1]
    assert divergent > tight


def test_collimated_spectrum_uses_the_target_window_for_its_angles():
    target = make_target(OutputRequest(OutputKind.COLLIMATED_SPECTRUM, (16, 8, 8)))
    ranges = auto_ranges(target, make_beam(), make_laser())[OutputKind.COLLIMATED_SPECTRUM]
    assert ranges[Axis.THETA_X] == (-target.m("theta_x_col"), target.m("theta_x_col"))
    assert ranges[Axis.THETA_Y] == (-target.m("theta_y_col"), target.m("theta_y_col"))
    assert ranges[Axis.ENERGY][1] > 0.0  # energy is still auto


def test_spatial_range_follows_the_smaller_of_beam_and_laser():
    # Photons come from the overlap, so each axis is set by whichever of the two is
    # narrower there. Beam: 20/30 um. Laser: 10 um in x (narrower), 50 um in y (wider).
    beam = make_beam()
    laser = make_laser(sigma_x=Q(10, "um"), sigma_y=Q(50, "um"))
    target = make_target(OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (16, 16)))
    ranges = auto_ranges(target, beam, laser)[OutputKind.SPATIAL_DISTRIBUTION]
    assert ranges[Axis.X][1] / ranges[Axis.Y][1] == pytest.approx(laser.m("sigma_x") / beam.m("sigma_y"))


def test_manual_spatial_range_overrides_only_the_axis_it_names():
    request = OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (16, 16), {Axis.X: (-1.0, 1.0)})
    ranges = auto_ranges(make_target(request), make_beam(), make_laser())[OutputKind.SPATIAL_DISTRIBUTION]
    assert ranges[Axis.X] == (-1.0, 1.0)
    assert ranges[Axis.Y] != (-1.0, 1.0)


def test_temporal_range_comes_from_the_actual_overlap():
    beam, laser = make_beam(), make_laser()
    bunch = sample_gaussian_bunch(beam, 5000, seed=1)
    target = make_target(OutputRequest(OutputKind.TEMPORAL_ENVELOPE, (64,)))
    low, high = auto_ranges(target, beam, laser, bunch)[OutputKind.TEMPORAL_ENVELOPE][Axis.TIME]
    assert low < high
    # The window is of order the time light needs to cross the bunch plus the pulse.
    assert (high - low) < 1e-10


def test_temporal_range_requires_the_bunch():
    target = make_target(OutputRequest(OutputKind.TEMPORAL_ENVELOPE, (64,)))
    with pytest.raises(ValueError, match="needs the bunch"):
        auto_ranges(target, make_beam(), make_laser())


def test_macroparticle_dump_has_no_range():
    target = make_target(OutputRequest(OutputKind.MACROPARTICLE_DUMP))
    assert auto_ranges(target, make_beam(), make_laser()) == {}


def test_slice_axis_values_are_bin_centres_inside_the_range():
    request = OutputRequest(OutputKind.SPECTRUM, (8,))
    ranges = {Axis.ENERGY: (0.0, 8.0)}
    values = slice_axis_values(request, ranges)[Axis.ENERGY]
    assert np.allclose(values, np.arange(8) + 0.5)
    assert values[0] > 0.0 and values[-1] < 8.0
    np.testing.assert_allclose(slice_axis_widths(request, ranges)[Axis.ENERGY], np.ones(8))


@pytest.mark.parametrize("resolution", [(1.5,), (True,), (0,)])
def test_output_request_rejects_non_integral_or_nonpositive_resolution(resolution):
    with pytest.raises(ValueError, match="positive integers"):
        OutputRequest(OutputKind.SPECTRUM, resolution)


@pytest.mark.parametrize("range_", [(1.0, 1.0), (2.0, 1.0), (math.nan, 1.0), (0.0, math.inf)])
def test_output_request_rejects_invalid_manual_spatial_ranges(range_):
    with pytest.raises(ValueError, match="finite and increasing"):
        OutputRequest(OutputKind.SPATIAL_DISTRIBUTION, (4, 4), {Axis.X: range_})


# ---------------------------------------------------------------------------
# Interaction (§3.5)
# ---------------------------------------------------------------------------
def test_build_interaction_samples_prefilters_and_sets_n_e():
    beam, laser = make_beam(), make_laser()
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    interaction = build_interaction(beam, laser, target, SamplingSpec(n_particles=20_000, seed=3))
    assert isinstance(interaction, InteractionParameters)
    assert interaction.N_e == pytest.approx(beam.n_electrons())
    assert interaction.bunch.n_particles < 20_000  # some were prefiltered away
    assert interaction.bunch.gaussian_fit is beam


def test_prefilter_off_keeps_every_particle():
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    spec = SamplingSpec(n_particles=5000, seed=3, prefilter=PREFILTER_OFF)
    interaction = build_interaction(make_beam(), make_laser(), target, spec)
    assert interaction.bunch.n_particles == 5000


def test_prefilter_does_not_change_n_e():
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    on = build_interaction(make_beam(), make_laser(), target, SamplingSpec(5000, 3, 1e-3))
    off = build_interaction(make_beam(), make_laser(), target, SamplingSpec(5000, 3, PREFILTER_OFF))
    assert on.N_e == off.N_e
    # The kept particles are exactly a subset of the unfiltered ones, values untouched.
    assert np.isin(on.bunch.x, off.bunch.x).all()
    assert on.bunch.n_particles < off.bunch.n_particles
    # And the weights are the unfiltered ones, never renormalized (§3.2).
    assert np.allclose(on.bunch.weight, 1.0 / 5000)


def test_same_seed_and_parameters_rebuild_the_same_interaction():
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    spec = SamplingSpec(n_particles=5000, seed=8)
    first = build_interaction(make_beam(), make_laser(), target, spec)
    second = build_interaction(make_beam(), make_laser(), target, spec)
    for a, b in zip(first.bunch.arrays(), second.bunch.arrays()):
        assert np.array_equal(a, b)


def test_charge_edit_reuses_the_bunch_and_only_moves_n_e():
    target = make_target(OutputRequest(OutputKind.SPECTRUM, (32,)))
    interaction = build_interaction(make_beam(), make_laser(), target, SamplingSpec(5000, 8))
    recharged = interaction.with_charge(2.0 * interaction.beam.bunch_charge)
    assert recharged.N_e == pytest.approx(2.0 * interaction.N_e)
    assert recharged.bunch is interaction.bunch  # the bunch is not resampled (§3.5)


def test_target_edit_reuses_the_bunch():
    # Target edits must not resample: they do not affect the sampled distribution, and
    # rebuilding would needlessly invalidate every engine cache keyed on the bunch (§3.5).
    interaction = build_interaction(
        make_beam(), make_laser(), make_target(OutputRequest(OutputKind.SPECTRUM, (32,))),
        SamplingSpec(5000, 8),
    )
    retargeted = dataclasses.replace(
        interaction, target=make_target(OutputRequest(OutputKind.SPECTRUM, (64,)))
    )
    assert retargeted.bunch is interaction.bunch


def test_sampling_spec_validates_its_fields():
    with pytest.raises(ValueError, match="n_particles"):
        SamplingSpec(n_particles=0)
    with pytest.raises(ValueError, match="prefilter"):
        SamplingSpec(prefilter=1.0)
    with pytest.raises(ValueError, match="prefilter"):
        SamplingSpec(prefilter=-0.1)


@pytest.mark.parametrize("bad", [1.5, True])
def test_sampling_spec_rejects_nonintegral_particle_counts(bad):
    with pytest.raises(ValueError, match="n_particles"):
        SamplingSpec(n_particles=bad)


@pytest.mark.parametrize("bad", [-1, 2**31, 1.5, True])
def test_sampling_spec_rejects_unsupported_seeds(bad):
    with pytest.raises(ValueError, match="seed"):
        SamplingSpec(seed=bad)


def test_sampling_spec_accepts_numpy_integer_inputs():
    spec = SamplingSpec(n_particles=np.int64(4), seed=np.int64(3))
    assert spec.n_particles == 4 and spec.seed == 3


def test_target_rejects_nonfinite_collimation_angles():
    with pytest.raises(ValueError, match="finite"):
        Target(theta_x_col=Q(np.nan, "rad"), theta_y_col=Q(1.0, "mrad"))
