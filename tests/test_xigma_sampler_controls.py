"""Public validation and plumbing for CuPy Stage-2 convergence controls."""

import pytest

from gammaforge.engines.xigma.schema import default_parameters
from gammaforge.engines.xigma.spectrum_sampler import calculate_angular_spectrum_gpu
from gammaforge.engines.xigma.stages import angular_spectrum_from_table


def test_sampler_schema_defaults_and_bounds():
    params = default_parameters()
    assert params.get_int("sampler_rings") == 32
    assert params.get_int("sampler_subsampling") == 32
    with pytest.raises(ValueError):
        type(params).from_specs(params.specs, sampler_rings=7)
    with pytest.raises(ValueError):
        type(params).from_specs(params.specs, sampler_rings=65)
    with pytest.raises(ValueError):
        type(params).from_specs(params.specs, sampler_subsampling=0)
    with pytest.raises(ValueError):
        type(params).from_specs(params.specs, sampler_subsampling=16_777_216)


@pytest.mark.parametrize("value", [True, 32.5, "32", float("nan")])
def test_direct_sampler_rejects_invalid_rings_before_gpu(value):
    with pytest.raises(ValueError):
        # Validation must precede the CUDA availability check.
        calculate_angular_spectrum_gpu(None, [0.0], [0.0], [0.5], rings=value)


@pytest.mark.parametrize("value", [True, 0, 16_777_216, 1.5, float("inf")])
def test_direct_sampler_rejects_invalid_subsampling_before_gpu(value):
    with pytest.raises(ValueError):
        calculate_angular_spectrum_gpu(None, [0.0], [0.0], [0.5], subsampling=value)


def test_stage2_signature_forwards_controls(monkeypatch):
    import gammaforge.engines.xigma.stages as stages

    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(stages, "stage2_backend", lambda *a, **k: "cupy")
    monkeypatch.setattr("gammaforge.engines.xigma.spectrum_sampler.calculate_angular_spectrum_gpu", fake)
    assert angular_spectrum_from_table(object(), [0.0], [0.0], [0.5], backend="cupy", rings=64, subsampling=7) == "ok"
    assert seen["rings"] == 64
    assert seen["subsampling"] == 7
