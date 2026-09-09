"""Phase 0 exit criterion: the package installs and imports cleanly."""

import pytest

import gammaforge

pytestmark = [pytest.mark.tier0, pytest.mark.fast]


def test_import():
    assert gammaforge.__version__


def test_subpackages_import():
    import gammaforge.engines
    import gammaforge.engines.analytical
    import gammaforge.engines.kascade
    import gammaforge.engines.xigma
    import gammaforge.gui
    import gammaforge.io
    import gammaforge.validation
    import gammaforge.validation.references
