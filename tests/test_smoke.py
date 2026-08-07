"""Phase 0 exit criterion: the package installs and imports cleanly."""

import gammaforge


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
