"""elegant / SDDS ``.ele`` 6D distributions (GRAND_PLAN.md §8).

The hand-rolled SDDS ASCII reader/writer enforces two project conventions:

* **SI → CGS at the boundary** (P1). ``.ele`` files are inherently SI/GeV; positions come
  in as metres and leave as metres, while the `Bunch` in between is centimetres.
* **Weights are normalized to the relative convention** ``1/n`` on load (§3.2). A plain
  ``.ele`` file carries no charge information at all, so the physical ``N_e`` comes from
  the separately entered charge field and never from the file; relative weights are all a
  `Bunch` can hold.

Longitudinal sign convention and column set (``x``, ``xp``, ``y``, ``yp``, ``z``, ``dP``)
follow the elegant SDDS convention so files round-trip with external accelerator tools.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..bunch import Bunch
from ..units import EV_CGS, MEC2_CGS

__all__ = ["load_elegant_ele", "save_elegant_ele"]

_M_TO_CM = 100.0

_NAME_RE = re.compile(r'name\s*=\s*(?:"([^"]+)"|(\S+))', re.IGNORECASE)
_VALUE_RE = re.compile(r'fix\s*=\s*("([^"]*)"|[+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)', re.IGNORECASE)
_REQUIRED_COLUMNS = ("x", "xp", "y", "yp", "z", "dP")


def _parse_header(path: Path, lines: list[str]) -> tuple[dict[str, float], list[str], list[str]]:
    params: dict[str, float] = {}
    columns: list[str] = []
    data_lines: list[str] = []
    in_data = False

    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        low = line.lower()
        if low.startswith("&description"):
            continue
        if low.startswith("&end"):
            break
        if low.startswith("&parameter"):
            name_match, value_match = _NAME_RE.search(line), _VALUE_RE.search(line)
            if name_match and value_match:
                name = name_match.group(1) or name_match.group(2)
                raw_value = value_match.group(2) if value_match.group(2) is not None else value_match.group(1)
                try:
                    params[name] = float(raw_value)
                except (TypeError, ValueError):
                    pass  # non-numeric header parameters are kept out of the numeric table
            continue
        if low.startswith("&column"):
            name_match = _NAME_RE.search(line)
            if name_match:
                columns.append(name_match.group(1) or name_match.group(2))
            continue
        if low.startswith("&data"):
            in_data = True
            continue
        if in_data:
            data_lines.append(raw)

    if not columns:
        raise ValueError(f"{path}: no &column definitions found")
    if not data_lines:
        raise ValueError(f"{path}: no data rows found before &end")
    return params, columns, data_lines


def load_elegant_ele(path: str | Path) -> Bunch:
    """Parse a 6D ``.ele`` file in SDDS ASCII format into a `Bunch` (CGS).

    Requires the columns ``x``, ``xp``, ``y``, ``yp``, ``z``, ``dP`` and a header
    ``Energy`` parameter in GeV; per-particle gamma is ``gamma0 * (1 + dP)``
    (ultrarelativistic, ``p ~ E``).

    The returned bunch has uniform relative weights ``1/n`` and ``gaussian_fit=None``:
    nothing has fit it yet, and pretending otherwise is what the ``None`` is for.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or not lines[0].strip().startswith("SDDS"):
        raise ValueError(f"{path}: not an SDDS file (missing SDDS1 header)")

    params, columns, data_lines = _parse_header(path, lines)

    rows = []
    for raw in data_lines:
        stripped = raw.strip()
        if not stripped or "=" in stripped.split()[0]:
            continue
        fields = stripped.split()
        if len(fields) < len(columns):
            raise ValueError(f"{path}: row has {len(fields)} fields, expected {len(columns)}")
        try:
            rows.append([float(value) for value in fields[: len(columns)]])
        except ValueError as exc:
            raise ValueError(f"{path}: cannot parse row {stripped!r}") from exc

    table = np.asarray(rows, dtype=float)
    column = {name: table[:, index] for index, name in enumerate(columns)}
    missing = [name for name in _REQUIRED_COLUMNS if name not in column]
    if missing:
        raise ValueError(f"{path}: required column(s) {missing} missing")

    if "Energy" not in params or not np.isfinite(params["Energy"]) or params["Energy"] <= 0:
        raise ValueError(f"{path}: cannot determine mean beam energy from the header")
    gamma0 = params["Energy"] * 1e9 * EV_CGS / MEC2_CGS
    gamma = np.clip(gamma0 * (1.0 + column["dP"]), 1.0 + 1e-9, None)

    n = column["x"].size
    return Bunch(
        x=column["x"] * _M_TO_CM,
        y=column["y"] * _M_TO_CM,
        z=column["z"] * _M_TO_CM,
        thx=column["xp"],
        thy=column["yp"],
        gamma=gamma,
        weight=np.full(n, 1.0 / n),
        meta={
            "source_path": str(path),
            "sdds_params": params,
            # The file has no charge; whoever needs N_e supplies it (§3.2/§3.5).
            "charge_from_file": False,
        },
    )


def save_elegant_ele(
    bunch: Bunch,
    path: str | Path,
    *,
    description: str = "6D electron bunch",
    reference_gamma: float | None = None,
) -> None:
    """Write a `Bunch` as SDDS ASCII ``.ele``, converting CGS back to SI at the boundary.

    ``reference_gamma`` is the gamma the ``dP`` column and the ``Energy`` header are
    relative to; it defaults to the bunch's own mean, which is right for a freshly sampled
    bunch. A caller round-tripping a loaded file should pass the reference from that
    file's header instead, so the reference does not drift by the sampling noise in the
    mean.

    Weights are **not** written: the format has nowhere to put them, and a `Bunch` holds
    only relative weights anyway (§3.2). Non-uniform weights are therefore lost, and the
    function says so rather than silently dropping them.
    """
    x = np.asarray(bunch.x, dtype=float).ravel() / _M_TO_CM
    y = np.asarray(bunch.y, dtype=float).ravel() / _M_TO_CM
    z = np.asarray(bunch.z, dtype=float).ravel() / _M_TO_CM
    xp = np.asarray(bunch.thx, dtype=float).ravel()
    yp = np.asarray(bunch.thy, dtype=float).ravel()
    gamma = np.asarray(bunch.gamma, dtype=float).ravel()
    n = x.size
    if not (xp.size == y.size == yp.size == z.size == gamma.size == n):
        raise ValueError("save_elegant_ele: all bunch arrays must be the same length")

    weight = np.asarray(bunch.weight, dtype=float).ravel()
    if weight.size and not np.allclose(weight, weight[0]):
        raise ValueError(
            "save_elegant_ele: the .ele format has no per-particle weight column, so a "
            "bunch with non-uniform weights cannot be written without losing information. "
            "Use the HDF5 macroparticle dump instead."
        )

    gamma0 = float(reference_gamma) if reference_gamma is not None else float(np.mean(gamma))
    dp = gamma / gamma0 - 1.0

    sigma_x, sigma_xp = float(np.std(x)), float(np.std(xp))
    sigma_y, sigma_yp = float(np.std(y)), float(np.std(yp))
    emit_x = sigma_x * sigma_xp
    emit_y = sigma_y * sigma_yp
    beta_x = sigma_x**2 / emit_x if emit_x > 0 else 0.0

    # Header scalars are written at 12 significant digits.
    # `Energy` is not decorative: every particle's gamma is reconstructed from it as
    # `gamma0 * (1 + dP)`, so header precision sets the round-trip accuracy of the whole
    # energy distribution -- at 6 digits it was ~1e-7, which a fit would see as noise.
    header = [
        "SDDS1",
        f'&description text="{description}" &',
        f'&parameter name=TotalParticles type=long description="Number of macroparticles" fix={n} &',
        f'&parameter name=Energy type=double units=GeV description="Mean beam energy" '
        f"fix={gamma0 * MEC2_CGS / EV_CGS / 1e9:.12e} &",
        f'&parameter name=NormEmittance type=double units=mm*mrad description="Normalized emittance" '
        f"fix={emit_x * gamma0 * 1e6:.12e} &",
        f'&parameter name=BetaFunction type=double units=m description="Beta function at location" '
        f"fix={beta_x:.12e} &",
        f'&parameter name=SigmaX type=double units=m description="RMS horizontal beam size" fix={sigma_x:.12e} &',
        f'&parameter name=SigmaXp type=double units=rad description="RMS horizontal divergence" fix={sigma_xp:.12e} &',
        f'&parameter name=SigmaY type=double units=m description="RMS vertical beam size" fix={sigma_y:.12e} &',
        f'&parameter name=SigmaYp type=double units=rad description="RMS vertical divergence" fix={sigma_yp:.12e} &',
        f'&parameter name=SigmaZ type=double units=m description="RMS bunch length" fix={float(np.std(z)):.12e} &',
        f'&parameter name=SigmaDp type=double description="RMS momentum spread" fix={float(np.std(dp)):.12e} &',
        '&column name=x type=double units=m description="Horizontal position" &',
        '&column name=xp type=double units=rad description="Horizontal angle (px/pz)" &',
        '&column name=y type=double units=m description="Vertical position" &',
        '&column name=yp type=double units=rad description="Vertical angle (py/pz)" &',
        '&column name=z type=double units=m description="Longitudinal position (head-tail, positive = ahead)" &',
        '&column name=dP type=double description="Relative momentum deviation (p-p0)/p0" &',
        "&data mode=ascii",
    ]
    rows = [
        f" {xi:.9e}  {xpi:.9e}  {yi:.9e}  {ypi:.9e}  {zi:.9e}  {dpi:.9e}"
        for xi, xpi, yi, ypi, zi, dpi in zip(x, xp, y, yp, z, dp)
    ]
    Path(path).write_text("\n".join([*header, *rows, "&end"]) + "\n")
