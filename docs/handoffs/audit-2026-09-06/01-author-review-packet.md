# 01 — Electron direction in the polarization factor: author review packet

This packet recorded the evidence for handoff 01's physics question. The author resolved
it on 2026-09-06; the approved convention is recorded below.

## Manuscript convention

The current manuscript defines all three vectors in the laboratory frame:

- Eq. `phaselin` says that $\mathbf v$ is the field-free electron velocity.
- Eq. `udef` defines

  $$
  \mathbf u_i =
  \frac{(\mathbf n-\mathbf v)(\mathbf n\mathbin\cdot\mathbf e_i)}
       {1-\mathbf v\mathbin\cdot\mathbf n}-\mathbf e_i.
  $$

- Eq. `wR` uses $\theta$ as the angle between $\mathbf n$ and the electron's own
  velocity. Eq. `smallangle` makes this explicit in the near-backscattering reduction:

  $$
  \theta^2=(n_y-\theta_y)^2+(n_z-\theta_z)^2.
  $$

The polarization basis vectors $\mathbf e_i$ are the laser basis in that same lab frame.
The manuscript therefore does not support mixing a particle-relative angle for one term
with a bunch-axis observer direction for another.

## Reproduction at the handoff code boundary

`stages.polarization_factor` accepts `theta_x` and `theta_y` as electron trajectory
angles, but never reads them. It constructs `v = (0, 0, beta)` and computes
`1 - v·n` from `theta_x_obs**2 + theta_y_obs**2`. With the project Python:

```python
from gammaforge.engines.xigma.stages import polarization_factor

for electron in ((0.0, 0.0), (0.010, -0.020)):
    print(electron, polarization_factor(2000, *electron, 0.0005, 0.0, 0.0, 0.0))
```

Both calls return `2.499999374183659e-07`. The handoff's coupled-angle probe is also
reproduced:

```python
for tilt in (0.0, 0.0005):
    print(polarization_factor(2000, tilt, 0.0, tilt, 0.0, 0.0, 0.0))
```

It returns `1.0` and `2.499999374183659e-07`. This latter change is not itself an
expected invariant; it only demonstrates that the observer angle is used while the
simultaneous electron angle is not.

`polarization_factor_vectorized` has no electron-direction input at all. In
`spectrum_from_table`, it receives only the requested observation direction, even though
the table has per-cell `theta_x` and `theta_y`. In `delta.resonance_spectrum`, the
resonance correctly uses

$$
r^2=(\theta_{x,e}-\theta_{x,\mathrm{obs}})^2+
    (\theta_{y,e}-\theta_{y,\mathrm{obs}})^2,
$$

but its subsequent call passes those relative values into `polarization_factor`'s ignored
electron-angle slots and passes the absolute observer coordinates as the active inputs.
Thus the Delta and xigma paths can agree while making the same coordinate mistake.

## Approved correction

The direct reading of the manuscript is to use per-particle lab vectors throughout:

$$
\mathbf v_e=\beta
\frac{(\theta_{x,e},\theta_{y,e},1)}
     {\sqrt{1+\theta_{x,e}^2+\theta_{y,e}^2}},
\qquad
\mathbf n=\frac{(\theta_{x,\mathrm{obs}},\theta_{y,\mathrm{obs}},1)}
     {\sqrt{1+\theta_{x,\mathrm{obs}}^2+\theta_{y,\mathrm{obs}}^2}},
$$

and to evaluate Eq. `udef` (or its dot-product expansion) with each rotated laser basis
vector $\mathbf e_i$ and the exact $1-\mathbf v_e\mathbin\cdot\mathbf n$. The resonance
small-angle term must use the corresponding particle-relative angle consistently.

An alternative is a particle-aligned coordinate frame. That is valid only if every vector
is transformed for each particle: the observer direction, laser propagation direction,
and rotated polarization basis. Setting only $\mathbf v=\beta\hat{\mathbf z}$ while
leaving $\mathbf n$ and $\mathbf e_i$ in lab coordinates is not such a transformation.

## Author resolution and implementation boundary

The author approved the direct lab-frame interpretation: “use proper definition for the
electron velocity” and do not make per-particle rotations. The implementation therefore
uses the vectors above, derives `beta` from each particle's `gamma`, and leaves the
observer direction and rotated laser basis in the shared lab frame.

`tests/test_stage0_delta.py` retains a direct Eq. `udef` calculation that does not call
the production factor, including a tilted electron, off-axis observer, rotated basis, and
ellipticity. It also verifies that Delta supplies the sampled particle direction. This
checks the implementation against the approved manuscript equation; it does not provide
an independent arbitrary-angle emission calculation.
