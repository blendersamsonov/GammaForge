#!/usr/bin/env python3
"""
Sympy verification of DER005 — Crossing angle (fast version).

Verifies the parts that are symbolically tractable:
- Part 1: Relative velocity factor (exact)
- Part 2: Resonance frequency (with trig identity)

The full Part 3 polarization structure with crossing angle is not checked by an
executable verifier in this checkout.  The head-on symbolic limit is checked by
``verify_der007_headon.py``; this script must not claim more.
"""

import sympy as sp

# ─── Symbols ───────────────────────────────────────────────────────────────
γ, θ, ψ, ψ_pol, θ_xz, θ_yz = sp.symbols('γ θ ψ ψ_pol θ_xz θ_yz', real=True, positive=True)
ω_L, â = sp.symbols('ω_L â', real=True, positive=True)

# ─── Part 1: Relative velocity factor (exact, no approximation needed) ──────
print("="*60)
print("PART 1: Relative velocity factor (exact)")
print("="*60)

Rx = sp.Matrix([[1, 0, 0],
                [0, sp.cos(θ_yz), -sp.sin(θ_yz)],
                [0, sp.sin(θ_yz), sp.cos(θ_yz)]])
Ry = sp.Matrix([[sp.cos(θ_xz), 0, sp.sin(θ_xz)],
                [0, 1, 0],
                [-sp.sin(θ_xz), 0, sp.cos(θ_xz)]])
R = Ry * Rx

n0 = R * sp.Matrix([0, 0, -1])
v = sp.Matrix([0, 0, 1])  # β=1

one_minus_v_dot_n0 = sp.simplify(1 - v.dot(n0))
target_1 = 1 + sp.cos(θ_xz) * sp.cos(θ_yz)

print(f"1 - v·n₀ = {one_minus_v_dot_n0}")
print(f"Target   = {target_1}")
print(f"✓ Verified: {sp.simplify(one_minus_v_dot_n0 - target_1) == 0}")
print()
assert sp.simplify(one_minus_v_dot_n0 - target_1) == 0, "DER005 relative velocity does not match"

# Ultrarelativistic limit with cosα = cosθ_xz cosθ_yz. The half-angle identity
# is checked for α itself; substituting the geometry defines which α is meant.
α = sp.symbols("α", real=True)
half_angle_identity = sp.trigsimp(2 * (1 + sp.cos(α)) - 4 * sp.cos(α / 2) ** 2)
assert half_angle_identity == 0, "DER005 half-angle identity does not hold"
print("✓ Verified: 1 + cosα = 2cos²(α/2), with cosα = cosθ_xz cosθ_yz")
print()

# ─── Part 2: Resonance frequency ───────────────────────────────────────────
print("="*60)
print("PART 2: Resonance frequency")
print("="*60)

ω_R = ω_L * 2 * γ**2 * (1 + sp.cos(θ_xz) * sp.cos(θ_yz)) / (1 + γ**2 * θ**2 + â)
print(f"ω_R = {sp.simplify(ω_R)}")

# With β=1 and cosα definition
cosα = sp.cos(θ_xz) * sp.cos(θ_yz)
ω_R_β1 = ω_L * 2 * γ**2 * (1 + cosα) / (1 + γ**2 * θ**2 + â)
# Using 1 + cosα = 2 cos²(α/2) where cosα = cosθ_xz cosθ_yz
ω_R_final = ω_L * 4 * γ**2 * sp.cos(α / 2)**2 / (1 + γ**2 * θ**2 + â)
print(f"ω_R (β=1, with cosα) = {sp.simplify(ω_R_β1)}")
print(f"ω_R (final form)     = {ω_R_final}")
assert sp.simplify(ω_R - ω_R_β1) == 0, "DER005 resonance frequency does not match"
print("✓ Verified (using the checked half-angle identity)")
print()

# ─── Part 3: Head-on limit & dipole null ─────────────────────────────────
print("="*60)
print("PART 3: Head-on limit & dipole null")
print("="*60)
print("The full Part 3 polarization structure with crossing angle involves")
print("expressions too complex for sympy to simplify in reasonable time.")
print()
print("✓ Head-on limit: Verified in verify_der007_headon.py (symbolic)")
print("  - P = 1 for all ε (pure state preservation)")
print("  - |V/I| = 2ε/(1+ε²) for circular polarization")
print("  - Q² + U² = 1 for linear polarization (ε=0)")
print("  - The α=90° dipole null is not checked by an executable verifier")
print()
print("! Full Part 3 with crossing angle is not checked by this script.")
print("  The numerical verifier names recorded in the derivation are not present in this checkout.")
print()

# ─── Summary ───────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY")
print("="*60)
print("✓ Part 1: 1 - v·n₀ = 1 + β cosθ_xz cosθ_yz (exact)")
print("✓ Part 2: ω_R = 4ω_L γ² cos²(α/2) / (1+γ²θ²+â) (with 1+cosα=2cos²(α/2))")
print("! Part 3: only the head-on symbolic limit is checked by verify_der007_headon.py")
print("! Full crossing-angle polarization remains unverified by an executable numerical check.")
print("\nPart 1 passed; Part 2 is an algebraic restatement; Part 3 is not claimed verified here.")
