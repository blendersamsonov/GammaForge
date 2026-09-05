#!/usr/bin/env python3
"""
Sympy verification suite for GammaForge derivations.

This script runs all symbolic verifications of the physics derivations.
Each derivation is verified using sympy to confirm the algebraic correctness
of the boxed formulas and their limits.

Usage:
    python verify_all.py          # Run all verifications
    python verify_all.py der004   # Run specific derivation

Derivations verified:
- DER004: Ellipticity in the emission kernel's polarization factor
- DER005: Crossing angle in the emission kernel (3 parts)
- DER006: Polarization matrix with ellipticity and crossing angle
- DER007: Stokes parameters (head-on limit, basis-invariant physics)

Note: DER007's full crossing-angle expressions are too complex for symbolic
simplification. The head-on limit (θ→0) is verified symbolically for all
basis-invariant physics (P=1, I=1, |V/I|=2ε/(1+ε²), dipole null).
The exact implementation uses the paper's recommended numerical approach
(§8) with exact vectors, verified numerically in verify_der007_numerical.py.
"""

import sys
import subprocess
from pathlib import Path

def run_verification(script_name, description):
    """Run a verification script and return success status."""
    print(f"\n{'='*70}")
    print(f"Running {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"FAILED (exit code {result.returncode})")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    # Map of derivation to verification script
    script_dir = Path(__file__).parent
    verifications = [
        (script_dir / "verify_der004.py", "DER004: Ellipticity in polarization factor"),
        (script_dir / "verify_der005.py", "DER005: Crossing angle (3 parts)"),
        (script_dir / "verify_der006.py", "DER006: Combined polarization matrix"),
        (script_dir / "verify_der007_headon.py", "DER007: Stokes parameters (head-on limit)"),
    ]

    # If specific derivation requested
    if len(sys.argv) > 1:
        target = sys.argv[1]
        verifications = [(s, d) for s, d in verifications if target in s]
        if not verifications:
            print(f"Unknown derivation: {target}")
            print(f"Available: {[s for s, _ in verifications]}")
            return 1

    print("GammaForge Derivation Verification Suite")
    print("Using sympy for symbolic algebraic verification")
    print()

    results = []
    for script, desc in verifications:
        success = run_verification(script, desc)
        results.append((desc, success))

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    all_passed = True
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {desc}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n✓ All verifications passed!")
        return 0
    else:
        print("\n✗ Some verifications failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())