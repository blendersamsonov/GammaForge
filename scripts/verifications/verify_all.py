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

DER005's full crossing-angle polarization numerical verifier is not present. DER006's
lab-frame convention is author-approved and covered by its direct Eq. `udef` pytest
reference; this wrapper verifies DER006's restricted symbolic algebra separately.
The wrapper reports a blocked derivation separately from a passed executable check.
"""

import sys
import subprocess
from pathlib import Path

def run_verification(script_path: Path, description: str) -> bool:
    """Run a verification script and return success status."""
    print(f"\n{'='*70}")
    print(f"Running {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
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

def main(argv: list[str] | None = None) -> int:
    # Map of derivation to verification script
    script_dir = Path(__file__).parent
    verifications = [
        (script_dir / "verify_der004.py", "DER004: Ellipticity in polarization factor", None),
        (script_dir / "verify_der005.py", "DER005: Crossing angle (3 parts)",
         "the full crossing-angle numerical verifier is absent"),
        (script_dir / "verify_der006.py", "DER006: Combined polarization matrix", None),
        (script_dir / "verify_der007_headon.py", "DER007: Stokes parameters (head-on limit)", None),
    ]

    # If specific derivation requested
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        print("usage: verify_all.py [der004|der005|der006|der007]")
        return 2
    if argv:
        target = argv[0].casefold()
        verifications = [
            (script, description, complete)
            for script, description, complete in verifications
            if script.stem.casefold() == f"verify_{target}" or script.stem.casefold().startswith(f"verify_{target}_")
        ]
        if not verifications:
            print(f"Unknown derivation: {target}")
            print("Available: der004, der005, der006, der007")
            return 1

    print("GammaForge Derivation Verification Suite")
    print("Using sympy for symbolic algebraic verification")
    print()

    results = []
    for script, desc, blocker in verifications:
        success = run_verification(script, desc)
        results.append((desc, success, blocker))

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    all_passed = True
    all_complete = True
    for desc, success, blocker in results:
        status = "✓ PASS" if success and blocker is None else ("! BLOCKED" if success else "✗ FAIL")
        suffix = f" — {blocker}" if blocker is not None else ""
        print(f"  {status}: {desc}{suffix}")
        if not success:
            all_passed = False
        if blocker is not None:
            all_complete = False

    if all_passed and all_complete:
        print("\n✓ All verifications passed!")
        return 0
    if not all_passed:
        print("\n✗ Some verifications failed!")
    else:
        print("\n! Some derivations are not fully executable in this checkout.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
