#!/usr/bin/env python3
"""Convert modular `# %%` Python scripts in notebooks/ into standard Jupyter .ipynb files.

Usage:
    python tools/build_notebooks.py
    python tools/build_notebooks.py --run  # also executes each notebook script to verify
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def py_to_notebook(py_path: Path) -> dict:
    """Parse a `# %%` formatted Python file into Jupyter Notebook v4 JSON structure."""
    content = py_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    cells = []
    current_type: str | None = None
    current_lines: list[str] = []

    def flush_cell():
        if current_type is not None and current_lines:
            # Strip trailing blank lines from cell
            cell_lines = list(current_lines)
            while cell_lines and cell_lines[-1].strip() == "":
                cell_lines.pop()
            if cell_lines:
                norm_lines = [l if l.endswith("\n") else l + "\n" for l in cell_lines]
                if current_type == "markdown":
                    md_lines = []
                    for l in norm_lines:
                        if l.startswith("# "):
                            md_lines.append(l[2:])
                        elif l.startswith("#\n"):
                            md_lines.append("\n")
                        elif l.startswith("#"):
                            md_lines.append(l[1:])
                        else:
                            md_lines.append(l)
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": md_lines,
                    })
                elif current_type == "code":
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": norm_lines,
                    })
        current_lines.clear()

    cell_marker_re = re.compile(r"^#\s*%%\s*(\[markdown\])?")

    for line in lines:
        match = cell_marker_re.match(line)
        if match:
            flush_cell()
            if match.group(1):
                current_type = "markdown"
            else:
                current_type = "code"
        else:
            if current_type is None:
                current_type = "code"
            current_lines.append(line)

    flush_cell()

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description="Build .ipynb notebooks from .py source scripts.")
    parser.add_argument("--run", action="store_true", help="Execute each script to verify it runs without error.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    notebooks_dir = repo_root / "notebooks"
    if not notebooks_dir.exists():
        print(f"Error: {notebooks_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    py_files = sorted(notebooks_dir.glob("*.py"))
    if not py_files:
        print(f"No .py files found in {notebooks_dir}")
        return

    for py_path in py_files:
        ipynb_path = py_path.with_suffix(".ipynb")
        print(f"Building {ipynb_path.name} from {py_path.name}...")
        nb_data = py_to_notebook(py_path)
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        print(f"  -> Generated {ipynb_path.name} ({len(nb_data['cells'])} cells)")

        if args.run:
            print(f"  -> Verifying execution of {py_path.name}...")
            res = subprocess.run([sys.executable, str(py_path)], cwd=repo_root, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Execution failed for {py_path.name}:", file=sys.stderr)
                print(res.stderr, file=sys.stderr)
                sys.exit(1)
            print(f"  -> {py_path.name} executed successfully.")

    print("\nAll notebooks built successfully!")


if __name__ == "__main__":
    main()
