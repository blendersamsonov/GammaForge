#!/usr/bin/env bash
# Run a command with the right interpreter and the right GammaForge, without relying on
# `source activate` having worked.
#
# Why this exists: the venv is created with `python3 -> /usr/bin/python3`, so if the checkout
# is copied to a machine whose Python lives elsewhere, `.venv/bin/python` becomes a
# dangling symlink. `which python` then silently reports the *system* interpreter, and
# `source .venv/bin/activate` appears to succeed while changing nothing usable. Meanwhile
# the venv's editable install points at whichever checkout it was created from, so a run can
# import the wrong GammaForge and produce numbers that look fine.
#
# Usage:  ./python.sh run_all.py --quick
#         ./python.sh summarize.py
#         ./python.sh -c "import gammaforge; print(gammaforge.__file__)"
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Locate a venv. A git worktree has no `.venv` of its own -- the real one lives in the main
# checkout, whose path is recoverable from the worktree's git common dir. So: this checkout
# first, then the main checkout, then whatever GAMMAFORGE_VENV names.
venv=""
for candidate in "$repo/.venv" "$(cd "$repo" 2>/dev/null && git rev-parse --git-common-dir 2>/dev/null | xargs -r dirname)/.venv"; do
    [[ -n "$candidate" && -d "$candidate" ]] && { venv="$candidate"; break; }
done
[[ -z "$venv" ]] && venv="${GAMMAFORGE_VENV:-$repo/.venv}"

# Pick an interpreter: the venv's if it actually executes, else the venv's python3, else
# whatever python3 is on PATH (and say so, loudly).
interpreter=""
for candidate in "$venv/bin/python" "$venv/bin/python3" python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "pass" >/dev/null 2>&1; then
        interpreter="$candidate"
        break
    fi
done
if [[ -z "$interpreter" ]]; then
    echo "no usable python3 found; set GAMMAFORGE_VENV or put python3 on PATH" >&2
    exit 1
fi
if [[ "$interpreter" != "$venv/bin/python" ]]; then
    echo "note: '$venv/bin/python' is not usable; falling back to $interpreter" >&2
    if [[ ! -d "$venv" ]]; then
        cat >&2 <<EOF
      No venv found at '$venv'. Either create one, or point at an existing one:

        export GAMMAFORGE_VENV=/path/to/venv
        # or:  python3 -m venv '$venv' && '$venv/bin/pip' install -e '$repo'
        # or:  pip install -e '$repo'   (into whatever python3 you are using)
EOF
    elif [[ ! -x "$venv/bin/python" ]]; then
        cat >&2 <<EOF
      That venv was almost certainly created on a different machine: its bin/python is a
      symlink to an interpreter that does not exist here, so 'which python' keeps reporting
      the system one. Either

        (a) point at a working venv:      export GAMMAFORGE_VENV=/path/to/venv
        (b) rebuild this one:            rm -rf '$venv' && python3 -m venv '$venv' \\
                                           && '$venv/bin/pip' install -e '$repo'
        (c) just use a python that has the deps:
                                            pip install -e '$repo'
EOF
    fi
fi

# Force this checkout's source ahead of whatever the editable install points at.
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$here"
exec "$interpreter" "$@"
