#!/usr/bin/env bash
# Run a command with a working interpreter and the right GammaForge.
#
# Two silent-failure modes make a bare `python` run produce plausible numbers from the wrong
# code, and this script exists to close both:
#
# 1. A venv is created with `python3 -> /usr/bin/python3`. Copied to a machine whose Python
#    lives elsewhere, `.venv/bin/python` is a *dangling* symlink — non-executable, so `which
#    python` keeps reporting the system interpreter and `source activate` appears to succeed
#    while changing nothing usable.
# 2. The venv's editable install points at whichever checkout it was created from, and
#    `PYTHONPATH` loses to it unless set. A run then measures a *different* GammaForge.
#
# So rather than guess a venv path, this **tests candidates for the capability it needs**
# (a Python that can import the project's dependencies) and forces this checkout's `src`
# onto `PYTHONPATH` on top. Selecting by capability is what makes it work across a real
# worktree, a bundle clone sitting next to the real repo, and a machine where the venv was
# never created at all.
#
# Usage:
#   ./python.sh preflight.py              # verify tree + code freshness
#   ./python.sh run_all.py --quick
#   ./python.sh summarize.py
#   ./python.sh --show-env                # report what was selected, then exit
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Print a candidate only if the directory looks like a venv.
looks_like_venv() { [[ -f "$1/pyvenv.cfg" ]]; }

candidates=()
[[ -n "${GAMMAFORGE_VENV:-}" ]] && candidates+=("$GAMMAFORGE_VENV")
candidates+=("$repo/.venv")

# The main checkout, when this is a real worktree of it.
common="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
if [[ -n "$common" ]]; then
    candidates+=("$(dirname "$common")/.venv")
fi

# Sibling checkouts: a bundle clone or a worktree is usually a sibling of the main repo, and
# that is where its venv will be. This is the case the git lookup above cannot see, because
# a bundle clone is its own repository rather than a worktree of anything.
#
# Ordered by specificity: a sibling whose *name matches this checkout's repo name* is
# strongly preferred over any other sibling, because a directory that merely happens to
# contain a .venv with pint and numpy is not necessarily this project's venv. A same-named
# sibling is, almost by definition, the main checkout of the same project.
repo_name="$(basename "$repo")"
preferred=()
other=()
for sibling in "$repo"/../*/.venv; do
    [[ -e "$sibling" ]] || continue
    if [[ "$(basename "$(dirname "$sibling")")" == "$repo_name" ]]; then
        preferred+=("$sibling")
    else
        other+=("$sibling")
    fi
done
candidates+=("${preferred[@]}")
candidates+=("${other[@]}")
# And a few levels up, for a venv kept at a project root above the checkout.
probe="$repo"
for _ in 1 2 3; do
    probe="$(dirname "$probe")"
    candidates+=("$probe/.venv")
done

interpreter=""
chosen_venv=""
for venv in "${candidates[@]}"; do
    looks_like_venv "$venv" || continue
    for exe in "$venv/bin/python" "$venv/bin/python3"; do
        # The test that matters: can this interpreter import the project's dependencies?
        if [[ -x "$exe" ]] && "$exe" -c "import pint, numpy" >/dev/null 2>&1; then
            interpreter="$exe"
            chosen_venv="$venv"
            break 2
        fi
    done
done

# Nothing usable among the venvs: fall back to a python3 on PATH that has the deps.
if [[ -z "$interpreter" ]]; then
    for exe in python3 python; do
        if command -v "$exe" >/dev/null 2>&1 \
           && "$exe" -c "import pint, numpy" >/dev/null 2>&1; then
            interpreter="$exe"
            chosen_venv="(none: using $exe from PATH)"
            break
        fi
    done
fi

if [[ -z "$interpreter" ]]; then
    cat >&2 <<EOF
No Python with the project's dependencies was found.

Looked for a venv at:
$(printf '  %s\n' "${candidates[@]}" | sort -u)
and for a python3/python on PATH able to 'import pint, numpy'.

Either point at a working one:
    export GAMMAFORGE_VENV=/path/to/venv
or create one from this checkout:
    python3 -m venv <venv> && <venv>/bin/pip install -e '$repo'
or install into whichever python3 you already use:
    pip install -e '$repo'

Then re-run ./python.sh preflight.py
EOF
    exit 1
fi

# Force this checkout's source ahead of whatever the editable install points at.
# Prepend without duplicating: the shell may already have it (e.g. set by hand), and a
# repeated entry is harmless but makes the diagnostic output confusing.
case ":${PYTHONPATH:-}:" in
    *":$repo/src:"*) ;;
    *) PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export PYTHONPATH
cd "$here"

if [[ "${1:-}" == "--show-env" ]]; then
    echo "interpreter : $interpreter"
    echo "venv        : $chosen_venv"
    echo "PYTHONPATH  : $PYTHONPATH"
    "$interpreter" - <<'PY'
import pathlib, sys
import gammaforge
print("gammaforge  :", gammaforge.__file__)
print("sys.prefix  :", sys.prefix)
print("site-pkgs   :", next((p for p in sys.path if p.endswith("site-packages")), "?"))
# Is this venv actually *this project's*? An editable install leaves a .pth behind.
for entry in pathlib.Path(sys.prefix, "lib").rglob("_editable_impl_*.pth"):
    for line in entry.read_text().splitlines():
        print("venv's own editable install points at:", line)
# Versions of the packages that actually shape the numbers.
for name in ("numpy", "pint", "h5py", "yaml"):
    try:
        module = __import__(name)
        print(f"  {name:6s} {getattr(module, '__version__', '?')}")
    except Exception:
        print(f"  {name:6s} -- not installed --")
PY
    exit 0
fi

exec "$interpreter" "$@"
