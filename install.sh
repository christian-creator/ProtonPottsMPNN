#!/usr/bin/env bash
# Build the Proton-PottsMPNN environment with uv.
#   ./install.sh                         # create .venv and install everything
#   VENV_DIR=~/venvs/ppm ./install.sh    # put the venv elsewhere (e.g. off a nearly full disk)
#   source "${VENV_DIR:-.venv}/bin/activate"
# The default .venv is rebuilt on every run. A VENV_DIR that already exists is only replaced
# with ./install.sh --clear, since uv would delete whatever that directory holds.
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
if [ "$VENV_DIR" != ".venv" ] && [ -e "$VENV_DIR" ] && [ "${1:-}" != "--clear" ]; then
  echo "$VENV_DIR already exists; rerun with './install.sh --clear' to replace it." >&2
  exit 1
fi

echo "[1/2] creating uv venv (Python 3.12) -> $VENV_DIR   (rebuilt if it exists)"
uv venv --clear --python 3.12 "$VENV_DIR"

echo "[2/2] installing the foundry package (mpnn + foundry core) + extras, in one resolution"
# one command so the resolver keeps BOTH sets — installing them separately can prune the extras.
uv pip install --python "$VENV_DIR/bin/python" -e ./foundry -r requirements-extra.txt

echo "[verify] import mpnn + foundry from the packaged tree"
"$VENV_DIR/bin/python" -P -c "
import inspect, pathlib, sys
import foundry, mpnn
root = pathlib.Path(sys.argv[1]).resolve()
where = pathlib.Path(inspect.getfile(mpnn)).resolve()
assert root in where.parents, f'mpnn resolves outside this repo: {where}'
print('  ok:', where)
" "$PWD"

echo "[kernel] register the venv as a Jupyter kernel (for the notebook)"
"$VENV_DIR/bin/python" -m ipykernel install --user --name protonpottsmpnn --display-name "ProtonPottsMPNN (venv)" \
  >/dev/null 2>&1 && echo "  ok: select the 'ProtonPottsMPNN (venv)' kernel" \
  || echo "  (skipped — just run:  $VENV_DIR/bin/jupyter lab inference/design_ph.ipynb)"

echo
echo "done.  activate with:  source $VENV_DIR/bin/activate"
echo "notebook:  jupyter lab inference/design_ph.ipynb   (pick the 'ProtonPottsMPNN (venv)' kernel)"
echo "script:    python inference/design_ph.py"
