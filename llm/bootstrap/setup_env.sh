#!/usr/bin/env bash
# Install Python dependencies on a bare vast.ai box.
#
# Deliberately pip, not conda: conda solves take minutes and we are billed by
# the hour. Also deliberately does NOT install torch when one is already
# present, so we inherit the image's CUDA-matched build instead of fighting it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${PYTHON:-python3}"

echo "[setup_env] python: $(${PY} --version 2>&1)"
echo "[setup_env] repo:   ${REPO_ROOT}"

if ${PY} -c "import torch" >/dev/null 2>&1; then
    TORCH_VER="$(${PY} -c 'import torch; print(torch.__version__)')"
    TORCH_CUDA="$(${PY} -c 'import torch; print(torch.version.cuda)')"
    echo "[setup_env] torch ${TORCH_VER} already present (cuda ${TORCH_CUDA}); leaving it alone"
else
    echo "[setup_env] no torch found; installing CPU-or-CUDA default wheel."
    echo "[setup_env] WARNING: on a GPU box you normally want an image that"
    echo "[setup_env]          already ships torch. This fallback may install a"
    echo "[setup_env]          build that does not match the driver."
    ${PY} -m pip install --no-cache-dir torch
fi

${PY} -m pip install --no-cache-dir --upgrade pip
${PY} -m pip install --no-cache-dir -r "${REPO_ROOT}/llm/requirements.txt"

echo "[setup_env] installed:"
${PY} - <<'PYEOF'
import importlib
for mod in ("torch", "transformers", "datasets", "sklearn", "numpy", "pandas"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod:<14} MISSING ({exc})")
PYEOF

echo "[setup_env] done"
