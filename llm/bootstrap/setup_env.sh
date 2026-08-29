#!/usr/bin/env bash
# Install Python dependencies on a bare GPU box.
#
# pip, not conda: conda solves take minutes and we are billed by the hour.
#
# The interesting part is torch. An image's preinstalled torch is usually the
# right one, but "usually" is not good enough on rented hardware: Vast hosts run
# a wide range of driver versions, and a torch built against a newer CUDA than
# the host driver supports imports fine and then fails at the first .cuda() call
# with "NVIDIA driver is too old". So we check whether torch can actually see the
# GPU, and reinstall a matching build if it cannot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${PYTHON:-python3}"
FORCE_CUDA="${FORCE_CUDA:-}"          # e.g. cu121 to override detection
SKIP_TORCH="${SKIP_TORCH:-0}"         # 1 to never touch torch

echo "[setup_env] python: $(${PY} --version 2>&1)"
echo "[setup_env] repo:   ${REPO_ROOT}"

# --- what CUDA does the host driver support? -------------------------------
# nvidia-smi reports the highest CUDA runtime the installed driver can run,
# which is the real ceiling on which torch wheel will work.
host_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi 2>/dev/null \
        | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1
}

# Map the driver ceiling to the closest wheel index that does not exceed it.
cuda_tag() {
    local version="$1"
    awk -v v="${version}" 'BEGIN {
        if (v >= 12.8)      print "cu128";
        else if (v >= 12.6) print "cu126";
        else if (v >= 12.4) print "cu124";
        else if (v >= 12.1) print "cu121";
        else if (v >= 11.8) print "cu118";
        else                print "";
    }'
}

torch_present()  { ${PY} -c "import torch" >/dev/null 2>&1; }
torch_sees_gpu() { ${PY} -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; }

HOST_CUDA="$(host_cuda)"
HAS_GPU=0
if [[ -n "${HOST_CUDA}" ]]; then
    HAS_GPU=1
    echo "[setup_env] host driver supports CUDA ${HOST_CUDA}"
else
    echo "[setup_env] no nvidia-smi; treating this as a CPU box"
fi

if [[ "${SKIP_TORCH}" == "1" ]]; then
    echo "[setup_env] SKIP_TORCH=1; not touching torch"
elif torch_present && { [[ "${HAS_GPU}" -eq 0 ]] || torch_sees_gpu; }; then
    echo "[setup_env] torch $(${PY} -c 'import torch; print(torch.__version__)')" \
         "(cuda $(${PY} -c 'import torch; print(torch.version.cuda)')) works; leaving it alone"
elif [[ "${HAS_GPU}" -eq 1 ]]; then
    TAG="${FORCE_CUDA:-$(cuda_tag "${HOST_CUDA}")}"
    if [[ -z "${TAG}" ]]; then
        echo "[setup_env] ERROR: driver only supports CUDA ${HOST_CUDA}, which is" >&2
        echo "[setup_env]        older than any current torch wheel. Pick another host." >&2
        exit 1
    fi
    if torch_present; then
        echo "[setup_env] torch is installed but cannot see the GPU" \
             "(built for cuda $(${PY} -c 'import torch; print(torch.version.cuda)' 2>/dev/null))."
        echo "[setup_env] reinstalling for ${TAG} to match the host driver"
    else
        echo "[setup_env] no torch; installing ${TAG} build"
    fi
    ${PY} -m pip install --no-cache-dir --force-reinstall \
        --index-url "https://download.pytorch.org/whl/${TAG}" torch

    if ! torch_sees_gpu; then
        echo "[setup_env] ERROR: torch still cannot see the GPU after installing ${TAG}." >&2
        echo "[setup_env]        Try FORCE_CUDA=cu121 (or another tag) and re-run." >&2
        ${PY} -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >&2 || true
        exit 1
    fi
    echo "[setup_env] torch now sees $(${PY} -c 'import torch; print(torch.cuda.device_count())') GPU(s)"
else
    echo "[setup_env] installing CPU torch"
    ${PY} -m pip install --no-cache-dir torch
fi

# --- everything else -------------------------------------------------------
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
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} device(s))")
except Exception:
    pass
PYEOF

echo "[setup_env] done"
