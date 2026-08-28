#!/usr/bin/env bash
# One command from a bare vast.ai box to a machine ready to run sweeps.
#
#   bash llm/bootstrap/bootstrap.sh --tier phase1
#
# Every step is idempotent, so re-running after an interruption is safe and
# mostly a no-op.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${PYTHON:-python3}"

TIER="phase1"
GPU="0"
SKIP_ENV=0
SKIP_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)           TIER="$2"; shift 2 ;;
        --gpu)            GPU="$2"; shift 2 ;;
        --skip-env)       SKIP_ENV=1; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Default the caches onto the persistent volume when one is mounted. Without
# this, model weights land on the container filesystem and are re-downloaded on
# every rental.
if [[ -d /workspace && -w /workspace ]]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    export RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=============================================================="
echo " WAMAL-LM bootstrap"
echo "   repo:    ${REPO_ROOT}"
echo "   tier:    ${TIER}"
echo "   HF_HOME: ${HF_HOME:-<default>}"
echo "   RESULTS: ${RESULTS_DIR:-<default>}"
echo "=============================================================="

if [[ "${SKIP_ENV}" -eq 0 ]]; then
    echo; echo "--- 1/4 environment ---"
    bash "${HERE}/setup_env.sh"
else
    echo; echo "--- 1/4 environment (skipped) ---"
fi

echo; echo "--- 2/4 models (tier: ${TIER}) ---"
${PY} -m llm.bootstrap.download_models --tier "${TIER}"

echo; echo "--- 3/4 data ---"
${PY} -m llm.bootstrap.download_data

if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    echo; echo "--- 4/4 preflight ---"
    ${PY} -m llm.bootstrap.preflight --tier "${TIER}" --gpu "${GPU}"
else
    echo; echo "--- 4/4 preflight (skipped) ---"
fi

cat <<EOF

=============================================================
 Ready.

   # see what a phase costs before committing
   python -m llm.scripts.sweep --phase ${TIER} --dry-run

   # run it
   python -m llm.scripts.sweep --phase ${TIER} --gpu ${GPU}

 Run llm/bootstrap/sync_results.sh and copy the archive off the
 box BEFORE destroying the instance.
=============================================================
EOF
