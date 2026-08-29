#!/usr/bin/env bash
# Package results for retrieval before the instance is destroyed.
#
#   bash llm/bootstrap/sync_results.sh
#   bash llm/bootstrap/sync_results.sh --with-checkpoints
#
# Checkpoints are excluded by default. metrics.jsonl, summary.json and the .npz
# dumps are everything the analysis needs and are roughly three orders of
# magnitude smaller, which matters when the transfer is over the public internet.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${PYTHON:-python3}"

WITH_CHECKPOINTS=0
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-checkpoints) WITH_CHECKPOINTS=1; shift ;;
        --out)              OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RESULTS="$(${PY} -c 'from llm import env; print(env.results_dir())')"
if [[ ! -d "${RESULTS}" ]]; then
    echo "no results directory at ${RESULTS}" >&2
    exit 1
fi

OUT_DIR="${OUT_DIR:-$(dirname "${RESULTS}")}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname | tr -c 'a-zA-Z0-9' '-' | sed 's/-*$//')"
ARCHIVE="${OUT_DIR}/wamal-results-${HOST}-${STAMP}.tar.gz"

COMPLETED="$(find "${RESULTS}" -name DONE -type f 2>/dev/null | wc -l | tr -d ' ')"
TOTAL="$(find "${RESULTS}" -name config.json -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "[sync] ${RESULTS}: ${COMPLETED}/${TOTAL} runs complete"

EXCLUDES=()
if [[ "${WITH_CHECKPOINTS}" -eq 0 ]]; then
    EXCLUDES+=(--exclude='*.pt')
fi

tar czf "${ARCHIVE}" "${EXCLUDES[@]}" -C "$(dirname "${RESULTS}")" "$(basename "${RESULTS}")"

SIZE="$(du -h "${ARCHIVE}" | cut -f1)"
echo "[sync] wrote ${ARCHIVE} (${SIZE})"

cat <<EOF

Copy it off the box now. From your laptop, one of:

  scp -P <ssh-port> root@<instance-host>:${ARCHIVE} .
  vastai copy <instance-id>:${ARCHIVE} .
  rclone copy ${ARCHIVE} <remote>:wamal/

Nothing outside the mounted volume survives teardown.
EOF
