#!/usr/bin/env bash
# Move results and code between your laptop and a rented box.
#
#   deploy/sync.sh pull                 # remote results -> ./vast_artifacts
#   deploy/sync.sh push                 # local repo     -> remote WORKDIR
#   deploy/sync.sh pull --with-checkpoints
#   deploy/sync.sh pull --host root@ssh5.vast.ai --port 41234
#   deploy/sync.sh status               # what's finished on the remote
#
# rsync, not tar+scp: it is incremental and resumable, so a dropped connection
# costs you the current file rather than the whole transfer, and re-running
# after more runs finish only moves the new ones.
set -euo pipefail

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

MODE="${1:-}"; shift || true
WITH_CHECKPOINTS=0
DRY=""

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" || -z "${MODE}" ]]; then
    usage
    [[ -z "${MODE}" ]] && die "first argument must be pull, push, or status"
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)             VAST_HOST="$2"; shift 2 ;;
        --port)             VAST_PORT="$2"; shift 2 ;;
        --alias)            VAST_SSH_ALIAS="$2"; shift 2 ;;
        --key)              SSH_KEY="$2"; shift 2 ;;
        --remote-results)   REMOTE_RESULTS="$2"; shift 2 ;;
        --local)            LOCAL_ARTIFACTS="$2"; shift 2 ;;
        --with-checkpoints) WITH_CHECKPOINTS=1; shift ;;
        --dry-run)          DRY="--dry-run"; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
done

case "${MODE}" in
    pull|push|status) ;;
    *) usage; die "first argument must be pull, push, or status" ;;
esac

command -v rsync >/dev/null 2>&1 || die "rsync is not installed locally"

# Resolved only after the mode and flags are known, so --help and a bad mode
# don't fail on a missing host instead of telling you what's wrong.
RSH="$(rsync_shell)"
TARGET="$(target)"

case "${MODE}" in
    pull)
        mkdir -p "${LOCAL_ARTIFACTS}"
        # Checkpoints are excluded by default: metrics.jsonl, summary.json and
        # the .npz dumps are everything the analysis reads and are ~1000x
        # smaller, which matters over a public-internet link.
        EXCLUDES=(--exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc')
        if [[ "${WITH_CHECKPOINTS}" -eq 0 ]]; then
            EXCLUDES+=(--exclude='*.pt')
        fi

        echo "[sync] pull ${TARGET}:${REMOTE_RESULTS}/ -> ${LOCAL_ARTIFACTS}/"
        rsync -avz --partial --progress ${DRY} \
            -e "${RSH}" "${EXCLUDES[@]}" \
            "${TARGET}:${REMOTE_RESULTS}/" "${LOCAL_ARTIFACTS}/"

        if [[ -z "${DRY}" ]]; then
            DONE_COUNT="$(find "${LOCAL_ARTIFACTS}" -name DONE -type f 2>/dev/null | wc -l | tr -d ' ')"
            echo "[sync] ${DONE_COUNT} completed runs now in ${LOCAL_ARTIFACTS}"
            echo "[sync] analyse with:"
            echo "         RESULTS_DIR=${LOCAL_ARTIFACTS} python -m llm.analysis.aggregate --figure"
        fi
        ;;

    push)
        # For iterating on code without a git round-trip. Excludes anything the
        # remote should own or regenerate.
        echo "[sync] push ${REPO_ROOT}/ -> ${TARGET}:${WORKDIR}/"
        remote_ssh "mkdir -p '${WORKDIR}'"
        rsync -avz --partial --progress ${DRY} \
            -e "${RSH}" \
            --exclude='.git/' --exclude='.venv*/' --exclude='__pycache__/' \
            --exclude='*.pyc' --exclude='results/' --exclude='vast_artifacts/' \
            --exclude='trained_models/' --exclude='deploy/env' \
            --exclude='.DS_Store' \
            "${REPO_ROOT}/" "${TARGET}:${WORKDIR}/"
        ;;

    status)
        echo "[sync] remote progress at ${TARGET}:${REMOTE_RESULTS}"
        remote_ssh "bash -lc '
            total=\$(find \"${REMOTE_RESULTS}\" -name config.json 2>/dev/null | wc -l);
            done=\$(find \"${REMOTE_RESULTS}\" -name DONE 2>/dev/null | wc -l);
            echo \"  \$done / \$total cells complete\";
            echo \"  disk: \$(du -sh \"${REMOTE_RESULTS}\" 2>/dev/null | cut -f1)\";
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader 2>/dev/null || echo \"  (no nvidia-smi)\";
        '"
        ;;

esac
