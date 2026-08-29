#!/usr/bin/env bash
# Collect API keys and persist them for the life of the instance.
#
#   bash llm/bootstrap/setup_secrets.sh          # prompts for anything missing
#   WANDB_API_KEY=xxx bash .../setup_secrets.sh  # non-interactive
#   SKIP_SECRETS=1 bash .../setup_secrets.sh     # skip entirely
#
# Order of preference for each key:
#   1. already exported in the environment
#   2. /workspace/.wamal_env from an earlier run on this volume
#   3. an interactive prompt, if we have a terminal
#   4. skipped, with a note -- never a hard failure
#
# Keys are written to /workspace/.wamal_env (chmod 600), which is outside the
# repo so it cannot be committed, and on the mounted volume so it survives a
# container restart. `wandb login` additionally persists to ~/.netrc, which is
# what makes unattended sweeps in a fresh shell work without sourcing anything.
set -euo pipefail

PY="${PYTHON:-python3}"

if [[ "${SKIP_SECRETS:-0}" == "1" ]]; then
    echo "[secrets] SKIP_SECRETS=1; skipping"
    exit 0
fi

if [[ -d /workspace && -w /workspace ]]; then
    SECRETS_FILE="${SECRETS_FILE:-/workspace/.wamal_env}"
else
    SECRETS_FILE="${SECRETS_FILE:-${HOME}/.wamal_env}"
fi

if [[ -f "${SECRETS_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a; source "${SECRETS_FILE}"; set +a
    echo "[secrets] loaded existing ${SECRETS_FILE}"
fi

touch "${SECRETS_FILE}"
chmod 600 "${SECRETS_FILE}"

persist() {
    local name="$1" value="$2"
    # Rewrite in place so re-running doesn't append duplicates.
    if [[ -f "${SECRETS_FILE}" ]]; then
        grep -v "^export ${name}=" "${SECRETS_FILE}" > "${SECRETS_FILE}.tmp" 2>/dev/null || true
        mv "${SECRETS_FILE}.tmp" "${SECRETS_FILE}"
    fi
    printf 'export %s=%q\n' "${name}" "${value}" >> "${SECRETS_FILE}"
    chmod 600 "${SECRETS_FILE}"
}

ask() {
    local name="$1" description="$2" current="${!1:-}"

    if [[ -n "${current}" ]]; then
        echo "[secrets] ${name} already set"
        persist "${name}" "${current}"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        echo "[secrets] ${name} not set and no terminal to ask on; skipping"
        return 0
    fi

    echo
    echo "  ${description}"

    # Keep the key out of the terminal scrollback, where it would survive in
    # screenshots and pasted logs. `read -s` alone is not enough: bash 3.2
    # accepts the flag and ignores it, echoing the key in full. Driving termios
    # directly works everywhere, so -s is only the second line of defence.
    local saved_tty=""
    saved_tty="$(stty -g 2>/dev/null || true)"
    restore_tty() { [[ -n "${saved_tty}" ]] && stty "${saved_tty}" 2>/dev/null || true; }
    # Ctrl-C at the prompt must not leave the terminal with echo off.
    trap 'restore_tty; echo; exit 130' INT
    [[ -n "${saved_tty}" ]] && stty -echo 2>/dev/null || true

    read -r -s -p "  ${name} (blank to skip): " value

    restore_tty
    trap - INT
    echo
    if [[ -z "${value}" ]]; then
        echo "[secrets] skipped ${name}"
        return 0
    fi
    export "${name}=${value}"
    persist "${name}" "${value}"
}

echo "=============================================================="
echo " API keys  (stored in ${SECRETS_FILE}, mode 600)"
echo "=============================================================="

ask WANDB_API_KEY "Weights & Biases key for --wandb runs. From https://wandb.ai/authorize
  Skip this and runs still record to metrics.jsonl, and to wandb offline."
ask HF_TOKEN "HuggingFace token. Optional: every model in the ladder is ungated,
  so this only matters for hub rate limits or if you add a gated model."

# --- make the keys usable without sourcing anything ------------------------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    if ${PY} -c "import wandb" >/dev/null 2>&1; then
        # Writes ~/.netrc, so a later `nohup ... sweep --wandb` in a fresh shell
        # authenticates with no environment setup at all.
        if ${PY} -m wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1; then
            echo "[secrets] wandb login ok (persisted to ~/.netrc)"
        else
            echo "[secrets] WARNING: wandb login failed; runs will record offline" >&2
        fi
    else
        echo "[secrets] wandb not installed yet; key saved, login skipped"
    fi
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    if ${PY} -c "import huggingface_hub" >/dev/null 2>&1; then
        ${PY} -c "
import os
from huggingface_hub import login
try:
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    print('[secrets] huggingface login ok')
except Exception as exc:
    print(f'[secrets] WARNING: huggingface login failed: {exc}')
" || true
    fi
fi

cat <<EOF

[secrets] done. To use these in a new shell:
    source ${SECRETS_FILE}
EOF
