#!/usr/bin/env bash
# Run a command on the rented box, or open a shell.
#
#   deploy/remote.sh                          # interactive shell in WORKDIR
#   deploy/remote.sh nvidia-smi
#   deploy/remote.sh 'tail -f /workspace/results/**/print.log'
#   deploy/remote.sh --logs                   # follow the newest run's log
#   deploy/remote.sh --gpu                    # one-shot GPU snapshot
#   deploy/remote.sh --clone                  # clone/update the repo on the box
#
# Saves retyping the ssh flags and puts you in WORKDIR with the right env vars
# already exported.
set -euo pipefail

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

REPO_URL="${REPO_URL:-https://github.com/MatthewSo/rl-aux.git}"
BRANCH="${BRANCH:-main}"

# Exported into every remote command so training and analysis agree on paths
# without you having to remember them.
remote_env() {
    echo "export HF_HOME='${REMOTE_HF_HOME:-/workspace/.cache/huggingface}';"
    echo "export RESULTS_DIR='${REMOTE_RESULTS}';"
    echo "export WAMAL_NUM_THREADS='${WAMAL_NUM_THREADS:-4}';"
    echo "export PYTHONPATH='${WORKDIR}';"
    [[ -n "${HF_TOKEN:-}" ]] && echo "export HF_TOKEN='${HF_TOKEN}';"
    [[ -n "${WANDB_API_KEY:-}" ]] && echo "export WANDB_API_KEY='${WANDB_API_KEY}';"
    [[ -n "${WANDB_PROJECT:-}" ]] && echo "export WANDB_PROJECT='${WANDB_PROJECT}';"
    [[ -n "${WANDB_ENTITY:-}" ]] && echo "export WANDB_ENTITY='${WANDB_ENTITY}';"
    # Picked up if an earlier bootstrap saved keys to the volume.
    echo "[ -f /workspace/.wamal_env ] && set -a && . /workspace/.wamal_env && set +a;"
    echo "cd '${WORKDIR}' 2>/dev/null || cd /workspace;"
}

case "${1:-}" in
    -h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 0
        ;;

    --gpu)
        remote_ssh "nvidia-smi"
        ;;

    --logs)
        # Follow whichever run started most recently.
        remote_ssh "bash -lc '
            latest=\$(ls -td ${REMOTE_RESULTS}/*/*/*/*/*/ 2>/dev/null | head -1);
            if [ -z \"\$latest\" ]; then echo \"no runs yet under ${REMOTE_RESULTS}\"; exit 1; fi;
            echo \"following \$latest\"; tail -f \"\$latest/print.log\";
        '"
        ;;

    --clone)
        # Token goes into a repo-local .git-credentials rather than the remote
        # URL, so it never shows up in `git remote -v`, in shell history, or in
        # any log line that echoes the URL.
        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            HOSTPART="$(echo "${REPO_URL}" | sed -E 's#https://([^/]+)/.*#\1#')"
            CRED="https://x-access-token:${GITHUB_TOKEN}@${HOSTPART}"
            remote_ssh "bash -lc '
                mkdir -p \"\$(dirname ${WORKDIR})\";
                if [ ! -d \"${WORKDIR}/.git\" ]; then
                    git clone ${REPO_URL} \"${WORKDIR}\";
                fi;
                cd \"${WORKDIR}\";
                printf \"%s\n\" \"${CRED}\" > .git-credentials;
                chmod 600 .git-credentials;
                git config credential.helper \"store --file=\$(pwd)/.git-credentials\";
                git fetch origin && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH};
                git log --oneline -1;
            '"
        else
            remote_ssh "bash -lc '
                mkdir -p \"\$(dirname ${WORKDIR})\";
                if [ ! -d \"${WORKDIR}/.git\" ]; then
                    git clone ${REPO_URL} \"${WORKDIR}\";
                fi;
                cd \"${WORKDIR}\";
                git fetch origin && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH};
                git log --oneline -1;
            '"
        fi
        ;;

    "")
        echo "[remote] opening a shell in ${WORKDIR}"
        build_ssh_opts
        exec ssh -t "${SSH_OPTS[@]}" "$(target)" "bash -lc '$(remote_env) exec bash'"
        ;;

    *)
        remote_ssh "bash -lc '$(remote_env) $*'"
        ;;
esac
