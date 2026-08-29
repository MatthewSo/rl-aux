#!/usr/bin/env bash
# Shared connection plumbing for deploy/*.sh. Sourced, not executed.
#
# Kept compatible with bash 3.2, which is what macOS still ships as /bin/bash.
# That rules out `mapfile`, associative arrays, and `${var,,}` — the scripts in
# here run on your laptop, so this is the constraint that matters, not whatever
# bash 5 is on the remote box.

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/.." && pwd)"

# deploy/env is optional; flags and exported vars work without it.
if [[ -f "${DEPLOY_DIR}/env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${DEPLOY_DIR}/env"
    set +a
fi

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-22}"
VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-}"
SSH_KEY="${SSH_KEY:-}"
WORKDIR="${WORKDIR:-/workspace/rl-aux}"
REMOTE_RESULTS="${REMOTE_RESULTS:-/workspace/results}"
LOCAL_ARTIFACTS="${LOCAL_ARTIFACTS:-./vast_artifacts}"

die() { echo "error: $*" >&2; exit 1; }

# Populated by build_ssh_opts. A global array rather than a function that prints
# lines, because reading lines back into an array portably is exactly what
# mapfile would have been for.
SSH_OPTS=()

build_ssh_opts() {
    SSH_OPTS=()
    if [[ -z "${VAST_SSH_ALIAS}" ]]; then
        SSH_OPTS+=(-p "${VAST_PORT}")
        if [[ -n "${SSH_KEY}" ]]; then
            SSH_OPTS+=(-i "${SSH_KEY/#\~/$HOME}")
        fi
    fi
    # Vast recycles host keys across instances, so a known_hosts entry from a
    # previous rental fails the check on a new box at the same address. These
    # are throwaway compute boxes, not a trust boundary worth the friction.
    SSH_OPTS+=(-o StrictHostKeyChecking=no)
    SSH_OPTS+=(-o UserKnownHostsFile=/dev/null)
    SSH_OPTS+=(-o LogLevel=ERROR)
    SSH_OPTS+=(-o ServerAliveInterval=30)
    SSH_OPTS+=(-o ServerAliveCountMax=6)
}

target() {
    if [[ -n "${VAST_SSH_ALIAS}" ]]; then
        echo "${VAST_SSH_ALIAS}"
    else
        [[ -n "${VAST_HOST}" ]] || die "set VAST_HOST (or VAST_SSH_ALIAS) in deploy/env, or pass --host"
        echo "${VAST_HOST}"
    fi
}

remote_ssh() {
    build_ssh_opts
    ssh "${SSH_OPTS[@]}" "$(target)" "$@"
}

# rsync wants the ssh invocation as a single -e string.
rsync_shell() {
    build_ssh_opts
    echo "ssh ${SSH_OPTS[*]}"
}
