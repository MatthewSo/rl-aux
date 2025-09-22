#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-./data/voc}"
DEST_DIR="${ROOT_DIR}/VOCdevkit"
FILE="VOCtrainval_11-May-2012.tar"
MD5_EXPECTED="6cd6e144f989b92b3379bac3b3de84fd"

# You can override or add mirrors by exporting VOC2012_URLS (space-separated)
# Example: VOC2012_URLS="https://my.mirror/VOCtrainval_11-May-2012.tar"
DEFAULT_URLS=(
  "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/${FILE}"
  "http://pjreddie.com/media/files/${FILE}"         # note: HTTP, not HTTPS
  "https://pjreddie.com/media/files/${FILE}"
)
URLS=(${VOC2012_URLS:-${DEFAULT_URLS[@]}})

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

# Already extracted?
if [[ -d "${DEST_DIR}/VOC2012" ]]; then
  echo "Found ${DEST_DIR}/VOC2012 — nothing to do."
  exit 0
fi

download_with() {
  local url="$1"
  echo "Downloading: ${url}"
  local out="${FILE}.part"

  if command -v aria2c >/dev/null 2>&1; then
    # parallel, handles flakey servers better
    aria2c -o "${out}" -x 8 -s 8 --max-connection-per-server=8 --continue=true "${url}" || return 1
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 3 -o "${out}" "${url}" || return 1
  else
    wget --tries=5 --waitretry=3 -O "${out}" "${url}" || return 1
  fi

  # move into place only if non-empty
  if [[ -s "${out}" ]]; then
    mv "${out}" "${FILE}"
    return 0
  fi
  rm -f "${out}"
  return 1
}

check_size() {
  # sanity: the real tar is ~1.9 GB; reject suspiciously small files
  local min_bytes=$((100 * 1024 * 1024)) # 100MB
  local sz
  sz=$(stat -c%s "${FILE}" 2>/dev/null || stat -f%z "${FILE}")
  if [[ "${sz}" -lt "${min_bytes}" ]]; then
    echo "Downloaded file is too small (${sz} bytes) — likely an HTML error page."
    return 1
  fi
  return 0
}

check_md5() {
  local got
  if command -v md5sum >/dev/null 2>&1; then
    got=$(md5sum "${FILE}" | awk '{print $1}')
  else
    got=$(md5 -q "${FILE}")
  fi
  if [[ "${got}" != "${MD5_EXPECTED}" ]]; then
    echo "MD5 mismatch! expected ${MD5_EXPECTED}, got ${got}"
    return 1
  fi
  echo "MD5 OK (${got})"
}

# If tar already exists, validate it; else try mirrors
if [[ -f "${FILE}" ]]; then
  echo "Found existing ${FILE}; validating…"
  (check_size && check_md5) || { echo "Existing file invalid; removing and re-downloading…"; rm -f "${FILE}"; }
fi

if [[ ! -f "${FILE}" ]]; then
  ok=0
  for url in "${URLS[@]}"; do
    set +e
    download_with "${url}"
    dl_rc=$?
    set -e
    if [[ "${dl_rc}" -ne 0 ]]; then
      echo "Download failed for: ${url}"
      continue
    fi
    if check_size && check_md5; then
      ok=1
      break
    else
      echo "Validation failed for: ${url}"
      rm -f "${FILE}"
    fi
  done

  if [[ "${ok}" -ne 1 ]]; then
    cat >&2 <<EOF

All automatic downloads failed.

Manual option:
  1) Download ${FILE} from any trusted mirror (e.g., university mirrors or a browser session that can access Oxford).
  2) Place it at: ${ROOT_DIR}/${FILE}
  3) Re-run this script.

After placing the file, we'll verify MD5 (${MD5_EXPECTED}) and extract.

EOF
    exit 1
  fi
fi

echo "Extracting to ${ROOT_DIR} …"
mkdir -p "${DEST_DIR}"
tar -xf "${FILE}" -C "${ROOT_DIR}"  # creates VOCdevkit/VOC2012
echo "Done. Layout:"
echo "${ROOT_DIR}/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/...}"
