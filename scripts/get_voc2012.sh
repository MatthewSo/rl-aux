#!/usr/bin/env bash
set -euo pipefail

# Where to put the dataset (matches your loader expectation)
ROOT_DIR="${1:-./data/voc}"
DEST_DIR="${ROOT_DIR}/VOCdevkit"
FILE="VOCtrainval_11-May-2012.tar"

# Official URL + fallback mirror
URL_PRIMARY="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/${FILE}"
URL_FALLBACK="https://pjreddie.com/media/files/${FILE}"

# MD5 from torchvision (VOC 2012 trainval)
MD5_EXPECTED="6cd6e144f989b92b3379bac3b3de84fd"  # torchvision reference
# ref: docs.pytorch.org vision voc dataset shows this exact file+md5. :contentReference[oaicite:2]{index=2}

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

download() {
  local url="$1"
  echo "Downloading: ${url}"
  # use curl if available, else wget
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 3 -o "${FILE}.part" "${url}"
  else
    wget --tries=5 --waitretry=3 -O "${FILE}.part" "${url}"
  fi
  mv "${FILE}.part" "${FILE}"
}

check_md5() {
  local got
  if command -v md5sum >/dev/null 2>&1; then
    got=$(md5sum "${FILE}" | awk '{print $1}')
  else
    # macOS fallback
    got=$(md5 -q "${FILE}")
  fi
  if [[ "${got}" != "${MD5_EXPECTED}" ]]; then
    echo "MD5 mismatch! expected ${MD5_EXPECTED}, got ${got}"
    return 1
  fi
  echo "MD5 OK (${got})"
}

# Skip if already extracted
if [[ -d "${DEST_DIR}/VOC2012" ]]; then
  echo "Found ${DEST_DIR}/VOC2012 — nothing to do."
  exit 0
fi

# Download if tar missing
if [[ ! -f "${FILE}" ]]; then
  set +e
  download "${URL_PRIMARY}" || {
    echo "Primary download failed, trying fallback mirror…"
    download "${URL_FALLBACK}" || {
      echo "Both downloads failed."
      exit 1
    }
  }
  set -e
fi

# Verify and extract
check_md5
echo "Extracting to ${ROOT_DIR} …"
mkdir -p "${DEST_DIR}"
tar -xf "${FILE}" -C "${ROOT_DIR}"  # creates VOCdevkit/VOC2012
echo "Done. Layout:"
echo "${ROOT_DIR}/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/...}"
