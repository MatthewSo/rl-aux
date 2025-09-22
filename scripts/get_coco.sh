#!/usr/bin/env bash
set -euo pipefail

# === Config ===
ROOT_DIR="${1:-./data/coco}"
IMAGES_DIR="${ROOT_DIR}"
ANN_DIR="${ROOT_DIR}/annotations"

# Filenames we need
TRAIN_ZIP="train2017.zip"
VAL_ZIP="val2017.zip"
PANOPTIC_ZIP="panoptic_annotations_trainval2017.zip"

# Official URLs
U_TRAIN_OFFICIAL="https://images.cocodataset.org/zips/${TRAIN_ZIP}"
U_VAL_OFFICIAL="https://images.cocodataset.org/zips/${VAL_ZIP}"
U_PANOPTIC_OFFICIAL="https://images.cocodataset.org/annotations/${PANOPTIC_ZIP}"

# Fallback mirrors (Internet Archive)
U_TRAIN_ARCHIVE="https://archive.org/download/MSCoco2017/${TRAIN_ZIP}"
U_VAL_ARCHIVE="https://archive.org/download/MSCoco2017/${VAL_ZIP}"
U_PANOPTIC_ARCHIVE="https://archive.org/download/MSCoco2017/${PANOPTIC_ZIP}"

# MD5 (from SJTU mirror page; used only for verification)
MD5_TRAIN="cced6f7f71b7629ddf16f17bbcfab6b2"
MD5_VAL="442b8da7639aecaf257c1dceb8ba8c80"
MD5_PANOPTIC="4170db65fc022c9c296af880dbca6055"

# Minimum size sanity checks (reject tiny HTML error pages)
MIN_TRAIN_BYTES=$((10 * 1024 * 1024 * 1024))   # 10 GB
MIN_VAL_BYTES=$((500 * 1024 * 1024))           # 500 MB
MIN_PANOPTIC_BYTES=$((500 * 1024 * 1024))      # 500 MB

# === Helpers ===
dl_tool() {
  if command -v aria2c >/dev/null 2>&1; then
    echo "aria2c"
  elif command -v curl >/dev/null 2>&1; then
    echo "curl"
  else
    echo "wget"
  fi
}

download_file() {
  local url="$1" out="$2"
  local tool; tool="$(dl_tool)"
  echo "Downloading: ${url}"
  case "${tool}" in
    aria2c)
      aria2c -o "${out}.part" -x 8 -s 8 --max-connection-per-server=8 --continue=true "${url}" || return 1
      ;;
    curl)
      curl -fL --retry 5 --retry-delay 3 -o "${out}.part" "${url}" || return 1
      ;;
    wget)
      wget --tries=5 --waitretry=3 -O "${out}.part" "${url}" || return 1
      ;;
  esac
  if [[ -s "${out}.part" ]]; then
    mv "${out}.part" "${out}"
    return 0
  fi
  rm -f "${out}.part"
  return 1
}

file_size() {
  # cross-platform stat
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

check_size_at_least() {
  local file="$1" min_bytes="$2"
  local sz; sz="$(file_size "${file}")"
  if [[ -z "${sz}" || "${sz}" -lt "${min_bytes}" ]]; then
    echo "File ${file} too small (${sz:-0} bytes) — likely an error page."
    return 1
  fi
}

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    md5 -q "$1"
  fi
}

check_md5() {
  local file="$1" expected="$2"
  local got; got="$(md5_of "${file}")"
  if [[ "${got}" != "${expected}" ]]; then
    echo "MD5 mismatch for ${file}! expected ${expected}, got ${got}"
    return 1
  fi
  echo "MD5 OK for ${file} (${got})"
}

safe_unzip() {
  local zip="$1" dest="$2"
  mkdir -p "${dest}"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${zip}" -d "${dest}"
  else
    # Python fallback if unzip is not available
    python3 - <<PY
import zipfile, sys, os
z = zipfile.ZipFile("${zip}")
z.extractall("${dest}")
PY
  fi
}

# === Begin ===
echo "Target root: ${ROOT_DIR}"
mkdir -p "${IMAGES_DIR}" "${ANN_DIR}"

# 1) train2017.zip
if [[ ! -d "${IMAGES_DIR}/train2017" ]]; then
  if [[ -f "${IMAGES_DIR}/${TRAIN_ZIP}" ]]; then
    echo "Found ${TRAIN_ZIP}; verifying…"
    check_size_at_least "${IMAGES_DIR}/${TRAIN_ZIP}" "${MIN_TRAIN_BYTES}" && check_md5 "${IMAGES_DIR}/${TRAIN_ZIP}" "${MD5_TRAIN}" \
      || { echo "Removing invalid ${TRAIN_ZIP}"; rm -f "${IMAGES_DIR}/${TRAIN_ZIP}"; }
  fi
  if [[ ! -f "${IMAGES_DIR}/${TRAIN_ZIP}" ]]; then
    download_file "${U_TRAIN_OFFICIAL}" "${IMAGES_DIR}/${TRAIN_ZIP}" || \
    download_file "${U_TRAIN_ARCHIVE}"  "${IMAGES_DIR}/${TRAIN_ZIP}" || {
      echo "ERROR: Could not download ${TRAIN_ZIP} from any source."; exit 1; }
    check_size_at_least "${IMAGES_DIR}/${TRAIN_ZIP}" "${MIN_TRAIN_BYTES}"
    check_md5 "${IMAGES_DIR}/${TRAIN_ZIP}" "${MD5_TRAIN}"
  fi
  echo "Extracting ${TRAIN_ZIP}…"
  safe_unzip "${IMAGES_DIR}/${TRAIN_ZIP}" "${IMAGES_DIR}"
else
  echo "Found ${IMAGES_DIR}/train2017 — skipping download."
fi

# 2) val2017.zip
if [[ ! -d "${IMAGES_DIR}/val2017" ]]; then
  if [[ -f "${IMAGES_DIR}/${VAL_ZIP}" ]]; then
    echo "Found ${VAL_ZIP}; verifying…"
    check_size_at_least "${IMAGES_DIR}/${VAL_ZIP}" "${MIN_VAL_BYTES}" && check_md5 "${IMAGES_DIR}/${VAL_ZIP}" "${MD5_VAL}" \
      || { echo "Removing invalid ${VAL_ZIP}"; rm -f "${IMAGES_DIR}/${VAL_ZIP}"; }
  fi
  if [[ ! -f "${IMAGES_DIR}/${VAL_ZIP}" ]]; then
    download_file "${U_VAL_OFFICIAL}" "${IMAGES_DIR}/${VAL_ZIP}" || \
    download_file "${U_VAL_ARCHIVE}"  "${IMAGES_DIR}/${VAL_ZIP}" || {
      echo "ERROR: Could not download ${VAL_ZIP} from any source."; exit 1; }
    check_size_at_least "${IMAGES_DIR}/${VAL_ZIP}" "${MIN_VAL_BYTES}"
    check_md5 "${IMAGES_DIR}/${VAL_ZIP}" "${MD5_VAL}"
  fi
  echo "Extracting ${VAL_ZIP}…"
  safe_unzip "${IMAGES_DIR}/${VAL_ZIP}" "${IMAGES_DIR}"
else
  echo "Found ${IMAGES_DIR}/val2017 — skipping download."
fi

# 3) panoptic_annotations_trainval2017.zip
if [[ ! -d "${ANN_DIR}/panoptic_train2017" || ! -d "${ANN_DIR}/panoptic_val2017" || ! -f "${ANN_DIR}/panoptic_train2017.json" || ! -f "${ANN_DIR}/panoptic_val2017.json" ]]; then
  if [[ -f "${ROOT_DIR}/${PANOPTIC_ZIP}" ]]; then
    echo "Found ${PANOPTIC_ZIP}; verifying…"
    check_size_at_least "${ROOT_DIR}/${PANOPTIC_ZIP}" "${MIN_PANOPTIC_BYTES}" && check_md5 "${ROOT_DIR}/${PANOPTIC_ZIP}" "${MD5_PANOPTIC}" \
      || { echo "Removing invalid ${PANOPTIC_ZIP}"; rm -f "${ROOT_DIR}/${PANOPTIC_ZIP}"; }
  fi
  if [[ ! -f "${ROOT_DIR}/${PANOPTIC_ZIP}" ]]; then
    download_file "${U_PANOPTIC_OFFICIAL}" "${ROOT_DIR}/${PANOPTIC_ZIP}" || \
    download_file "${U_PANOPTIC_ARCHIVE}"  "${ROOT_DIR}/${PANOPTIC_ZIP}" || {
      echo "ERROR: Could not download ${PANOPTIC_ZIP} from any source."; exit 1; }
    check_size_at_least "${ROOT_DIR}/${PANOPTIC_ZIP}" "${MIN_PANOPTIC_BYTES}"
    check_md5 "${ROOT_DIR}/${PANOPTIC_ZIP}" "${MD5_PANOPTIC}"
  fi

  echo "Extracting ${PANOPTIC_ZIP}…"
  TMP_EXTRACT="${ROOT_DIR}/.panoptic_extract"
  rm -rf "${TMP_EXTRACT}"
  mkdir -p "${TMP_EXTRACT}"
  safe_unzip "${ROOT_DIR}/${PANOPTIC_ZIP}" "${TMP_EXTRACT}"

  # Find panoptic dirs and JSONs regardless of exact nesting in the zip
  mkdir -p "${ANN_DIR}"
  # move dirs
  for d in panoptic_train2017 panoptic_val2017; do
    found_dir="$(find "${TMP_EXTRACT}" -type d -name "${d}" | head -n 1 || true)"
    if [[ -n "${found_dir}" ]]; then
      rm -rf "${ANN_DIR}/${d}"
      mv "${found_dir}" "${ANN_DIR}/${d}"
    else
      echo "ERROR: Could not locate directory ${d} in extracted panoptic zip."
      exit 1
    fi
  done
  # move JSONs
  for j in panoptic_train2017.json panoptic_val2017.json; do
    found_json="$(find "${TMP_EXTRACT}" -type f -name "${j}" | head -n 1 || true)"
    if [[ -n "${found_json}" ]]; then
      mv "${found_json}" "${ANN_DIR}/${j}"
    else
      echo "ERROR: Could not locate ${j} in extracted panoptic zip."
      exit 1
    fi
  done
  rm -rf "${TMP_EXTRACT}"
else
  echo "Found panoptic folders & JSONs — skipping download."
fi

echo
echo "✅ COCO 2017 (images + panoptic) ready at: ${ROOT_DIR}"
echo "   ├── train2017/            (images)"
echo "   ├── val2017/              (images)"
echo "   └── annotations/"
echo "       ├── panoptic_train2017/  (PNG maps)"
echo "       ├── panoptic_val2017/    (PNG maps)"
echo "       ├── panoptic_train2017.json"
echo "       └── panoptic_val2017.json"
