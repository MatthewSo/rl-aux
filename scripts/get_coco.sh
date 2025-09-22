#!/usr/bin/env bash
set -euo pipefail

# =========================
# COCO 2017 + Panoptic data
# =========================
# Usage:
#   bash scripts/get_coco_panoptic_2017.sh                 # installs to ./data/coco
#   bash scripts/get_coco_panoptic_2017.sh /path/to/dir    # custom target dir
#
# Requires ~40–45 GB free space.

ROOT_DIR="${1:-./data/coco}"
IMAGES_DIR="${ROOT_DIR}"
ANN_DIR="${ROOT_DIR}/annotations"

TRAIN_ZIP="train2017.zip"
VAL_ZIP="val2017.zip"
PANOPTIC_ZIP="panoptic_annotations_trainval2017.zip"

# Prefer archive.org first (your env worked with it), then official.
U_TRAIN_ARCHIVE="https://archive.org/download/MSCoco2017/${TRAIN_ZIP}"
U_VAL_ARCHIVE="https://archive.org/download/MSCoco2017/${VAL_ZIP}"
U_PANOPTIC_ARCHIVE="https://archive.org/download/MSCoco2017/${PANOPTIC_ZIP}"

U_TRAIN_OFFICIAL="https://images.cocodataset.org/zips/${TRAIN_ZIP}"
U_VAL_OFFICIAL="https://images.cocodataset.org/zips/${VAL_ZIP}"
U_PANOPTIC_OFFICIAL="https://images.cocodataset.org/annotations/${PANOPTIC_ZIP}"

# MD5 checksums (publicly documented; used for integrity verification)
MD5_TRAIN="cced6f7f71b7629ddf16f17bbcfab6b2"
MD5_VAL="442b8da7639aecaf257c1dceb8ba8c80"
MD5_PANOPTIC="4170db65fc022c9c296af880dbca6055"

# Sanity thresholds (reject tiny HTML error pages)
MIN_TRAIN_BYTES=$((10 * 1024 * 1024 * 1024))   # 10 GB
MIN_VAL_BYTES=$((500 * 1024 * 1024))           # 500 MB
MIN_PANOPTIC_BYTES=$((500 * 1024 * 1024))      # 500 MB

mkdir -p "${IMAGES_DIR}" "${ANN_DIR}"

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
      # parallel + resume
      aria2c -o "${out}.part" -x 8 -s 8 --max-connection-per-server=8 --continue=true "${url}" || return 1
      ;;
    curl)
      # retry; quietly handle SSL issues by letting archive.org succeed
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
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

check_size_at_least() {
  local file="$1" min_bytes="$2"
  local sz; sz="$(file_size "${file}")"
  if [[ -z "${sz}" || "${sz}" -lt "${min_bytes}" ]]; then
    echo "File ${file} too small (${sz:-0} bytes) — likely an HTML error page."
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
    # Python fallback
    python3 - <<PY
import zipfile
z=zipfile.ZipFile("${zip}")
z.extractall("${dest}")
PY
  fi
}

# Generic download+verify helper
ensure_zip() {
  local out_dir="$1" zip_name="$2" min_bytes="$3" md5="$4" url1="$5" url2="$6"

  if [[ -d "${out_dir}" ]]; then
    echo "Found ${out_dir} — skipping download."
    return 0
  fi

  local out_zip="${IMAGES_DIR}/${zip_name}"
  if [[ -f "${out_zip}" ]]; then
    echo "Found ${zip_name}; verifying…"
    check_size_at_least "${out_zip}" "${min_bytes}" && check_md5 "${out_zip}" "${md5}" \
      || { echo "Removing invalid ${zip_name}"; rm -f "${out_zip}"; }
  fi
  if [[ ! -f "${out_zip}" ]]; then
    download_file "${url1}" "${out_zip}" || download_file "${url2}" "${out_zip}" || {
      echo "ERROR: Could not download ${zip_name} from any source."; exit 1; }
    check_size_at_least "${out_zip}" "${min_bytes}"
    check_md5 "${out_zip}" "${md5}"
  fi
}

echo "Target root: ${ROOT_DIR}"

# 1) train2017
ensure_zip "${IMAGES_DIR}/train2017" "${TRAIN_ZIP}" "${MIN_TRAIN_BYTES}" "${MD5_TRAIN}" \
  "${U_TRAIN_ARCHIVE}" "${U_TRAIN_OFFICIAL}"
if [[ ! -d "${IMAGES_DIR}/train2017" ]]; then
  echo "Extracting ${TRAIN_ZIP}…"
  safe_unzip "${IMAGES_DIR}/${TRAIN_ZIP}" "${IMAGES_DIR}"
fi

# 2) val2017
ensure_zip "${IMAGES_DIR}/val2017" "${VAL_ZIP}" "${MIN_VAL_BYTES}" "${MD5_VAL}" \
  "${U_VAL_ARCHIVE}" "${U_VAL_OFFICIAL}"
if [[ ! -d "${IMAGES_DIR}/val2017" ]]; then
  echo "Extracting ${VAL_ZIP}…"
  safe_unzip "${IMAGES_DIR}/${VAL_ZIP}" "${IMAGES_DIR}"
fi

# 3) Panoptic annotations (robust extraction)
if [[ ! -d "${ANN_DIR}/panoptic_train2017" || ! -d "${ANN_DIR}/panoptic_val2017" || \
      ! -f "${ANN_DIR}/panoptic_train2017.json" || ! -f "${ANN_DIR}/panoptic_val2017.json" ]]; then

  local_zip="${ROOT_DIR}/${PANOPTIC_ZIP}"
  if [[ -f "${local_zip}" ]]; then
    echo "Found ${PANOPTIC_ZIP}; verifying…"
    check_size_at_least "${local_zip}" "${MIN_PANOPTIC_BYTES}" && check_md5 "${local_zip}" "${MD5_PANOPTIC}" \
      || { echo "Removing invalid ${PANOPTIC_ZIP}"; rm -f "${local_zip}"; }
  fi
  if [[ ! -f "${local_zip}" ]]; then
    download_file "${U_PANOPTIC_ARCHIVE}" "${local_zip}" || download_file "${U_PANOPTIC_OFFICIAL}" "${local_zip}" || {
      echo "ERROR: Could not download ${PANOPTIC_ZIP} from any source."; exit 1; }
    check_size_at_least "${local_zip}" "${MIN_PANOPTIC_BYTES}"
    check_md5 "${local_zip}" "${MD5_PANOPTIC}"
  fi

  echo "Extracting ${PANOPTIC_ZIP}…"
  TMP_EXTRACT="${ROOT_DIR}/.panoptic_extract"
  rm -rf "${TMP_EXTRACT}"; mkdir -p "${TMP_EXTRACT}"
  safe_unzip "${local_zip}" "${TMP_EXTRACT}"

  mkdir -p "${ANN_DIR}"

  # helpers to robustly find paths regardless of nesting
  find_one_dir() {
    local base="$1" name="$2"
    local p
    p="$(find "${base}" -type d -name "${name}" -print -quit || true)"
    if [[ -z "${p}" ]]; then
      echo "ERROR: Could not locate directory ${name} in extracted panoptic zip."
      echo "Archive content preview (top 3 levels):"
      find "${base}" -maxdepth 3 -print | sed 's/^/  /' | head -n 200
      exit 1
    fi
    echo "${p}"
  }
  find_one_json() {
    local base="$1" name="$2"
    local p
    p="$(find "${base}" -type f -name "${name}" -print -quit || true)"
    if [[ -z "${p}" ]]; then
      echo "ERROR: Could not locate file ${name} in extracted panoptic zip."
      echo "Archive content preview (top 3 levels):"
      find "${base}" -maxdepth 3 -print | sed 's/^/  /' | head -n 200
      exit 1
    fi
    echo "${p}"
  }

  DIR_TRN="$(find_one_dir  "${TMP_EXTRACT}" "panoptic_train2017")"
  DIR_VAL="$(find_one_dir  "${TMP_EXTRACT}" "panoptic_val2017")"
  JSON_TRN="$(find_one_json "${TMP_EXTRACT}" "panoptic_train2017.json")"
  JSON_VAL="$(find_one_json "${TMP_EXTRACT}" "panoptic_val2017.json")"

  rm -rf "${ANN_DIR}/panoptic_train2017" "${ANN_DIR}/panoptic_val2017"
  mv "${DIR_TRN}" "${ANN_DIR}/panoptic_train2017"
  mv "${DIR_VAL}" "${ANN_DIR}/panoptic_val2017"
  mv -f "${JSON_TRN}" "${ANN_DIR}/panoptic_train2017.json"
  mv -f "${JSON_VAL}" "${ANN_DIR}/panoptic_val2017.json"
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
