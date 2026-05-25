#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/bangj/miniconda3/envs/ai/bin/python}"
EXPORT_METHOD="${EXPORT_METHOD:-tracing}"
FORMAT="${FORMAT:-torchscript}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_e2e}"
SAMPLE_IMAGE="${SAMPLE_IMAGE:-}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml}"
WEIGHTS="${WEIGHTS:-detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl}"

usage() {
  echo "Usage: $0 <input.jpg> [export_method]"
  echo
  echo "Example:"
  echo "  $0 input.jpg tracing"
  echo
  echo "Env overrides:"
  echo "  PYTHON_BIN, OUTPUT_DIR, CONFIG_FILE, WEIGHTS, SAMPLE_IMAGE, FORMAT"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

INPUT_IMAGE="$1"
if [[ $# -eq 2 ]]; then
  EXPORT_METHOD="$2"
fi

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Input image not found: ${INPUT_IMAGE}"
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${SAMPLE_IMAGE}" ]]; then
  SAMPLE_IMAGE="${INPUT_IMAGE}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_model.py" \
  --config-file "${CONFIG_FILE}" \
  --output "${OUTPUT_DIR}" \
  --export-method "${EXPORT_METHOD}" \
  --format "${FORMAT}" \
  --sample-image "${SAMPLE_IMAGE}" \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.DEVICE cuda

"${SCRIPT_DIR}/run_torchscript_mask_rcnn.sh" \
  "${OUTPUT_DIR}/model.ts" "${INPUT_IMAGE}" "${EXPORT_METHOD}"
