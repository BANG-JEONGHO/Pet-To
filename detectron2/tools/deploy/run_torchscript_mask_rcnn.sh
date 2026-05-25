#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LOCAL_PREFIX="${REPO_ROOT}/runtime/cuda-toolkit"
PYTHON_BIN="${PYTHON_BIN:-/home/bangj/miniconda3/envs/ai/bin/python}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_tmp}"
BIN_PATH="${BUILD_DIR}/torchscript_mask_rcnn"

usage() {
  echo "Usage: $0 <model.ts> <input.jpg> <tracing|scripting|caffe2_tracing>"
}

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN to a Python environment with torch installed."
  exit 1
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DPython3_EXECUTABLE="${PYTHON_BIN}"
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

TORCH_LIB_DIR="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

exec "${BIN_PATH}" "$1" "$2" "$3"
