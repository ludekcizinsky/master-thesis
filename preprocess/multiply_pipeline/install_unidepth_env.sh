#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-unidepth}
CONDA_ROOT="/home/cizinsky/miniconda3"
UNIDEPTH_ROOT="/home/cizinsky/master-thesis/preprocess/multiply_pipeline/unidepth"

if [[ ! -d "${UNIDEPTH_ROOT}" ]]; then
  echo "UniDepth repo not found at ${UNIDEPTH_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Unable to source conda from ${CONDA_ROOT}" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -e "${UNIDEPTH_ROOT}" --extra-index-url https://download.pytorch.org/whl/cu118

python -m pip uninstall -y pillow >/dev/null 2>&1 || true
if ! CC="cc -mavx2" python -m pip install -U --force-reinstall pillow-simd >/tmp/pillow-simd-install.log 2>&1; then
  cat /tmp/pillow-simd-install.log >&2
  echo "pillow-simd build failed; falling back to standard Pillow." >&2
  python -m pip install --upgrade pillow
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module load gcc/11.3.0 >/dev/null 2>&1 || true
  module load cuda/11.8.0 >/dev/null 2>&1 || true
fi

pushd "${UNIDEPTH_ROOT}/unidepth/ops/knn" >/dev/null
bash compile.sh
popd >/dev/null

python - <<'PY'
import torch
import importlib
import unidepth
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("unidepth root:", unidepth.__file__)
PY

echo "UniDepth environment (${ENV_NAME}) is ready."
