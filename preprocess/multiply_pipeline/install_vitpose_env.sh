#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-vitpose}
CONDA_ROOT="/home/cizinsky/miniconda3"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Unable to source conda from ${CONDA_ROOT}. Update CONDA_ROOT and retry." >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[1/8] Resetting torch stack"
pip uninstall -y torch torchvision torchaudio || true
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "[2/8] Installing low-level runtimes"
conda install -y -c conda-forge ittapi libgomp intel-openmp

echo "[3/8] Ensuring libittnotify is available for libtorch_cpu"
pushd "${CONDA_PREFIX}/lib" >/dev/null
if [[ ! -f libittnotify.so ]]; then
  gcc -shared -o libittnotify.so -Wl,--whole-archive libittnotify.a -Wl,--no-whole-archive -lpthread -ldl
fi
TORCH_LIB_DIR="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib"
patchelf --add-needed libittnotify.so "${TORCH_LIB_DIR}/libtorch_cpu.so"
ln -sf "${CONDA_PREFIX}/lib/libittnotify.so" "${TORCH_LIB_DIR}/libittnotify.so"
popd >/dev/null

echo "[4/8] Base scientific stack (NumPy 1.x + OpenCV 4.8)"
conda install -y numpy==1.26.4
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78

echo "[5/8] OpenMMLab tooling"
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"

echo "[6/8] Re-pin NumPy & OpenCV after mim installs"
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78

echo "[7/8] Housekeeping"
unset LD_PRELOAD

echo "[7.5/8] Installing ViTPose repo (editable)"
VITPOSE_ROOT="/home/cizinsky/source_installs"
mkdir -p "${VITPOSE_ROOT}"
if [[ ! -d "${VITPOSE_ROOT}/ViTPose/.git" ]]; then
  git clone https://github.com/ViTAE-Transformer/ViTPose.git "${VITPOSE_ROOT}/ViTPose"
fi
pushd "${VITPOSE_ROOT}/ViTPose" >/dev/null
pip install -r requirements.txt
pip install -e .
popd >/dev/null

echo "[7.6/8] Fetching ViTPose-H checkpoint"
PRETRAIN_DIR="/scratch/izar/cizinsky/pretrained"
mkdir -p "${PRETRAIN_DIR}"
VITPOSE_CKPT="${PRETRAIN_DIR}/vitpose-h-multi-coco.pth"
VITPOSE_URL="https://my.microsoftpersonalcontent.com/personal/e534267b85818129/_layouts/15/download.aspx?UniqueId=85818129-267b-2034-80e5-ae0000000000&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiIwNTdkZGVkNC01MDEzLTRmOGMtYWE3OC1hMWQxOTFkYzkwMmMiLCJhcHBpZCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDA0ODE3MTBhNCIsImF1ZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMC9teS5taWNyb3NvZnRwZXJzb25hbGNvbnRlbnQuY29tQDkxODgwNDBkLTZjNjctNGM1Yi1iMTEyLTM2YTMwNGI2NmRhZCIsImV4cCI6IjE3NjI2MDk5ODEifQ.BHyHKOAL8z6HqEwIqnL7nbOCmabQVN2Ta2Xt76Q1yJYYfNCfRx1eQIXyljXro1kf2H4wXS8ylKsGBmxs5mngSbsbj0o6eVlnMK0lq8mr2q4DV0_mT8Eh2-ro-trpJQnb1H2SAQ-19nGSbqeiVUlDo3trazv51OMpuWSWilakodCitn6G6Es0gU3x07yiqYCPydPg2D_oIDGh26ujpDGLxjUFYQVh6P0nJECGe4e_g29mVEilKw0CW1ZAwRHbUAfRaakd_wG1AnhdUHjGsA9N8Jjfm2M9s3PhF0VGx1_oaX0dQhHCE6x31TQCTVRjzdHXcFchHH1E-EPOh7vIEdwBf5V8XT_1GPMmIe3WTJI0Uepi3PoBSmAzuAfeqb4Yk9WJ02pOJZHx_H63CKcwrmGC3SMWgs6gLcukFw_3ajAkPCg.2BAi-46wwT3FCCgLIn_lohtM0XMmwmZVIZTiTl0fakM&ApiVersion=2.0"
if [[ ! -f "${VITPOSE_CKPT}" || $(stat -c%s "${VITPOSE_CKPT}" 2>/dev/null || echo 0) -lt 100000000 ]]; then
  TMP_CKPT="$(mktemp)"
  curl -L "${VITPOSE_URL}" \
    -o "${TMP_CKPT}"
  mv "${TMP_CKPT}" "${VITPOSE_CKPT}"
fi

echo "[8/8] Sanity check"
python - <<'PY'
import torch, numpy as np, mmengine, mmcv, mmdet, mmpose
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("numpy:", np.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
PY

echo "vitpose environment ready."
