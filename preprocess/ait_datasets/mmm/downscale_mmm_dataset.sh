#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: downscale_mmm_dataset.sh DATASET_ROOT [--org-factor N] [--stage-factor M]

  DATASET_ROOT  Path to a sequence directory that contains org_image/, stage_img/, cameras/, opt_cam/.
  --org-factor  Downscale factor for org_image/ + opt_cam/ (default: 2).
  --stage-factor
                Downscale factor for stage_img/ + cameras/ (default: 4).

Example:
  ./downscale_mmm_dataset.sh /scratch/.../mmm/dance --org-factor 2 --stage-factor 4
USAGE
}

DATASET_ROOT=""
ORG_FACTOR=2
STAGE_FACTOR=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --org-factor)
      if [[ $# -lt 2 ]]; then
        echo "--org-factor requires an integer argument" >&2
        exit 1
      fi
      ORG_FACTOR=$2
      shift 2
      ;;
    --stage-factor)
      if [[ $# -lt 2 ]]; then
        echo "--stage-factor requires an integer argument" >&2
        exit 1
      fi
      STAGE_FACTOR=$2
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$DATASET_ROOT" ]]; then
        DATASET_ROOT=$1
        shift
      else
        echo "Multiple dataset roots provided: $DATASET_ROOT and $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/downscale_mmm_dataset.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

conda run -n thesis python "$PY_SCRIPT" "$DATASET_ROOT" \
  --org-factor "$ORG_FACTOR" \
  --stage-factor "$STAGE_FACTOR"
