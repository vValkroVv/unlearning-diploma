#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/home/vkropoti/unlearning/artifacts/dualcf}"
if [[ -z "${IDK_DPO_ARTIFACT_ROOT:-}" ]]; then
  if [[ "$(basename "${ARTIFACT_ROOT}")" == "dualcf" ]]; then
    IDK_DPO_ARTIFACT_ROOT="$(dirname "${ARTIFACT_ROOT}")/idk_dpo"
  else
    IDK_DPO_ARTIFACT_ROOT="${ARTIFACT_ROOT}/idk_dpo"
  fi
fi

FORGET_LABEL="${FORGET_LABEL:-rare}"
if [[ -z "${CF_DATASET_DATA_FILES:-}" ]]; then
  case "${FORGET_LABEL}" in
    rare|popular|merged)
      export CF_DATASET_DATA_FILES="${IDK_DPO_ARTIFACT_ROOT}/duet/${FORGET_LABEL}_llama31_8b_v2/idk_dpo_${FORGET_LABEL}_v1.jsonl"
      ;;
    *)
      echo "[duet][IdkDPO] Unsupported FORGET_LABEL=${FORGET_LABEL}" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -f "${CF_DATASET_DATA_FILES}" ]]; then
  echo "[duet][IdkDPO] missing IdkDPO artifact: ${CF_DATASET_DATA_FILES}" >&2
  echo "Build it with src/tools/build_idk_dpo_artifact.py." >&2
  exit 1
fi

export TRAINER=DPO
export METHOD_NAME=idk_dpo
export RUN_LABEL=IdkDPO
export EXPERIMENT=unlearn/duet/idk_dpo_lora.yaml
export CF_DATASET_PATH="${CF_DATASET_PATH:-json}"
export CF_DATASET_SPLIT="${CF_DATASET_SPLIT:-train}"
export BETAS="${IDK_DPO_BETAS:-${BETAS:-0.1}}"
export ALPHAS="${IDK_DPO_ALPHAS:-${ALPHAS:-1.0}}"
export GAMMAS="${IDK_DPO_GAMMAS:-${GAMMAS:-1.0}}"

bash "${script_dir}/dual_cf_duet.sh"
