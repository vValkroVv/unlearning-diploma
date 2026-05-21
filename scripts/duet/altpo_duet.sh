#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/home/vkropoti/unlearning/artifacts}"
if [[ -z "${ALTPO_ARTIFACT_ROOT:-}" ]]; then
  if [[ "$(basename "${ARTIFACT_ROOT}")" == "dualcf" ]]; then
    ALTPO_ARTIFACT_ROOT="$(dirname "${ARTIFACT_ROOT}")/altpo"
  else
    ALTPO_ARTIFACT_ROOT="${ARTIFACT_ROOT}/altpo"
  fi
fi
ALTPO_REPEATS="${ALTPO_REPEATS:-5}"
ALTPO_ARTIFACT_SEED_EFFECTIVE="${ALTPO_ARTIFACT_SEED:-${TRAIN_SEED:-${SEED:-0}}}"
FORGET_LABEL="${FORGET_LABEL:-merged}"

if [[ -z "${CF_DATASET_DATA_FILES:-}" ]]; then
  case "${FORGET_LABEL}" in
    rare)
      export CF_DATASET_DATA_FILES="${ALTPO_ARTIFACT_ROOT}/duet/rare_llama31_8b/altpo_rare_alt${ALTPO_REPEATS}_seed${ALTPO_ARTIFACT_SEED_EFFECTIVE}.jsonl"
      ;;
    popular)
      export CF_DATASET_DATA_FILES="${ALTPO_ARTIFACT_ROOT}/duet/popular_llama31_8b/altpo_popular_alt${ALTPO_REPEATS}_seed${ALTPO_ARTIFACT_SEED_EFFECTIVE}.jsonl"
      ;;
    merged)
      export CF_DATASET_DATA_FILES="${ALTPO_ARTIFACT_ROOT}/duet/merged_llama31_8b/altpo_merged_alt${ALTPO_REPEATS}_seed${ALTPO_ARTIFACT_SEED_EFFECTIVE}.jsonl"
      ;;
    *)
      echo "[duet][AltPO] unsupported FORGET_LABEL=${FORGET_LABEL}" >&2
      exit 2
      ;;
  esac
fi

if [[ ! -f "${CF_DATASET_DATA_FILES}" ]]; then
  echo "[duet][AltPO] missing generated AltPO artifact: ${CF_DATASET_DATA_FILES}" >&2
  echo "Run: SEEDS='${ALTPO_ARTIFACT_SEED_EFFECTIVE}' bash scripts/altpo/prepare_altpo_artifacts.sh duet_${FORGET_LABEL}" >&2
  exit 1
fi

export METHOD_VARIANT=altpo
export CF_DATASET_PATH=json
export CF_DATASET_SPLIT="${CF_DATASET_SPLIT:-train}"

# AltPO defaults from the paper quick-start.
export BETAS="${ALTPO_BETAS:-0.1}"
export ALPHAS="${ALTPO_ALPHAS:-1.0}"
export GAMMAS="${ALTPO_GAMMAS:-1.0}"
export LRS=${LRS:-5e-5}
export NUM_EPOCHS=${NUM_EPOCHS:-2}

bash "${script_dir}/run_dualcf_ablation_v2.sh"
