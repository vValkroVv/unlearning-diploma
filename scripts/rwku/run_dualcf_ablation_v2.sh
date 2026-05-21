#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

METHOD_VARIANT=${METHOD_VARIANT:-full}

case "${METHOD_VARIANT}" in
  full)
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  d_only)
    export DISABLE_ATTRIBUTION_ROUTES=${DISABLE_ATTRIBUTION_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  a_only)
    export DISABLE_DIFFICULTY_ROUTES=${DISABLE_DIFFICULTY_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  dpo)
    export TRAINER=${TRAINER:-DPO}
    export METHOD_NAME=${METHOD_NAME:-dpo_cf}
    export RUN_LABEL=${RUN_LABEL:-DPO}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  general_cf)
    export TRAINER=${TRAINER:-GeneralCF}
    export METHOD_NAME=${METHOD_NAME:-general_cf}
    export RUN_LABEL=${RUN_LABEL:-GeneralCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/general_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  simple_ce)
    exec bash "${script_dir}/simple_ce_rwku.sh"
    ;;
  multicf)
    export TRAINER=${TRAINER:-MultiCF}
    export METHOD_NAME=${METHOD_NAME:-multicf}
    export RUN_LABEL=${RUN_LABEL:-MultiCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/multicf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  boundary_cf)
    export TRAINER=${TRAINER:-BoundaryCF}
    export METHOD_NAME=${METHOD_NAME:-boundary_cf}
    export RUN_LABEL=${RUN_LABEL:-BoundaryCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/boundary_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf)
    export TRAINER=${TRAINER:-SpanCF}
    export METHOD_NAME=${METHOD_NAME:-span_cf}
    export RUN_LABEL=${RUN_LABEL:-SpanCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_simnpo)
    export TRAINER=${TRAINER:-SpanCFSimNPO}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPO}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_simnpo_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_local_retain)
    export TRAINER=${TRAINER:-SpanCFLocalRetain}
    export METHOD_NAME=${METHOD_NAME:-span_cf_local_retain}
    export RUN_LABEL=${RUN_LABEL:-SpanCFLocalRetain}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_local_retain_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_samnpo)
    export TRAINER=${TRAINER:-SpanCFSAMNPO}
    export METHOD_NAME=${METHOD_NAME:-span_cf_samnpo}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSAMNPO}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_samnpo_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_simnpo_local_retain)
    export TRAINER=${TRAINER:-SpanCFSimNPOLocalRetain}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_local_retain}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOLocalRetain}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_simnpo_local_retain_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_simnpo_sam)
    export TRAINER=${TRAINER:-SpanCFSimNPOSAM}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_sam}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOSAM}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_simnpo_sam_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf_simnpo_projected)
    export TRAINER=${TRAINER:-SpanCFSimNPOProjected}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_projected}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOProjected}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_simnpo_projected_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  altpo)
    export TRAINER=DPO
    export METHOD_NAME=altpo
    export RUN_LABEL=altpo
    export EXPERIMENT=unlearn/rwku/altpo_lora.yaml
    export CF_DATASET_PATH="${CF_DATASET_PATH:-json}"
    export CF_DATASET_SPLIT="${CF_DATASET_SPLIT:-train}"
    export BETAS="${ALTPO_BETAS:-0.1}"
    export ALPHAS="${ALTPO_ALPHAS:-1.0}"
    export GAMMAS="${ALTPO_GAMMAS:-1.0}"
    if [[ -z "${CF_DATASET_DATA_FILES:-}" ]]; then
      echo "[rwku][AltPO] ERROR: CF_DATASET_DATA_FILES must point to generated AltPO JSONL." >&2
      exit 1
    fi
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  ga)
    exec bash "${script_dir}/ga_rwku.sh"
    ;;
  grad_diff|gd)
    exec bash "${script_dir}/gd_rwku.sh"
    ;;
  ceu)
    exec bash "${script_dir}/ceu_rwku.sh"
    ;;
  pdu)
    exec bash "${script_dir}/pdu_rwku.sh"
    ;;
  idk_dpo)
    exec bash "${script_dir}/idk_dpo_rwku.sh"
    ;;
  ada_pop)
    exec bash "${script_dir}/ada_pop_rwku.sh"
    ;;
  npo)
    exec bash "${script_dir}/npo_rwku.sh"
    ;;
  simnpo)
    exec bash "${script_dir}/simnpo_rwku.sh"
    ;;
  tpo)
    exec bash "${script_dir}/tpo_rwku.sh"
    ;;
  adaptive_rmu)
    exec bash "${script_dir}/adaptive_rmu_rwku.sh"
    ;;
  flat)
    exec bash "${script_dir}/flat_rwku.sh"
    ;;
  undial)
    exec bash "${script_dir}/undial_rwku.sh"
    ;;
  rmu)
    exec bash "${script_dir}/rmu_rwku.sh"
    ;;
  wga)
    exec bash "${script_dir}/wga_rwku.sh"
    ;;
  unilogit)
    exec bash "${script_dir}/unilogit_rwku.sh"
    ;;
  stat)
    exec bash "${script_dir}/stat_rwku.sh"
    ;;
  satimp)
    exec bash "${script_dir}/satimp_rwku.sh"
    ;;
  npo_sam)
    exec bash "${script_dir}/npo_sam_rwku.sh"
    ;;
  loku)
    exec bash "${script_dir}/loku_rwku.sh"
    ;;
  *)
    echo "[run_dualcf_ablation_v2] Unsupported METHOD_VARIANT=${METHOD_VARIANT}" >&2
    exit 1
    ;;
esac
