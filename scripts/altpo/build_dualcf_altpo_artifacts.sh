#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/altpo/build_dualcf_altpo_artifacts.sh DUALCF_OR_ARTIFACTS_ROOT ALTPO_ROOT OUTPUT_ROOT [TARGET]

Example:
  ALTPO_ARTIFACT_SEED=0 ALTPO_REPEATS=5 \
  bash scripts/altpo/build_dualcf_altpo_artifacts.sh \
    /data/home/vkropoti/unlearning/artifacts \
    /data/home/vkropoti/unlearning/artifacts/altpo \
    /data/home/vkropoti/unlearning/artifacts-dualcf-altpo \
    all

Targets:
  duet_rare, duet_popular, duet_merged, duet, rwku, all

Notes:
  - The first argument may be either the parent artifacts root or artifacts/dualcf.
  - ALTPO_ARTIFACT_SEED defaults to 0 and is independent from training SEEDS.
  - Set ALTPO_SEEDS for an intentional multi-artifact-seed build.
  - Multi-seed builds write under OUTPUT_ROOT/seed<seed>/...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "${script_dir}/../..")"
cd "${repo_root}"

artifacts_arg="$1"
altpo_root="$2"
if [[ $# -ge 3 ]]; then
  output_root="$3"
else
  if [[ "$(basename "${artifacts_arg}")" == "dualcf" ]]; then
    output_root="$(dirname "$(dirname "${artifacts_arg}")")/artifacts-dualcf-altpo"
  else
    output_root="$(dirname "${artifacts_arg}")/artifacts-dualcf-altpo"
  fi
fi
target="${4:-all}"

resolve_dualcf_root() {
  local root="$1"
  if [[ -f "${root}/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl" ]]; then
    printf '%s\n' "${root}"
    return
  fi
  if [[ -d "${root}/dualcf" ]]; then
    printf '%s\n' "${root}/dualcf"
    return
  fi
  if [[ -f "${root}/dualcf/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl" ]]; then
    printf '%s\n' "${root}/dualcf"
    return
  fi
  printf '%s\n' "${root}"
}

dualcf_root="$(resolve_dualcf_root "${artifacts_arg}")"

if [[ -n "${SEEDS:-}" && -z "${ALTPO_SEEDS:-}" && -z "${ALTPO_ARTIFACT_SEED:-}" ]]; then
  echo "[dualcf-altpo] NOTE: ignoring training SEEDS for artifact composition; set ALTPO_SEEDS for multiple artifact seeds." >&2
fi
raw_seeds="${ALTPO_SEEDS:-${ALTPO_ARTIFACT_SEED:-0}}"
raw_seeds="${raw_seeds//,/ }"
raw_seeds="${raw_seeds//\"/}"
raw_seeds="${raw_seeds//\'/}"
read -r -a seeds <<< "${raw_seeds}"
if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "[dualcf-altpo] No seeds parsed from ALTPO_ARTIFACT_SEED/ALTPO_SEEDS." >&2
  exit 2
fi

altpo_repeats="${ALTPO_REPEATS:-5}"
altpo_repeat_select="${ALTPO_REPEAT_SELECT:-0}"
layout="${DUALCF_ALTPO_LAYOUT:-}"
validate="${VALIDATE:-1}"
validate_strict="${VALIDATE_STRICT:-1}"
reject_gold_substring="${REJECT_GOLD_SUBSTRING:-0}"
max_overlap_ratio="${MAX_OVERLAP_RATIO:-}"

if [[ -z "${layout}" ]]; then
  if [[ ${#seeds[@]} -eq 1 ]]; then
    layout="flat"
  else
    layout="seed_subdirs"
  fi
fi

if [[ "${layout}" == "flat" && ${#seeds[@]} -ne 1 ]]; then
  echo "[dualcf-altpo] DUALCF_ALTPO_LAYOUT=flat requires exactly one seed." >&2
  exit 2
fi

mkdir -p "${output_root}"

compose_one() {
  local seed="$1"
  local label="$2"
  local question_key="$3"
  local dualcf_rel="$4"
  local altpo_rel="$5"

  local seed_root="${output_root}/seed${seed}"
  if [[ "${layout}" == "flat" ]]; then
    seed_root="${output_root}"
  fi

  local dualcf_path="${dualcf_root}/${dualcf_rel}"
  local altpo_path="${altpo_root}/${altpo_rel}"
  local output_path="${seed_root}/${dualcf_rel}"

  if [[ ! -f "${dualcf_path}" ]]; then
    echo "[dualcf-altpo] missing DualCF artifact for ${label}: ${dualcf_path}" >&2
    exit 1
  fi
  if [[ ! -f "${altpo_path}" ]]; then
    echo "[dualcf-altpo] missing AltPO artifact for ${label}: ${altpo_path}" >&2
    exit 1
  fi

  echo "[dualcf-altpo] compose ${label} seed=${seed}"
  cmd=(
    python src/tools/build_dualcf_altpo_artifact.py
    --dualcf-path "${dualcf_path}"
    --altpo-path "${altpo_path}"
    --output-path "${output_path}"
    --question-key "${question_key}"
    --repeats "${altpo_repeats}"
    --altpo-repeat "${altpo_repeat_select}"
  )
  if [[ "${reject_gold_substring}" == "1" ]]; then
    cmd+=(--reject-gold-substring)
  fi
  if [[ -n "${max_overlap_ratio}" ]]; then
    cmd+=(--max-overlap-ratio "${max_overlap_ratio}")
  fi
  "${cmd[@]}"

  if [[ "${validate}" == "1" ]]; then
    validate_cmd=(
      python src/tools/validate_dual_cf_artifact.py
      --artifact-path "${output_path}"
      --question-key "${question_key}"
    )
    if [[ "${validate_strict}" == "1" ]]; then
      validate_cmd+=(--strict)
    fi
    "${validate_cmd[@]}"
  fi
}

run_duet_rare() {
  local seed="$1"
  compose_one \
    "${seed}" \
    "duet_rare" \
    "question" \
    "duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl" \
    "duet/rare_llama31_8b/altpo_rare_alt${altpo_repeats}_seed${seed}.jsonl"
}

run_duet_popular() {
  local seed="$1"
  compose_one \
    "${seed}" \
    "duet_popular" \
    "question" \
    "duet/popular_llama31_8b_v2/dualcf_popular_v2.jsonl" \
    "duet/popular_llama31_8b/altpo_popular_alt${altpo_repeats}_seed${seed}.jsonl"
}

run_duet_merged() {
  local seed="$1"
  compose_one \
    "${seed}" \
    "duet_merged" \
    "question" \
    "duet/merged_llama31_8b_v2/dualcf_merged_v2.jsonl" \
    "duet/merged_llama31_8b/altpo_merged_alt${altpo_repeats}_seed${seed}.jsonl"
}

run_rwku() {
  local seed="$1"
  compose_one \
    "${seed}" \
    "rwku_level2" \
    "query" \
    "rwku/llama31_8b_level2_v2/dualcf_forget_level2_v2.jsonl" \
    "rwku/llama31_8b_level2/altpo_forget_level2_alt${altpo_repeats}_seed${seed}.jsonl"
}

run_target_for_seed() {
  local seed="$1"
  case "${target}" in
    duet_rare|rare) run_duet_rare "${seed}" ;;
    duet_popular|popular) run_duet_popular "${seed}" ;;
    duet_merged|merged) run_duet_merged "${seed}" ;;
    duet) run_duet_rare "${seed}"; run_duet_popular "${seed}"; run_duet_merged "${seed}" ;;
    rwku|rwku_level2) run_rwku "${seed}" ;;
    all) run_duet_rare "${seed}"; run_duet_popular "${seed}"; run_duet_merged "${seed}"; run_rwku "${seed}" ;;
    *)
      echo "[dualcf-altpo] Unsupported target=${target}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

echo "[dualcf-altpo] dualcf_root=${dualcf_root}"
echo "[dualcf-altpo] altpo_root=${altpo_root}"
echo "[dualcf-altpo] output_root=${output_root}"
echo "[dualcf-altpo] seeds=${seeds[*]} repeats=${altpo_repeats} repeat_select=${altpo_repeat_select} layout=${layout}"

for seed in "${seeds[@]}"; do
  run_target_for_seed "${seed}"
done

if [[ "${layout}" == "seed_subdirs" ]]; then
  echo "[dualcf-altpo] done. Multi-artifact-seed build wrote ARTIFACT_ROOT=${output_root}/seed<artifact_seed>."
else
  echo "[dualcf-altpo] done. Use ARTIFACT_ROOT=${output_root}."
fi
