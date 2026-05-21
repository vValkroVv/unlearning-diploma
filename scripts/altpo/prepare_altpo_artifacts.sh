#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "${script_dir}/../..")"
cd "${repo_root}"

target="${1:-all}"

: "${DUET_LOCAL_SFT_BASE:?Set DUET_LOCAL_SFT_BASE=/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models}"
: "${HF_BASE_MODEL_PATH:?Set HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/home/vkropoti/unlearning/artifacts}"
if [[ -z "${ALTPO_ARTIFACT_ROOT:-}" ]]; then
  if [[ "$(basename "${ARTIFACT_ROOT}")" == "dualcf" ]]; then
    ALTPO_ARTIFACT_ROOT="$(dirname "${ARTIFACT_ROOT}")/altpo"
  else
    ALTPO_ARTIFACT_ROOT="${ARTIFACT_ROOT}/altpo"
  fi
fi
ALTPO_REPEATS="${ALTPO_REPEATS:-5}"
ALTPO_MAX_NEW_TOKENS="${ALTPO_MAX_NEW_TOKENS:-200}"
ALTPO_TEMPERATURE="${ALTPO_TEMPERATURE:-1.0}"
ALTPO_BATCH_SIZE="${ALTPO_BATCH_SIZE:-1}"
ALTPO_MAX_EXAMPLES="${ALTPO_MAX_EXAMPLES:-0}"
ALTPO_PROMPT_TEMPLATE="${ALTPO_PROMPT_TEMPLATE:-llama3_altpo}"
ALTPO_TORCH_DTYPE="${ALTPO_TORCH_DTYPE:-bf16}"
ALTPO_ATTN_IMPLEMENTATION="${ALTPO_ATTN_IMPLEMENTATION:-flash_attention_2}"
ALTPO_DEVICE="${ALTPO_DEVICE:-cuda}"
ALTPO_DEVICE_MAP="${ALTPO_DEVICE_MAP:-auto}"

DUET_SFT_SUBFOLDER="${DUET_SFT_SUBFOLDER:-llama-3.1-8b-instruct-tripunlamb-ft}"
DUET_TOKENIZER_PATH="${DUET_TOKENIZER_PATH:-${DUET_LOCAL_SFT_BASE}}"
DUET_TOKENIZER_SUBFOLDER="${DUET_TOKENIZER_SUBFOLDER:-${DUET_SFT_SUBFOLDER}}"

RWKU_TOKENIZER_PATH="${TOKENIZER_MODEL_PATH:-${HF_BASE_MODEL_PATH}}"

if [[ -n "${SEEDS:-}" && -z "${ALTPO_SEEDS:-}" && -z "${ALTPO_ARTIFACT_SEED:-}" ]]; then
  echo "[altpo][prep] NOTE: ignoring training SEEDS for artifact generation; set ALTPO_SEEDS to generate multiple artifact seeds." >&2
fi
raw_seeds="${ALTPO_SEEDS:-${ALTPO_ARTIFACT_SEED:-0}}"
raw_seeds="${raw_seeds//,/ }"
raw_seeds="${raw_seeds//\"/}"
raw_seeds="${raw_seeds//\'/}"
read -r -a seeds <<< "${raw_seeds}"

mkdir -p "${ALTPO_ARTIFACT_ROOT}"

run_gen() {
  local label="$1"
  local dataset_path="$2"
  local dataset_name="$3"
  local split="$4"
  local question_key="$5"
  local model_path="$6"
  local tokenizer_path="$7"
  local model_subfolder="$8"
  local tokenizer_subfolder="$9"
  local out_dir="${10}"
  local out_prefix="${11}"

  mkdir -p "${out_dir}"

  for seed in "${seeds[@]}"; do
    local out_path="${out_dir}/${out_prefix}_alt${ALTPO_REPEATS}_seed${seed}.jsonl"
    if [[ -f "${out_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
      echo "[altpo][prep] skip existing ${label} seed=${seed}: ${out_path}"
      continue
    fi

    echo "[altpo][prep] generate ${label} seed=${seed} -> ${out_path}"
    cmd=(
      python src/tools/generate_altpo_artifacts.py
      --dataset-path "${dataset_path}"
      --split "${split}"
      --question-key "${question_key}"
      --answer-key answer
      --model-path "${model_path}"
      --tokenizer-path "${tokenizer_path}"
      --prompt-template "${ALTPO_PROMPT_TEMPLATE}"
      --output-path "${out_path}"
      --seed "${seed}"
      --repeats "${ALTPO_REPEATS}"
      --batch-size "${ALTPO_BATCH_SIZE}"
      --max-examples "${ALTPO_MAX_EXAMPLES}"
      --max-new-tokens "${ALTPO_MAX_NEW_TOKENS}"
      --temperature "${ALTPO_TEMPERATURE}"
      --do-sample
      --torch-dtype "${ALTPO_TORCH_DTYPE}"
      --attn-implementation "${ALTPO_ATTN_IMPLEMENTATION}"
      --device "${ALTPO_DEVICE}"
      --device-map "${ALTPO_DEVICE_MAP}"
      --write-compat-metadata
    )
    if [[ -n "${dataset_name}" && "${dataset_name}" != "null" ]]; then
      cmd+=(--dataset-name "${dataset_name}")
    fi
    if [[ -n "${model_subfolder}" && "${model_subfolder}" != "null" ]]; then
      cmd+=(--model-subfolder "${model_subfolder}")
    fi
    if [[ -n "${tokenizer_subfolder}" && "${tokenizer_subfolder}" != "null" ]]; then
      cmd+=(--tokenizer-subfolder "${tokenizer_subfolder}")
    fi
    "${cmd[@]}"
  done
}

run_duet_rare() {
  run_gen \
    "duet_rare" \
    "SwetieePawsss/DUET" \
    "" \
    "city_forget_rare_5" \
    "question" \
    "${DUET_LOCAL_SFT_BASE}" \
    "${DUET_TOKENIZER_PATH}" \
    "${DUET_SFT_SUBFOLDER}" \
    "${DUET_TOKENIZER_SUBFOLDER}" \
    "${ALTPO_ARTIFACT_ROOT}/duet/rare_llama31_8b" \
    "altpo_rare"
}

run_duet_popular() {
  run_gen \
    "duet_popular" \
    "SwetieePawsss/DUET" \
    "" \
    "city_forget_popular_5" \
    "question" \
    "${DUET_LOCAL_SFT_BASE}" \
    "${DUET_TOKENIZER_PATH}" \
    "${DUET_SFT_SUBFOLDER}" \
    "${DUET_TOKENIZER_SUBFOLDER}" \
    "${ALTPO_ARTIFACT_ROOT}/duet/popular_llama31_8b" \
    "altpo_popular"
}

run_duet_merged() {
  run_gen \
    "duet_merged" \
    "SwetieePawsss/DUET" \
    "" \
    "city_forget_rare_5+city_forget_popular_5" \
    "question" \
    "${DUET_LOCAL_SFT_BASE}" \
    "${DUET_TOKENIZER_PATH}" \
    "${DUET_SFT_SUBFOLDER}" \
    "${DUET_TOKENIZER_SUBFOLDER}" \
    "${ALTPO_ARTIFACT_ROOT}/duet/merged_llama31_8b" \
    "altpo_merged"
}

run_rwku() {
  run_gen \
    "rwku_level2" \
    "SwetieePawsss/exp_r" \
    "forget_level2" \
    "test" \
    "query" \
    "${HF_BASE_MODEL_PATH}" \
    "${RWKU_TOKENIZER_PATH}" \
    "" \
    "" \
    "${ALTPO_ARTIFACT_ROOT}/rwku/llama31_8b_level2" \
    "altpo_forget_level2"
}

case "${target}" in
  duet_rare|rare) run_duet_rare ;;
  duet_popular|popular) run_duet_popular ;;
  duet_merged|merged) run_duet_merged ;;
  rwku|rwku_level2) run_rwku ;;
  duet) run_duet_rare; run_duet_popular; run_duet_merged ;;
  all) run_duet_rare; run_duet_popular; run_duet_merged; run_rwku ;;
  *)
    echo "Usage: bash scripts/altpo/prepare_altpo_artifacts.sh [duet_rare|duet_popular|duet_merged|duet|rwku|all]" >&2
    exit 2
    ;;
esac
