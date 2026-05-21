#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//\//_}"
  value="${value//:/_}"
  echo "${value}"
}

resolve_num_rows() {
  local dataset_name="$1"
  python - <<PY
import datasets
name = ${dataset_name@Q}
ds = datasets.load_dataset("SwetieePawsss/exp_r", name=name, split="test")
print(len(ds))
PY
}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

method_name="${METHOD_NAME:?set METHOD_NAME}"
run_label="${RUN_LABEL:-${method_name}}"
trainer="${TRAINER:?set TRAINER}"
experiment="${EXPERIMENT:?set EXPERIMENT}"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${base_model_path}}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[rwku][${run_label}] Using Hugging Face base checkpoint ${base_model_path}"

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/rwku/${method_name}}"
mkdir -p "${output_root}"

forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-16}
gradient_accumulation_steps=${GRAD_ACCUM:-2}
eval_batch_size=${EVAL_BATCH_SIZE:-512}
num_train_epochs=${NUM_EPOCHS:-5}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
max_steps="${MAX_STEPS:-0}"
checkpoint_every_half_epoch="${CHECKPOINT_EVERY_HALF_EPOCH:-0}"
save_total_limit="${SAVE_TOTAL_LIMIT:-2}"
checkpoint_epochs_raw="${CHECKPOINT_EPOCHS:-2}"
checkpoint_epochs_csv=""
if [[ -n "${checkpoint_epochs_raw}" ]]; then
  checkpoint_epochs_raw="${checkpoint_epochs_raw//,/ }"
  checkpoint_epochs_raw="${checkpoint_epochs_raw//\"/}"
  checkpoint_epochs_raw="${checkpoint_epochs_raw//\'/}"
  read -r -a checkpoint_epochs <<< "${checkpoint_epochs_raw}"
  checkpoint_epochs_csv=$(IFS=,; echo "${checkpoint_epochs[*]}")
fi
run_tag_extra="${RUN_TAG_EXTRA:-}"

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

if [[ "${trainer}" == "TPO" ]]; then
  raw_alphas="${TPO_ALPHAS:-${ALPHAS:-1.0}}"
else
  raw_alphas="${ALPHAS:-${GRAD_DIFF_ALPHAS:-1.0}}"
fi
raw_alphas="${raw_alphas//,/ }"
raw_alphas="${raw_alphas//\"/}"
raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

if [[ "${trainer}" == "TPO" ]]; then
  raw_gammas="${TPO_GAMMAS:-${GAMMAS:-0.1}}"
else
  raw_gammas="${GAMMAS:-${GRAD_DIFF_GAMMAS:-1.0}}"
fi
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

raw_ceu_ignore_first_ns="${CEU_IGNORE_FIRST_NS:-1}"
raw_ceu_ignore_first_ns="${raw_ceu_ignore_first_ns//,/ }"
read -r -a ceu_ignore_first_ns <<< "${raw_ceu_ignore_first_ns}"

raw_pdu_retain_loss_eps="${PDU_RETAIN_LOSS_EPS_VALUES:-${PDU_RETAIN_LOSS_EPS:-1.0}}"
raw_pdu_retain_loss_eps="${raw_pdu_retain_loss_eps//,/ }"
raw_pdu_retain_loss_eps="${raw_pdu_retain_loss_eps//\"/}"
raw_pdu_retain_loss_eps="${raw_pdu_retain_loss_eps//\'/}"
read -r -a pdu_retain_loss_eps_values <<< "${raw_pdu_retain_loss_eps}"

raw_pdu_dual_step_sizes="${PDU_DUAL_STEP_SIZES:-${PDU_DUAL_STEP_SIZE:-0.05}}"
raw_pdu_dual_step_sizes="${raw_pdu_dual_step_sizes//,/ }"
raw_pdu_dual_step_sizes="${raw_pdu_dual_step_sizes//\"/}"
raw_pdu_dual_step_sizes="${raw_pdu_dual_step_sizes//\'/}"
read -r -a pdu_dual_step_sizes <<< "${raw_pdu_dual_step_sizes}"

pdu_primal_dual="${PDU_PRIMAL_DUAL:-true}"
pdu_dual_update_upon="${PDU_DUAL_UPDATE_UPON:-step}"
pdu_dual_warmup_epochs="${PDU_DUAL_WARMUP_EPOCHS:-0}"

raw_tpo_betas="${TPO_BETAS:-${BETAS:-0.2}}"
raw_tpo_betas="${raw_tpo_betas//,/ }"
raw_tpo_betas="${raw_tpo_betas//\"/}"
raw_tpo_betas="${raw_tpo_betas//\'/}"
read -r -a tpo_betas <<< "${raw_tpo_betas}"

raw_tpo_pl_coeffs="${TPO_PL_COEFFS:-1.0}"
raw_tpo_pl_coeffs="${raw_tpo_pl_coeffs//,/ }"
raw_tpo_pl_coeffs="${raw_tpo_pl_coeffs//\"/}"
raw_tpo_pl_coeffs="${raw_tpo_pl_coeffs//\'/}"
read -r -a tpo_pl_coeffs <<< "${raw_tpo_pl_coeffs}"

tpo_identifier_mode="${TPO_IDENTIFIER_MODE:-stopword}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for lr in "${lrs[@]}"; do
  for lora_r in "${lora_rs[@]}"; do
    for lora_alpha in "${lora_alphas[@]}"; do
      for lora_dropout in "${lora_dropouts[@]}"; do
        dropout_tag=$(tag_value "${lora_dropout}")

        if [[ "${trainer}" == "CEU" ]]; then
          alpha_values=("none")
          gamma_values=("none")
          ceu_values=("${ceu_ignore_first_ns[@]}")
          pdu_eps_values=("none")
          pdu_step_values=("none")
          tpo_beta_values=("none")
          tpo_pl_values=("none")
        elif [[ "${trainer}" == "PDU" ]]; then
          alpha_values=("${alphas[@]}")
          gamma_values=("${gammas[@]}")
          ceu_values=("none")
          pdu_eps_values=("${pdu_retain_loss_eps_values[@]}")
          pdu_step_values=("${pdu_dual_step_sizes[@]}")
          tpo_beta_values=("none")
          tpo_pl_values=("none")
        elif [[ "${trainer}" == "TPO" ]]; then
          alpha_values=("${alphas[@]}")
          gamma_values=("${gammas[@]}")
          ceu_values=("none")
          pdu_eps_values=("none")
          pdu_step_values=("none")
          tpo_beta_values=("${tpo_betas[@]}")
          tpo_pl_values=("${tpo_pl_coeffs[@]}")
        else
          alpha_values=("${alphas[@]}")
          gamma_values=("${gammas[@]}")
          ceu_values=("none")
          pdu_eps_values=("none")
          pdu_step_values=("none")
          tpo_beta_values=("none")
          tpo_pl_values=("none")
        fi

        for alpha in "${alpha_values[@]}"; do
          for gamma in "${gamma_values[@]}"; do
            for ceu_ignore in "${ceu_values[@]}"; do
              for pdu_eps in "${pdu_eps_values[@]}"; do
                for pdu_step in "${pdu_step_values[@]}"; do
                  for tpo_beta in "${tpo_beta_values[@]}"; do
                    for tpo_pl in "${tpo_pl_values[@]}"; do
                      method_suffix=""
                      extra_method_args=()
                      if [[ "${trainer}" == "CEU" ]]; then
                        method_suffix="ign$(tag_value "${ceu_ignore}")"
                        extra_method_args+=(trainer.method_args.ignore_first_n_answer_tokens=${ceu_ignore})
                      elif [[ "${trainer}" == "PDU" ]]; then
                        alpha_tag=$(tag_value "${alpha}")
                        gamma_tag=$(tag_value "${gamma}")
                        eps_tag=$(tag_value "${pdu_eps}")
                        step_tag=$(tag_value "${pdu_step}")
                        method_suffix="alpha${alpha_tag}_gamma${gamma_tag}_eps${eps_tag}_ds${step_tag}_du${pdu_dual_update_upon}_dw${pdu_dual_warmup_epochs}"
                        extra_method_args+=(
                          trainer.method_args.alpha=${alpha}
                          trainer.method_args.gamma=${gamma}
                          trainer.method_args.retain_loss_type=NLL
                          trainer.method_args.retain_loss_eps=${pdu_eps}
                          trainer.method_args.primal_dual=${pdu_primal_dual}
                          trainer.method_args.dual_step_size=${pdu_step}
                          trainer.method_args.dual_update_upon=${pdu_dual_update_upon}
                          trainer.method_args.dual_warmup_epochs=${pdu_dual_warmup_epochs}
                        )
                      elif [[ "${trainer}" == "TPO" ]]; then
                        alpha_tag=$(tag_value "${alpha}")
                        gamma_tag=$(tag_value "${gamma}")
                        beta_tag=$(tag_value "${tpo_beta}")
                        pl_tag=$(tag_value "${tpo_pl}")
                        id_tag=$(tag_value "${tpo_identifier_mode}")
                        method_suffix="beta${beta_tag}_pl${pl_tag}_alpha${alpha_tag}_gamma${gamma_tag}_id${id_tag}"
                        extra_method_args+=(
                          trainer.method_args.beta=${tpo_beta}
                          trainer.method_args.pl_coeff=${tpo_pl}
                          trainer.method_args.alpha=${alpha}
                          trainer.method_args.gamma=${gamma}
                          trainer.method_args.retain_loss_type=NLL
                          trainer.method_args.identifier_mode=${tpo_identifier_mode}
                        )
                      else
                        alpha_tag=$(tag_value "${alpha}")
                        gamma_tag=$(tag_value "${gamma}")
                        method_suffix="alpha${alpha_tag}_gamma${gamma_tag}"
                        extra_method_args+=(
                          trainer.method_args.alpha=${alpha}
                          trainer.method_args.gamma=${gamma}
                          trainer.method_args.retain_loss_type=NLL
                        )
                      fi

                      task_name=rwku_${base_model}_${forget_split}_${method_name}_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_${method_suffix}
                      if [[ -n "${run_tag_extra}" ]]; then
                        task_name="${task_name}_${run_tag_extra}"
                      fi
                      run_dir=${output_root}/${task_name}
                      eval_dir=${run_dir}/evals
                      summary_path=${eval_dir}/DUET_SUMMARY.json

                      if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                        echo "[rwku][${run_label}] Skipping ${task_name}: found existing summary at ${summary_path}"
                        continue
                      fi

                      echo "[rwku][${run_label}] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                      adapter_path=${run_dir}/adapter_model.safetensors
                      if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                        mkdir -p "${run_dir}"
                        extra_schedule_args=()
                        if [[ -n "${checkpoint_epochs_csv}" ]]; then
                          extra_schedule_args+=(
                            ++trainer.args.save_strategy=no
                            ++trainer.args.save_total_limit=${save_total_limit}
                            ++trainer.args.save_safetensors=true
                            ++trainer.save_on_epochs=[${checkpoint_epochs_csv}]
                          )
                        elif [[ "${checkpoint_every_half_epoch}" == "1" && "${max_steps}" == "0" ]]; then
                          num_rows=$(resolve_num_rows "${forget_split}")
                          global_batch=$(( per_device_train_batch_size * gradient_accumulation_steps ))
                          steps_per_epoch=$(( (num_rows + global_batch - 1) / global_batch ))
                          half_epoch_steps=$(( (steps_per_epoch + 1) / 2 ))
                          logging_steps=$(( half_epoch_steps / 2 ))
                          if [[ "${logging_steps}" -lt 1 ]]; then
                            logging_steps=1
                          fi
                          extra_schedule_args+=(
                            ++trainer.args.save_strategy=steps
                            ++trainer.args.save_steps=${half_epoch_steps}
                            ++trainer.args.save_total_limit=${save_total_limit}
                            ++trainer.args.logging_strategy=steps
                            ++trainer.args.logging_steps=${logging_steps}
                            ++trainer.args.save_safetensors=true
                            ++trainer.args.load_best_model_at_end=false
                          )
                        fi

                        train_cmd=(
                          src/train.py
                          --config-name=unlearn.yaml
                          experiment=${experiment}
                          trainer=${trainer}
                          task_name=${task_name}
                          model=${lora_model}
                          forget_split=${forget_split}
                          retain_split=${retain_split}
                          model.model_args.pretrained_model_name_or_path=${base_model_path}
                          model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                          model.model_args.device_map="auto"
                          ++model.model_args.low_cpu_mem_usage=true
                          model.lora_config.r=${lora_r}
                          model.lora_config.lora_alpha=${lora_alpha}
                          model.lora_config.lora_dropout=${lora_dropout}
                          ++trainer.args.seed=${TRAIN_SEED:-42}
                          ++trainer.args.data_seed=${DATA_SEED:-${TRAIN_SEED:-42}}
                          trainer.args.per_device_train_batch_size=${per_device_train_batch_size}
                          trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}
                          trainer.args.num_train_epochs=${num_train_epochs}
                          trainer.args.gradient_checkpointing=${gradient_checkpointing}
                          trainer.args.learning_rate=${lr}
                          +trainer.trace_jsonl=true
                          retain_logs_path=null
                          "${extra_method_args[@]}"
                          "${extra_schedule_args[@]}"
                          paths.output_dir=${run_dir}
                        )
                        if [[ "${max_steps}" != "0" ]]; then
                          train_cmd+=(+trainer.args.max_steps=${max_steps})
                        fi
                        if [[ "${FULL_DETERMINISM:-0}" == "1" ]]; then
                          train_cmd+=(++trainer.args.full_determinism=true)
                        fi
                        python "${train_cmd[@]}"
                      fi

                      mkdir -p "${eval_dir}"
                      if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                        rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                      fi

                      eval_cmd=(
                        experiment=eval/rwku/default.yaml
                        model=${lora_model}
                        forget_split=${forget_split}
                        holdout_split=${retain_split}
                        task_name=${task_name}
                        model.model_args.pretrained_model_name_or_path=${run_dir}
                        ++model.model_args.base_model_name_or_path=${base_model_path}
                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                        model.model_args.device_map="auto"
                        ++model.model_args.low_cpu_mem_usage=true
                        model.lora_config.r=${lora_r}
                        model.lora_config.lora_alpha=${lora_alpha}
                        model.lora_config.lora_dropout=${lora_dropout}
                        eval.duet.batch_size=${eval_batch_size}
                        eval.duet.overwrite=true
                        paths.output_dir=${eval_dir}
                        retain_logs_path=null
                      )
                      python src/eval.py "${eval_cmd[@]}"

                      if [[ "${run_checkpoint_eval}" == "1" ]]; then
                        bash "${script_dir}/eval_checkpoints_rwku.sh" \
                          "${run_dir}" \
                          "${forget_split}" \
                          "${retain_split}" \
                          "${base_model_path}" \
                          "${tokenizer_model_path}" \
                          "${lora_model}" \
                          "${base_model}"
                      fi

                      if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                          rm -f "${run_dir}"/*.safetensors
                          echo "[rwku][${run_label}] Removed safetensors from ${run_dir}"
                        fi
                      fi
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
