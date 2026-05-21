# Production Gemma GPU Runs

This runbook runs the 19 requested methods on `gemma-7b-it`, one method at a
time.

Important artifact rule:

- `DPO` uses the existing DualCF artifact paths under
  `/data/home/vkropoti/unlearning/artifacts/dualcf/*llama31_8b*`.
- `AltPO` uses the existing AltPO artifact paths under
  `/data/home/vkropoti/unlearning/artifacts/altpo/*llama31_8b*`.
- These artifact paths are intentionally Llama-named because the preference /
  counterfactual rows are treated as model-independent training artifacts.

Gemma uses one model family for all phases. DUET uses the Gemma DUET SFT
subfolder, while RWKU uses the Gemma base model path directly through the
campaign wrapper.

## Common setup

Run this once per shell.

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss \
  /home/vkropoti/diploma/open-unlearning/SwetieePawsss

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
export TMPDIR=/data/home/vkropoti/unlearning/tmp
export XDG_CACHE_HOME=/data/home/vkropoti/unlearning/.cache
export PYTHONDONTWRITEBYTECODE=1
export ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf
export ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo
export IDK_DPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/idk_dpo
export OUTPUT_ROOT=/data/home/vkropoti/unlearning/saves/unlearn-gemma
export UTILITY=${UTILITY:-3k}
export UTILITY_ROOT=${UTILITY_ROOT:-/data/home/vkropoti/unlearning/evals/utility_3k_v1}
export BASELINE_CACHE_ROOT=/data/home/vkropoti/unlearning/saves/eval/utility_baselines
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" \
  "$ARTIFACT_ROOT" "$ALTPO_ARTIFACT_ROOT" "$OUTPUT_ROOT" \
  "$UTILITY_ROOT" "$BASELINE_CACHE_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export BASE_MODEL=gemma-7b-it
export MODEL_CONFIG=gemma-7b-it-lora
export MODEL_CFG=configs/model/gemma-7b-it.yaml
export LORA_MODEL_CFG=configs/model/gemma-7b-it-lora.yaml
export HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it
export BASE_MODEL_PATH=${HF_BASE_MODEL_PATH}
export BASE_MODEL_EVAL_CONFIG=gemma-7b-it
export LORA_MODEL_EVAL_CONFIG=gemma-7b-it-lora
export DUET_LOCAL_SFT_BASE=/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models
export DUET_SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft

export LORA_RS=32
export LORA_ALPHAS=64
export LORA_DROPOUTS=0.0

export NUM_EPOCHS=5
export GRADIENT_CHECKPOINTING=false
export CHECKPOINT_EVERY_HALF_EPOCH=0
export CHECKPOINT_EPOCHS=2
export SAVE_TOTAL_LIMIT=2
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1
export DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=1
export RUN_CHECKPOINT_EVAL=1
export RUN_UTILITY_EVAL=1
export EVAL_RUN_BASE_MODEL=0
export EVAL_BATCH_SIZE=512
export UTILITY_EVAL_BATCH_SIZE=512
export UTILITY_APPLY_CHAT_TEMPLATE=true
export PER_DEVICE_TRAIN_BS=16
export GRAD_ACCUM=2

export ALTPO_ARTIFACT_SEED=${ALTPO_ARTIFACT_SEED:-0}
export ALTPO_REPEATS=${ALTPO_REPEATS:-5}

export_if_set() {
  for name in "$@"; do
    if [[ -n "${!name+x}" ]]; then
      export "${name}"
    else
      unset "${name}"
    fi
  done
}

export_method_env() {
  export_if_set \
    SEEDS METHOD_VARIANTS METHOD_NAME RUN_LABEL FORCE_RERUN MAX_STEPS LR \
    BETAS ALPHAS GAMMAS DELTAS ALPHA_CONST BETA_CONST BETA_A BETA_B \
    ADDITIONAL_LOSS ROUTING SPAN_ADDITIONAL SPAN_CF_BRANCH \
    DISABLE_RARITY_ROUTES DISABLE_DIFFICULTY_ROUTES DISABLE_ATTRIBUTION_ROUTES \
    RARITY_NEG_GAINS RARITY_CF_GAINS \
    SPAN_MODE SPAN_ALT_SHARED_TOKEN_WEIGHT SPAN_ALT_UNIQUE_TOKEN_WEIGHT \
    SPAN_ORIG_SHARED_TOKEN_WEIGHT SPAN_ORIG_UNIQUE_TOKEN_WEIGHT \
    SPAN_SAM_RHO SPAN_SAM_ADAPTIVE \
    SAM_RHOS SAM_ADAPTIVES SAM_EPS \
    FORGET_COEFS KL_DIRECTIONS \
    STAT_FORGET_WEIGHTS STAT_RETAIN_WEIGHTS STAT_SYNTHETIC_MODES \
    STAT_EXCLUDE_SPECIAL_TOKENS STAT_PRESERVE_EOS \
    UNDIAL_BETAS UNDIAL_ALPHAS UNDIAL_GAMMAS \
    RMU_STEERING_COEFFS RMU_ALPHAS RMU_GAMMAS \
    SATIMP_BETA1S SATIMP_BETA2S \
    FLAT_DIVERGENCES FLAT_TEMPLATE_TEXT FLAT_TEMPLATE_TAG FLAT_TEMPLATE_ADD_EOS FLAT_ALPHAS \
    ADAPTIVE_RMU_ALPHAS ADAPTIVE_RMU_GAMMAS ADAPTIVE_RMU_STEERING_COEFFS ADAPTIVE_RMU_SCALES \
    ALTPO_ARTIFACT_ROOT ALTPO_ARTIFACT_SEED ALTPO_REPEATS ALTPO_BETAS ALTPO_ALPHAS ALTPO_GAMMAS \
    GRAD_DIFF_ALPHAS GRAD_DIFF_GAMMAS \
    CEU_IGNORE_FIRST_NS \
    PDU_RETAIN_LOSS_EPS PDU_DUAL_STEP_SIZE PDU_DUAL_UPDATE_UPON PDU_DUAL_WARMUP_EPOCHS PDU_PRIMAL_DUAL \
    TPO_BETAS TPO_PL_COEFFS TPO_ALPHAS TPO_GAMMAS TPO_IDENTIFIER_MODE
}

run_gemma_method() {
  local gpu="${GPU_ID:-0}"
  local lr="${LR:-1e-4}"
  export_method_env
  bash scripts/dualcf/run_campaign_one_lr.sh "${gpu}" "${lr}" all
}

run_gemma_duet_method() {
  local gpu="${GPU_ID:-0}"
  local lr="${LR:-1e-4}"
  export_method_env
  bash scripts/dualcf/run_campaign_one_lr.sh "${gpu}" "${lr}" duet_all
}

run_gemma_rwku_method() {
  local gpu="${GPU_ID:-0}"
  local lr="${LR:-1e-4}"
  export_method_env
  bash scripts/dualcf/run_campaign_one_lr.sh "${gpu}" "${lr}" rwku
}
```

## DPO baseline

Uses the validated Llama-named DualCF artifacts through `ARTIFACT_ROOT`.

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="dpo"
  BETAS="0.5"
  ALPHAS="1.0"
  GAMMAS="1.0"
  run_gemma_method
)
```

## GA baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="ga"
  run_gemma_method
)
```

## NPO baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="npo"
  BETAS="0.5"
  ALPHAS="1.0"
  GAMMAS="1.0"
  run_gemma_method
)
```

## NPO-SAM baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="npo_sam"
  BETAS="0.1"
  ALPHAS="1.0"
  GAMMAS="1.0"
  SAM_RHOS="0.01"
  SAM_ADAPTIVES="false"
  run_gemma_method
)
```

## LoKU baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="loku"
  run_gemma_method
)
```

## SimNPO baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="simnpo"
  DELTAS="0.0"
  BETAS="4.5"
  ALPHAS="1.0"
  GAMMAS="0.125"
  run_gemma_method
)
```

## Unilogit baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="unilogit"
  FORGET_COEFS="1.0"
  KL_DIRECTIONS="model_to_target"
  ALPHAS="1.0"
  GAMMAS="1.0"
  run_gemma_method
)
```

## STAT baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="stat"
  STAT_FORGET_WEIGHTS="1.0"
  STAT_RETAIN_WEIGHTS="1.0"
  STAT_SYNTHETIC_MODES="uniform"
  STAT_EXCLUDE_SPECIAL_TOKENS="true"
  STAT_PRESERVE_EOS="false"
  run_gemma_method
)
```

## UnDIAL baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="undial"
  UNDIAL_BETAS="3.0"
  UNDIAL_ALPHAS="0.0"
  UNDIAL_GAMMAS="1.0"
  run_gemma_method
)
```

## RMU baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="rmu"
  RMU_STEERING_COEFFS="2.0"
  RMU_ALPHAS="1.0"
  RMU_GAMMAS="1.0"
  run_gemma_method
)
```

## SatImp baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="satimp"
  SATIMP_BETA1S="5.0"
  SATIMP_BETA2S="0.1"
  ALPHAS="0.1"
  GAMMAS="1.0"
  run_gemma_method
)
```

## WGA baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="wga"
  BETAS="1.0"
  ALPHAS="1.0"
  GAMMAS="1.0"
  run_gemma_method
)
```

## AltPO baseline

Uses the existing Llama-named AltPO artifacts through `ALTPO_ARTIFACT_ROOT`.
AltPO is slow, so run DUET and RWKU as separate commands. `duet_all` runs DUET
`rare`, `popular`, and `merged` on the selected GPU; the RWKU command can run in
a second shell on another GPU after the common setup block is loaded there.

DUET command:

```bash
(
  GPU_ID=4
  SEEDS="42 179 1137"
  METHOD_VARIANTS="altpo"
  ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo
  ALTPO_ARTIFACT_SEED=0
  ALTPO_REPEATS=5
  ALTPO_BETAS="0.1"
  ALTPO_ALPHAS="1.0"
  ALTPO_GAMMAS="1.0"
  export PER_DEVICE_TRAIN_BS=8
  export GRAD_ACCUM=4
  run_gemma_duet_method
)
```

RWKU command:

```bash
(
  GPU_ID=5
  SEEDS="42 179 1137"
  METHOD_VARIANTS="altpo"
  ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo
  ALTPO_ARTIFACT_SEED=0
  ALTPO_REPEATS=5
  ALTPO_BETAS="0.1"
  ALTPO_ALPHAS="1.0"
  ALTPO_GAMMAS="1.0"
  export PER_DEVICE_TRAIN_BS=8
  export GRAD_ACCUM=4
  run_gemma_rwku_method
)
```

## FLAT baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="flat"
  FLAT_DIVERGENCES="Total-Variation"
  FLAT_TEMPLATE_TEXT="I don't know."
  FLAT_TEMPLATE_TAG="idk"
  FLAT_TEMPLATE_ADD_EOS="true"
  FLAT_ALPHAS="0.0"
  GAMMAS="1.0"
  run_gemma_method
)
```

## Adaptive RMU baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="adaptive_rmu"
  ADAPTIVE_RMU_ALPHAS="1200.0"
  ADAPTIVE_RMU_GAMMAS="1.0"
  ADAPTIVE_RMU_STEERING_COEFFS="1.0"
  ADAPTIVE_RMU_SCALES="5.0"
  run_gemma_method
)
```

## TPO baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="tpo"
  TPO_BETAS="0.2"
  TPO_PL_COEFFS="1.0"
  TPO_ALPHAS="1.0"
  TPO_GAMMAS="0.1"
  TPO_IDENTIFIER_MODE=stopword
  run_gemma_method
)
```

## GradDiff baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="grad_diff"
  GRAD_DIFF_ALPHAS="1.0"
  GRAD_DIFF_GAMMAS="1.0"
  run_gemma_method
)
```

## CE-U baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="ceu"
  CEU_IGNORE_FIRST_NS="1"
  run_gemma_method
)
```

## PDU baseline

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="pdu"
  ALPHAS="1.0"
  GAMMAS="1.0"
  PDU_RETAIN_LOSS_EPS="1.0"
  PDU_DUAL_STEP_SIZE="0.05"
  PDU_DUAL_UPDATE_UPON="step"
  PDU_DUAL_WARMUP_EPOCHS="0"
  PDU_PRIMAL_DUAL="true"
  run_gemma_method
)
```

## Additional DualCF and AdaPop runs

Run these before the final post-run sweeps if you also want the GeneralCF
DualCF variant and AdaPop in the Gemma output tree.

### GeneralCF / DualCF

This is the requested `general_cf` DualCF configuration. Run DUET and RWKU as
separate commands so they can occupy separate GPUs. The DUET command runs
`rare`, `popular`, and `merged`.

DUET command:

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="general_cf"
  ADDITIONAL_LOSS=NPO-SAM
  ROUTING=full
  SPAN_ADDITIONAL=true
  SPAN_CF_BRANCH=true
  DISABLE_RARITY_ROUTES=true
  DISABLE_DIFFICULTY_ROUTES=false
  DISABLE_ATTRIBUTION_ROUTES=false
  RARITY_NEG_GAINS="0.0"
  RARITY_CF_GAINS="0.0"
  SPAN_MODE=lcs
  SPAN_ALT_SHARED_TOKEN_WEIGHT=0.0
  SPAN_ALT_UNIQUE_TOKEN_WEIGHT=1.0
  SPAN_ORIG_SHARED_TOKEN_WEIGHT=0.00
  SPAN_ORIG_UNIQUE_TOKEN_WEIGHT=1.0
  BETAS=0.1
  GAMMAS=1.0
  SPAN_SAM_RHO=0.01
  SPAN_SAM_ADAPTIVE=false
  run_gemma_duet_method
)
```

RWKU command:

```bash
(
  GPU_ID=1
  SEEDS="42 179 1137"
  METHOD_VARIANTS="general_cf"
  ADDITIONAL_LOSS=NPO-SAM
  ROUTING=full
  SPAN_ADDITIONAL=true
  SPAN_CF_BRANCH=true
  DISABLE_RARITY_ROUTES=true
  DISABLE_DIFFICULTY_ROUTES=false
  DISABLE_ATTRIBUTION_ROUTES=false
  RARITY_NEG_GAINS="0.0"
  RARITY_CF_GAINS="0.0"
  SPAN_MODE=lcs
  SPAN_ALT_SHARED_TOKEN_WEIGHT=0.0
  SPAN_ALT_UNIQUE_TOKEN_WEIGHT=1.0
  SPAN_ORIG_SHARED_TOKEN_WEIGHT=0.00
  SPAN_ORIG_UNIQUE_TOKEN_WEIGHT=1.0
  BETAS=0.1
  GAMMAS=1.0
  SPAN_SAM_RHO=0.01
  SPAN_SAM_ADAPTIVE=false
  run_gemma_rwku_method
)
```

### AdaPop comparison

AdaPop is artifact-free. It uses the same Gemma `OUTPUT_ROOT` as the other
methods: `/data/home/vkropoti/unlearning/saves/unlearn-gemma`.

DUET command:

```bash
(
  GPU_ID=0
  SEEDS="42 179 1137"
  METHOD_VARIANTS="ada_pop"
  run_gemma_duet_method
)
```

RWKU command:

```bash
(
  GPU_ID=1
  SEEDS="42 179 1137"
  METHOD_VARIANTS="ada_pop"
  run_gemma_rwku_method
)
```

## Final post-run sweeps

Run these after all Gemma jobs finish. Use the Gemma output root directly; do not
use `${OUTPUT_ROOT%/unlearn}` because this runbook saves into `unlearn-gemma`,
not the old `unlearn` directory.

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export OUTPUT_ROOT=/data/home/vkropoti/unlearning/saves/unlearn-gemma
export COS_SIM_CUDA_VISIBLE_DEVICES=${COS_SIM_CUDA_VISIBLE_DEVICES:-0}

python scripts/calc_cos_sim.py \
  --path_to_saves "${OUTPUT_ROOT}" \
  --gpu "${COS_SIM_CUDA_VISIBLE_DEVICES}"

python scripts/calc_wrong_generations.py \
  --path_to_saves "${OUTPUT_ROOT}"

bash package_saves.sh \
  --path_to_saves "${OUTPUT_ROOT}" \
  --out_path /data/home/vkropoti/unlearning/saves-clean-gemma \
  --save_eval 0
```
