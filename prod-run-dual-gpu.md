# DualCF v2 public GPU runbook

This is the public entry point for the diploma production workflow. The original
H100 operator command log is preserved at `docs/operator_logs/h100_commands.md`
for auditability, but new runs should use environment variables instead of
machine-specific absolute paths.

## Environment

Copy the template and edit paths:

```bash
cp scripts/env/example.env .env
source .env
cd "${REPO_ROOT}"
source "${VENV_PATH:-${REPO_ROOT}/.venv}/bin/activate"
```

The runbooks assume that model weights, local dataset mirrors, generated
artifacts, and training outputs live under `${DATA_ROOT}` or explicitly supplied
environment variables.

## Utility-3K panel

Build or provide the Utility-3K panel under `${UTILITY_ROOT}`. The campaign
wrapper defaults to `UTILITY=3k` and reads utility task configuration from
`configs/lm_eval_tasks/utility_3k`.

## vLLM generator

Start an OpenAI-compatible generator service and export:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL="${MODEL_ROOT}/GENERATOR/Qwen3.5-27B"
```

Use the generator only for artifact preparation. The generated JSONL artifacts
should remain outside Git.

## Prepare DualCF artifacts

DUET:

```bash
export ARTIFACT_ROOT="${DATA_ROOT}/artifacts/dualcf"
export HF_BASE_MODEL_PATH="${MODEL_ROOT}/BASE/Llama-3.1-8B-Instruct"
export DUET_LOCAL_SFT_BASE="${DATA_ROOT}/SwetieePawsss/DUET_ft_models"
bash scripts/duet/prepare_dual_cf_duet_v2.sh
```

RWKU:

```bash
export ARTIFACT_ROOT="${DATA_ROOT}/artifacts/dualcf"
export HF_BASE_MODEL_PATH="${MODEL_ROOT}/BASE/Llama-3.1-8B-Instruct"
bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

Validate the final artifacts before training:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ARTIFACT_ROOT}/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl" \
  --question-key question
```

## Train and evaluate

Run one phase at one learning rate:

```bash
bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 duet_rare 42
bash scripts/dualcf/run_campaign_one_lr.sh 1 5e-6 duet_popular 42
bash scripts/dualcf/run_campaign_one_lr.sh 2 5e-6 duet_merged 42
bash scripts/dualcf/run_campaign_one_lr.sh 3 5e-6 rwku 42
```

Run all main phases serially on one GPU:

```bash
bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 all 42
```

## Package summaries

Create a reviewer/public summary copy without checkpoints:

```bash
bash package_saves.sh \
  --path_to_saves "${OUTPUT_ROOT}" \
  --out_path "${DATA_ROOT}/saves-clean-diploma" \
  --save_eval 0
```
