# Reproducing the diploma experiments

## Scope

This guide describes the public reproduction path for the diploma experiments.
It assumes local access to model weights, DUET/RWKU data mirrors, generated
artifacts, and GPU hardware. Full checkpoints, model caches, and generated
counterfactual JSONL files are intentionally not stored in Git.

## Environment variables

Copy `scripts/env/example.env` to a local `.env` file and edit paths:

```bash
source .env
cd "${REPO_ROOT}"
```

At minimum, set `REPO_ROOT`, `DATA_ROOT`, `HF_BASE_MODEL_PATH`,
`DUET_LOCAL_SFT_BASE`, `ARTIFACT_ROOT`, `OUTPUT_ROOT`, and `UTILITY_ROOT`.

## Step 1: build the Utility-3K panel

See `prod-run-dual-gpu.md#utility-3k-panel`. The campaign wrapper expects the
panel under `${UTILITY_ROOT}` unless overridden.

## Step 2: start the vLLM generator

See `prod-run-dual-gpu.md#vllm-generator`. Configure `VLLM_BASE_URL`,
`VLLM_API_KEY`, and `VLLM_MODEL` for the generator service used to build
counterfactual artifacts.

## Step 3: prepare artifacts

Run DUET Phase A, DUET Phase B, RWKU Phase A, and RWKU Phase B using the public
runbook commands or the direct scripts:

```bash
bash scripts/duet/prepare_dual_cf_duet_v2.sh
bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

## Step 4: validate artifacts

Validate each final JSONL before training:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ARTIFACT_ROOT}/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl" \
  --question-key question
```

## Step 5: train and evaluate

Use the one-LR campaign wrapper for the main diploma runs:

```bash
bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 duet_rare 42
bash scripts/dualcf/run_campaign_one_lr.sh 1 5e-6 duet_popular 42
bash scripts/dualcf/run_campaign_one_lr.sh 2 5e-6 duet_merged 42
bash scripts/dualcf/run_campaign_one_lr.sh 3 5e-6 rwku 42
```

You can also use the direct DUET/RWKU launchers when debugging a single method:

```bash
bash scripts/duet/dual_cf_duet.sh
bash scripts/rwku/dual_cf_rwku.sh
```

## Step 6: post-run summaries

After training and evaluation, run the summary utilities used by the diploma:

```bash
python scripts/calc_cos_sim.py --help
python scripts/calc_wrong_generations.py --help
bash package_saves.sh \
  --path_to_saves "${OUTPUT_ROOT}" \
  --out_path "${DATA_ROOT}/saves-clean-diploma" \
  --save_eval 0
```

`package_saves.sh` creates a compact summary-only copy and ZIP. Keep full
checkpoints and per-example generated artifacts outside Git.
