# AltPO Faithful Generation Integration Diff

AltPO is integrated as a faithful generated-artifact preference baseline for
DUET and RWKU. AltPO artifacts are generated directly from benchmark forget
splits with the AltPO prompt family; DualCF artifacts are not read or converted
for this method.

## Files added

- `src/tools/generate_altpo_artifacts.py`
- `src/tools/build_dualcf_altpo_artifact.py`
- `scripts/altpo/prepare_altpo_artifacts.sh`
- `scripts/altpo/build_dualcf_altpo_artifacts.sh`
- `src/trainer/unlearn/altpo.py`
- `configs/trainer/AltPO.yaml`
- `configs/experiment/unlearn/duet/altpo_lora.yaml`
- `configs/experiment/unlearn/rwku/altpo_lora.yaml`
- `scripts/duet/altpo_duet.sh`
- `scripts/rwku/altpo_rwku.sh`

## Files modified

- `src/trainer/__init__.py`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `check_saves.py`
- `src/tools/build_structured_saves.py`
- `src/tools/analyze_wrong_generations.py`
- `src/tools/export_unlearning_sanity_checks.py`
- `src/tools/build_results_combine_tables.py`
- `prod-run-dual-gpu.md`
- `prod-run-dual-vast.md`

## Objective

AltPO training uses the repo-native DPO objective:

- preferred answer: generated `sub_answer`, exposed as `forget.alternate`
- rejected answer: original benchmark answer, exposed as `forget.original`
- retain loss: NLL with weight `alpha`

Default method parameters:

- `beta=0.1`
- `alpha=1.0`
- `gamma=1.0`
- `retain_loss_type=NLL`
- paper-default LR: `5e-5`
- paper-default epochs: `2`
- generation repeats: `5`
- generation temperature: `1.0`
- generation max new tokens: `200`

## Artifact Contract

The campaign wrapper expects generated AltPO artifacts under
`${ALTPO_ARTIFACT_ROOT}`:

- `duet/rare_llama31_8b/altpo_rare_alt5_seed<SEED>.jsonl`
- `duet/popular_llama31_8b/altpo_popular_alt5_seed<SEED>.jsonl`
- `duet/merged_llama31_8b/altpo_merged_alt5_seed<SEED>.jsonl`
- `rwku/llama31_8b_level2/altpo_forget_level2_alt5_seed<SEED>.jsonl`

Each generated row includes `sub_answer` and the training alias `alternate`.
Compatibility metadata (`difficulty_score`, `attribution_score`, and
`rarity_score`) is written by default so the existing alternate-metadata dataset
configs can load the file, but DPO ignores those fields.

For the "DualCF routing/weights with AltPO alternates" ablation, the composer
reads the scored DualCF artifact and a generated AltPO artifact, preserves the
DualCF row count plus routing metadata, and replaces only `alternate` with one
AltPO generation matched by `source_index`. By default the generated artifact
seed is fixed at `0` and independent from training seeds. The standard output
layout is:

- `artifacts-dualcf-altpo/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl`
- `artifacts-dualcf-altpo/duet/popular_llama31_8b_v2/dualcf_popular_v2.jsonl`
- `artifacts-dualcf-altpo/duet/merged_llama31_8b_v2/dualcf_merged_v2.jsonl`
- `artifacts-dualcf-altpo/rwku/llama31_8b_level2_v2/dualcf_forget_level2_v2.jsonl`

## Commands

Smoke generation:

```bash
ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo \
DUET_LOCAL_SFT_BASE=/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models \
DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
ALTPO_ARTIFACT_SEED=0 \
ALTPO_MAX_EXAMPLES=4 \
ALTPO_REPEATS=5 \
ALTPO_BATCH_SIZE=32 \
FORCE_RERUN=1 \
bash scripts/altpo/prepare_altpo_artifacts.sh duet_rare
```

Full fixed-artifact generation:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
ALTPO_ARTIFACT_SEED=0 \
ALTPO_MAX_EXAMPLES=0 \
ALTPO_REPEATS=5 \
ALTPO_BATCH_SIZE=32 \
FORCE_RERUN=1 \
bash scripts/altpo/prepare_altpo_artifacts.sh all
```

Build DualCF-score / AltPO-alternate ablation artifacts:

```bash
ALTPO_ARTIFACT_SEED=0 \
ALTPO_REPEATS=5 \
ALTPO_REPEAT_SELECT=0 \
bash scripts/altpo/build_dualcf_altpo_artifacts.sh \
  /data/home/vkropoti/unlearning/artifacts \
  /data/home/vkropoti/unlearning/artifacts/altpo \
  /data/home/vkropoti/unlearning/artifacts-dualcf-altpo \
  all
```

Run DualCF on those composed artifacts:

```bash
SEEDS="42 179 1137" \
METHOD_VARIANTS="full" \
METHOD_NAME=dual_cf_altpo \
RUN_LABEL=DualCFAltPO \
ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts-dualcf-altpo \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

Train AltPO at the shared production LR:

```bash
SEEDS="42 179 1137" \
METHOD_VARIANTS="altpo" \
ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo \
ALTPO_ARTIFACT_SEED=0 \
ALTPO_REPEATS=5 \
ALTPO_BETAS="0.1" \
ALTPO_ALPHAS="1.0" \
ALTPO_GAMMAS="1.0" \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

Run the paper-default AltPO LR:

```bash
SEEDS="42 179 1137" \
METHOD_VARIANTS="altpo" \
ALTPO_ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/altpo \
ALTPO_ARTIFACT_SEED=0 \
ALTPO_REPEATS=5 \
ALTPO_BETAS="0.1" \
ALTPO_ALPHAS="1.0" \
ALTPO_GAMMAS="1.0" \
PER_DEVICE_TRAIN_BS=5 \
GRAD_ACCUM=7 \
bash scripts/dualcf/run_campaign_one_lr.sh 2 5e-5 all
```

## Validation

- Python compile checks passed for the AltPO generator, AltPO trainer,
  DualCF/AltPO composer, trainer registry, save checker, and parser/table
  tooling.
- Shell syntax checks passed for the AltPO prep script, direct launchers,
  DualCF/AltPO composer wrapper, DUET/RWKU dispatchers, DualCF-family
  launchers, and shared campaign wrapper.
- YAML load checks and direct Hydra composition passed for both AltPO experiment
  configs with `trainer=DPO`.
- Parser and trainer-registry smokes passed.
- Local JSONL composer smoke passed and verified that `alternate` changes while
  DualCF scores remain unchanged.
- No GPU generation or training smoke was run in this edit pass.
