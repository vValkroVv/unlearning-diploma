# DualCF artifact schema

Each DualCF artifact row must contain:

- `index`: source row index;
- `question` or `query`: benchmark prompt;
- `answer`: original answer to forget;
- `alternate`: counterfactual or negative answer;
- `difficulty_score`: normalized difficulty or risk score;
- `attribution_score`: normalized retain-attribution risk score;
- optional `rarity_score`: normalized rarity score;
- optional sidecar metadata fields depending on the artifact builder.

Before training, validate artifacts with:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /path/to/dualcf_rare_v2.jsonl \
  --question-key question
```

Generated DUET and RWKU artifacts are release inputs, not Git-tracked outputs.
Store full JSONL files under `${ARTIFACT_ROOT}` or another external artifact
location.
