#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  setup_data.py \
  src/train.py \
  src/eval.py \
  src/tools/validate_dual_cf_artifact.py \
  src/tools/build_structured_saves.py \
  src/tools/build_results_combine_tables.py

bash -n scripts/dualcf/run_campaign_one_lr.sh
bash -n scripts/duet/dual_cf_duet.sh
bash -n scripts/rwku/dual_cf_rwku.sh
bash -n package_saves.sh

python setup_data.py --help >/dev/null
python src/train.py --help >/dev/null 2>/dev/null || true
python src/eval.py --help >/dev/null 2>/dev/null || true

echo "release smoke OK"
