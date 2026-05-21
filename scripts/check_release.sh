#!/usr/bin/env bash
set -euo pipefail

fail=0
tmp_dir="${TMPDIR:-/tmp}"

check_no_matches() {
  local name="$1"
  local pattern="$2"
  echo "[release-check] ${name}"
  if git grep -nE "${pattern}" -- . \
      ':!docs/operator_logs/**' \
      ':!requirements-gpu-cu124.txt' \
      ':!RELEASE_MANIFEST.md' \
      ':!README.md' \
      ':!scripts/check_release.sh' >"${tmp_dir}/release_check_matches.txt"; then
    cat "${tmp_dir}/release_check_matches.txt"
    fail=1
  fi
}

check_no_matches_in_release_paths() {
  local name="$1"
  local pattern="$2"
  shift 2
  local paths=("$@")

  echo "[release-check] ${name}"
  if git grep -nE "${pattern}" -- "${paths[@]}" >"${tmp_dir}/release_check_matches.txt"; then
    cat "${tmp_dir}/release_check_matches.txt"
    fail=1
  fi
}

release_paths=(
  README.md
  FORK.md
  CITATION.cff
  RELEASE_MANIFEST.md
  prod-run-dual-gpu.md
  docs/diploma_repro.md
  docs/artifact_schema.md
  docs/counterfactual_generation_api.md
  scripts/env/example.env
  scripts/dualcf/run_campaign_one_lr.sh
  scripts/duet/dual_cf_duet.sh
  scripts/rwku/dual_cf_rwku.sh
  package_saves.sh
  setup.py
  setup_data.py
)

check_no_matches "possible literal secrets" '(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|OPENAI_API_KEY=[A-Za-z0-9_-]+|HF_TOKEN=[A-Za-z0-9_-]+|HUGGINGFACE_HUB_TOKEN=[A-Za-z0-9_-]+)'
check_no_matches_in_release_paths "personal absolute paths in public entry points" '(/home/vkropoti|/data/home/vkropoti|/Users/|/mnt/extremessd|/workspace/unlearning)' "${release_paths[@]}"

echo "[release-check] tracked model-weight files"
if git ls-files | grep -E '\.(safetensors|bin|pt|pth|ckpt)$' >"${tmp_dir}/release_model_files.txt"; then
  cat "${tmp_dir}/release_model_files.txt"
  fail=1
fi

find . -type f -size +50M \
  -not -path './.git/*' \
  -not -path './docs/operator_logs/*' \
  -not -path './results_public/*' \
  -print | tee "${tmp_dir}/release_large_files.txt"
if [[ -s "${tmp_dir}/release_large_files.txt" ]]; then
  echo "[release-check] large files found; remove them or use external storage"
  fail=1
fi

bash -n scripts/dualcf/run_campaign_one_lr.sh
bash -n package_saves.sh
python -m py_compile setup_data.py setup.py

if [[ ${fail} -ne 0 ]]; then
  echo "[release-check] FAILED"
  exit 1
fi

echo "[release-check] OK"
