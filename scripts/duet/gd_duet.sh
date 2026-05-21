#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHOD_NAME=grad_diff
export RUN_LABEL=GradDiff
export TRAINER=GradDiff
export EXPERIMENT=unlearn/duet/grad_diff_lora.yaml

bash "${script_dir}/gd_family_duet.sh"
