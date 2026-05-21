#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHOD_NAME=ceu
export RUN_LABEL=CEU
export TRAINER=CEU
export EXPERIMENT=unlearn/duet/ceu_lora.yaml

bash "${script_dir}/gd_family_duet.sh"
