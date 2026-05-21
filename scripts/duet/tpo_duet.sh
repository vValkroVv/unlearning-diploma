#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHOD_NAME=tpo
export RUN_LABEL=TPO
export TRAINER=TPO
export EXPERIMENT=unlearn/duet/tpo_lora.yaml

bash "${script_dir}/gd_family_duet.sh"
