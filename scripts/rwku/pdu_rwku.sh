#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHOD_NAME=pdu
export RUN_LABEL=PDU
export TRAINER=PDU
export EXPERIMENT=unlearn/rwku/pdu_lora.yaml

bash "${script_dir}/gd_family_rwku.sh"
