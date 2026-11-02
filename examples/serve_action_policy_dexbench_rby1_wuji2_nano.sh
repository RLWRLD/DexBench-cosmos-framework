#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Serve a DexBench RBY1-M + Wuji 2 Cosmos3-Nano checkpoint
# (cosmos_framework.scripts.action_policy_server_dexbench; docs/action_policy_dexbench_rby1_wuji2_server.md).
#
#   bash examples/serve_action_policy_dexbench_rby1_wuji2_nano.sh <run dir | checkpoint dir> [server options]
#
# <run dir>: a training run's job dir ($OUTPUT_ROOT/<project>/<group>/<name>, holding config.yaml,
#   policy_contract.json and checkpoints/iter_*); its latest checkpoint is served.
# <checkpoint dir>: one checkpoints/iter_<N> of such a run, or a copy holding model/, config.yaml and
#   policy_contract.json side by side.
# Server options pass through, e.g. --port 8000 --api-token-file <file> --host 127.0.0.1.
# Settings for this machine (HF_TOKEN, HF_HOME) come from examples/dexbench.env, as for training.
# ============================================================================
set -euo pipefail
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$WORKDIR/examples/_dexbench_env.sh"

[[ $# -ge 1 ]] || { sed -n '8,16p' "${BASH_SOURCE[0]}" >&2; exit 2; }
target="$(cd "$1" && pwd)"; shift

if [[ -d "$target/model" ]]; then
    run="$target" checkpoint="$target/model"
elif [[ -d "$target/checkpoints" ]]; then
    run="$target" checkpoint="$(ls -d "$target"/checkpoints/iter_* 2>/dev/null | sort | tail -1)"
    [[ -n "$checkpoint" ]] || { echo "ERROR: no checkpoints/iter_* in $target" >&2; exit 1; }
elif [[ "$(basename "$(dirname "$target")")" == checkpoints ]]; then
    run="$(dirname "$(dirname "$target")")" checkpoint="$target"
else
    echo "ERROR: $target is neither a run dir (checkpoints/), an iter_<N> checkpoint nor a copy with model/" >&2
    exit 1
fi
for f in "$run/config.yaml" "$run/policy_contract.json"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

cd "$WORKDIR"
echo "serving $checkpoint ($run/config.yaml)"
PYTHONPATH=. exec python -m cosmos_framework.scripts.action_policy_server_dexbench \
    --checkpoint-path "$checkpoint" --config-file "$run/config.yaml" --contract-path "$run/policy_contract.json" "$@"
