#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for DexBench RBY1-M + Wuji 2 action-policy SFT on Cosmos3-Nano.
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_dexbench_rby1_wuji2_nano.toml (selects the
# registered `action_policy_dexbench_rby1_wuji2_nano` experiment; 256 tier, 62-D absolute joints
# + use_state, head stereo pair). See docs/action_policy_dexbench_rby1_wuji2_posttrain.md.
#
# Env vars (override for your filesystem):
#   DEXBENCH_COSMOS3_DATASET  LeRobot v3.0 root(s) (62-D joints, 256x144 stereo), space-separated, no default
#   DEXBENCH_NARROW_JOINTS    refuse (default) | widen: joints with a q01..q99 half-range under 1e-3
#   POLICY_CONTRACT           default <job dir>/policy_contract.json, the dir the trainer writes
#                             config.yaml to ($OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>);
#                             written from the roots before training; training normalizes with it
#                             (action_stats_path) and the policy server reads it (--contract-path)
#   BASE_CHECKPOINT_PATH      DCP of nvidia/Cosmos3-Nano (default examples/checkpoints/Cosmos3-Nano)
#   WAN_VAE_PATH              Wan2.2 VAE .pth (default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth)
#   OUTPUT_ROOT               default outputs/train
#   WANDB_API_KEY             for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE            torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES      space-separated Hydra overrides
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20 \
#                                trainer.grad_accum_iter=1 dataloader_train.max_samples_per_batch=4"
#   bash examples/launch_sft_action_policy_dexbench_rby1_wuji2_nano.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_dexbench_rby1_wuji2_nano.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export DEXBENCH_COSMOS3_DATASET="${DEXBENCH_COSMOS3_DATASET:-}"

# The run's [job] key: the TOML's value unless a tail override (job.<key>=...) sets it, as the
# trainer applies the overrides after the TOML. The job dir is $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>.
job_key() {
    local v o
    v=$(sed -n "/^\[job\]/,/^\[/s/^$1 *= *\"\(.*\)\"/\1/p" "$TOML_FILE")
    for o in "${TAIL_OVERRIDES[@]}"; do o=${o#+}; o=${o#+}; [[ $o == job.$1=* ]] && v=${o#*=}; done
    echo "$v"
}

read -ra DEXBENCH_ROOTS <<< "$DEXBENCH_COSMOS3_DATASET"
DS=dataloader_train.dataloader.datasets.dexbench.dataset
# After the input checks: write the policy contract from the roots and train with it.
EXTRA_DATASET_CHECK='for r in "${DEXBENCH_ROOTS[@]:-}"; do [[ -f "$r/meta/info.json" ]] || { echo "ERROR: DEXBENCH_COSMOS3_DATASET must be local LeRobot v3.0 dirs containing meta/info.json (got: '\''$DEXBENCH_COSMOS3_DATASET'\''). See docs/action_policy_dexbench_rby1_wuji2_posttrain.md" >&2; exit 1; }; done
POLICY_CONTRACT="${POLICY_CONTRACT:-$IMAGINAIRE_OUTPUT_ROOT/$(job_key project)/$(job_key group)/$(job_key name)/policy_contract.json}"
(cd "$WORKDIR" && mkdir -p "$(dirname "$POLICY_CONTRACT")" && PYTHONPATH=. python -m cosmos_framework.scripts.write_dexbench_policy_contract -o "$POLICY_CONTRACT" \
    --dataset-root "${DEXBENCH_ROOTS[@]}" --narrow-joints "${DEXBENCH_NARROW_JOINTS:-refuse}") || exit 1
TAIL_OVERRIDES+=("+$DS.action_stats_path=$POLICY_CONTRACT")
(( ${#DEXBENCH_ROOTS[@]} > 1 )) && TAIL_OVERRIDES+=("$DS.root=[$(IFS=,; echo "${DEXBENCH_ROOTS[*]}")]")'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
