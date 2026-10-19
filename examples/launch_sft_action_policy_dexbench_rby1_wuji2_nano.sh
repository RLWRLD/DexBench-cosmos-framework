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
#   DEXBENCH_COSMOS3_DATASET  LeRobot v3.0 root (62-D joints, 256x144 stereo), no default
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

EXTRA_DATASET_CHECK='[[ -f "$DEXBENCH_COSMOS3_DATASET/meta/info.json" ]] || { echo "ERROR: DEXBENCH_COSMOS3_DATASET must be a local LeRobot v3.0 dir containing meta/info.json (got: '\''$DEXBENCH_COSMOS3_DATASET'\''). See docs/action_policy_dexbench_rby1_wuji2_posttrain.md" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
