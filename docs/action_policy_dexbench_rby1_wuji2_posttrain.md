# Cosmos3 DexBench RBY1-M + Wuji 2 Action-Policy Post-Training

This example post-trains [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) into a bimanual dexterous policy for the DexBench RBY1-M robot with two Wuji 2 hands. The model predicts 62 absolute joint positions conditioned on the measured joint state and the head stereo pair at the 256 resolution tier (`256×144` per camera). Experiment `action_policy_dexbench_rby1_wuji2_nano`, launcher `launch_sft_action_policy_dexbench_rby1_wuji2_nano.sh`. Serving a trained checkpoint is covered in [action_policy_dexbench_rby1_wuji2_server.md](./action_policy_dexbench_rby1_wuji2_server.md).

Two external inputs are required: (1) a DexBench dataset in LeRobotDataset v3.0 format, and (2) a DCP base checkpoint converted from Cosmos3-Nano.

The recipe runs on one node with 8 GPUs (FSDP).

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Inputs You Provide](#inputs-you-provide)
- [Recipe](#recipe)
- [Full Reproduction](#full-reproduction)
- [Checkpoints](#checkpoints)

______________________________________________________________________

<!--TOC-->

## Prerequisites

- [Setup](../README.md#setup) — clone the repo, install the training extras (`uv sync --all-extras --group=cu128-train` was used for this recipe), and activate the environment.
- [Environment Variables](./environment_variables.md) — set environment variables.
- [FAQ](./faq.md) — troubleshooting (OOM during SFT, defaults) and common pitfalls.

The runnable artifacts (TOML recipe, paired launch shell) live in [`examples/`](../examples); all commands below run from the repo root with the environment activated.

## Inputs You Provide

This package ships the training stack — the registered `action_policy_dexbench_rby1_wuji2_nano`
experiment, the `DexBenchRby1Wuji2LeRobotDataset` dataset class with the recipe knobs
(`use_state`, `stereo_stack`, `stereo_gap`) and the `dexbench_rby1_wuji2` domain. Two inputs are
external and must be provided per environment:

1. **DexBench dataset (in LeRobotDataset v3.0 format)** — one or more roots with `action` and
   `observation.state` as the 62 joint positions in the order
   `left_arm(7), right_arm(7), head(2), hands(40), torso(6)`, the head stereo pair as
   `observation.images.camera_ego_left` / `camera_ego_right` at `256×144`, recorded at 30 Hz. The
   DexBench-Arena repository produces it from a 66-joint RRC LeRobot v2.1 export with
   `scripts/training/relayout_lerobot_62d_320x180.py --size 256x144` (joints selected by name,
   frames area-downscaled) and `scripts/training/convert_lerobot_v21_to_v30.py`. Point
   `DEXBENCH_COSMOS3_DATASET` at the v3.0 root, or at several roots separated by spaces.
2. **DCP base checkpoint** — convert [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) to DCP and point
   `BASE_CHECKPOINT_PATH` at it (see [Full Reproduction](#full-reproduction)). Action heads are
   not loaded from it (they init fresh).

## Recipe

| knob         | value                                                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| init         | `Cosmos3-Nano` (public Hugging Face repo)                                                                                    |
| action space | 62 absolute joint positions (rad): `left_arm 7, right_arm 7, head 2, hands 40, torso 6`                                      |
| state        | `use_state=true`; the current state is the clean conditioning row, one quantile normalizer pooled over `action` and `state`  |
| task mode    | `wam`                                                                                                                        |
| resolution   | `256`: both cameras at `256×144`, stacked left over right with a 32-row band (`stereo_gap=32`) into the `256×320` 3:4 canvas |
| viewpoint    | `concat_view`, `camera_set=stereo_stack`                                                                                     |
| rate / chunk | native 30 Hz, chunk `32` (tokenizer `encode_exact_durations=[33]`)                                                           |
| rf shift     | training `3`; the policy server samples at `5` with 4 denoising steps                                                        |
| domain       | `dexbench_rby1_wuji2` = 33 (`num_embodiment_domains=34`; the action projections train from scratch)                          |
| lr           | `5e-5`, 500 warm-up iterations, `LambdaLinear` cycle 16000                                                                   |
| global batch | `256` = `max_samples_per_batch` 16 × 8 ranks × `grad_accum_iter` 2                                                           |
| iterations   | `4000`, saved every 1000                                                                                                     |

Without the band, two `256×144` cameras stack to `256×288`, exactly halfway between the tier's
1:1 and 3:4 canvases; `find_closest_target_size` keeps 1:1 and would shrink the stack to `228×256`.

## Full Reproduction

```shell
# Step 1: convert the base checkpoint -> $BASE_CHECKPOINT_PATH
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o $BASE_CHECKPOINT_PATH

# Step 2: download the Wan2.2 VAE at the revision cosmos_framework/inference/common/checkpoints.py pins
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir /path/to/wan22_vae

# Step 3: launch. The TOML selects the experiment + scalars; the dataset/action
# knobs come from the registered experiment.
export DEXBENCH_COSMOS3_DATASET=/path/to/dexbench_62d_256x144_lerobotv30
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/wan22_vae/Wan2.2_VAE.pth
export OUTPUT_ROOT=/path/to/outputs
export NPROC_PER_NODE=8
export EXTRA_TAIL_OVERRIDES="job.name=<run-name>"
bash examples/launch_sft_action_policy_dexbench_rby1_wuji2_nano.sh
```

Before training, the launcher writes the run's policy contract from the dataset roots and trains
with it (`action_stats_path`), so the trainer and the policy server normalize with the same file:

- `POLICY_CONTRACT` — where it goes; default
  `$OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/policy_contract.json`, next to the
  run's `config.yaml`.
- `DEXBENCH_NARROW_JOINTS` — `refuse` (default) stops on a joint whose q01..q99 half-range is
  under 1e-3; `widen` widens such a joint's range instead and records it in the contract.
- `DEXBENCH_RELATIVE_JOINT_SLICES` — unset (default) trains absolute actions; JSON column ranges
  such as `"[[0, 14]]"` (both arms) train those columns as `action - state`, with the state of the
  window's first frame. The contract then states the representation and carries the action rows'
  statistics (`action_rows`, the arm deltas on those columns), and the run's config carries
  `relative_joint_slices`, from which the policy server adds the state back.

The recipe TOML ([`action_policy_dexbench_rby1_wuji2_nano.toml`](../examples/toml/sft_config/action_policy_dexbench_rby1_wuji2_nano.toml)) sets the scalar
knobs (`max_iter`, `save_iter`, `grad_accum_iter`, parallelism, wandb); the dataset/action knobs
(62-D joints, `use_state`, stereo stack and band, 256 tier, chunk 32) live in the registered
`action_policy_dexbench_rby1_wuji2_nano` experiment. W&B logs to project `cosmos3_action_dexbench`
under `WANDB_ENTITY` when it is set. On 8× H100 an iteration takes 4.3 s; the 4000 iterations
with four saves took 5 h 16 min.

## Checkpoints

- Saved every `save_iter` iters (1000) to
  `$OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/checkpoints/iter_<N>/`, next to the
  run's `config.yaml`; one save (model + optimizer) is about 166 GB.
- Export to HF safetensors (EMA weights, about 30 GB) via `cosmos_framework.scripts.export_model`:

  ```shell
  python -m cosmos_framework.scripts.export_model \
    --checkpoint-path $OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/checkpoints/iter_000004000 \
    --config-file $OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/config.yaml \
    -o /path/to/export
  ```

- Serve a checkpoint (DCP or exported) with
  [action_policy_dexbench_rby1_wuji2_server.md](./action_policy_dexbench_rby1_wuji2_server.md).
