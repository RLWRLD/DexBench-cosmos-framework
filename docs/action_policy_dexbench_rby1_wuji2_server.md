# Cosmos3 DexBench RBY1-M + Wuji 2 Policy Server

`cosmos_framework.scripts.action_policy_server_dexbench` serves a checkpoint of the
`action_policy_dexbench_rby1_wuji2_nano` recipe ([post-training](./action_policy_dexbench_rby1_wuji2_posttrain.md))
over HTTP on one GPU. A request carries the left and right head-camera frames, the 62 measured
joint positions and the prompt; the reply is the next 32 absolute joint targets. Each request goes
through the training `ActionTransformPipeline` with the dataset settings read from the run's
`config.yaml`, so the stacked frame, the JSON prompt and the normalized state row are built by the
same code that built them in training.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Policy Contract](#policy-contract)
- [Policy Server](#policy-server)
  - [Endpoints](#endpoints)
- [Simulation Client](#simulation-client)

______________________________________________________________________

<!--TOC-->

## Policy Contract

The server normalizes the state and denormalizes the predicted chunk with the run's
`policy_contract.json`: the q01/q99/mean/std/min/max statistics pooled over `action` and
`observation.state` (the training normalizer), plus the joint order, frame size, rate and prompts
of the data. Write it from the run's saved `config.yaml`, which names the dataset roots the run
trained on (`--dataset-root` replaces them if that data has moved):

```shell
python -m cosmos_framework.scripts.write_dexbench_policy_contract \
  --config-file $OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/config.yaml \
  -o /path/to/policy_contract.json
```

`tools/compute_action_stats.py` covers the statistics of `action` alone; the contract pools state
and action into one normalizer, as the recipe trains, and adds the names and frame size the server
checks requests against.

## Policy Server

```shell
python -m cosmos_framework.scripts.action_policy_server_dexbench \
  --checkpoint-path /path/to/export \
  --config-file $OUTPUT_ROOT/cosmos3_action_dexbench/action_sft/<job.name>/config.yaml \
  --contract-path /path/to/policy_contract.json \
  --host 127.0.0.1 --port 8000 --api-token-file /path/to/token
```

- `--checkpoint-path` takes a DCP checkpoint directory (`.../checkpoints/iter_<N>`) or an export
  of `cosmos_framework.scripts.export_model`; `--config-file` is the run's `config.yaml` either way.
- `--api-token-file`: every request must carry `Authorization: Bearer <token>`. The server binds
  `--host` (default `127.0.0.1`); keep it off `0.0.0.0` unless the port is firewalled.
- `--num-steps` (default 4) denoising steps per chunk; `--seed` is the base seed (a request
  without `seed` uses `seed + request number`).
- `--shift` (default 5) is the sampler shift. The recipe trains at shift 3; at 10 steps on 64
  training windows, shift 5 predicted the recorded actions slightly better than 3 on every joint
  group, and 4 steps at shift 5 matched 10 steps there (mean left-arm error 0.0150 vs 0.0153 rad).

Loading takes about 4 min. On an H100 a chunk takes about 0.25 s at 4 steps (0.45 s at 10), and
the server peaks at about 32 GiB of GPU memory.

### Endpoints

All JSON:

- `GET /info`: the joint names, frame size (`frame_hw`), `stereo_gap`, rate, horizon, prompts and
  sampling settings, for a client to bind joints by name and resize its frames.
- `POST /predict`:
  `{"images": {"left": <b64 PNG>, "right": <b64 PNG>}, "state": [62 floats], "prompt": str, "seed": int (optional)}`
  → `{"action": [[62 floats] × 32], "state_echo_error": float, "seed": int, "ms_infer": float, ...}`.
  The PNGs must be RGB at the contract's `frame_hw` (`144×256`), area-downscaled from the render as
  in the dataset. `state_echo_error` is the largest difference (rad) between the input state and
  the model's denormalized conditioning row, which should come back unchanged.

## Simulation Client

DexBench-Arena's `dexbench_arena.policy.cosmos3_joint_policy.Cosmos3JointPolicy` is a client for
this server: it reads `/info`, area-resizes the robot's head-camera renders to `frame_hw`, sends the
joint state in contract order and executes a configurable prefix of each 32-step chunk.
