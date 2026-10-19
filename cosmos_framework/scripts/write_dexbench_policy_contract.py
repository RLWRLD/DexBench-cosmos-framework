# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Write the policy contract of a DexBench RBY1-M + Wuji 2 training run.

``action_policy_server_dexbench`` reads it with ``--contract-path``: the q01/q99/mean/std/min/max
statistics pooled over ``action`` and ``observation.state`` (the training normalizer), plus the
joint order, frame size, rate and prompts of the data. The dataset roots are taken from the run's
saved ``config.yaml`` (the resolved roots its dataloader read), so the statistics are those of the
data the run trained on; ``--dataset-root`` replaces them when that data has moved, or names them
before the run exists (the launcher writes the contract this way before training). A run trained
with ``relative_joint_slices`` needs the same ``--relative-joint-slices`` here: the contract then
states that representation and carries the action rows' statistics (``build_policy_contract``).

Usage
-----
    python -m cosmos_framework.scripts.write_dexbench_policy_contract \\
        --config-file <run dir>/cosmos3_action_dexbench/action_sft/<job.name>/config.yaml \\
        -o <run dir>/policy_contract.json
"""

import json
from pathlib import Path
from typing import Annotated, Literal

import tyro

from cosmos_framework.data.generator.action.datasets.dexbench_rby1_wuji2_lerobot_dataset import (
    build_policy_contract,
    find_dataset_node,
)
from cosmos_framework.inference.common.config import deserialize_config_dict


def main(
    output: Annotated[Path, tyro.conf.arg(aliases=("-o",), help="Output policy_contract.json path.")],
    config_file: Annotated[Path | None, tyro.conf.arg(help="The training run's saved config.yaml.")] = None,
    dataset_root: Annotated[
        list[Path] | None,
        tyro.conf.arg(help="LeRobot v3.0 roots to read instead of the ones config.yaml names."),
    ] = None,
    narrow_joints: Annotated[
        Literal["refuse", "widen"],
        tyro.conf.arg(help="Joints with a q01..q99 half-range under 1e-3: leave them (training refuses) or widen."),
    ] = "refuse",
    relative_joint_slices: Annotated[
        str | None,
        tyro.conf.arg(help='JSON column ranges trained relative to the state, e.g. "[[0, 14]]" (both arms).'),
    ] = None,
    chunk_length: Annotated[
        int, tyro.conf.arg(help="The run's chunk_length (relative statistics only; recorded with them).")
    ] = 32,
) -> None:
    """Write policy_contract.json for a DexBench RBY1-M + Wuji 2 run."""
    if dataset_root:
        roots = [str(root) for root in dataset_root]
    else:
        root = find_dataset_node(deserialize_config_dict(config_file))["root"]
        roots = [root] if isinstance(root, str) else list(root)
    slices = json.loads(relative_joint_slices) if relative_joint_slices else None
    contract = build_policy_contract(roots, narrow_joints, slices, chunk_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(contract, indent=1) + "\n")
    partial.replace(output)
    print(
        f"policy contract: {output} ({len(contract['contract']['joint_names'])} joints, "
        f"frames {contract['contract']['frame_hw']}, tasks {contract['contract']['tasks']}, roots {roots})"
    )


if __name__ == "__main__":
    tyro.cli(main)
