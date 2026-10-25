# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torchvision.transforms.v2 as T

from cosmos_framework.data.generator.action.datasets.dexbench_rby1_wuji2_lerobot_dataset import (
    ACTION_DIM,
    DexBenchRby1Wuji2LeRobotDataset,
    StateActionRowNormalizer,
    action_row_stats,
    compute_joint_quantile_stats,
    compute_relative_action_stats,
    relative_joint_mask,
)
from cosmos_framework.data.generator.action.utils.action_processing import resolve_action_normalization
from cosmos_framework.scripts.action_policy_server_dexbench import DexBenchRequestBuilder

_SLICES = [[0, 14]]
_CHUNK = 8
_SPIKE = 100.0


def _episodes() -> list[tuple[np.ndarray, np.ndarray]]:
    """Two random-walk ``(action, state)`` episodes whose last action jumps by ``_SPIKE`` on every joint.

    No training window reaches an episode's last frame, so the jump must not show in the statistics.
    """
    rng = np.random.default_rng(0)
    episodes = []
    for length in (40, 29):
        state = np.cumsum(rng.normal(0.0, 0.02, (length, ACTION_DIM)), axis=0)
        action = state + rng.normal(0.0, 0.01, (length, ACTION_DIM))
        action[-1] += _SPIKE
        episodes.append((action, state))
    return episodes


def _write_root(root: Path, episodes: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """A LeRobot v3.0 ``data/`` shard of ``episodes``, the columns the statistics read."""
    action = np.concatenate([a for a, _ in episodes])
    state = np.concatenate([s for _, s in episodes])
    table = pa.table(
        {
            "action": list(action),
            "observation.state": list(state),
            "episode_index": np.concatenate([np.full(len(a), i) for i, (a, _) in enumerate(episodes)]),
            "frame_index": np.concatenate([np.arange(len(a)) for a, _ in episodes]),
        }
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return str(root)


@pytest.mark.L0
def test_relative_actions_round_trip_to_the_recorded_actions(tmp_path: Path) -> None:
    """Delta statistics cover exactly the training windows, and normalize -> to_absolute restores the actions."""
    episodes = _episodes()
    root = _write_root(tmp_path, episodes)
    mask = relative_joint_mask(_SLICES)

    # Windows start at 0 <= t < L - chunk_length and hold the actions t .. t + chunk_length - 1.
    expected = np.array(
        [
            action[t + k, mask.numpy()] - state[t, mask.numpy()]
            for action, state in episodes
            for t in range(len(action) - _CHUNK)
            for k in range(_CHUNK)
        ]
    )
    deltas = compute_relative_action_stats([root], _SLICES, _CHUNK)
    for key, value in (("mean", expected.mean(0)), ("min", expected.min(0)), ("max", expected.max(0))):
        np.testing.assert_allclose(deltas[key], value, rtol=0, atol=1e-12, err_msg=key)
    assert deltas["max"].max() < _SPIKE / 2  # no window reached an episode's last frame

    def normalizer(raw: dict[str, np.ndarray]):
        return resolve_action_normalization(
            "quantile", {k: torch.as_tensor(v, dtype=torch.float32) for k, v in raw.items()}
        )

    pooled = compute_joint_quantile_stats([root])
    rows = StateActionRowNormalizer(normalizer(pooled), normalizer(action_row_stats(pooled, deltas, _SLICES)))
    builder = object.__new__(DexBenchRequestBuilder)  # to_absolute reads only these two attributes
    builder.normalizer, builder.relative_mask = rows, mask

    action, state = (torch.as_tensor(x, dtype=torch.float32) for x in episodes[0])
    t = 5
    recorded, state_t = action[t : t + _CHUNK], state[t]
    # The dataset's sample: the state row, then the actions with the state subtracted on the relative columns.
    window = torch.cat([state_t[None], recorded.clone()])
    window[1:, mask] -= state_t[mask]

    normalized = rows.normalize_action(window)
    torch.testing.assert_close(normalized[0], rows.state.normalize_action(state_t))
    torch.testing.assert_close(rows.denormalize_action(normalized), window, rtol=0, atol=1e-5)
    absolute = builder.to_absolute(normalized, state_t)
    torch.testing.assert_close(absolute[0], state_t, rtol=0, atol=1e-5)
    torch.testing.assert_close(absolute[1:], recorded, rtol=0, atol=1e-5)


@pytest.mark.L0
def test_color_jitter_draws_once_for_both_cameras_and_every_frame() -> None:
    dataset = object.__new__(DexBenchRby1Wuji2LeRobotDataset)  # _compose_video reads only these attributes
    dataset._camera_set, dataset._stereo_gap = "stereo_stack", 4
    dataset._color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    frame = torch.rand(3, 8, 8)
    sample = {
        "observation.images.camera_ego_left": frame.expand(5, -1, -1, -1).clone(),
        "observation.images.camera_ego_right": frame.expand(5, -1, -1, -1).clone(),
    }
    torch.manual_seed(0)
    video = dataset._compose_video(sample)  # [5,3,20,8]
    left, band, right = video[:, :, :8], video[:, :, 8:12], video[:, :, 12:]
    assert not torch.equal(left[0], frame)
    assert torch.equal(left, right) and torch.equal(left, left[:1].expand_as(left))
    assert not band.any()

