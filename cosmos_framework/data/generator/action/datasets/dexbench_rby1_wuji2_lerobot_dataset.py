# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DexBench RBY1-M + Wuji 2 hands LeRobot action-policy dataset.

Reads the DexBench 62-joint training contract written by DexBench-Arena's
``scripts/training/relayout_lerobot_62d_320x180.py`` (``--size 256x144``) and converted to
LeRobot v3.0 by ``scripts/training/convert_lerobot_v21_to_v30.py``:

* ``action`` and ``observation.state`` are 62 absolute joint positions in radians, in the order
  ``left_arm(7), right_arm(7), head(2), hands(40), torso(6)`` (``meta/modality.json``; the hand
  block is DexBench's stage-major interleave of both Wuji 2 hands). ``action`` is the commanded
  joint target, ``observation.state`` the measured position at the same frame;
* ``observation.images.camera_ego_{left,right}`` are the head-mounted ZED Mini stereo pair,
  area-downscaled from the render to 256x144;
* the data is recorded at 30 Hz and is read at its native rate. ``fps`` must equal the dataset's
  rate: ``build_episode_spans`` trims ``chunk_length`` raw rows per episode regardless of the
  delta-timestamp spacing, so reading 30 Hz data at 15 Hz would run windows past episode ends.

The sample follows DROID's ``joint_pos`` + ``use_state`` contract: the current joint state is
prepended to the ``chunk_length`` future actions as a clean conditioning row (same width, same
joint space), so the model sees ``[chunk_length + 1, 62]``. Because state and actions share one
normalizer, the default quantile statistics are pooled over both columns.

``relative_joint_slices`` (default ``None``: every action absolute) makes the listed joint columns
relative, as GR00T N1.7's arm-relative modality does: ``[[0, 14]]`` turns both arms' action rows into
``action - state`` with the state of the window's first frame (the conditioning row), while hands, head
and torso stay absolute. The state row stays absolute. The two kinds of row then normalize with their
own statistics (``StateActionRowNormalizer``): the state row with the pooled absolute ones, the action
rows with ``action_rows`` -- the pooled arm deltas on the relative columns and the absolute statistics on
the rest -- which ``build_policy_contract`` writes beside them. The policy server reads the same setting
from the run's config, denormalizes the generated rows with ``action_rows`` and adds the request's state
back on the relative columns, so a client still receives absolute joint targets.

``camera_set`` selects the observation:

* ``"stereo_stack"`` (default) -- left over right, full width: ``[T, C, 2H + stereo_gap, W]``;
* ``"left"`` -- the left camera alone: ``[T, C, H, W]``.

``stereo_gap`` puts that many black rows between the two cameras of the stack (default 0, no
band). Two 256x144 cameras stack to 256x288, whose H/W of 1.125 is exactly halfway between the
256 tier's 1:1 (256x256) and 3:4 (256x320) canvases, and ``find_closest_target_size`` keeps the
first, 1:1, so the stack would be shrunk to 228x256. The recipe's ``stereo_gap=32`` makes it
256x320, the 3:4 canvas itself: both cameras reach the model at 256x144 with no resize and no
padding. The policy server stacks request frames with the same ``stack_stereo`` and reads
``stereo_gap`` from the run's config (``find_dataset_node``).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Joint,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint
from cosmos_framework.utils import log

EMBODIMENT = "dexbench_rby1_wuji2"
ACTION_DIM = 62
# (count, label, prefix) per block of the 62-joint contract, in column order.
JOINT_BLOCKS: tuple[tuple[int, str, str], ...] = (
    (7, "joint", "left_arm"),
    (7, "joint", "right_arm"),
    (2, "joint", "head"),
    (40, "joint", "hands"),
    (6, "joint", "torso"),
)

_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_IMAGE_FEATURES = {
    "left": "observation.images.camera_ego_left",
    "right": "observation.images.camera_ego_right",
}
_CAMERA_SETS = ("stereo_stack", "left")
# The JSON prompt's framing text per camera set; the policy server sends the same words.
VIEW_DESCRIPTIONS = {
    "stereo_stack": (
        "The top half is the left camera and the bottom half the right camera of a "
        "stereo pair mounted on the robot's head."
    ),
    "left": "The view is the left camera of a stereo pair mounted on the robot's head.",
}
# A joint whose pooled q01..q99 half-range is below this would be scaled by ~1/0 under quantile
# normalization; refuse rather than train on exploding values.
_MIN_QUANTILE_HALF_RANGE = 1e-3


def stack_stereo(left: torch.Tensor, right: torch.Tensor, gap: int = 0) -> torch.Tensor:
    """Left camera over right along the height axis, with ``gap`` black rows between them.

    ``[..., H, W]`` frames of either dtype; the band is zeros of the frames' dtype.
    """
    if gap == 0:
        return torch.cat([left, right], dim=-2)
    band = left.new_zeros((*left.shape[:-2], gap, left.shape[-1]))
    return torch.cat([left, band, right], dim=-2)


def find_dataset_node(config: dict) -> dict:
    """The ``get_action_dexbench_rby1_wuji2_sft_dataset`` call in a run config's ``dataloader_train``.

    ``config`` is the plain dict of a saved ``config.yaml``; the node holds the dataset keyword
    arguments the run trained with (resolved ``root`` included).
    """
    found: list[dict] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if str(obj.get("_target_", "")).endswith("get_action_dexbench_rby1_wuji2_sft_dataset"):
                found.append(obj)
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for value in obj:
                _walk(value)

    _walk(config.get("dataloader_train"))
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one get_action_dexbench_rby1_wuji2_sft_dataset node in dataloader_train, "
            f"found {len(found)}; not a DexBench RBY1-M + Wuji 2 run config"
        )
    return found[0]


JointSlices = list[list[int]] | None
"""``[[start, stop], ...]`` column ranges of the 62-joint contract whose actions are relative."""


def relative_joint_mask(slices: JointSlices) -> torch.Tensor:
    """``[62]`` bool mask of the relative columns; all False for ``None``."""
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for item in slices or []:
        start, stop = (int(v) for v in item)
        if not 0 <= start < stop <= ACTION_DIM:
            raise ValueError(f"relative_joint_slices entry {item!r} is not a column range within [0, {ACTION_DIM}]")
        mask[start:stop] = True
    return mask


def _stats_of(values: np.ndarray) -> dict[str, np.ndarray]:
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    return {
        "q01": q01,
        "q99": q99,
        "mean": values.mean(0),
        "std": values.std(0),
        "min": values.min(0),
        "max": values.max(0),
    }


def _episode_arrays(root: str):
    """``(action, state)`` ``[L, 62]`` float64 arrays per episode of a LeRobot v3.0 root."""
    files = sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data/*/*.parquet under {root!r}")
    for path in files:
        table = pq.read_table(path, columns=[_ACTION_FEATURE, _STATE_FEATURE, "episode_index", "frame_index"])
        episodes = np.asarray(table.column("episode_index").to_numpy())
        frames = np.asarray(table.column("frame_index").to_numpy())
        action = np.stack(table.column(_ACTION_FEATURE).to_numpy(zero_copy_only=False)).astype(np.float64)
        state = np.stack(table.column(_STATE_FEATURE).to_numpy(zero_copy_only=False)).astype(np.float64)
        for episode in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode)
            rows = rows[np.argsort(frames[rows], kind="stable")]
            yield action[rows], state[rows]


def compute_relative_action_stats(roots: list[str], slices: JointSlices, chunk_length: int) -> dict[str, np.ndarray]:
    """Statistics of ``action[t + k] - state[t]`` on the relative columns, over every training window.

    A window starts at frame ``t`` of an episode of ``L`` frames for ``0 <= t < L - chunk_length``
    (``build_episode_spans``) and holds the actions of frames ``t .. t + chunk_length - 1``; every
    ``(t, k)`` pair contributes one delta. Returns ``{q01, q99, mean, std, min, max}`` of shape
    ``[n_relative_columns]`` in column order.
    """
    mask = relative_joint_mask(slices).numpy()
    deltas = []
    for root in roots:
        for action, state in _episode_arrays(root):
            starts = len(action) - chunk_length
            if starts <= 0:
                continue
            ref = state[:starts, mask]  # [S, R]
            for k in range(chunk_length):
                deltas.append(action[k : k + starts, mask] - ref)
    values = np.concatenate(deltas, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("non-finite relative actions")
    return _stats_of(values)


class StateActionRowNormalizer:
    """Row 0 of a ``[T, D]`` window (the absolute state) with ``state``, rows ``1..`` with ``action``.

    A 1-D ``[D]`` input, or a ``[1, D]`` one, is a state row. Both normalizers are affine, so
    ``denormalize(normalize(x)) == x`` row by row.
    """

    def __init__(self, state, action) -> None:
        self.state = state
        self.action = action

    def _apply(self, x: torch.Tensor, fn: str) -> torch.Tensor:
        if x.dim() == 1:
            return getattr(self.state, fn)(x)
        out = torch.empty_like(x)
        out[..., :1, :] = getattr(self.state, fn)(x[..., :1, :])
        if x.shape[-2] > 1:
            out[..., 1:, :] = getattr(self.action, fn)(x[..., 1:, :])
        return out

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, "normalize_action")

    def denormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, "denormalize_action")


def compute_joint_quantile_stats(roots: list[str]) -> dict[str, np.ndarray]:
    """Pooled ``action`` + ``observation.state`` statistics over every frame of ``roots``.

    Returns the flat ``{q01, q99, mean, std, min, max}`` layout ``load_action_stats`` reads.
    """
    rows = []
    for root in roots:
        files = sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no data/*/*.parquet under {root!r}")
        for path in files:
            table = pq.read_table(path, columns=[_ACTION_FEATURE, _STATE_FEATURE])
            for column in (_ACTION_FEATURE, _STATE_FEATURE):
                rows.append(np.stack(table.column(column).to_numpy(zero_copy_only=False)))
    values = np.concatenate(rows, axis=0).astype(np.float64)  # [N, 62]
    if values.shape[1] != ACTION_DIM or not np.isfinite(values).all():
        raise ValueError(f"expected finite [N, {ACTION_DIM}] joint values, got {values.shape}")
    return _stats_of(values)


def action_row_stats(pooled: dict[str, Any], relative: dict[str, Any], slices: JointSlices) -> dict[str, np.ndarray]:
    """The action rows' statistics: ``pooled`` (absolute) with the relative columns replaced."""
    mask = relative_joint_mask(slices).numpy()
    out = {}
    for key in ("q01", "q99", "mean", "std", "min", "max"):
        row = np.asarray(pooled[key], dtype=np.float64).copy()
        row[mask] = np.asarray(relative[key], dtype=np.float64)
        out[key] = row
    return out


def representation(slices: JointSlices) -> dict[str, Any]:
    """The contract's and the server's statement of the action representation."""
    if not slices:
        return {"type": "absolute"}
    return {
        "type": "relative_joints",
        "relative_joint_slices": [[int(a), int(b)] for a, b in slices],
        "reference": "observation.state at the window's first frame (the conditioning row)",
        "absolute_elsewhere": True,
    }


def representation_mismatch(stated: dict[str, Any] | None, slices: JointSlices) -> str | None:
    """Why a contract stating ``stated`` does not fit a run of ``relative_joint_slices=slices``, or None.

    Compares ``type`` and the relative slices only, as DexBench-Arena's ``representation_mismatch`` does,
    so rewording the descriptive fields keeps existing contracts valid. No ``representation`` is absolute.
    """
    stated = stated or {"type": "absolute"}
    if stated.get("type") not in ("absolute", "relative_joints"):
        return f"unknown action representation {stated!r}"
    stated_slices = stated.get("relative_joint_slices") if stated["type"] == "relative_joints" else None
    found = [[int(a), int(b)] for a, b in stated_slices or ()]
    expected = [[int(a), int(b)] for a, b in slices or ()]
    if found == expected:
        return None

    def name(s: list[list[int]]) -> str:
        return f"relative_joints {s}" if s else "absolute"

    return f"states {name(found)} actions, the run is configured for {name(expected)}"


def build_policy_contract(
    roots: list[str],
    narrow_joints: Literal["refuse", "widen"] = "refuse",
    relative_joint_slices: JointSlices = None,
    chunk_length: int = 32,
) -> dict[str, Any]:
    """What a policy server needs from the training data, in one JSON-ready dict.

    The pooled statistics sit at the top level, so ``load_action_stats`` reads the same file the
    server denormalizes with; ``contract`` carries the joint order, frame size, rate and task
    strings of ``roots``. Writing this when a run is submitted pins the normalizer to the data the
    run trained on, instead of recomputing it later from whatever the root then holds.

    A joint whose q01..q99 half-range is under ``_MIN_QUANTILE_HALF_RANGE`` makes the dataset refuse
    to train (``narrow_joints="refuse"``, the default). ``"widen"`` keeps its centre and sets its
    half-range to ``_MIN_QUANTILE_HALF_RANGE`` or, if larger, its furthest min/max from the centre,
    so no recorded value normalizes beyond +-1; ``normalization_adjustment.half_range_rad`` records it.

    With ``relative_joint_slices`` the contract also states the ``representation`` and carries
    ``action_rows``, the statistics the action rows normalize with: ``action[t + k] - state[t]``
    (``k < chunk_length``) on the relative columns and the top-level statistics, widened or not, on
    the rest. Without it the contract is the absolute one, unchanged.
    """
    names = frame_hw = fps = None
    tasks: list[str] = []
    for root in roots:
        info = json.loads((Path(root) / "meta" / "info.json").read_text())
        features = info["features"]
        shard_names = list(features[_ACTION_FEATURE]["names"])
        shard_hw = [int(v) for v in features[_IMAGE_FEATURES["left"]]["shape"][1:]]
        if features[_IMAGE_FEATURES["right"]]["shape"][1:] != shard_hw:
            raise ValueError(f"{root!r}: left and right camera frames differ in size")
        if names is None:
            names, frame_hw, fps = shard_names, shard_hw, float(info["fps"])
        elif (shard_names, shard_hw, float(info["fps"])) != (names, frame_hw, fps):
            raise ValueError(f"{root!r} disagrees with {roots[0]!r} on joint order, frame size or fps")
        for task in pq.read_table(Path(root) / "meta" / "tasks.parquet").to_pandas().index:
            if task not in tasks:
                tasks.append(str(task))
    stats = compute_joint_quantile_stats(roots)
    adjusted = {}
    if narrow_joints == "widen":
        centre = (stats["q01"] + stats["q99"]) / 2
        reach = np.maximum(np.abs(stats["min"] - centre), np.abs(stats["max"] - centre))
        for i in np.flatnonzero((stats["q99"] - stats["q01"]) / 2 < _MIN_QUANTILE_HALF_RANGE):
            half = adjusted[names[i]] = max(_MIN_QUANTILE_HALF_RANGE, float(reach[i]))
            # Rounded outwards in float32, the precision the dataset normalizes in.
            stats["q01"][i] = np.nextafter(np.float32(centre[i] - half), np.float32(-np.inf))
            stats["q99"][i] = np.nextafter(np.float32(centre[i] + half), np.float32(np.inf))
    relative: dict[str, Any] = {}
    if relative_joint_slices:
        deltas = compute_relative_action_stats(roots, relative_joint_slices, chunk_length)
        relative = {
            "representation": representation(relative_joint_slices),
            "action_rows": {k: v.tolist() for k, v in action_row_stats(stats, deltas, relative_joint_slices).items()},
            "action_rows_note": (
                f"action rows: q01/q99/mean/std/min/max of action[t+k] - state[t] (k < {chunk_length}) on the "
                "relative columns, the top-level statistics elsewhere; the state row uses the top level"
            ),
        }
    return {
        **{key: value.tolist() for key, value in stats.items()},
        **relative,
        "contract": {
            "embodiment": EMBODIMENT,
            "roots": [str(Path(root).resolve()) for root in roots],
            "joint_names": names,
            "frame_hw": frame_hw,
            "fps": fps,
            "tasks": tasks,
            "statistics": "q01/q99/mean/std/min/max pooled over action and observation.state",
        },
        **({"normalization_adjustment": {"half_range_rad": adjusted}} if adjusted else {}),
    }


class DexBenchRby1Wuji2LeRobotDataset(BaseActionLeRobotDataset):
    """RBY1-M + Wuji 2 bimanual dexterous dataset: 62-D absolute joint actions, stereo head cameras."""

    def __init__(
        self,
        root: str | list[str],
        fps: float = 30.0,
        chunk_length: int = 32,
        split_seed: int = 42,
        split_val_ratio: float = 0.0,
        split: str = "train",
        mode: str = "wam",
        action_normalization: ActionNormalization | None = "quantile",
        action_stats_path: str | None = None,
        tolerance_s: float = 1e-4,
        viewpoint: Viewpoint = "concat_view",
        camera_set: str = "stereo_stack",
        stereo_gap: int = 0,
        use_state: bool = True,
        enable_fast_init: bool = False,
        relative_joint_slices: JointSlices = None,
    ) -> None:
        if relative_joint_slices and not use_state:
            raise ValueError("relative_joint_slices needs use_state=True (the state row is the reference)")
        if camera_set not in _CAMERA_SETS:
            raise ValueError(f"Unsupported camera_set={camera_set!r}. Use one of {_CAMERA_SETS}.")
        if not isinstance(stereo_gap, int) or stereo_gap < 0:
            raise ValueError(f"stereo_gap must be a non-negative int, got {stereo_gap!r}")
        if stereo_gap and camera_set != "stereo_stack":
            raise ValueError(f"stereo_gap={stereo_gap} only applies to camera_set='stereo_stack'")
        roots = [root] if isinstance(root, str) else list(root)
        if not roots:
            raise ValueError("root must name at least one LeRobot dataset")
        for shard in roots:
            info_path = Path(shard) / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"not a LeRobot dataset root (no meta/info.json): {shard!r}")
            info = json.loads(info_path.read_text())
            if float(info["fps"]) != float(fps):
                raise ValueError(f"fps={fps} must equal the dataset rate {info['fps']} of {shard!r}")
            for feature in (_ACTION_FEATURE, _STATE_FEATURE):
                if info["features"][feature]["shape"] != [ACTION_DIM]:
                    raise ValueError(f"{shard!r}: {feature} must be {ACTION_DIM}-D")
        # Read by _load_norm_stats, which the base __init__ calls.
        self._roots = roots
        self._action_stats_path = action_stats_path
        self._relative_joint_slices = (
            [[int(a), int(b)] for a, b in relative_joint_slices] if relative_joint_slices else None
        )
        self._relative_mask = relative_joint_mask(self._relative_joint_slices)
        if action_stats_path:
            # Either way round: an absolute run refuses a relative contract as a relative run refuses an absolute one.
            contract = json.loads(Path(action_stats_path).read_text())
            problem = representation_mismatch(contract.get("representation"), self._relative_joint_slices)
            if problem:
                raise ValueError(f"{action_stats_path} {problem}")
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=EMBODIMENT,
            viewpoint=viewpoint,
            pose_convention="absolute",  # absolute joint targets; no frame-wise idle detection
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            enable_fast_init=enable_fast_init,
        )
        if self._relative_joint_slices and self._action_normalizer is not None:
            row_stats = self._load_action_row_stats(action_normalization, chunk_length)
            self._action_normalizer = StateActionRowNormalizer(
                self._action_normalizer, resolve_action_normalization(action_normalization, row_stats)
            )
        self._use_state = use_state
        self._camera_set = camera_set
        self._stereo_gap = stereo_gap
        self._all_shard_roots = roots

        # chunk_length future actions; chunk_length + 1 observation frames (current + future).
        observation_ts = [i * self._dt for i in range(self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(self._chunk_length)]
        self._delta_timestamps: dict[str, list[float]] = {_ACTION_FEATURE: action_ts}
        if self._use_state:
            self._delta_timestamps[_STATE_FEATURE] = observation_ts
        self._delta_timestamps[_IMAGE_FEATURES["left"]] = observation_ts
        if self._camera_set == "stereo_stack":
            self._delta_timestamps[_IMAGE_FEATURES["right"]] = observation_ts

        self._register_sources()

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _build_action_spec(self) -> ActionSpec:
        return build_action_spec(*(Joint(n=n, label=label, prefix=prefix) for n, label, prefix in JOINT_BLOCKS))

    def _load_norm_stats(self, action_normalization: ActionNormalization) -> dict[str, torch.Tensor]:
        """Stats from ``action_stats_path`` if given, else pooled from the dataset itself."""
        if action_normalization not in ("quantile", "meanstd", "minmax"):
            raise ValueError(
                f"{type(self).__name__} supports quantile/meanstd/minmax for absolute joints, "
                f"got {action_normalization!r}"
            )
        if self._action_stats_path:
            raw = load_action_stats(self._action_stats_path)
            source = self._action_stats_path
        else:
            raw = compute_joint_quantile_stats(self._roots)
            source = f"pooled action+state of {self._roots}"
        stats = {key: torch.from_numpy(np.asarray(value)).float() for key, value in raw.items()}
        half_range = (stats["q99"] - stats["q01"]) / 2
        if action_normalization == "quantile" and bool((half_range < _MIN_QUANTILE_HALF_RANGE).any()):
            narrow = [int(i) for i in torch.nonzero(half_range < _MIN_QUANTILE_HALF_RANGE).flatten()]
            raise ValueError(f"degenerate q01..q99 range for joints {narrow} ({source}); pass action_stats_path")
        log.info(f"{type(self).__name__}: {action_normalization} stats from {source}")
        return stats

    def _load_action_row_stats(
        self, action_normalization: ActionNormalization, chunk_length: int
    ) -> dict[str, torch.Tensor]:
        """The action rows' statistics: ``action_rows`` of ``action_stats_path``, else computed."""
        if self._action_stats_path:
            contract = json.loads(Path(self._action_stats_path).read_text())
            raw, source = contract["action_rows"], f"{self._action_stats_path} action_rows"
        else:
            pooled = compute_joint_quantile_stats(self._roots)
            deltas = compute_relative_action_stats(self._roots, self._relative_joint_slices, chunk_length)
            raw = action_row_stats(pooled, deltas, self._relative_joint_slices)
            source = f"relative action rows of {self._roots}"
        stats = {
            key: torch.as_tensor(np.asarray(raw[key]), dtype=torch.float32)
            for key in ("q01", "q99", "mean", "std", "min", "max")
        }
        half_range = (stats["q99"] - stats["q01"]) / 2
        if action_normalization == "quantile" and bool((half_range < _MIN_QUANTILE_HALF_RANGE).any()):
            narrow = [int(i) for i in torch.nonzero(half_range < _MIN_QUANTILE_HALF_RANGE).flatten()]
            raise ValueError(f"degenerate q01..q99 range of the action rows for joints {narrow} ({source})")
        log.info(f"{type(self).__name__}: {action_normalization} action-row stats from {source}")
        return stats

    def _compose_video(self, sample: dict[str, Any]) -> torch.Tensor:
        left = sample[_IMAGE_FEATURES["left"]]  # [T,C,H,W]
        if self._camera_set == "left":
            return left
        right = sample[_IMAGE_FEATURES["right"]]  # [T,C,H,W]
        return stack_stereo(left, right, self._stereo_gap)  # [T,C,2H+gap,W]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(idx)
        action = sample[_ACTION_FEATURE].float()  # [chunk, 62]
        if self._use_state:
            # Index 0 of the chunk+1 observation window is the current frame.
            state = sample[_STATE_FEATURE][0].float()  # [62]
            if self._relative_joint_slices:
                # Relative columns: each action minus the state of the window's first frame.
                action = action.clone()
                action[:, self._relative_mask] -= state[self._relative_mask]
            action = torch.cat([state.unsqueeze(0), action], dim=0)  # [chunk+1, 62]

        video = None if self._skip_video_loading else self._compose_video(sample)
        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=sample["task"],
            additional_view_description=VIEW_DESCRIPTIONS[self._camera_set],
        )
