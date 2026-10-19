# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP policy server for the DexBench RBY1-M + Wuji 2 Cosmos3 action policy.

Serves a checkpoint of ``action_policy_dexbench_rby1_wuji2_nano``: 62 absolute joint targets per
step, the current joint state as the clean conditioning row, the head stereo pair stacked left
over right. Every request goes through the training-time ``ActionTransformPipeline`` with the
dataset settings read from the checkpoint's own config, so the padded frame, the JSON prompt and
the normalized state row are built by the same code that built them in training; only the
diffusion start (frames 1.. and action rows 1.. are generated) differs.

A run trained with ``relative_joint_slices`` predicts those columns relative to the state. The
server denormalizes them with the contract's ``action_rows`` and adds the request's state back, so
the reply is absolute joint targets either way; ``/info``'s ``action_representation`` states which
representation the checkpoint was trained with, and a contract stating another one is refused.

``--contract-path`` is the run's ``policy_contract.json``
(``cosmos_framework.scripts.write_dexbench_policy_contract``): pooled q01/q99 statistics plus the
joint order, frame size and rate of the data the run trained on. The server normalizes the state
and denormalizes the chunk with it and refuses frames of another size.

Endpoints (JSON; with ``--api-token-file`` every request needs ``Authorization: Bearer <token>``):

- ``GET /info``: contract, sampling settings, checkpoint and ``action_representation``.
- ``POST /predict``:
  ``{"images": {"left": <b64 PNG>, "right": <b64 PNG>}, "state": [62 floats], "prompt": str,
  "seed": int (optional)}`` ->
  ``{"action": [[62 floats] x chunk], "state_echo_error": float, "seed": int, "ms_infer": float}``.
  ``state_echo_error`` is the max abs difference (rad) between the input state and the model's
  denormalized row 0, which is the clean conditioning row and should come back unchanged.

Example (DCP checkpoint written by the training run; see docs/action_policy_dexbench_rby1_wuji2_server.md):

  python -m cosmos_framework.scripts.write_dexbench_policy_contract \\
    --config-file RUN/cosmos3_action_dexbench/action_sft/EXP/config.yaml -o RUN/policy_contract.json
  python -m cosmos_framework.scripts.action_policy_server_dexbench \\
    --checkpoint-path RUN/cosmos3_action_dexbench/action_sft/EXP/checkpoints/iter_000004000 \\
    --config-file RUN/cosmos3_action_dexbench/action_sft/EXP/config.yaml \\
    --contract-path RUN/policy_contract.json --port 8000
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import base64
import binascii
import inspect
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import tyro
from PIL import Image

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_dexbench_rby1_wuji2_sft_dataset,
)
from cosmos_framework.data.generator.action.datasets.dexbench_rby1_wuji2_lerobot_dataset import (
    ACTION_DIM,
    EMBODIMENT,
    VIEW_DESCRIPTIONS,
    StateActionRowNormalizer,
    find_dataset_node,
    relative_joint_mask,
    representation,
    stack_stereo,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    load_action_stats,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ConfigFileType, tyro_cli
from cosmos_framework.inference.common.config import deserialize_config_dict
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log

# Dataset keywords the server reproduces; absent ones take the SFT builder's defaults.
_DATASET_KEYS = (
    "fps",
    "chunk_length",
    "mode",
    "viewpoint",
    "camera_set",
    "stereo_gap",
    "use_state",
    "action_normalization",
    "resolution",
    "append_viewpoint_info",
    "append_duration_fps_timestamps",
    "append_resolution_info",
    "append_idle_frames",
    "format_prompt_as_json",
    "relative_joint_slices",
)


def _dataset_settings(config: dict) -> dict[str, Any]:
    node = find_dataset_node(config)
    defaults = inspect.signature(get_action_dexbench_rby1_wuji2_sft_dataset).parameters
    settings = {key: node.get(key, defaults[key].default) for key in _DATASET_KEYS}
    if settings["mode"] != "wam" or not settings["use_state"]:
        raise ValueError(f"served recipe must be mode='wam' with use_state=True, got {settings}")
    return settings


def _load_config_dict(setup_args: OmniSetupArgs) -> dict:
    """The experiment config as a plain dict (``OmniInference`` saves module configs as YAML)."""
    if setup_args.config_file_type == ConfigFileType.MODULE:
        return deserialize_config_dict(Path(setup_args.output_dir) / "config.yaml")
    return deserialize_config_dict(Path(setup_args.config_file))


def _decode_png(b64: str, expected_hw: tuple[int, int], name: str) -> torch.Tensor:
    """``[3,H,W]`` uint8 RGB from a base64 PNG of exactly ``expected_hw``."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"images.{name}: invalid base64: {e}") from e
    with Image.open(io.BytesIO(raw)) as img:
        if img.mode != "RGB":
            raise ValueError(f"images.{name}: expected an RGB PNG, got mode {img.mode}")
        arr = np.asarray(img, dtype=np.uint8).copy()
    if arr.shape[:2] != tuple(expected_hw):
        raise ValueError(
            f"images.{name}: frame is {arr.shape[1]}x{arr.shape[0]}, the policy was trained on "
            f"{expected_hw[1]}x{expected_hw[0]} (area-downscaled from the render)"
        )
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DexBenchServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint: tyro.conf.OmitArgPrefixes[CheckpointOverrides] = CheckpointOverrides.model_construct()
    """Checkpoint and config. ``use_ema_weights`` defaults True at inference."""

    contract_path: Path
    """The run's ``policy_contract.json`` (``write_dexbench_policy_contract``): stats, joint order, frame size."""

    output_dir: Path | None = None
    """``OmniInference`` output directory (saved config.yaml). Defaults to /tmp/cosmos3_action_server."""

    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler."""
    seed: int = 0
    """Base seed. A request without ``seed`` uses ``seed + request number``, so a repeated
    observation does not return the same chunk forever."""
    guidance: float = 1.0
    """Guidance scale; 1.0 runs one forward per step (no classifier-free guidance)."""
    num_steps: int = 4
    """Denoising steps per chunk."""
    shift: float = 5.0
    """UniPC sampler shift. The recipe trains at shift 3; at 10 steps, sampling at 5 matched the recorded
    actions of 64 training windows slightly better than 3."""

    host: str = "127.0.0.1"
    """Bind address. Keep it off 0.0.0.0 unless the port is firewalled; pair with --api-token-file."""
    port: int = 8000
    """Bind port."""
    api_token_file: Path | None = None
    """File holding a bearer token every request must carry."""
    dump_dir: Path | None = None
    """If set, save each request's stacked frame, state and returned chunk here."""


class DexBenchRequestBuilder:
    """Turns a /predict request into the training pipeline's sample dict; no model needed."""

    def __init__(self, config: dict, contract_path: Path) -> None:
        self.settings = _dataset_settings(config)
        contract_file = json.loads(Path(contract_path).read_text())
        self.contract = contract_file["contract"]
        if self.contract["embodiment"] != EMBODIMENT or len(self.contract["joint_names"]) != ACTION_DIM:
            raise ValueError(f"{contract_path} is not a {ACTION_DIM}-joint {EMBODIMENT} contract")
        if float(self.contract["fps"]) != float(self.settings["fps"]):
            raise ValueError(f"contract fps {self.contract['fps']} != training fps {self.settings['fps']}")
        stats = {k: torch.from_numpy(np.asarray(v)).float() for k, v in load_action_stats(str(contract_path)).items()}
        self.normalizer = resolve_action_normalization(self.settings["action_normalization"], stats)
        # Relative joints (config.yaml's relative_joint_slices): the contract must state the same
        # representation; action rows denormalize with its action_rows block and get the state added back.
        slices = self.settings["relative_joint_slices"]
        self.representation = representation(slices)
        if contract_file.get("representation", {"type": "absolute"}) != self.representation:
            raise ValueError(
                f"{contract_path} states representation {contract_file.get('representation')}, the run's config "
                f"trains {self.representation}"
            )
        self.relative_mask = relative_joint_mask(slices)
        if slices:
            rows = {k: torch.as_tensor(np.asarray(contract_file["action_rows"][k]), dtype=torch.float32)
                    for k in ("q01", "q99", "mean", "std", "min", "max")}
            self.normalizer = StateActionRowNormalizer(
                self.normalizer, resolve_action_normalization(self.settings["action_normalization"], rows)
            )
        self.frame_hw = tuple(int(v) for v in self.contract["frame_hw"])
        self.chunk = int(self.settings["chunk_length"])
        self.max_action_dim = int(config["model"]["config"]["max_action_dim"])
        self.domain_id = get_domain_id(EMBODIMENT)
        # The training transform, minus tokenization (the model tokenizes the caption itself)
        # and minus caption dropout.
        self.transform = ActionTransformPipeline(
            tokenizer_config=None,
            cfg_dropout_rate=0.0,
            max_action_dim=self.max_action_dim,
            append_viewpoint_info=self.settings["append_viewpoint_info"],
            append_duration_fps_timestamps=self.settings["append_duration_fps_timestamps"],
            append_resolution_info=self.settings["append_resolution_info"],
            append_idle_frames=self.settings["append_idle_frames"],
            format_prompt_as_json=self.settings["format_prompt_as_json"],
        )

    @property
    def camera_names(self) -> list[str]:
        return ["left"] if self.settings["camera_set"] == "left" else ["left", "right"]

    def to_absolute(self, pred_normalized: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``[chunk+1, 62]`` model rows -> denormalized ``[chunk+1, 62]``: row 0 the state echo, rows 1..
        absolute joint targets (the state added back on the relative columns)."""
        pred = self.normalizer.denormalize_action(pred_normalized)
        if bool(self.relative_mask.any()):
            pred = pred.clone()
            pred[1:, self.relative_mask] += state[self.relative_mask]
        return pred

    def build_sample(self, req: dict[str, Any]) -> dict[str, Any]:
        """One request as the training pipeline's output dict (padded video, prompt, action)."""
        images = req.get("images")
        state = req.get("state")
        prompt = req.get("prompt")
        if not isinstance(images, dict) or not isinstance(prompt, str) or not prompt:
            raise ValueError("request needs 'images' {left, right}, 'state' and a non-empty 'prompt'")
        frames = [_decode_png(images.get(n, ""), self.frame_hw, n) for n in self.camera_names]
        # [3,H,W], or [3,2H+gap,W] with the left camera on top, built as the training dataset builds it
        frame = frames[0] if len(frames) == 1 else stack_stereo(*frames, gap=int(self.settings["stereo_gap"]))
        state_t = torch.as_tensor(state, dtype=torch.float32).reshape(-1)  # [62]
        if state_t.shape[0] != ACTION_DIM or not torch.isfinite(state_t).all():
            raise ValueError(f"'state' must be {ACTION_DIM} finite joint positions in contract order")
        t = self.chunk + 1
        video = frame.unsqueeze(1).repeat(1, t, 1, 1)  # [3,T,H,W]; frames 1.. are generated
        action = torch.zeros((t, ACTION_DIM), dtype=torch.float32)  # [T,62]; rows 1.. are generated
        action[0] = self.normalizer.normalize_action(state_t.unsqueeze(0))[0]
        data = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.settings["fps"], dtype=torch.long),
            "mode": "wam",
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": self.settings["viewpoint"],
            "additional_view_description": VIEW_DESCRIPTIONS[self.settings["camera_set"]],
        }
        out = self.transform(data, self.settings["resolution"])
        caption = out["ai_caption"]
        out["ai_caption"] = json.dumps(caption) if isinstance(caption, dict) else caption
        out["state"] = state_t
        out["stacked_frame"] = frame
        return out


class DexBenchPolicyService:
    def __init__(self, args: DexBenchServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if not getattr(args.checkpoint, "checkpoint_path", ""):
            raise ValueError("--checkpoint-path is required")
        maybe_init_distributed()
        overrides = OmniSetupOverrides.model_validate(args.checkpoint.model_dump())
        overrides.output_dir = args.output_dir or DEFAULT_FALLBACK_OUTPUT_DIR
        overrides.sampler = args.sampler
        # The server returns actions, never media, so OmniInference's text/video guardrails would never
        # run; building them downloads a blocklist from the Hub through `uv`, which an offline eval pod
        # (HF_HUB_OFFLINE, no uv in the image) cannot do. action_policy_server_robolab skips them too.
        overrides.guardrails = False
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)
        log.info(f"[dexbench-server] loading {setup_args.checkpoint_path} ({setup_args.config_file})")
        self.pipe = OmniInference.create(setup_args)
        self.model = self.pipe.model
        self.model.eval()
        self.setup_args = self.pipe.setup_args
        self.builder = DexBenchRequestBuilder(_load_config_dict(self.setup_args), args.contract_path)
        if self.builder.max_action_dim != int(self.model.config.max_action_dim):
            raise ValueError("config max_action_dim disagrees with the loaded model")
        self.input_video_key = getattr(self.model, "input_video_key", None) or self.model.config.input_video_key
        self.args = args
        self.token = args.api_token_file.read_text().strip() if args.api_token_file else None
        self._lock = threading.Lock()
        self._requests = 0
        log.info(f"[dexbench-server] dataset settings {self.builder.settings}; frames {self.builder.frame_hw}; "
                 f"domain {self.builder.domain_id}")

    def info(self) -> dict[str, Any]:
        b = self.builder
        return {
            "checkpoint": str(self.setup_args.checkpoint_path),
            "config_file": str(self.setup_args.config_file),
            "embodiment": EMBODIMENT,
            "joint_names": b.contract["joint_names"],
            "frame_hw": list(b.frame_hw),
            "camera_set": b.settings["camera_set"],
            "stereo_gap": b.settings["stereo_gap"],
            "fps": b.settings["fps"],
            "action_horizon": b.chunk,
            "tasks": b.contract["tasks"],
            "num_steps": self.args.num_steps,
            "shift": self.args.shift,
            "guidance": self.args.guidance,
            "sampler": self.args.sampler,
            "seed": self.args.seed,
            "action_representation": b.representation,
        }

    def predict(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        with self._lock:
            self._requests += 1
            request_id = self._requests
        seed = req.get("seed")
        seed = int(seed) if isinstance(seed, int) else self.args.seed + request_id
        sample = self.builder.build_sample(req)
        batch = {
            self.input_video_key: [[sample["video"]]],
            "raw_action_dim": [sample["raw_action_dim"]],
            "action_processing_record": [sample["action_processing_record"]],
            "action": [[sample["action"]]],
            "mode": ["wam"],
            "ai_caption": [sample["ai_caption"]],
            "prompt": [sample["ai_caption"]],
            "conditioning_fps": [sample["conditioning_fps"]],
            "image_size": sample["image_size"].unsqueeze(0).to(device="cuda"),
            "domain_id": [sample["domain_id"]],
            "sequence_plan": [sample["sequence_plan"]],
        }
        t1 = time.monotonic()
        with self._lock, torch.inference_mode():
            out = self.model.generate_samples_from_batch(
                batch,
                guidance=self.args.guidance,
                seed=[seed],
                num_steps=self.args.num_steps,
                shift=self.args.shift,
                has_negative_prompt=False,
            )
        t2 = time.monotonic()
        pred = out["action"][0].float().reshape(-1, out["action"][0].shape[-1])[:, :ACTION_DIM].cpu()  # [T,62]
        pred = self.builder.to_absolute(pred, sample["state"])  # [T,62] radians, absolute targets
        if pred.shape[0] != self.builder.chunk + 1 or not torch.isfinite(pred).all():
            raise RuntimeError(f"model returned {tuple(pred.shape)} actions, expected {self.builder.chunk + 1} finite rows")
        echo = float((pred[0] - sample["state"]).abs().max())
        chunk = pred[1:]
        if self.args.dump_dir is not None:
            dump = self.args.dump_dir / f"req{request_id:06d}"
            dump.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sample["stacked_frame"].permute(1, 2, 0).numpy()).save(dump / "frame.png")
            (dump / "io.json").write_text(json.dumps({
                "prompt": req["prompt"], "caption": sample["ai_caption"], "seed": seed,
                "state": sample["state"].tolist(), "action": chunk.tolist(), "state_echo_error": echo,
            }))
        result = {
            "action": chunk.tolist(),
            "state_echo_error": echo,
            "seed": seed,
            "request_id": request_id,
            "ms_prep": (t1 - t0) * 1e3,
            "ms_infer": (t2 - t1) * 1e3,
        }
        log.info(f"[dexbench-server] request {request_id} seed {seed} prep {result['ms_prep']:.0f} ms "
                 f"infer {result['ms_infer']:.0f} ms echo {echo:.4f} rad")
        return result


class _Handler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self) -> bool:
        token = self.server.service.token  # type: ignore[attr-defined]
        if token is None or self.headers.get("Authorization", "") == f"Bearer {token}":
            return True
        self._send(401, {"error": "missing or wrong bearer token"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path == "/info":
            self._send(200, self.server.service.info())  # type: ignore[attr-defined]
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            self._send(200, self.server.service.predict(req))  # type: ignore[attr-defined]
        except Exception as e:  # report to the client instead of dropping the connection
            traceback.print_exc()
            self._send(400 if isinstance(e, ValueError) else 500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def main() -> None:
    args = tyro_cli(
        DexBenchServerArgs,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes, tyro.conf.CascadeSubcommandArgs, tyro.conf.OmitSubcommandPrefixes),
    )
    service = DexBenchPolicyService(args)
    httpd = ThreadingHTTPServer((args.host, int(args.port)), _Handler)
    setattr(httpd, "service", service)
    log.info(f"[dexbench-server] serving http://{args.host}:{args.port} (node {get_local_ip()}); "
             f"token {'required' if service.token else 'off'}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
