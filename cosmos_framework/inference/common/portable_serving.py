# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serving a checkpoint on a machine other than the one that trained it.

Two things a training machine supplies that a serving machine may not:

- a CUDA toolkit, where Transformer Engine looks for CUDA's libraries before the environment's own;
- the absolute path the training run read the Wan2.2 VAE from, which it writes into the checkpoint's config.

Kept free of torch and Transformer Engine imports: :func:`python_cuda_library_env` must run before either loads.
"""

import os
import sysconfig
from pathlib import Path, PurePath
from typing import Any

CUDA_TOOLKIT = Path("/usr/local/cuda")
"""Transformer Engine's last place to look for a toolkit."""
WAN22_VAE_URI = "s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
"""The registered Wan2.2 VAE (``register_checkpoints``): the same file, pinned on the Hub."""


def python_cuda_library_env() -> dict[str, str]:
    """Environment that points Transformer Engine at the CUDA libraries installed from PyPI, on a machine with no CUDA
    toolkit; empty where one is found, so such a machine keeps using it.

    Transformer Engine looks for nvrtc, curand and cudnn under ``<LIB>_HOME``, ``CUDA_HOME``, ``CUDA_PATH`` and
    ``/usr/local/cuda``. Finding none, it falls back to the PyPI packages and requires cudart under ``nvidia/cudart`` or
    ``nvidia/cuda_cudart``, while the CUDA 12 wheels install it under ``nvidia/cuda_runtime``: it raises "cudart shared
    object not found" and the process stops. Setting ``<LIB>_HOME`` to the wheels' own directories lets the first
    search succeed. Pass the result as ``init_script(default_env=...)``, so a variable already set wins.
    """
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_TOOLKIT.is_dir():
        return {}
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    candidates = {
        "NVRTC_HOME": nvidia / "cuda_nvrtc",
        "CURAND_HOME": nvidia / "curand",
        "CUDNN_HOME": nvidia / "cudnn",
        "NVTE_CUDA_INCLUDE_DIR": nvidia / "cuda_runtime" / "include",
    }
    return {name: str(path) for name, path in candidates.items() if path.is_dir()}


def registered_vae_override(config_dict: dict[str, Any]) -> list[str]:
    """An experiment override that points the video tokenizer at the registered Wan2.2 VAE, when the checkpoint's
    config names a local file of that name that is not on this machine; otherwise none.

    The training recipe reads the VAE from ``WAN_VAE_PATH`` and its config keeps that absolute path, so a checkpoint
    served elsewhere could not load. The registry pins the same file on the Hub (``WAN22_VAE_URI``), which is
    downloaded here with huggingface_hub (from its cache under ``HF_HUB_OFFLINE``) rather than through the registry's
    own download, which runs the ``hf`` CLI through ``uv``: a serving machine may have neither.
    """
    tokenizer = ((config_dict.get("model") or {}).get("config") or {}).get("tokenizer") or {}
    vae_path = tokenizer.get("vae_path")
    if not isinstance(vae_path, str) or "://" in vae_path or Path(vae_path).exists():
        return []
    if PurePath(vae_path).name != PurePath(WAN22_VAE_URI).name:
        return []
    from huggingface_hub import hf_hub_download

    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import CheckpointConfig

    register_checkpoints()
    hf = CheckpointConfig.from_uri(WAN22_VAE_URI).hf
    local = hf_hub_download(repo_id=hf.repository, filename=hf.filename, revision=hf.revision)
    return [f"model.config.tokenizer.vae_path={local}"]
