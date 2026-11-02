# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sysconfig

import huggingface_hub

from cosmos_framework.inference.common import portable_serving
from cosmos_framework.inference.common.portable_serving import python_cuda_library_env, registered_vae_override


def _no_toolkit(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(portable_serving, "CUDA_TOOLKIT", tmp_path / "no-toolkit")
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path / "site-packages"))


def test_without_a_toolkit_transformer_engine_is_pointed_at_the_wheels(monkeypatch, tmp_path) -> None:
    _no_toolkit(monkeypatch, tmp_path)
    nvidia = tmp_path / "site-packages" / "nvidia"
    for name in ("cuda_nvrtc", "curand", "cudnn", "cuda_runtime/include"):
        (nvidia / name).mkdir(parents=True)
    assert python_cuda_library_env() == {
        "NVRTC_HOME": str(nvidia / "cuda_nvrtc"),
        "CURAND_HOME": str(nvidia / "curand"),
        "CUDNN_HOME": str(nvidia / "cudnn"),
        "NVTE_CUDA_INCLUDE_DIR": str(nvidia / "cuda_runtime" / "include"),
    }


def test_a_machine_with_a_toolkit_keeps_it(monkeypatch, tmp_path) -> None:
    _no_toolkit(monkeypatch, tmp_path)
    (tmp_path / "site-packages" / "nvidia" / "curand").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
    assert python_cuda_library_env() == {}


def _config(vae_path: str) -> dict:
    return {"model": {"config": {"tokenizer": {"vae_path": vae_path}}}}


def test_a_vae_path_from_another_machine_becomes_the_registered_file(monkeypatch, tmp_path) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/hub-cache/Wan2.2_VAE.pth"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    override = registered_vae_override(_config(str(tmp_path / "gone" / "Wan2.2_VAE.pth")))
    assert override == ["model.config.tokenizer.vae_path=/hub-cache/Wan2.2_VAE.pth"]
    assert calls == [
        {
            "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
            "filename": "Wan2.2_VAE.pth",
            "revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        }
    ]


def test_a_vae_that_is_here_or_unknown_is_left_as_it_is(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    here = tmp_path / "Wan2.2_VAE.pth"
    here.write_bytes(b"")
    assert registered_vae_override(_config(str(here))) == []
    assert registered_vae_override(_config(str(tmp_path / "gone" / "other_vae.pth"))) == []
    assert registered_vae_override(_config("s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth")) == []
    assert registered_vae_override({}) == []
