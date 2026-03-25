import hashlib
import json
import logging
import os
import sys

import torch
import comfy.model_management

log = logging.getLogger(__name__)

# The directory this node package lives in — used to resolve relative paths
# like "cd_prismaudio_source" and "ckpts" from the node folder, not CWD.
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Module-level cache keyed on (resolved_ckpt_dir, dtype)
_PIPELINE_CACHE: dict = {}

# PrismAudio frame-rate / resolution constants (from app.py)
_CLIP_FPS = 4
_CLIP_SIZE = 288
_SYNC_FPS = 25
_SYNC_SIZE = 224


class PrismAudioLoader:
    """
    Load the PrismAudio feature extractor and diffusion model once,
    cache them, and return a PRISMAUDIO_PIPELINE handle for the sampler.
    """

    CATEGORY = "PrismAudio"
    FUNCTION = "execute"
    RETURN_TYPES = ("PRISMAUDIO_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prismaudio_source_dir": (
                    "STRING",
                    {"default": "cd_prismaudio_source"},
                ),
                "ckpt_dir": (
                    "STRING",
                    {"default": "ckpts"},
                ),
                "dtype": (
                    ["bf16", "fp16", "fp32"],
                    {"default": "bf16"},
                ),
                "offload_features_after_extract": (
                    "BOOLEAN",
                    {"default": True},
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, prismaudio_source_dir, ckpt_dir, dtype, offload_features_after_extract):
        # Resolve relative paths from the node package dir, not CWD
        resolved = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(_NODE_DIR, ckpt_dir)
        key_str = f"{os.path.abspath(resolved)}:{dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def execute(
        self,
        prismaudio_source_dir: str,
        ckpt_dir: str,
        dtype: str,
        offload_features_after_extract: bool,
    ):
        device = comfy.model_management.get_torch_device()

        # Resolve relative paths from the node package dir, not CWD.
        # Absolute paths are used as-is so users can point elsewhere.
        if os.path.isabs(prismaudio_source_dir):
            source_dir = prismaudio_source_dir
        else:
            source_dir = os.path.join(_NODE_DIR, prismaudio_source_dir)
        source_dir = os.path.abspath(source_dir)

        if os.path.isabs(ckpt_dir):
            ckpt_dir_abs = ckpt_dir
        else:
            ckpt_dir_abs = os.path.join(_NODE_DIR, ckpt_dir)
        ckpt_dir_abs = os.path.abspath(ckpt_dir_abs)

        cache_key = (ckpt_dir_abs, dtype)
        if cache_key in _PIPELINE_CACHE:
            log.info("PrismAudioLoader: returning cached pipeline")
            # Update the offload flag in the config dict in case it changed
            cached = _PIPELINE_CACHE[cache_key]
            cached[2]["offload_features_after_extract"] = offload_features_after_extract
            return (cached,)

        # Ensure the PrismAudio source tree is importable
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)

        # Resolve all file paths
        ckpt_path = os.path.join(ckpt_dir_abs, "prismaudio.ckpt")
        vae_ckpt_path = os.path.join(ckpt_dir_abs, "vae.ckpt")
        synchformer_ckpt_path = os.path.join(ckpt_dir_abs, "synchformer_state_dict.pth")
        model_config_path = os.path.join(source_dir, "PrismAudio", "configs", "model_configs", "prismaudio.json")
        vae_config_path = os.path.join(source_dir, "PrismAudio", "configs", "model_configs", "stable_audio_2_0_vae.json")

        for label, path in [
            ("prismaudio.ckpt", ckpt_path),
            ("vae.ckpt", vae_ckpt_path),
            ("synchformer_state_dict.pth", synchformer_ckpt_path),
            ("prismaudio.json", model_config_path),
            ("stable_audio_2_0_vae.json", vae_config_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"PrismAudioLoader: required file not found: {label} at {path}")

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        # ------------------------------------------------------------------
        # VRAM baseline
        # ------------------------------------------------------------------
        vram_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

        # ------------------------------------------------------------------
        # 1. Feature extractor  (FeaturesUtils)
        # ------------------------------------------------------------------
        # JAX must be pinned to CPU *before* it is imported, otherwise it
        # will claim the GPU and fight with PyTorch for VRAM.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "False")

        log.info("PrismAudioLoader: loading FeaturesUtils ...")
        from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils

        feature_extractor = FeaturesUtils(
            vae_ckpt=None,
            vae_config=vae_config_path,
            enable_conditions=True,
            synchformer_ckpt=synchformer_ckpt_path,
        )
        # NOTE: FeaturesUtils is NOT cast to torch_dtype — T5 and Synchformer
        # manage their own precision internally.  Only the diffusion model
        # benefits from explicit dtype control.
        feature_extractor = feature_extractor.eval().to(device)
        log.info("PrismAudioLoader: FeaturesUtils ready")

        # ------------------------------------------------------------------
        # 2. Diffusion model
        # ------------------------------------------------------------------
        log.info("PrismAudioLoader: loading diffusion model ...")
        from PrismAudio.models import create_model_from_config
        from PrismAudio.models.utils import load_ckpt_state_dict

        with open(model_config_path) as f:
            model_config = json.load(f)

        diffusion = create_model_from_config(model_config)
        diffusion.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        vae_state = load_ckpt_state_dict(vae_ckpt_path, prefix="autoencoder.")
        diffusion.pretransform.load_state_dict(vae_state)

        diffusion = diffusion.eval().to(device=device, dtype=torch_dtype)
        log.info("PrismAudioLoader: diffusion model ready")

        # ------------------------------------------------------------------
        # VRAM after
        # ------------------------------------------------------------------
        vram_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        log.info(
            f"PrismAudioLoader: VRAM delta = {(vram_after - vram_before) / 1024**3:.2f} GB "
            f"(before={vram_before / 1024**3:.2f} GB, after={vram_after / 1024**3:.2f} GB)"
        )

        config_dict = {
            "source_dir": source_dir,
            "ckpt_dir": ckpt_dir_abs,
            "dtype": dtype,
            "torch_dtype": torch_dtype,
            "offload_features_after_extract": offload_features_after_extract,
            "ckpt_path": ckpt_path,
            "vae_ckpt_path": vae_ckpt_path,
            "synchformer_ckpt_path": synchformer_ckpt_path,
            "model_config_path": model_config_path,
            "vae_config_path": vae_config_path,
            "model_config": model_config,
            # Frame processing constants consumed by the sampler
            "clip_fps": _CLIP_FPS,
            "clip_size": _CLIP_SIZE,
            "sync_fps": _SYNC_FPS,
            "sync_size": _SYNC_SIZE,
        }

        pipeline = (feature_extractor, diffusion, config_dict)
        _PIPELINE_CACHE[cache_key] = pipeline

        return (pipeline,)
