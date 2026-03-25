import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

import comfy.model_management

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100


def _prepare_video_frames(
    images: torch.Tensor,
    fps: float,
    duration: float,
    clip_fps: int,
    clip_size: int,
    sync_fps: int,
    sync_size: int,
) -> tuple:
    """
    Convert a ComfyUI IMAGE batch (B, H, W, C) float32 0-1 into the two frame
    tensors expected by FeaturesUtils.

    Returns:
        clip_chunk : (1, T_clip, H=clip_size, W=clip_size, C=3) float32  [0, 1]
        sync_chunk : (1, T_sync, C=3, H=sync_size, W=sync_size) float32  normalised
    """
    # images: (B, H, W, C) — reorder to (B, C, H, W) for torchvision ops
    frames_bchw = images.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    src_fps = fps  # frames per second of the input batch

    # ---- Build sync transform (mirrors app.py) ----
    sync_transform = v2.Compose([
        v2.Resize(sync_size, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(sync_size),
        v2.ToDtype(torch.float32, scale=False),  # already float32, no rescale needed
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def _resample_frames(frames_bchw: torch.Tensor, src_fps: float, target_fps: float, target_n: int) -> torch.Tensor:
        """Nearest-frame resampling: pick indices at target_fps cadence."""
        indices = [min(round(i * src_fps / target_fps), len(frames_bchw) - 1) for i in range(target_n)]
        return frames_bchw[indices]  # (target_n, C, H, W)

    def _pad_or_trim(frames: torch.Tensor, target_n: int) -> torch.Tensor:
        n = frames.shape[0]
        if n >= target_n:
            return frames[:target_n]
        # Repeat last frame to pad
        pad = frames[-1:].expand(target_n - n, -1, -1, -1)
        return torch.cat([frames, pad], dim=0)

    # ---- Clip stream ----
    clip_n = round(clip_fps * duration)
    clip_frames = _resample_frames(frames_bchw, src_fps, clip_fps, clip_n)
    clip_frames = _pad_or_trim(clip_frames, clip_n)

    # Pad to square then resize to clip_size
    _, c, h, w = clip_frames.shape
    max_side = max(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    clip_frames = F.pad(clip_frames, pad=padding, mode="constant", value=0)
    clip_frames = F.interpolate(clip_frames, size=(clip_size, clip_size), mode="bilinear", align_corners=False)

    # (T, C, H, W) -> (T, H, W, C) as expected by FeaturesUtils.encode_video_and_text_with_videoprism
    clip_frames_hwc = clip_frames.permute(0, 2, 3, 1)  # (T, H, W, C)
    clip_chunk = clip_frames_hwc.unsqueeze(0)           # (1, T, H, W, C)

    # ---- Sync stream ----
    sync_n = round(sync_fps * duration)
    sync_frames = _resample_frames(frames_bchw, src_fps, sync_fps, sync_n)
    sync_frames = _pad_or_trim(sync_frames, sync_n)
    # Apply sync transform: resize+crop+normalize, input expected as uint8 or float
    # v2 transforms work on (C, H, W) or batches; apply per-frame then stack
    sync_processed = torch.stack([sync_transform(f) for f in sync_frames], dim=0)  # (T, C, H, W)
    sync_chunk = sync_processed.unsqueeze(0)  # (1, T, C, H, W)

    return clip_chunk, sync_chunk


class PrismAudioSampler:
    """
    Core inference node: ComfyUI IMAGE batch in, ambient audio out.
    """

    CATEGORY = "PrismAudio"
    FUNCTION = "execute"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PRISMAUDIO_PIPELINE",),
                "images": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.1},
                ),
                "duration_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999},
                ),
                "steps": (
                    "INT",
                    {"default": 24, "min": 1, "max": 200},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "seed": (
                    "INT",
                    {"default": 42},
                ),
                "caption_cot": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            }
        }

    def execute(
        self,
        pipeline,
        images: torch.Tensor,
        fps: float,
        duration_frames: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        caption_cot: str,
    ):
        feature_extractor, diffusion, config_dict = pipeline

        device = comfy.model_management.get_torch_device()

        # Determine duration: 0 = use all input frames
        if duration_frames <= 0:
            num_frames = images.shape[0]
        else:
            num_frames = duration_frames
        duration = num_frames / fps

        log.info(f"PrismAudioSampler: generating {duration:.2f}s of audio from {images.shape[0]} frames @ {fps} fps")

        clip_fps = config_dict["clip_fps"]
        clip_size = config_dict["clip_size"]
        sync_fps = config_dict["sync_fps"]
        sync_size = config_dict["sync_size"]
        torch_dtype = config_dict["torch_dtype"]
        offload = config_dict["offload_features_after_extract"]

        # ------------------------------------------------------------------
        # Step 1: Prepare frame tensors
        # ------------------------------------------------------------------
        clip_chunk, sync_chunk = _prepare_video_frames(
            images, fps, duration, clip_fps, clip_size, sync_fps, sync_size
        )
        # clip_chunk: (1, T, H, W, C) — FeaturesUtils takes this on CPU (JAX handles it)
        # sync_chunk: (1, T, C, H, W) — needs to go to the feature extractor device

        # ------------------------------------------------------------------
        # Step 2: Feature extraction
        # ------------------------------------------------------------------
        # Ensure feature extractor is on the target device
        feature_extractor_device = next(feature_extractor.parameters()).device
        if str(feature_extractor_device) != str(device):
            feature_extractor = feature_extractor.to(device)

        log.info("PrismAudioSampler: extracting features ...")
        t0 = time.time()

        with torch.no_grad():
            # T5 text features
            text_features = feature_extractor.encode_t5_text([caption_cot])
            text_features_cpu = text_features[0].cpu()

            # Video features via VideoPrism (JAX, takes CPU numpy internally)
            video_feat, frame_embed, _, text_feat = \
                feature_extractor.encode_video_and_text_with_videoprism(
                    clip_chunk.cpu(), [caption_cot]
                )

            global_video_features = torch.tensor(np.array(video_feat)).squeeze(0).cpu()
            video_features = torch.tensor(np.array(frame_embed)).squeeze(0).cpu()
            global_text_features = torch.tensor(np.array(text_feat)).squeeze(0).cpu()

            # Sync features via Synchformer (PyTorch, needs CUDA)
            sync_input = sync_chunk.to(device)
            sync_features = feature_extractor.encode_video_with_sync(sync_input)[0].cpu()

        log.info(f"PrismAudioSampler: feature extraction done in {time.time() - t0:.2f}s")

        # ------------------------------------------------------------------
        # Step 3: Optionally offload feature extractor
        # ------------------------------------------------------------------
        if offload:
            feature_extractor.to("cpu")
            comfy.model_management.soft_empty_cache()
            log.info("PrismAudioSampler: feature extractor offloaded to CPU")

        # ------------------------------------------------------------------
        # Step 4: Ensure diffusion model is on device
        # ------------------------------------------------------------------
        diffusion_device = next(diffusion.parameters()).device
        if str(diffusion_device) != str(device):
            diffusion = diffusion.to(device)
            log.info("PrismAudioSampler: diffusion model moved to device")

        # ------------------------------------------------------------------
        # Step 5: Build metadata dict for the conditioner
        # ------------------------------------------------------------------
        latent_length = round(SAMPLE_RATE * duration / 2048)

        meta = {
            "id": "comfy",
            "relpath": "comfy.npz",
            "path": "comfy.npz",
            "caption_cot": caption_cot,
            "video_exist": torch.tensor(True),
            "text_features": text_features_cpu,
            "global_video_features": global_video_features,
            "video_features": video_features,
            "global_text_features": global_text_features,
            "sync_features": sync_features,
        }

        # ------------------------------------------------------------------
        # Step 6: Diffusion sampling
        # ------------------------------------------------------------------
        from PrismAudio.inference.sampling import sample, sample_discrete_euler

        model_config = config_dict["model_config"]
        diffusion_objective = model_config["model"]["diffusion"]["diffusion_objective"]

        meta_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in meta.items()
        }
        metadata = (meta_on_device,)

        log.info(f"PrismAudioSampler: running diffusion ({steps} steps, cfg={cfg_scale}, seed={seed}) ...")
        t0 = time.time()

        torch.manual_seed(seed)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                conditioning = diffusion.conditioner(metadata, device)

            video_exist = torch.stack([item["video_exist"] for item in metadata], dim=0)
            if "metaclip_features" in conditioning:
                conditioning["metaclip_features"][~video_exist] = \
                    diffusion.model.model.empty_clip_feat
            if "sync_features" in conditioning:
                conditioning["sync_features"][~video_exist] = \
                    diffusion.model.model.empty_sync_feat

            cond_inputs = diffusion.get_conditioning_inputs(conditioning)
            noise = torch.randn([1, diffusion.io_channels, latent_length], device=device)

            with torch.amp.autocast("cuda"):
                if diffusion_objective == "v":
                    fakes = sample(
                        diffusion.model, noise, steps, 0,
                        **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True,
                    )
                else:  # rectified_flow (default for PrismAudio)
                    fakes = sample_discrete_euler(
                        diffusion.model, noise, steps,
                        **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True,
                    )

                if diffusion.pretransform is not None:
                    fakes = diffusion.pretransform.decode(fakes)

        log.info(f"PrismAudioSampler: diffusion done in {time.time() - t0:.2f}s")

        # ------------------------------------------------------------------
        # Step 7: Post-process → ComfyUI AUDIO dict
        # ------------------------------------------------------------------
        # fakes shape: (1, 2, T_samples) — stereo float
        waveform = fakes.to(torch.float32)

        # Normalise to [-1, 1] (matches app.py logic but keep as float for ComfyUI)
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        waveform = waveform.clamp(-1.0, 1.0).cpu()

        audio_out = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        return (audio_out,)
