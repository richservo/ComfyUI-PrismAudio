import logging

import torch
import torchaudio.transforms as T

log = logging.getLogger(__name__)


class PrismAudioMix:
    """
    Mix ambient audio (PrismAudio output) with a music track at controlled
    per-stream gains. Output matches the music sample rate.
    """

    CATEGORY = "PrismAudio"
    FUNCTION = "execute"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_ambient": ("AUDIO",),
                "audio_music": ("AUDIO",),
                "ambient_gain_db": (
                    "FLOAT",
                    {"default": -12.0, "min": -40.0, "max": 12.0, "step": 0.1},
                ),
                "music_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -40.0, "max": 12.0, "step": 0.1},
                ),
                "match_duration": (
                    ["pad_ambient", "trim_ambient", "trim_music"],
                ),
            }
        }

    def execute(
        self,
        audio_ambient: dict,
        audio_music: dict,
        ambient_gain_db: float,
        music_gain_db: float,
        match_duration: str,
    ):
        # Unpack ComfyUI AUDIO dicts — waveform shape: (B, C, T)
        ambient_wave = audio_ambient["waveform"].float()
        music_wave = audio_music["waveform"].float()
        ambient_sr = audio_ambient["sample_rate"]
        music_sr = audio_music["sample_rate"]

        # dB → linear
        ambient_gain = 10.0 ** (ambient_gain_db / 20.0)
        music_gain = 10.0 ** (music_gain_db / 20.0)

        # ---- Resample ambient to music sample rate if needed ----
        if ambient_sr != music_sr:
            log.info(f"PrismAudioMix: resampling ambient {ambient_sr} Hz -> {music_sr} Hz")
            resampler = T.Resample(orig_freq=ambient_sr, new_freq=music_sr)
            # Resample operates on (C, T) or (B, C, T); apply per-batch-item
            resampled = []
            for b in range(ambient_wave.shape[0]):
                resampled.append(resampler(ambient_wave[b]))
            ambient_wave = torch.stack(resampled, dim=0)

        # ---- Align batch dimensions: broadcast to the larger batch ----
        # ComfyUI typically uses B=1, but handle gracefully
        if ambient_wave.shape[0] != music_wave.shape[0]:
            max_b = max(ambient_wave.shape[0], music_wave.shape[0])
            ambient_wave = ambient_wave.expand(max_b, -1, -1)
            music_wave = music_wave.expand(max_b, -1, -1)

        # ---- Channel matching ----
        a_ch = ambient_wave.shape[1]
        m_ch = music_wave.shape[1]
        if a_ch != m_ch:
            target_ch = max(a_ch, m_ch)
            if a_ch < target_ch:
                # Mono ambient -> stereo: duplicate channel
                ambient_wave = ambient_wave.expand(-1, target_ch, -1)
            else:
                # Stereo ambient -> mono music: mix down ambient
                ambient_wave = ambient_wave.mean(dim=1, keepdim=True)

        # ---- Length matching ----
        a_len = ambient_wave.shape[2]
        m_len = music_wave.shape[2]

        if a_len != m_len:
            if match_duration == "pad_ambient":
                # Zero-pad ambient to music length
                if a_len < m_len:
                    pad = torch.zeros(
                        ambient_wave.shape[0], ambient_wave.shape[1], m_len - a_len,
                        dtype=ambient_wave.dtype
                    )
                    ambient_wave = torch.cat([ambient_wave, pad], dim=2)
                else:
                    ambient_wave = ambient_wave[:, :, :m_len]
            elif match_duration == "trim_ambient":
                ambient_wave = ambient_wave[:, :, :m_len]
                # If ambient is shorter than music, pad with zeros to avoid shape mismatch
                if ambient_wave.shape[2] < m_len:
                    pad = torch.zeros(
                        ambient_wave.shape[0], ambient_wave.shape[1], m_len - ambient_wave.shape[2],
                        dtype=ambient_wave.dtype
                    )
                    ambient_wave = torch.cat([ambient_wave, pad], dim=2)
            elif match_duration == "trim_music":
                music_wave = music_wave[:, :, :a_len]
                # If music was shorter, pad zeros
                if music_wave.shape[2] < a_len:
                    pad = torch.zeros(
                        music_wave.shape[0], music_wave.shape[1], a_len - music_wave.shape[2],
                        dtype=music_wave.dtype
                    )
                    music_wave = torch.cat([music_wave, pad], dim=2)

        # ---- Mix ----
        mixed = ambient_wave * ambient_gain + music_wave * music_gain
        mixed = mixed.clamp(-1.0, 1.0)

        return ({"waveform": mixed, "sample_rate": music_sr},)
