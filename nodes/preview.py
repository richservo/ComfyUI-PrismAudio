import torch


class PrismAudioPreview:
    """
    Preview audio with a gain trim before passing to ComfyUI's built-in
    PreviewAudio node. OUTPUT_NODE = True marks it as a terminal node so
    ComfyUI will execute it even when nothing is connected downstream.
    """

    CATEGORY = "PrismAudio"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -20.0, "max": 6.0, "step": 0.1},
                ),
            }
        }

    def execute(self, audio: dict, gain_db: float):
        waveform = audio["waveform"].float()
        sample_rate = audio["sample_rate"]

        gain = 10.0 ** (gain_db / 20.0)
        waveform = (waveform * gain).clamp(-1.0, 1.0)

        return ({"waveform": waveform, "sample_rate": sample_rate},)
