from .nodes.loader import PrismAudioLoader
from .nodes.sampler import PrismAudioSampler
from .nodes.mix import PrismAudioMix
from .nodes.preview import PrismAudioPreview

NODE_CLASS_MAPPINGS = {
    "PrismAudioLoader": PrismAudioLoader,
    "PrismAudioSampler": PrismAudioSampler,
    "PrismAudioMix": PrismAudioMix,
    "PrismAudioPreview": PrismAudioPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrismAudioLoader": "PrismAudio Loader",
    "PrismAudioSampler": "PrismAudio Sampler",
    "PrismAudioMix": "PrismAudio Mix",
    "PrismAudioPreview": "PrismAudio Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
