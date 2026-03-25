# ComfyUI-PrismAudio

ComfyUI node pack for [PrismAudio](https://github.com/liuhuadai/ThinkSound) — video-to-ambient-audio generation. Feed in video frames and get synchronized sound effects, ambience, and foley audio back.

## Nodes

| Node | Description |
|------|-------------|
| **PrismAudio Loader** | Loads the PrismAudio feature extractor and diffusion model. Caches the pipeline across runs. |
| **PrismAudio Sampler** | Takes video frames (IMAGE) and generates matching ambient audio. Configurable steps, CFG scale, seed, and optional text caption. |
| **PrismAudio Mix** | Mixes generated ambient audio with a music track at independent dB gains, with duration matching options. |
| **PrismAudio Preview** | Preview/passthrough node with gain trim. Marked as an output node so it executes without downstream connections. |

## Installation

### Via ComfyUI Manager

Search for **ComfyUI-PrismAudio** and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/richservo/ComfyUI-PrismAudio.git
cd ComfyUI-PrismAudio
pip install -r requirements.txt
python install.py
```

The install script clones the official PrismAudio source and installs additional dependencies (VideoPrism, facenet_pytorch).

### Model Checkpoints

You must download the checkpoints separately:

```bash
cd ComfyUI/custom_nodes/ComfyUI-PrismAudio
git lfs install
git clone https://huggingface.co/FunAudioLLM/PrismAudio ckpts
```

Required files in `ckpts/`:
- `prismaudio.ckpt`
- `vae.ckpt`
- `synchformer_state_dict.pth`

## Usage

1. Add **PrismAudio Loader** — point it at your source and checkpoint directories (defaults work if installed via the install script).
2. Connect the pipeline output to **PrismAudio Sampler** along with your video frames (IMAGE batch).
3. Set FPS to match your input video, adjust steps/CFG/seed as needed.
4. Optionally add a text caption in `caption_cot` to guide generation.
5. Use **PrismAudio Preview** to listen, or **PrismAudio Mix** to blend with a music track.

## Requirements

- CUDA GPU with sufficient VRAM (~12 GB+ recommended)
- Python 3.10+
- ComfyUI

## Credits

Based on [PrismAudio / ThinkSound](https://github.com/liuhuadai/ThinkSound) by FunAudioLLM.

## License

This wrapper follows the license of the original PrismAudio project.
