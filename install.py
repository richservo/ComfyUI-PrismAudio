"""
Post-install hook for ComfyUI-PrismAudio.

Clones the PrismAudio source code (prismaudio branch) if not already present,
and installs the videoprism package.
"""
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIR = os.path.join(_THIS_DIR, "cd_prismaudio_source")


def _run(cmd, cwd=None):
    print(f"[ComfyUI-PrismAudio install] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ComfyUI-PrismAudio install] STDERR: {result.stderr}")
    return result.returncode == 0


def main():
    # 1. Clone PrismAudio source if not present
    if not os.path.isdir(_SOURCE_DIR):
        print("[ComfyUI-PrismAudio install] Cloning PrismAudio source (prismaudio branch)...")
        ok = _run([
            "git", "clone", "-b", "prismaudio",
            "https://github.com/liuhuadai/ThinkSound.git",
            _SOURCE_DIR,
        ])
        if not ok:
            print("[ComfyUI-PrismAudio install] WARNING: Failed to clone PrismAudio source.")
            print("  You can clone manually:")
            print(f"  git clone -b prismaudio https://github.com/liuhuadai/ThinkSound.git {_SOURCE_DIR}")
    else:
        print("[ComfyUI-PrismAudio install] PrismAudio source already present.")

    # 2. Install videoprism if not already installed
    try:
        import videoprism  # noqa: F401
        print("[ComfyUI-PrismAudio install] videoprism already installed.")
    except ImportError:
        print("[ComfyUI-PrismAudio install] Installing videoprism...")
        videoprism_dir = os.path.join(_THIS_DIR, "_videoprism_src")
        if not os.path.isdir(videoprism_dir):
            _run(["git", "clone", "https://github.com/google-deepmind/videoprism.git", videoprism_dir])
        if os.path.isdir(videoprism_dir):
            _run([sys.executable, "-m", "pip", "install", "."], cwd=videoprism_dir)

    # 3. Install facenet_pytorch (needed by synchformer, no-deps to avoid conflicts)
    try:
        import facenet_pytorch  # noqa: F401
        print("[ComfyUI-PrismAudio install] facenet_pytorch already installed.")
    except ImportError:
        print("[ComfyUI-PrismAudio install] Installing facenet_pytorch...")
        _run([sys.executable, "-m", "pip", "install", "facenet_pytorch==2.6.0", "--no-deps"])

    print("[ComfyUI-PrismAudio install] Done.")
    print()
    print("  IMPORTANT: You still need to download the model checkpoints:")
    print("    git lfs install")
    print("    git clone https://huggingface.co/FunAudioLLM/PrismAudio ckpts")
    print()
    print("  Required files in ckpts/:")
    print("    - prismaudio.ckpt")
    print("    - vae.ckpt")
    print("    - synchformer_state_dict.pth")


if __name__ == "__main__":
    main()
