"""
One-shot Colab bootstrap for liya_diplomCC.

Run this in the FIRST cell of any notebook on Google Colab when you want a
clean fresh setup (clones the repo into Drive, installs deps, mounts, sets paths).

After running once per Colab runtime, each notebook's existing Cell 0 will
work as-is (it expects DRIVE_ROOT to point at the project inside MyDrive).

Usage:
    !curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/liya_diplomCC/main/setup_colab.py | python -

Or, after you've cloned the repo into Drive once:
    !python /content/drive/MyDrive/liya_diplomCC/setup_colab.py
"""

import os
import subprocess
import sys

GITHUB_URL = "https://github.com/superpevel/liya_diploma.git"  # EDIT THIS
DRIVE_ROOT = "/content/drive/MyDrive/liya_diploma"
AI_TOOLKIT = "/content/ai-toolkit"


def sh(cmd: str) -> int:
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    # Mount Drive
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")

    # Clone or update the project in Drive
    if not os.path.exists(DRIVE_ROOT):
        sh(f"git clone {GITHUB_URL} {DRIVE_ROOT}")
    else:
        print(f"Project already in Drive at {DRIVE_ROOT}; running git pull.")
        sh(f"cd {DRIVE_ROOT} && git pull --ff-only")

    # Install Python deps
    sh(f"pip install -q -r {DRIVE_ROOT}/requirements.txt")

    # Clone & install ai-toolkit (LoRA training engine)
    if not os.path.exists(AI_TOOLKIT):
        sh(f"git clone https://github.com/ostris/ai-toolkit {AI_TOOLKIT}")
    sh(f"pip install -q -r {AI_TOOLKIT}/requirements.txt")
    # ai-toolkit pins numpy<2 / scipy<1.13 which break Colab's stack.
    # Pin a known-compatible pair (numpy 2.1.x + scipy 1.14/1.15) and
    # force-reinstall so they're built against each other consistently.
    sh('pip install -q --force-reinstall --no-deps '
       '"numpy>=2.1,<2.2" "scipy>=1.14,<1.16"')

    # Make modules importable
    for p in (DRIVE_ROOT, AI_TOOLKIT):
        if p not in sys.path:
            sys.path.insert(0, p)

    print("\n=== Setup complete ===")
    print(f"DRIVE_ROOT: {DRIVE_ROOT}")
    print(f"AI_TOOLKIT: {AI_TOOLKIT}")
    print("Now run the notebook's Cell 0 (it will reuse this setup).")
    print("If FLUX.1-dev is needed, also run: !huggingface-cli login")


if __name__ == "__main__":
    main()
