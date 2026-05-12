"""
Бутстрап для Colab: клонит репо в Drive, ставит зависимости, монтирует Drive.
Запускается один раз на runtime, после этого ноутбучные Cell 0 работают как есть.
"""

import os
import subprocess
import sys


GITHUB_URL = "https://github.com/superpevel/liya_diploma.git"
DRIVE_ROOT = "/content/drive/MyDrive/liya_diploma"
AI_TOOLKIT = "/content/ai-toolkit"


def sh(cmd: str) -> int:
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")

    if not os.path.exists(DRIVE_ROOT):
        sh(f"git clone {GITHUB_URL} {DRIVE_ROOT}")
    else:
        print(f"Project already in Drive at {DRIVE_ROOT}; running git pull.")
        sh(f"cd {DRIVE_ROOT} && git pull --ff-only")

    sh(f"pip install -q -r {DRIVE_ROOT}/requirements.txt")

    # свой requirements вместо ai-toolkit/requirements.txt — там scipy==1.12 ломает numpy 2.x
    if not os.path.exists(AI_TOOLKIT):
        sh(f"git clone https://github.com/ostris/ai-toolkit {AI_TOOLKIT}")
    sh(f"pip install -q -r {DRIVE_ROOT}/requirements_aitoolkit_colab.txt")

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
