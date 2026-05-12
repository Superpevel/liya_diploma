"""
Бутстрап для Colab: клонит репо в Drive, ставит зависимости, монтирует Drive.
Запускается один раз на runtime, после этого ноутбучные Cell 0 работают как есть.

Список зависимостей сознательно НЕ вынесен в requirements_*.txt — иначе
при редактировании списка приходится отдельным шагом синхронизировать его
на Drive, а старая версия молча продолжает использоваться. См. cell 1 в
notebooks/04_train_sdxl_lora.ipynb с теми же пакетами и подробностями.
"""

import os
import subprocess
import sys


GITHUB_URL = "https://github.com/superpevel/liya_diploma.git"
DRIVE_ROOT = "/content/drive/MyDrive/liya_diploma"
AI_TOOLKIT = "/content/ai-toolkit"

# Не пиннуем — Colab сам ставит совместимые версии diffusers/torch/numpy/etc.
# k-diffusion ставится отдельно с --no-deps (его clean-fid тянет scipy<1.12
# который под py3.12 собирается из source и падает на metadata-generation).
EXTRA_PACKAGES = [
    'hf_transfer',
    'oyaml',
    'flatten_json',
    'lycoris-lora',
    'kornia',
    'invisible-watermark',
    'bitsandbytes',
    'prodigyopt',
    'open_clip_torch',
    'lpips',
    'optimum-quanto',
    'av',
    # k-diffusion runtime deps:
    'torchsde',
    'torchdiffeq',
    'jsonmerge',
    'resize-right',
    'clip-anytorch',
    # FLUX:
    'sentencepiece',
]
KDIFF_VERSION = 'k-diffusion==0.1.1.post1'


def sh(cmd):
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True)


def pip_install(args, label):
    print(f"\n=== {label} ===", flush=True)
    rc = subprocess.run([sys.executable, '-m', 'pip', 'install', *args]).returncode
    if rc:
        raise RuntimeError(f"{label}: pip упал (exit {rc}) — см. вывод выше.")


def main():
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")

    if not os.path.exists(DRIVE_ROOT):
        sh(f"git clone {GITHUB_URL} {DRIVE_ROOT}")
    else:
        print(f"Project already in Drive at {DRIVE_ROOT}; running git pull.")
        sh(f"cd {DRIVE_ROOT} && git pull --ff-only")

    if not os.path.exists(AI_TOOLKIT):
        sh(f"git clone https://github.com/ostris/ai-toolkit {AI_TOOLKIT}")

    pip_install(
        ['--upgrade-strategy', 'only-if-needed', *EXTRA_PACKAGES],
        "Installing extras (Colab-preinstalled core не трогаем)",
    )
    pip_install(['--no-deps', KDIFF_VERSION], "Installing k-diffusion (--no-deps)")

    for p in (DRIVE_ROOT, AI_TOOLKIT):
        if p not in sys.path:
            sys.path.insert(0, p)

    print("\n=== Setup complete ===")
    print(f"DRIVE_ROOT: {DRIVE_ROOT}")
    print(f"AI_TOOLKIT: {AI_TOOLKIT}")
    print("Если что-то импортируется с ABI mismatch — Runtime -> Restart session.")
    print("Если нужен FLUX.1-dev: !huggingface-cli login")


if __name__ == "__main__":
    main()
