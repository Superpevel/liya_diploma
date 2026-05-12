from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
STORAGE_DIR = PROJECT_ROOT / "bot_storage" / "images"
DB_PATH = PROJECT_ROOT / "bot_storage" / "bot.db"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    key: str
    title: str
    base: str          # "sdxl" | "flux"
    base_model_id: str
    lora_path: Path
    trigger: str = ""  # trigger token prepended to user prompt


SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
FLUX_BASE = "black-forest-labs/FLUX.1-dev"

MODELS: Dict[str, ModelConfig] = {
    "sdxl_r16": ModelConfig(
        key="sdxl_r16",
        title="SDXL LoRA r16 (s4000) — основная",
        base="sdxl",
        base_model_id=SDXL_BASE,
        lora_path=RESULTS_DIR
        / "sdxl_r16_s4000"
        / "sdxl_logo_lora_r16_s4000"
        / "sdxl_logo_lora_r16_s4000.safetensors",
        trigger="logo, ",
    ),
    "sdxl_r32": ModelConfig(
        key="sdxl_r32",
        title="SDXL LoRA r32 (s1000)",
        base="sdxl",
        base_model_id=SDXL_BASE,
        lora_path=RESULTS_DIR
        / "sdxl_r32"
        / "sdxl_logo_lora_r32"
        / "sdxl_logo_lora_r32_000001000.safetensors",
        trigger="logo, ",
    ),
    "sdxl_r4": ModelConfig(
        key="sdxl_r4",
        title="SDXL LoRA r4 (лёгкая)",
        base="sdxl",
        base_model_id=SDXL_BASE,
        lora_path=RESULTS_DIR
        / "sdxl_r4"
        / "sdxl_logo_lora_r4"
        / "sdxl_logo_lora_r4.safetensors",
        trigger="logo, ",
    ),
    "flux_r16": ModelConfig(
        key="flux_r16",
        title="FLUX.1 LoRA r16 (3500 шагов)",
        base="flux",
        base_model_id=FLUX_BASE,
        lora_path=RESULTS_DIR
        / "flux_r16"
        / "checkpoint-3500"
        / "pytorch_lora_weights.safetensors",
        trigger="logo, ",
    ),
}

DEFAULT_MODEL_KEY = "sdxl_r16"


@dataclass(frozen=True)
class Settings:
    bot_token: str
    num_variants: int = 4
    history_ttl_hours: int = 24
    image_width: int = 1024
    image_height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 7.0
    negative_prompt: str = (
        "low quality, blurry, watermark, text artifacts, jpeg artifacts, "
        "deformed, ugly, signature"
    )
    device: str = field(default_factory=lambda: os.getenv("BOT_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: os.getenv("BOT_DTYPE", "fp16"))


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не задан. Скопируйте .env.example в .env и "
            "пропишите токен от @BotFather."
        )
    return Settings(bot_token=token)
