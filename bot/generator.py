from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from . import config


@dataclass
class GenerationResult:
    images: List[Image.Image]
    seeds: List[int]


class _StubPipeline:
    """Заглушка, отдающая цветные placeholder-картинки.

    Используется автоматически, если в окружении нет ``torch``/``diffusers``
    или модели не помещаются в память. Позволяет проверить весь Telegram-флоу
    без GPU.
    """

    def __init__(self, model_key: str) -> None:
        self.model_key = model_key

    def __call__(self, *, prompt: str, negative_prompt: str,
                 num_inference_steps: int, guidance_scale: float,
                 width: int, height: int, generator) -> "_StubOutput":
        seed = int(getattr(generator, "initial_seed", lambda: 0)())
        rng = random.Random(f"{prompt}:{seed}")
        bg = tuple(rng.randint(40, 215) for _ in range(3))
        img = Image.new("RGB", (width, height), bg)
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size=42)
        except OSError:
            font = ImageFont.load_default()
        text = f"[stub:{self.model_key}]\nseed={seed}\n{prompt[:80]}"
        draw.multiline_text((40, 40), text, fill=(255, 255, 255), font=font,
                            spacing=8)
        return _StubOutput([img])


@dataclass
class _StubOutput:
    images: List[Image.Image]


class ImageGenerator:
    """Ленивая загрузка SDXL/FLUX пайплайнов с LoRA-весами."""

    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self._pipelines: dict[str, object] = {}
        self._lock = asyncio.Lock()
        self._device, self._dtype, self._reason = self._resolve_runtime()

    def _resolve_runtime(self) -> Tuple[str, object, Optional[str]]:
        try:
            import torch  # type: ignore
        except ImportError:
            return "stub", None, "torch не установлен"
        requested = self.settings.device
        if requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested
        if device == "cuda" and not torch.cuda.is_available():
            return "stub", None, "CUDA недоступна"
        dtype = torch.float16 if self.settings.dtype == "fp16" else torch.float32
        if device == "cpu":
            dtype = torch.float32
        return device, dtype, None

    @property
    def backend_info(self) -> str:
        if self._device == "stub":
            return f"stub ({self._reason})"
        return f"{self._device}/{self.settings.dtype}"

    async def _get_pipeline(self, model_key: str):
        async with self._lock:
            if model_key in self._pipelines:
                return self._pipelines[model_key]
            cfg = config.MODELS[model_key]
            pipe = await asyncio.to_thread(self._load_pipeline, cfg)
            self._pipelines[model_key] = pipe
            return pipe

    def _load_pipeline(self, cfg: config.ModelConfig):
        if self._device == "stub":
            return _StubPipeline(cfg.key)
        import torch  # type: ignore
        from diffusers import (  # type: ignore
            StableDiffusionXLPipeline,
            FluxPipeline,
        )

        if cfg.base == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                cfg.base_model_id,
                torch_dtype=self._dtype,
                use_safetensors=True,
                variant="fp16" if self._dtype == torch.float16 else None,
                add_watermarker=False,
            )
        elif cfg.base == "flux":
            pipe = FluxPipeline.from_pretrained(
                cfg.base_model_id, torch_dtype=self._dtype,
            )
        else:
            raise ValueError(f"Неизвестный base: {cfg.base}")

        if not cfg.lora_path.exists():
            raise FileNotFoundError(f"LoRA не найдена: {cfg.lora_path}")
        pipe.load_lora_weights(
            str(cfg.lora_path.parent), weight_name=cfg.lora_path.name,
        )
        pipe.to(self._device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        return pipe

    async def generate(
        self,
        prompt: str,
        model_key: str,
        num_variants: int,
    ) -> GenerationResult:
        cfg = config.MODELS[model_key]
        full_prompt = f"{cfg.trigger}{prompt}".strip()
        pipe = await self._get_pipeline(model_key)
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_variants)]
        images = await asyncio.to_thread(
            self._run_pipeline, pipe, full_prompt, seeds
        )
        return GenerationResult(images=images, seeds=seeds)

    def _run_pipeline(
        self, pipe, prompt: str, seeds: List[int],
    ) -> List[Image.Image]:
        results: List[Image.Image] = []
        s = self.settings
        if self._device == "stub":
            class _G:
                def __init__(self, seed: int) -> None:
                    self._seed = seed

                def initial_seed(self) -> int:
                    return self._seed

            for seed in seeds:
                out = pipe(
                    prompt=prompt,
                    negative_prompt=s.negative_prompt,
                    num_inference_steps=s.num_inference_steps,
                    guidance_scale=s.guidance_scale,
                    width=s.image_width,
                    height=s.image_height,
                    generator=_G(seed),
                )
                results.extend(out.images)
            return results

        import torch  # type: ignore

        for seed in seeds:
            gen = torch.Generator(device=self._device).manual_seed(seed)
            out = pipe(
                prompt=prompt,
                negative_prompt=s.negative_prompt,
                num_inference_steps=s.num_inference_steps,
                guidance_scale=s.guidance_scale,
                width=s.image_width,
                height=s.image_height,
                generator=gen,
            )
            results.extend(out.images)
        return results
