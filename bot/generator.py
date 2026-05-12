import asyncio
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from . import config


@dataclass
class GenerationResult:
    images: list
    seeds: list


class ImageGenerator:
    """SDXL/FLUX + LoRA. Если torch/CUDA нет — рисуем placeholder."""

    def __init__(self, settings: config.Settings):
        self.settings = settings
        self._pipes: dict = {}
        self._lock = asyncio.Lock()
        self._device, self._dtype, self._stub_reason = self._detect_runtime()

    def _detect_runtime(self):
        try:
            import torch
        except ImportError:
            return "stub", None, "torch не установлен"

        requested = self.settings.device
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if requested == "auto" else requested

        if device == "cuda" and not torch.cuda.is_available():
            return "stub", None, "CUDA недоступна"

        if device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16 if self.settings.dtype == "fp16" else torch.float32
        return device, dtype, None

    @property
    def is_stub(self) -> bool:
        return self._device == "stub"

    @property
    def backend_info(self) -> str:
        if self.is_stub:
            return f"stub ({self._stub_reason})"
        return f"{self._device}/{self.settings.dtype}"

    async def _get_pipeline(self, model_key: str):
        async with self._lock:
            if model_key in self._pipes:
                return self._pipes[model_key]
            cfg = config.MODELS[model_key]
            pipe = await asyncio.to_thread(self._load_pipeline, cfg)
            self._pipes[model_key] = pipe
            return pipe

    def _load_pipeline(self, cfg: config.ModelConfig):
        import torch
        from diffusers import StableDiffusionXLPipeline, FluxPipeline

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
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        return pipe

    async def generate(self, prompt: str, model_key: str,
                       num_variants: int) -> GenerationResult:
        cfg = config.MODELS[model_key]
        full_prompt = f"{cfg.trigger}{prompt}".strip()
        seeds = [random.randint(0, 2**31 - 1) for _ in range(num_variants)]

        if self.is_stub:
            images = [self._stub_image(full_prompt, s, cfg.key) for s in seeds]
            return GenerationResult(images=images, seeds=seeds)

        pipe = await self._get_pipeline(model_key)
        images = await asyncio.to_thread(self._run_pipeline, pipe, full_prompt, seeds)
        return GenerationResult(images=images, seeds=seeds)

    def _run_pipeline(self, pipe, prompt: str, seeds: list) -> list:
        import torch
        s = self.settings
        result = []
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
            result.extend(out.images)
        return result

    def _stub_image(self, prompt: str, seed: int, model_key: str) -> Image.Image:
        s = self.settings
        rng = random.Random(f"{prompt}:{seed}")
        bg = tuple(rng.randint(40, 215) for _ in range(3))
        img = Image.new("RGB", (s.image_width, s.image_height), bg)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", size=42)
        except OSError:
            font = ImageFont.load_default()
        text = f"[stub:{model_key}]\nseed={seed}\n{prompt[:80]}"
        draw.multiline_text((40, 40), text, fill=(255, 255, 255), font=font,
                            spacing=8)
        return img
