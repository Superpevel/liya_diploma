import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

PROMPT = (
    "USER: <image>\n"
    "Describe this logo image concisely. Include: shape/geometry, color palette, "
    "style (minimalist/detailed/geometric/organic), industry/theme if recognizable, "
    "typography presence. Output 1-2 sentences, max 77 tokens.\nASSISTANT:"
)


def load_model():
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return processor, model, next(model.parameters()).device


def caption_image(processor, model, device, image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(PROMPT, image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text = processor.decode(output[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip()


def _load_done(output_jsonl: str) -> set:
    done = set()
    p = Path(output_jsonl)
    if not p.exists():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["png_path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def caption_batch(input_jsonl: str, output_jsonl: str) -> None:
    """Дописывает caption в output_jsonl. Уже подписанные пары пропускает."""
    done = _load_done(output_jsonl)
    with open(input_jsonl) as f:
        pairs = [json.loads(l) for l in f]

    remaining = [p for p in pairs if p["png_path"] not in done]
    print(f"Captioning {len(remaining)} images ({len(done)} already done)")
    if not remaining:
        print("Nothing to do — all images already captioned.")
        return

    processor, model, device = load_model()
    with open(output_jsonl, "a") as out:
        for item in tqdm(remaining):
            item["caption"] = caption_image(processor, model, device, item["png_path"])
            out.write(json.dumps(item) + "\n")
