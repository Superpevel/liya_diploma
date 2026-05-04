from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


CAPTION_PROMPT = (
    "USER: <image>\n"
    "Describe this logo image concisely. Include: shape/geometry, color palette, "
    "style (minimalist/detailed/geometric/organic), industry/theme if recognizable, "
    "typography presence. Output 1-2 sentences, max 77 tokens.\nASSISTANT:"
)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


def load_model():
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    device = next(model.parameters()).device
    return processor, model, device


def caption_image(processor, model, device, image_path: str) -> str:
    """Generate a caption for a single logo image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(CAPTION_PROMPT, image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    full = processor.decode(output[0], skip_special_tokens=True)
    return full.split("ASSISTANT:")[-1].strip()


def caption_batch(input_jsonl: str, output_jsonl: str) -> None:
    """
    Read filtered_pairs.jsonl, add caption field, write to output_jsonl.
    Supports resuming: skips already-captioned entries.
    """
    done = set()
    if Path(output_jsonl).exists():
        with open(output_jsonl) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done.add(item["png_path"])
                except json.JSONDecodeError:
                    continue

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
            caption = caption_image(processor, model, device, item["png_path"])
            item["caption"] = caption
            out.write(json.dumps(item) + "\n")
