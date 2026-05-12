import json
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


def _load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), preprocess, tokenizer, device


def _clip_score_single(model, preprocess, tokenizer, device, image_path, caption):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([caption]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (img_feat * txt_feat).sum().item()


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_clip_scores(jsonl_path: str, sample_size: int = 200,
                        seed: int = 42) -> list[float]:
    """CLIP Score по случайной выборке. Возвращает список значений."""
    random.seed(seed)
    pairs = _read_jsonl(jsonl_path)
    sample = random.sample(pairs, min(sample_size, len(pairs)))

    model, preprocess, tokenizer, device = _load_clip()
    return [
        _clip_score_single(model, preprocess, tokenizer, device,
                           item["png_path"], item["caption"])
        for item in tqdm(sample, desc="CLIP scoring")
    ]


def filter_by_clip_score(jsonl_path: str, output_path: str,
                         threshold: float = 0.25) -> dict:
    """Отбрасывает пары с CLIP Score < threshold. Сохраняет остальное."""
    pairs = _read_jsonl(jsonl_path)
    model, preprocess, tokenizer, device = _load_clip()

    kept, removed = [], 0
    for item in tqdm(pairs, desc="CLIP filtering"):
        score = _clip_score_single(model, preprocess, tokenizer, device,
                                   item["png_path"], item["caption"])
        if score >= threshold:
            item["clip_score"] = score
            kept.append(item)
        else:
            removed += 1

    with open(output_path, "w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")

    return {"kept": len(kept), "removed": removed, "total": len(pairs)}
