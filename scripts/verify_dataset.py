import open_clip
import torch
from PIL import Image
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm


def _load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), preprocess, tokenizer, device


def _clip_score_single(model, preprocess, tokenizer, device,
                        image_path: str, caption: str) -> float:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([caption]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (img_feat * txt_feat).sum().item()


def compute_clip_scores(jsonl_path: str, sample_size: int = 200,
                        seed: int = 42) -> list[float]:
    """Compute CLIP Score for a random sample. Returns list of scores."""
    random.seed(seed)
    with open(jsonl_path) as f:
        pairs = [json.loads(l) for l in f]
    sample = random.sample(pairs, min(sample_size, len(pairs)))

    model, preprocess, tokenizer, device = _load_clip()
    return [
        _clip_score_single(model, preprocess, tokenizer, device,
                           item["png_path"], item["caption"])
        for item in tqdm(sample, desc="CLIP scoring")
    ]


def filter_by_clip_score(jsonl_path: str, output_path: str,
                          threshold: float = 0.25) -> dict:
    """Filter pairs where CLIP Score < threshold. Writes kept pairs to output_path."""
    with open(jsonl_path) as f:
        pairs = [json.loads(l) for l in f]

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
