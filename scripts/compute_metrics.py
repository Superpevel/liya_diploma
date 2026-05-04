import torch
import open_clip
from PIL import Image
from pathlib import Path
import json
import numpy as np
from torch_fidelity import calculate_metrics
import lpips as lpips_lib


def compute_clip_score(image_paths: list[str], captions: list[str]) -> float:
    """Mean CLIP Score (ViT-B/32, openai) for image-caption pairs."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to("cuda")

    scores = []
    for img_path, caption in zip(image_paths, captions):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda")
        text = tokenizer([caption]).to("cuda")
        with torch.no_grad():
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat * txt_feat).sum().item())
    return float(np.mean(scores))


def compute_fid(real_dir: str, fake_dir: str) -> float:
    """FID between real_dir and fake_dir image directories."""
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=True,
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def compute_lpips(real_paths: list[str], fake_paths: list[str]) -> float:
    """Mean LPIPS (AlexNet) for paired real/fake images."""
    loss_fn = lpips_lib.LPIPS(net="alex").to("cuda")
    scores = []
    for r, f in zip(real_paths, fake_paths):
        img_r = lpips_lib.im2tensor(lpips_lib.load_image(r)).to("cuda")
        img_f = lpips_lib.im2tensor(lpips_lib.load_image(f)).to("cuda")
        with torch.no_grad():
            scores.append(loss_fn(img_r, img_f).item())
    return float(np.mean(scores))
