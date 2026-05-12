import numpy as np
import open_clip
import torch
import lpips as lpips_lib
from PIL import Image
from torch_fidelity import calculate_metrics


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_clip_score(image_paths, captions) -> float:
    """Средний CLIP Score (ViT-B/32, openai) по парам картинка/подпись."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = _pick_device()
    model.eval().to(device)

    scores = []
    for img_path, caption in zip(image_paths, captions):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        text = tokenizer([caption]).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat * txt_feat).sum().item())
    return float(np.mean(scores))


def compute_fid(real_dir: str, fake_dir: str) -> float:
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def compute_lpips(real_paths, fake_paths) -> float:
    """LPIPS (AlexNet) по парам real/fake."""
    device = _pick_device()
    loss_fn = lpips_lib.LPIPS(net="alex").to(device)
    scores = []
    for r, f in zip(real_paths, fake_paths):
        img_r = lpips_lib.im2tensor(lpips_lib.load_image(r)).to(device)
        img_f = lpips_lib.im2tensor(lpips_lib.load_image(f)).to(device)
        with torch.no_grad():
            scores.append(loss_fn(img_r, img_f).item())
    return float(np.mean(scores))
