from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


SCALES = [16, 32, 64, 512]


def ssim_at_scale(img_path: str, scale: int) -> float:
    # ужимаем до scale×scale, растягиваем обратно и сравниваем с оригиналом
    orig = Image.open(img_path).convert("RGB")
    small = orig.resize((scale, scale), Image.LANCZOS)
    restored = small.resize(orig.size, Image.NEAREST)
    return float(
        ssim(np.array(orig), np.array(restored), channel_axis=2, data_range=255)
    )


def scale_test_batch(image_dir: str) -> dict[int, float]:
    images = list(Path(image_dir).glob("*.png"))
    if not images:
        raise ValueError(f"No PNG files found in {image_dir}")

    bucket = {s: [] for s in SCALES}
    for img_path in images:
        for s in SCALES:
            bucket[s].append(ssim_at_scale(str(img_path), s))
    return {s: float(np.mean(vals)) for s, vals in bucket.items()}
