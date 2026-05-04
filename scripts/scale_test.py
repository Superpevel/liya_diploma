from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pathlib import Path


SCALES = [16, 32, 64, 512]


def ssim_at_scale(img_path: str, scale: int) -> float:
    """
    Downscale image to `scale`×`scale`, upscale back, compute SSIM vs original.
    Higher SSIM = more readable at that scale.
    """
    orig = Image.open(img_path).convert("RGB")
    small = orig.resize((scale, scale), Image.LANCZOS)
    restored = small.resize(orig.size, Image.NEAREST)
    orig_arr = np.array(orig)
    restored_arr = np.array(restored)
    return float(ssim(orig_arr, restored_arr, channel_axis=2, data_range=255))


def scale_test_batch(image_dir: str) -> dict[int, float]:
    """
    Compute mean SSIM at each scale for all PNGs in image_dir.
    Returns {scale: mean_ssim}.
    """
    images = list(Path(image_dir).glob("*.png"))
    if not images:
        raise ValueError(f"No PNG files found in {image_dir}")
    results = {s: [] for s in SCALES}

    for img_path in images:
        for scale in SCALES:
            results[scale].append(ssim_at_scale(str(img_path), scale))

    return {scale: float(np.mean(scores)) for scale, scores in results.items()}
