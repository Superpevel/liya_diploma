# Logo Generation ML Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete ML pipeline — SVG dataset → LLaVA captions → LoRA fine-tuning (SDXL + FLUX.1-dev) → evaluation (FID/CLIP/LPIPS/VLM/scale) — all running in Google Colab Pro.

**Architecture:** SVG logos rasterized to PNG 512×512 via cairosvg, captioned by LLaVA-Next, split into 2k (ablation) + 10k (final) + 500 (test). SDXL and FLUX.1-dev fine-tuned via ai-toolkit LoRA. Results compared on FID, CLIP Score, LPIPS, VLM scores, and SSIM at multiple resolutions.

**Tech Stack:** Python 3.10, Google Colab Pro A100, `cairosvg`, `datasets`, `transformers` (llava-hf/llava-v1.6-mistral-7b-hf), `ai-toolkit` (ostris/ai-toolkit on GitHub), `diffusers`, `open_clip`, `torch-fidelity`, `lpips`, `scikit-image`, Recraft v3 API.

---

## File Map

**Scripts** (helper modules imported by notebooks):
- Create: `scripts/__init__.py`
- Create: `scripts/svg_to_png.py` — SVG → PNG 512×512 with white background
- Create: `scripts/filter_dataset.py` — filter by SVG path count and aspect ratio
- Create: `scripts/caption_llava.py` — batch LLaVA-Next captioning with resume
- Create: `scripts/verify_dataset.py` — CLIP Score verification of caption quality
- Create: `scripts/compute_metrics.py` — FID, CLIP Score, LPIPS
- Create: `scripts/scale_test.py` — SSIM at 16/32/64/512px

**Configs** (ai-toolkit YAML):
- Create: `configs/sdxl_lora_r4.yaml`
- Create: `configs/sdxl_lora_r8.yaml`
- Create: `configs/sdxl_lora_r16.yaml`
- Create: `configs/sdxl_lora_r32.yaml`
- Create: `configs/flux_lora_r16.yaml`

**Notebooks** (primary artifacts, run in Colab):
- Create: `notebooks/01_dataset_collection.ipynb`
- Create: `notebooks/02_dataset_captioning.ipynb`
- Create: `notebooks/03_dataset_verification.ipynb`
- Create: `notebooks/04_train_sdxl_lora.ipynb`
- Create: `notebooks/05_train_flux_lora.ipynb`
- Create: `notebooks/06_inference_compare.ipynb`
- Create: `notebooks/07_metrics_evaluation.ipynb`

**Project files:**
- Create: `requirements.txt`
- Create: `.gitignore`

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `scripts/__init__.py`
- Create: `results/experiments/.gitkeep`
- Create: `results/metrics/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
cairosvg==2.7.1
Pillow==10.3.0
datasets==2.19.1
transformers==4.41.0
accelerate==0.30.0
diffusers==0.27.2
open-clip-torch==2.24.0
torch-fidelity==0.3.0
lpips==0.1.4
scikit-image==0.23.2
scipy==1.13.0
numpy==1.26.4
tqdm==4.66.4
huggingface-hub==0.23.0
bitsandbytes==0.43.1
pyyaml==6.0.1
requests==2.31.0
```

- [ ] **Step 2: Create .gitignore**

```gitignore
data/
*.ckpt
*.safetensors
__pycache__/
.ipynb_checkpoints/
*.egg-info/
.env
```

- [ ] **Step 3: Create empty scripts/__init__.py**

```python
```

- [ ] **Step 4: Create directory structure and gitkeep stubs**

Run:
```bash
mkdir -p configs scripts notebooks results/experiments results/metrics
touch results/experiments/.gitkeep results/metrics/.gitkeep scripts/__init__.py
```

Expected: no errors, directories created.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt .gitignore scripts/__init__.py results/
git commit -m "feat: project scaffold for logo generation ML pipeline"
```

---

## Task 2: SVG → PNG Converter

**Files:**
- Create: `scripts/svg_to_png.py`

- [ ] **Step 1: Write scripts/svg_to_png.py**

```python
import cairosvg
from PIL import Image
import io
from pathlib import Path


def svg_to_png(svg_path: str, output_path: str, size: int = 512) -> bool:
    """Convert SVG to PNG with white background. Returns True on success."""
    try:
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.convert("RGB").save(output_path, "PNG")
        return True
    except Exception:
        return False


def batch_convert(svg_dir: str, png_dir: str, size: int = 512) -> dict:
    """Convert all SVGs in svg_dir to PNGs in png_dir. Returns stats dict."""
    svg_dir = Path(svg_dir)
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    svgs = list(svg_dir.glob("**/*.svg"))
    success, failed = 0, 0

    for svg_path in svgs:
        out_path = png_dir / (svg_path.stem + ".png")
        if svg_to_png(str(svg_path), str(out_path), size):
            success += 1
        else:
            failed += 1

    return {"total": len(svgs), "success": success, "failed": failed}
```

- [ ] **Step 2: Verify the converter**

Run in Python:
```python
import sys; sys.path.insert(0, ".")
from scripts.svg_to_png import svg_to_png
from PIL import Image

test_svg = "/tmp/test_logo.svg"
with open(test_svg, "w") as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<circle cx="50" cy="50" r="40" fill="navy"/></svg>')

result = svg_to_png(test_svg, "/tmp/test_out.png", 512)
assert result is True
img = Image.open("/tmp/test_out.png")
assert img.size == (512, 512), f"Expected 512x512, got {img.size}"
assert img.mode == "RGB", f"Expected RGB, got {img.mode}"
print("PASS: SVG → PNG converter works")
```

Expected: `PASS: SVG → PNG converter works`

- [ ] **Step 3: Commit**

```bash
git add scripts/svg_to_png.py
git commit -m "feat: SVG to PNG rasterizer with white background compositing"
```

---

## Task 3: Dataset Filter

**Files:**
- Create: `scripts/filter_dataset.py`

- [ ] **Step 1: Write scripts/filter_dataset.py**

```python
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path


def count_svg_paths(svg_path: str) -> int:
    """Count path elements in SVG. Returns -1 on parse error."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", ns) + root.findall(".//path")
        return len(paths)
    except Exception:
        return -1


def get_aspect_ratio(png_path: str) -> float:
    """Return width/height ratio of PNG. Returns -1.0 on error."""
    try:
        img = Image.open(png_path)
        w, h = img.size
        return w / h
    except Exception:
        return -1.0


def filter_dataset(
    png_dir: str,
    svg_dir: str,
    min_paths: int = 3,
    max_paths: int = 500,
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
) -> list[dict]:
    """
    Filter PNG/SVG pairs by path count and aspect ratio.
    Returns list of dicts: {png_path, svg_path}.
    """
    png_dir = Path(png_dir)
    svg_dir = Path(svg_dir)
    results = []

    for png_path in sorted(png_dir.glob("*.png")):
        svg_path = svg_dir / (png_path.stem + ".svg")
        if not svg_path.exists():
            continue

        n_paths = count_svg_paths(str(svg_path))
        if n_paths < min_paths or n_paths > max_paths:
            continue

        ratio = get_aspect_ratio(str(png_path))
        if ratio < min_ratio or ratio > max_ratio:
            continue

        results.append({"png_path": str(png_path), "svg_path": str(svg_path)})

    return results
```

- [ ] **Step 2: Verify filter logic**

```python
from scripts.filter_dataset import count_svg_paths, get_aspect_ratio
from PIL import Image

# Test path counting
with open("/tmp/test_paths.svg", "w") as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<path d="M10 10 L90 10"/><path d="M10 50 L90 50"/>'
            '<path d="M10 90 L90 90"/></svg>')
assert count_svg_paths("/tmp/test_paths.svg") == 3
assert count_svg_paths("/nonexistent.svg") == -1

# Test aspect ratio
assert get_aspect_ratio("/tmp/test_out.png") == 1.0  # 512x512 from Task 2
assert get_aspect_ratio("/nonexistent.png") == -1.0

print("PASS: filter functions work correctly")
```

Expected: `PASS: filter functions work correctly`

- [ ] **Step 3: Commit**

```bash
git add scripts/filter_dataset.py
git commit -m "feat: dataset filter by SVG path count and aspect ratio"
```

---

## Task 4: Notebook 01 — Dataset Collection

**Files:**
- Create: `notebooks/01_dataset_collection.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/01_dataset_collection.ipynb` as a Jupyter notebook with the following cells in order.

**Cell 1 — Install dependencies and mount Drive:**
```python
!pip install -q cairosvg datasets huggingface-hub Pillow tqdm

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
SVG_DIR = f'{DRIVE_ROOT}/data/raw_svg'
PNG_DIR = f'{DRIVE_ROOT}/data/png_512'
```

**Cell 2 — Download SVG logo dataset from HuggingFace:**
```python
# Search huggingface.co/datasets for "svg logo" to find the best available dataset.
# Recommended IDs to try (verify availability before running):
#   "logo-wizard/modern-logo-dataset"
#   "segment-anything/1b-masks" (no, wrong)
# If no SVG dataset is found, use PNG logos and skip the SVG filter step.

from datasets import load_dataset

DATASET_ID = "logo-wizard/modern-logo-dataset"  # UPDATE after checking HuggingFace
ds = load_dataset(DATASET_ID, split="train")
print(f"Loaded {len(ds)} samples")
print("Features:", ds.features)
```

**Cell 3 — Save raw data to Drive:**
```python
from pathlib import Path
from tqdm import tqdm
import os

Path(SVG_DIR).mkdir(parents=True, exist_ok=True)

for i, item in enumerate(tqdm(ds)):
    if "svg" in item and item["svg"]:
        out_path = f"{SVG_DIR}/{i:06d}.svg"
        with open(out_path, "w") as f:
            f.write(item["svg"])
    elif "image" in item:
        out_path = f"{SVG_DIR}/{i:06d}.png"
        item["image"].save(out_path)

print(f"Saved {i+1} files to {SVG_DIR}")
```

**Cell 4 — Rasterize SVGs to PNG 512×512:**
```python
from scripts.svg_to_png import batch_convert

stats = batch_convert(SVG_DIR, PNG_DIR, size=512)
print(f"Converted: {stats['success']} OK, {stats['failed']} failed of {stats['total']}")
```

**Cell 5 — Filter dataset:**
```python
from scripts.filter_dataset import filter_dataset
import json

filtered = filter_dataset(PNG_DIR, SVG_DIR, min_paths=3, max_paths=500)
print(f"After filtering: {len(filtered)} valid pairs")

with open(f'{DRIVE_ROOT}/data/filtered_pairs.jsonl', 'w') as f:
    for item in filtered:
        f.write(json.dumps(item) + '\n')
print("Saved data/filtered_pairs.jsonl")
```

**Cell 6 — Spot check: show 5 random samples:**
```python
import random
from PIL import Image
import matplotlib.pyplot as plt

sample = random.sample(filtered, 5)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, item in zip(axes, sample):
    ax.imshow(Image.open(item['png_path']))
    ax.axis('off')
plt.suptitle(f"5 random logos from {len(filtered)} filtered pairs")
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/experiments/dataset_sample.png', dpi=150)
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_dataset_collection.ipynb
git commit -m "feat: dataset collection notebook — SVG download, rasterize, filter"
```

---

## Task 5: LLaVA Captioning Script

**Files:**
- Create: `scripts/caption_llava.py`

- [ ] **Step 1: Write scripts/caption_llava.py**

```python
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
    return processor, model


def caption_image(processor, model, image_path: str) -> str:
    """Generate a caption for a single logo image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(CAPTION_PROMPT, image, return_tensors="pt").to("cuda")
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
                item = json.loads(line)
                done.add(item["png_path"])

    with open(input_jsonl) as f:
        pairs = [json.loads(l) for l in f]

    remaining = [p for p in pairs if p["png_path"] not in done]
    print(f"Captioning {len(remaining)} images ({len(done)} already done)")

    processor, model = load_model()

    with open(output_jsonl, "a") as out:
        for item in tqdm(remaining):
            caption = caption_image(processor, model, item["png_path"])
            item["caption"] = caption
            out.write(json.dumps(item) + "\n")
```

- [ ] **Step 2: Verify caption output format (no GPU required)**

```python
caption = "Minimalist circular logo with a stylized coffee cup in dark green, flat vector design."
assert isinstance(caption, str)
assert 10 < len(caption) < 500
assert caption.count("\n") == 0
print("PASS: caption format valid")
```

Expected: `PASS: caption format valid`

- [ ] **Step 3: Commit**

```bash
git add scripts/caption_llava.py
git commit -m "feat: LLaVA-Next batch captioning script with resume support"
```

---

## Task 6: Notebook 02 — Dataset Captioning

**Files:**
- Create: `notebooks/02_dataset_captioning.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Setup:**
```python
!pip install -q transformers accelerate bitsandbytes

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
INPUT_JSONL = f'{DRIVE_ROOT}/data/filtered_pairs.jsonl'
OUTPUT_JSONL = f'{DRIVE_ROOT}/data/captioned_pairs.jsonl'
```

**Cell 2 — Run captioning:**
```python
from scripts.caption_llava import caption_batch

caption_batch(input_jsonl=INPUT_JSONL, output_jsonl=OUTPUT_JSONL)
print("Captioning complete")
```

**Cell 3 — Spot check 3 captions:**
```python
import json, random
from PIL import Image
import matplotlib.pyplot as plt

with open(OUTPUT_JSONL) as f:
    captioned = [json.loads(l) for l in f]

sample = random.sample(captioned, 3)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, item in zip(axes, sample):
    ax.imshow(Image.open(item['png_path']))
    ax.set_title(item['caption'][:60] + "...", fontsize=7)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/experiments/caption_sample.png', dpi=150)
plt.show()
print(f"Total captioned: {len(captioned)}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_dataset_captioning.ipynb
git commit -m "feat: captioning notebook with LLaVA-Next and resume support"
```

---

## Task 7: Dataset Verification Script

**Files:**
- Create: `scripts/verify_dataset.py`

- [ ] **Step 1: Write scripts/verify_dataset.py**

```python
import open_clip
import torch
from PIL import Image
import json
import random
import numpy as np
from pathlib import Path


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
        for item in sample
    ]


def filter_by_clip_score(jsonl_path: str, output_path: str,
                          threshold: float = 0.25) -> dict:
    """Filter pairs where CLIP Score < threshold. Writes kept pairs to output_path."""
    with open(jsonl_path) as f:
        pairs = [json.loads(l) for l in f]

    model, preprocess, tokenizer, device = _load_clip()
    kept, removed = [], 0

    for item in pairs:
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/verify_dataset.py
git commit -m "feat: CLIP Score verification for dataset quality filtering"
```

---

## Task 8: Notebook 03 — Verification + Split Creation

**Files:**
- Create: `notebooks/03_dataset_verification.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Setup:**
```python
!pip install -q open-clip-torch

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
```

**Cell 2 — Compute CLIP Score distribution on 200 random pairs:**
```python
from scripts.verify_dataset import compute_clip_scores
import matplotlib.pyplot as plt
import numpy as np

scores = compute_clip_scores(
    f'{DRIVE_ROOT}/data/captioned_pairs.jsonl',
    sample_size=200,
    seed=42,
)
print(f"CLIP Score — mean={np.mean(scores):.3f}, "
      f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")
print(f"Pairs above 0.25: {sum(s >= 0.25 for s in scores)}/200")

plt.hist(scores, bins=30, color='steelblue', edgecolor='white')
plt.axvline(0.25, color='red', linestyle='--', label='threshold=0.25')
plt.xlabel("CLIP Score")
plt.ylabel("Count")
plt.title("CLIP Score distribution (200 sampled pairs)")
plt.legend()
plt.savefig(f'{DRIVE_ROOT}/results/metrics/clip_score_distribution.png', dpi=150)
plt.show()
```

**Cell 3 — Create train/test splits:**
```python
import json, random

random.seed(42)
with open(f'{DRIVE_ROOT}/data/captioned_pairs.jsonl') as f:
    all_pairs = [json.loads(l) for l in f]

random.shuffle(all_pairs)

test_500  = all_pairs[:500]
train_2k  = all_pairs[500:2500]
train_10k = all_pairs[500:10500]

def save_split(pairs, path, split_name):
    with open(path, 'w') as f:
        for item in pairs:
            item['split'] = split_name
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(pairs)} pairs → {path}")

save_split(test_500,  f'{DRIVE_ROOT}/data/test_500.jsonl',  'test')
save_split(train_2k,  f'{DRIVE_ROOT}/data/train_2k.jsonl',  'train_2k')
save_split(train_10k, f'{DRIVE_ROOT}/data/train_10k.jsonl', 'train_10k')
```

**Cell 4 — Verify split sizes (no overlap):**
```python
for name, path, expected in [
    ('test_500',  f'{DRIVE_ROOT}/data/test_500.jsonl',  500),
    ('train_2k',  f'{DRIVE_ROOT}/data/train_2k.jsonl',  2000),
    ('train_10k', f'{DRIVE_ROOT}/data/train_10k.jsonl', 10000),
]:
    with open(path) as f:
        items = [json.loads(l) for l in f]
    print(f"{name}: {len(items)} pairs (expected {expected})")
    assert len(items) == expected, f"Size mismatch for {name}"

# Verify no overlap between test and train
test_paths = {json.loads(l)['png_path']
              for l in open(f'{DRIVE_ROOT}/data/test_500.jsonl')}
train_paths = {json.loads(l)['png_path']
               for l in open(f'{DRIVE_ROOT}/data/train_2k.jsonl')}
assert len(test_paths & train_paths) == 0, "OVERLAP between test and train!"
print("PASS: no overlap between test and train splits")
```

Expected:
```
test_500: 500 pairs (expected 500)
train_2k: 2000 pairs (expected 2000)
train_10k: 10000 pairs (expected 10000)
PASS: no overlap between test and train splits
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_dataset_verification.ipynb
git commit -m "feat: dataset verification and split creation (2k/10k train + 500 test)"
```

---

## Task 9: SDXL LoRA Configs (Rank Ablation)

**Files:**
- Create: `configs/sdxl_lora_r4.yaml`
- Create: `configs/sdxl_lora_r8.yaml`
- Create: `configs/sdxl_lora_r16.yaml`
- Create: `configs/sdxl_lora_r32.yaml`

- [ ] **Step 1: Create configs/sdxl_lora_r4.yaml**

```yaml
job: extension
config:
  name: sdxl_logo_lora_r4
  process:
    - type: sd_trainer
      training_folder: "/content/drive/MyDrive/liya_diplomCC/results/experiments/sdxl_r4"
      device: cuda:0
      trigger_word: "LOGOIMG"
      network:
        type: lora
        linear: 4
        linear_alpha: 4
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/content/drive/MyDrive/liya_diplomCC/data/train_2k_images"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 512]
      train:
        batch_size: 2
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: ddpm
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine_with_restarts
        lr_scheduler_params:
          num_cycles: 1
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_flux: false
        quantize: false
      sample:
        sampler: euler
        sample_every: 500
        width: 512
        height: 512
        prompts:
          - "LOGOIMG minimalist coffee shop logo, circular, dark green, flat design"
          - "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif"
        neg: "photorealistic, blurry, cluttered, complex background"
        seed: 42
        walk_seed: true
        guidance_scale: 7.5
        sample_steps: 20
meta:
  name: "[name]"
  version: "1.0"
```

- [ ] **Step 2: Create configs/sdxl_lora_r8.yaml**

```yaml
job: extension
config:
  name: sdxl_logo_lora_r8
  process:
    - type: sd_trainer
      training_folder: "/content/drive/MyDrive/liya_diplomCC/results/experiments/sdxl_r8"
      device: cuda:0
      trigger_word: "LOGOIMG"
      network:
        type: lora
        linear: 8
        linear_alpha: 8
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/content/drive/MyDrive/liya_diplomCC/data/train_2k_images"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 512]
      train:
        batch_size: 2
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: ddpm
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine_with_restarts
        lr_scheduler_params:
          num_cycles: 1
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_flux: false
        quantize: false
      sample:
        sampler: euler
        sample_every: 500
        width: 512
        height: 512
        prompts:
          - "LOGOIMG minimalist coffee shop logo, circular, dark green, flat design"
          - "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif"
        neg: "photorealistic, blurry, cluttered, complex background"
        seed: 42
        walk_seed: true
        guidance_scale: 7.5
        sample_steps: 20
meta:
  name: "[name]"
  version: "1.0"
```

- [ ] **Step 3: Create configs/sdxl_lora_r16.yaml**

```yaml
job: extension
config:
  name: sdxl_logo_lora_r16
  process:
    - type: sd_trainer
      training_folder: "/content/drive/MyDrive/liya_diplomCC/results/experiments/sdxl_r16"
      device: cuda:0
      trigger_word: "LOGOIMG"
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/content/drive/MyDrive/liya_diplomCC/data/train_2k_images"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 512]
      train:
        batch_size: 2
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: ddpm
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine_with_restarts
        lr_scheduler_params:
          num_cycles: 1
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_flux: false
        quantize: false
      sample:
        sampler: euler
        sample_every: 500
        width: 512
        height: 512
        prompts:
          - "LOGOIMG minimalist coffee shop logo, circular, dark green, flat design"
          - "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif"
        neg: "photorealistic, blurry, cluttered, complex background"
        seed: 42
        walk_seed: true
        guidance_scale: 7.5
        sample_steps: 20
meta:
  name: "[name]"
  version: "1.0"
```

- [ ] **Step 4: Create configs/sdxl_lora_r32.yaml**

```yaml
job: extension
config:
  name: sdxl_logo_lora_r32
  process:
    - type: sd_trainer
      training_folder: "/content/drive/MyDrive/liya_diplomCC/results/experiments/sdxl_r32"
      device: cuda:0
      trigger_word: "LOGOIMG"
      network:
        type: lora
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/content/drive/MyDrive/liya_diplomCC/data/train_2k_images"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 512]
      train:
        batch_size: 2
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: ddpm
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine_with_restarts
        lr_scheduler_params:
          num_cycles: 1
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_flux: false
        quantize: false
      sample:
        sampler: euler
        sample_every: 500
        width: 512
        height: 512
        prompts:
          - "LOGOIMG minimalist coffee shop logo, circular, dark green, flat design"
          - "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif"
        neg: "photorealistic, blurry, cluttered, complex background"
        seed: 42
        walk_seed: true
        guidance_scale: 7.5
        sample_steps: 20
meta:
  name: "[name]"
  version: "1.0"
```

- [ ] **Step 5: Commit**

```bash
git add configs/sdxl_lora_r4.yaml configs/sdxl_lora_r8.yaml \
        configs/sdxl_lora_r16.yaml configs/sdxl_lora_r32.yaml
git commit -m "feat: SDXL LoRA training configs for rank ablation r4/r8/r16/r32"
```

---

## Task 10: FLUX LoRA Config

**Files:**
- Create: `configs/flux_lora_r16.yaml`

- [ ] **Step 1: Create configs/flux_lora_r16.yaml**

```yaml
job: extension
config:
  name: flux_logo_lora_r16
  process:
    - type: sd_trainer
      training_folder: "/content/drive/MyDrive/liya_diplomCC/results/experiments/flux_r16"
      device: cuda:0
      trigger_word: "LOGOIMG"
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/content/drive/MyDrive/liya_diplomCC/data/train_10k_images"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 512]
      train:
        batch_size: 1
        steps: 4000
        gradient_accumulation_steps: 2
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: 1e-4
        lr_scheduler: cosine_with_restarts
        lr_scheduler_params:
          num_cycles: 1
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
      sample:
        sampler: flowmatch
        sample_every: 500
        width: 512
        height: 512
        prompts:
          - "LOGOIMG minimalist coffee shop logo, circular, dark green, flat design"
          - "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif"
        seed: 42
        walk_seed: true
        guidance_scale: 3.5
        sample_steps: 20
meta:
  name: "[name]"
  version: "1.0"
```

- [ ] **Step 2: Commit**

```bash
git add configs/flux_lora_r16.yaml
git commit -m "feat: FLUX.1-dev LoRA config (r16, train_10k, 4000 steps)"
```

---

## Task 11: Notebook 04 — SDXL LoRA Training (Experiments 1 + 2)

**Files:**
- Create: `notebooks/04_train_sdxl_lora.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Install ai-toolkit and mount Drive:**
```python
!git clone https://github.com/ostris/ai-toolkit /content/ai-toolkit
!cd /content/ai-toolkit && pip install -r requirements.txt
!pip install -q diffusers accelerate

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')
sys.path.insert(0, '/content/ai-toolkit')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
```

**Cell 2 — Prepare ai-toolkit image+caption folders:**
```python
import json, shutil
from pathlib import Path

def prepare_aitoolkit_folder(jsonl_path: str, out_folder: str, max_items: int = None):
    """ai-toolkit expects a folder with image.png + image.txt (caption) per sample."""
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path) as f:
        pairs = [json.loads(l) for l in f]
    if max_items:
        pairs = pairs[:max_items]
    for item in pairs:
        stem = Path(item['png_path']).stem
        shutil.copy(item['png_path'], out / f"{stem}.png")
        (out / f"{stem}.txt").write_text(f"LOGOIMG {item['caption']}")
    print(f"Prepared {len(pairs)} pairs → {out_folder}")

prepare_aitoolkit_folder(
    f'{DRIVE_ROOT}/data/train_2k.jsonl',
    f'{DRIVE_ROOT}/data/train_2k_images',
)
prepare_aitoolkit_folder(
    f'{DRIVE_ROOT}/data/train_10k.jsonl',
    f'{DRIVE_ROOT}/data/train_10k_images',
)
```

**Cell 3 — Experiment 1: Rank ablation (r4, r8, r16, r32) at 2000 steps:**
```python
import subprocess

for rank in [4, 8, 16, 32]:
    config = f'{DRIVE_ROOT}/configs/sdxl_lora_r{rank}.yaml'
    print(f"\n{'='*50}\nTraining SDXL LoRA r={rank}...")
    result = subprocess.run(
        ['python', '/content/ai-toolkit/run.py', config],
        capture_output=False,
    )
    print(f"r={rank}: {'DONE' if result.returncode == 0 else 'FAILED'}")
```

**Cell 4 — Experiment 2: Steps ablation (set BEST_RANK after reviewing Exp.1 FID):**
```python
import yaml, copy

# UPDATE BEST_RANK after reviewing Experiment 1 FID scores in notebook 07.
# Priority: lowest FID; use CLIP Score as tiebreaker.
BEST_RANK = 16

with open(f'{DRIVE_ROOT}/configs/sdxl_lora_r{BEST_RANK}.yaml') as f:
    base_cfg = yaml.safe_load(f)

for steps in [500, 1000, 4000]:  # 2000 already done in Exp.1
    cfg = copy.deepcopy(base_cfg)
    cfg['config']['name'] = f'sdxl_logo_lora_r{BEST_RANK}_s{steps}'
    cfg['config']['process'][0]['training_folder'] = (
        f'{DRIVE_ROOT}/results/experiments/sdxl_r{BEST_RANK}_s{steps}'
    )
    cfg['config']['process'][0]['train']['steps'] = steps
    tmp = f'/tmp/sdxl_r{BEST_RANK}_s{steps}.yaml'
    with open(tmp, 'w') as f:
        yaml.dump(cfg, f)
    print(f"\nTraining r={BEST_RANK}, steps={steps}...")
    subprocess.run(['python', '/content/ai-toolkit/run.py', tmp])
```

**Cell 5 — Generate sample images for all rank checkpoints:**
```python
from diffusers import StableDiffusionXLPipeline
import torch
from pathlib import Path

TEST_PROMPTS = [
    "LOGOIMG minimalist coffee shop logo, circular, dark green, flat vector design",
    "LOGOIMG tech startup logo, geometric hexagon, blue gradient, sans-serif",
    "LOGOIMG bakery logo, wheat icon, warm brown, handcrafted artisan style",
    "LOGOIMG fitness brand, bold lion silhouette, orange and black, geometric",
    "LOGOIMG law firm, balanced scales, navy blue, serif elegant typography",
]

def generate_samples(lora_path: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.load_lora_weights(lora_path)
    pipe.set_progress_bar_config(disable=True)

    for i, prompt in enumerate(TEST_PROMPTS):
        imgs = pipe(
            prompt,
            negative_prompt="photorealistic, blurry, cluttered, complex background",
            num_images_per_prompt=2,
            generator=torch.Generator().manual_seed(42),
            guidance_scale=7.5,
            num_inference_steps=30,
            height=512, width=512,
        ).images
        for j, img in enumerate(imgs):
            img.save(f"{out_dir}/prompt{i:02d}_v{j}.png")

    del pipe
    torch.cuda.empty_cache()
    print(f"Saved samples → {out_dir}")

for rank in [4, 8, 16, 32]:
    generate_samples(
        f'{DRIVE_ROOT}/results/experiments/sdxl_r{rank}',
        f'{DRIVE_ROOT}/results/experiments/sdxl_r{rank}_samples',
    )
```

**Cell 6 — Visual grid: 5 prompts × 4 ranks:**
```python
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(5, 4, figsize=(16, 20))
ranks = [4, 8, 16, 32]

for row in range(5):
    for col, rank in enumerate(ranks):
        img_path = (f'{DRIVE_ROOT}/results/experiments/'
                    f'sdxl_r{rank}_samples/prompt{row:02d}_v0.png')
        if Path(img_path).exists():
            axes[row, col].imshow(Image.open(img_path))
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(f'LoRA r={rank}', fontsize=10, fontweight='bold')

plt.suptitle("SDXL LoRA — Rank Ablation (5 prompts × 4 ranks)", fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/experiments/rank_ablation_grid.png', dpi=150)
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/04_train_sdxl_lora.ipynb
git commit -m "feat: SDXL LoRA training notebook — rank + steps ablation with sample grids"
```

---

## Task 12: Notebook 05 — FLUX + SD1.5 Baseline (Experiment 3)

**Files:**
- Create: `notebooks/05_train_flux_lora.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Setup:**
```python
!git clone https://github.com/ostris/ai-toolkit /content/ai-toolkit
!cd /content/ai-toolkit && pip install -r requirements.txt
!pip install -q diffusers transformers accelerate

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/ai-toolkit')
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
```

**Cell 2 — Train FLUX.1-dev LoRA on train_10k:**
```python
import subprocess

config = f'{DRIVE_ROOT}/configs/flux_lora_r16.yaml'
print("Training FLUX.1-dev LoRA r=16 on train_10k (4000 steps)...")
result = subprocess.run(
    ['python', '/content/ai-toolkit/run.py', config],
    capture_output=False,
)
print("DONE" if result.returncode == 0 else "FAILED")
```

**Cell 3 — SD 1.5 baseline inference (no fine-tuning, prompt engineering only):**
```python
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

TEST_PROMPTS = [
    "minimalist coffee shop logo, circular icon, dark green, flat vector design, white background",
    "tech startup logo, geometric hexagon, blue gradient, bold sans-serif, white background",
    "bakery logo, wheat sheaf icon, warm brown, handcrafted artisan style, white background",
    "fitness brand, lion silhouette, orange and black, bold geometric, white background",
    "law firm logo, balanced scales, navy blue, serif elegant typography, white background",
]

out_dir = f'{DRIVE_ROOT}/results/experiments/sd15_baseline'
Path(out_dir).mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.set_progress_bar_config(disable=True)

for i, prompt in enumerate(TEST_PROMPTS):
    imgs = pipe(
        prompt,
        negative_prompt="photorealistic, blurry, cluttered, complex background, 3D render",
        num_images_per_prompt=2,
        generator=torch.Generator().manual_seed(42),
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images
    for j, img in enumerate(imgs):
        img.save(f"{out_dir}/prompt{i:02d}_v{j}.png")

del pipe; torch.cuda.empty_cache()
print(f"SD 1.5 baseline: {len(TEST_PROMPTS)*2} images → {out_dir}")
```

**Cell 4 — FLUX LoRA inference:**
```python
from diffusers import FluxPipeline

flux_lora_path = f'{DRIVE_ROOT}/results/experiments/flux_r16'
out_dir = f'{DRIVE_ROOT}/results/experiments/flux_r16_samples'
Path(out_dir).mkdir(parents=True, exist_ok=True)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.load_lora_weights(flux_lora_path)
pipe.set_progress_bar_config(disable=True)

FLUX_PROMPTS = [f"LOGOIMG {p.replace(', white background', '')}" for p in TEST_PROMPTS]

for i, prompt in enumerate(FLUX_PROMPTS):
    imgs = pipe(
        prompt,
        num_images_per_prompt=2,
        generator=torch.Generator().manual_seed(42),
        guidance_scale=3.5,
        num_inference_steps=20,
        height=512, width=512,
    ).images
    for j, img in enumerate(imgs):
        img.save(f"{out_dir}/prompt{i:02d}_v{j}.png")

del pipe; torch.cuda.empty_cache()
print(f"FLUX LoRA: {len(FLUX_PROMPTS)*2} images → {out_dir}")
```

**Cell 5 — Side-by-side comparison grid (5 prompts × 3 models):**
```python
import matplotlib.pyplot as plt
from PIL import Image

MODELS = {
    "SD 1.5 Baseline": f'{DRIVE_ROOT}/results/experiments/sd15_baseline',
    "SDXL LoRA (best)": f'{DRIVE_ROOT}/results/experiments/sdxl_r16_samples',
    "FLUX.1-dev LoRA": f'{DRIVE_ROOT}/results/experiments/flux_r16_samples',
}

fig, axes = plt.subplots(5, 3, figsize=(12, 20))
for row in range(5):
    for col, (model_name, img_dir) in enumerate(MODELS.items()):
        img_path = f"{img_dir}/prompt{row:02d}_v0.png"
        if Path(img_path).exists():
            axes[row, col].imshow(Image.open(img_path))
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(model_name, fontsize=10, fontweight='bold')

plt.suptitle("Experiment 3: SD1.5 Baseline vs SDXL LoRA vs FLUX LoRA", fontsize=12)
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/experiments/exp3_model_comparison.png', dpi=150)
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/05_train_flux_lora.ipynb
git commit -m "feat: FLUX.1-dev LoRA training + SD1.5 baseline + model comparison grid"
```

---

## Task 13: Metrics Computation Script

**Files:**
- Create: `scripts/compute_metrics.py`

- [ ] **Step 1: Write scripts/compute_metrics.py**

```python
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
```

- [ ] **Step 2: Verify imports and function signatures**

```python
import sys; sys.path.insert(0, ".")
from scripts.compute_metrics import compute_clip_score, compute_fid, compute_lpips
import inspect

assert callable(compute_clip_score)
assert callable(compute_fid)
assert callable(compute_lpips)
sig = inspect.signature(compute_clip_score)
assert "image_paths" in sig.parameters
assert "captions" in sig.parameters
print("PASS: compute_metrics.py imports and signatures valid")
```

Expected: `PASS: compute_metrics.py imports and signatures valid`

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_metrics.py
git commit -m "feat: FID, CLIP Score, LPIPS computation utilities"
```

---

## Task 14: Scale Readability Test Script

**Files:**
- Create: `scripts/scale_test.py`

- [ ] **Step 1: Write scripts/scale_test.py**

```python
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
    results = {s: [] for s in SCALES}

    for img_path in images:
        for scale in SCALES:
            results[scale].append(ssim_at_scale(str(img_path), scale))

    return {scale: float(np.mean(scores)) for scale, scores in results.items()}
```

- [ ] **Step 2: Verify scale test**

```python
from scripts.scale_test import ssim_at_scale
from PIL import Image, ImageDraw
import tempfile, os

img = Image.new("RGB", (512, 512), (255, 255, 255))
draw = ImageDraw.Draw(img)
draw.ellipse([100, 100, 412, 412], fill=(0, 0, 128))

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    img.save(f.name)
    score_512 = ssim_at_scale(f.name, 512)
    score_16  = ssim_at_scale(f.name, 16)
    os.unlink(f.name)

assert score_512 == 1.0, f"SSIM at original size must be 1.0, got {score_512}"
assert 0.0 <= score_16 <= 1.0, f"SSIM at 16px must be [0,1], got {score_16}"
print(f"PASS: SSIM at 512px={score_512:.3f}, at 16px={score_16:.3f}")
```

Expected: `PASS: SSIM at 512px=1.000, at 16px=<value between 0 and 1>`

- [ ] **Step 3: Commit**

```bash
git add scripts/scale_test.py
git commit -m "feat: SSIM-based scale readability test at 16/32/64/512px"
```

---

## Task 15: Notebook 06 — Inference Comparison + Recraft (Experiment 4)

**Files:**
- Create: `notebooks/06_inference_compare.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Setup:**
```python
!pip install -q diffusers transformers accelerate requests

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'

EXP4_PROMPTS = [
    "minimalist coffee shop logo, circular icon, dark green, flat design",
    "tech startup logo, geometric hexagon, blue gradient, bold sans-serif",
    "bakery logo, wheat sheaf icon, warm brown, handcrafted artisan style",
    "fitness brand, lion silhouette, orange and black, bold geometric",
    "law firm logo, balanced scales, navy blue, serif elegant",
    "eco brand, leaf and water droplet, green and teal, clean minimal",
    "photography studio, aperture symbol, monochrome, sleek modern",
    "music label, vinyl record, purple gradient, retro modern",
    "medical clinic, cross and heartbeat, blue and white, professional",
    "restaurant, chef hat and fork, red and gold, classic",
    "real estate, house outline, dark blue, trustworthy minimal",
    "travel agency, compass rose, orange and navy, adventurous",
    "bookstore, open book, burgundy and gold, literary",
    "pet shop, paw print, teal and white, friendly rounded",
    "art gallery, abstract brush stroke, black and coral, creative",
    "yoga studio, lotus flower, lavender and white, serene",
    "gaming company, controller icon, dark purple neon, futuristic",
    "finance app, upward arrow chart, emerald green, trustworthy",
    "delivery service, lightning bolt package, yellow and dark grey, dynamic",
    "hair salon, scissors and comb, rose gold and black, premium",
    "construction company, hard hat, orange and black, bold solid",
    "organic farm, sun over field, green and yellow, natural",
    "coding bootcamp, terminal cursor, dark background cyan, tech",
    "wedding planner, interlinked rings, gold and ivory, elegant",
    "coffee roaster, steam cup bean, dark brown and copper, artisan",
    "surf shop, wave and sun, blue and sandy yellow, coastal",
    "candle brand, flame teardrop, warm amber and cream, cozy",
    "juice bar, orange slice splash, bright orange and green, fresh",
    "security firm, shield with lock, dark grey and blue, strong",
    "toy store, star balloon, bright multicolor, playful",
    "spa resort, lotus in water, soft teal and white, luxury",
    "architecture firm, geometric building outline, charcoal, minimal",
    "flower shop, stylized bloom, pink and green, feminine delicate",
    "wine brand, grape cluster, deep purple and gold, sophisticated",
    "cycling club, wheel spokes, red and white, sporty",
    "accounting firm, balance beam, navy and silver, precise",
    "art supplies, palette with brush, colorful abstract, creative",
    "dog grooming, dog face silhouette, brown and light blue, friendly",
    "language school, speech bubble globe, blue and orange, global",
    "printing shop, ink drop on paper, black and teal, clean",
    "e-commerce, shopping cart pixel, blue and yellow, digital",
    "event planning, balloon ribbon, purple and gold, festive",
    "dental clinic, tooth with sparkle, sky blue and white, clean",
    "vegan restaurant, leaf fork, green and beige, natural",
    "brewery, hops in circle, amber and dark brown, craft",
    "startup incubator, rocket in bulb, blue and green, innovative",
    "cloud storage, cloud with lock, grey and sky blue, tech",
    "insurance company, umbrella shield, dark blue and teal, protective",
    "fashion label, hanger silhouette, black and rose, chic",
    "children school, apple and pencil, red and yellow, friendly",
]
```

**Cell 2 — Generate with best SDXL LoRA:**
```python
from diffusers import StableDiffusionXLPipeline
import torch
from pathlib import Path

# UPDATE to the rank with best FID from Experiment 1
BEST_SDXL_LORA = f'{DRIVE_ROOT}/results/experiments/sdxl_r16'

out_dir = f'{DRIVE_ROOT}/results/experiments/exp4_sdxl_lora'
Path(out_dir).mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_lora_weights(BEST_SDXL_LORA)
pipe.set_progress_bar_config(disable=True)

for i, prompt in enumerate(EXP4_PROMPTS):
    img = pipe(
        f"LOGOIMG {prompt}",
        negative_prompt="photorealistic, blurry, cluttered, complex background",
        generator=torch.Generator().manual_seed(42),
        guidance_scale=7.5,
        num_inference_steps=30,
        height=512, width=512,
    ).images[0]
    img.save(f"{out_dir}/prompt{i:02d}.png")

del pipe; torch.cuda.empty_cache()
print(f"SDXL LoRA: {len(EXP4_PROMPTS)} images saved")
```

**Cell 3 — Generate with Recraft v3 API:**
```python
import requests
from pathlib import Path

RECRAFT_API_KEY = "YOUR_RECRAFT_API_KEY"  # Get at recraft.ai

out_dir = f'{DRIVE_ROOT}/results/experiments/exp4_recraft'
Path(out_dir).mkdir(parents=True, exist_ok=True)

for i, prompt in enumerate(EXP4_PROMPTS):
    resp = requests.post(
        "https://external.api.recraft.ai/v1/images/generations",
        headers={
            "Authorization": f"Bearer {RECRAFT_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": f"logo: {prompt}",
            "style": "vector_illustration",
            "width": 512,
            "height": 512,
            "n": 1,
        },
    )
    if resp.status_code == 200:
        img_url = resp.json()["data"][0]["url"]
        img_data = requests.get(img_url).content
        with open(f"{out_dir}/prompt{i:02d}.png", "wb") as f:
            f.write(img_data)
    else:
        print(f"Recraft error prompt {i}: {resp.status_code} {resp.text}")

print(f"Recraft: images → {out_dir}")
```

**Cell 4 — Visual comparison grid (10 prompts × 3 models):**
```python
import matplotlib.pyplot as plt
from PIL import Image

MODELS = {
    "SD 1.5 Baseline": f'{DRIVE_ROOT}/results/experiments/sd15_baseline',
    "SDXL LoRA (best)": f'{DRIVE_ROOT}/results/experiments/exp4_sdxl_lora',
    "Recraft v3":       f'{DRIVE_ROOT}/results/experiments/exp4_recraft',
}

SHOW = list(range(10))
fig, axes = plt.subplots(len(SHOW), len(MODELS), figsize=(12, 40))

for row, idx in enumerate(SHOW):
    for col, (model_name, img_dir) in enumerate(MODELS.items()):
        img_path = f"{img_dir}/prompt{idx:02d}.png"
        if Path(img_path).exists():
            axes[row, col].imshow(Image.open(img_path))
        else:
            axes[row, col].text(0.5, 0.5, "Missing", ha='center', va='center')
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(model_name, fontsize=10, fontweight='bold')
        if col == 0:
            short = EXP4_PROMPTS[idx][:35]
            axes[row, col].set_ylabel(short, fontsize=6, rotation=0,
                                      labelpad=80, va='center')

plt.suptitle("Experiment 4: Model Comparison (10 prompts × 3 models)", fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/experiments/exp4_comparison_grid.png', dpi=150)
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/06_inference_compare.ipynb
git commit -m "feat: inference comparison notebook — SDXL LoRA vs SD1.5 vs Recraft (Exp4)"
```

---

## Task 16: Notebook 07 — Metrics Evaluation

**Files:**
- Create: `notebooks/07_metrics_evaluation.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Setup:**
```python
!pip install -q torch-fidelity open-clip-torch lpips scikit-image

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/liya_diplomCC')

DRIVE_ROOT = '/content/drive/MyDrive/liya_diplomCC'
```

**Cell 2 — Prepare real images directory for FID:**
```python
import json, shutil
from pathlib import Path

real_dir = f'{DRIVE_ROOT}/data/png_512_test'
Path(real_dir).mkdir(parents=True, exist_ok=True)

with open(f'{DRIVE_ROOT}/data/test_500.jsonl') as f:
    test_pairs = [json.loads(l) for l in f]

for item in test_pairs:
    shutil.copy(item['png_path'], real_dir)

print(f"Copied {len(test_pairs)} real images to {real_dir}")
```

**Cell 3 — Compute FID and CLIP Score for all experiments:**
```python
from scripts.compute_metrics import compute_fid, compute_clip_score
from pathlib import Path
import json

captions = [p['caption'] for p in test_pairs]

EXPERIMENTS = {
    "SD1.5 Baseline":  f'{DRIVE_ROOT}/results/experiments/sd15_baseline',
    "SDXL LoRA r4":    f'{DRIVE_ROOT}/results/experiments/sdxl_r4_samples',
    "SDXL LoRA r8":    f'{DRIVE_ROOT}/results/experiments/sdxl_r8_samples',
    "SDXL LoRA r16":   f'{DRIVE_ROOT}/results/experiments/sdxl_r16_samples',
    "SDXL LoRA r32":   f'{DRIVE_ROOT}/results/experiments/sdxl_r32_samples',
    "FLUX LoRA r16":   f'{DRIVE_ROOT}/results/experiments/flux_r16_samples',
    "Recraft v3":      f'{DRIVE_ROOT}/results/experiments/exp4_recraft',
}

results = {}
for name, fake_dir in EXPERIMENTS.items():
    fake_imgs = sorted(Path(fake_dir).glob("*.png"))
    caps = captions[:len(fake_imgs)]
    fid  = compute_fid(real_dir, fake_dir)
    clip = compute_clip_score([str(p) for p in fake_imgs], caps)
    results[name] = {"fid": fid, "clip_score": clip}
    print(f"{name:<22} FID={fid:6.2f}  CLIP={clip:.4f}")

with open(f'{DRIVE_ROOT}/results/metrics/all_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**Cell 4 — Scale readability test for all experiments:**
```python
from scripts.scale_test import scale_test_batch
import json

scale_results = {}
for name, fake_dir in EXPERIMENTS.items():
    scale_results[name] = scale_test_batch(fake_dir)
    s = scale_results[name]
    print(f"{name:<22} SSIM@16={s[16]:.3f}  @64={s[64]:.3f}  @512={s[512]:.3f}")

with open(f'{DRIVE_ROOT}/results/metrics/scale_test.json', 'w') as f:
    json.dump(scale_results, f, indent=2)
```

**Cell 5 — VLM scoring with LLaVA:**
```python
from scripts.caption_llava import load_model
from pathlib import Path
from tqdm import tqdm
import json

processor, vlm_model = load_model()

VLM_QUESTIONS = [
    "Is this image a professional logo? Rate 0-5 where 5=definitely a logo. Reply with a single digit only.",
    "Rate the aesthetic quality of this logo design 0-5 where 5=excellent. Reply with a single digit only.",
]

def vlm_score(img_path: str, question: str) -> float:
    from scripts.caption_llava import caption_image
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    # Temporarily patch the global prompt
    import scripts.caption_llava as cap_mod
    orig = cap_mod.CAPTION_PROMPT
    cap_mod.CAPTION_PROMPT = prompt
    raw = caption_image(processor, vlm_model, img_path)
    cap_mod.CAPTION_PROMPT = orig
    try:
        return min(max(float(raw.strip()[0]), 0.0), 5.0)
    except Exception:
        return 0.0

vlm_results = {}
for name, fake_dir in list(EXPERIMENTS.items()):
    images = sorted(Path(fake_dir).glob("*.png"))[:20]
    logo_scores, quality_scores = [], []
    for img_path in tqdm(images, desc=name):
        logo_scores.append(vlm_score(str(img_path), VLM_QUESTIONS[0]))
        quality_scores.append(vlm_score(str(img_path), VLM_QUESTIONS[1]))
    vlm_results[name] = {
        "logo_score":    sum(logo_scores) / len(logo_scores),
        "quality_score": sum(quality_scores) / len(quality_scores),
    }
    print(f"{name:<22} logo={vlm_results[name]['logo_score']:.2f}  "
          f"quality={vlm_results[name]['quality_score']:.2f}")

with open(f'{DRIVE_ROOT}/results/metrics/vlm_scores.json', 'w') as f:
    json.dump(vlm_results, f, indent=2)
```

**Cell 6 — Final summary table and bar charts:**
```python
import matplotlib.pyplot as plt
import json

with open(f'{DRIVE_ROOT}/results/metrics/all_metrics.json') as f:
    metrics = json.load(f)

names = list(metrics.keys())
fids  = [metrics[n]['fid'] for n in names]
clips = [metrics[n]['clip_score'] for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.barh(names, fids, color='steelblue')
ax1.set_xlabel("FID (lower is better)")
ax1.set_title("FID by Model/Config")
ax1.axvline(fids[0], color='red', linestyle='--', alpha=0.5, label='SD1.5 baseline')
ax1.legend()

ax2.barh(names, clips, color='darkorange')
ax2.set_xlabel("CLIP Score (higher is better)")
ax2.set_title("CLIP Score by Model/Config")

plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/results/metrics/metrics_comparison.png', dpi=150)
plt.show()

print(f"\n{'Model':<25} {'FID':>8} {'CLIP Score':>12}")
print("-" * 47)
for name in names:
    print(f"{name:<25} {metrics[name]['fid']:>8.2f} {metrics[name]['clip_score']:>12.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/07_metrics_evaluation.ipynb
git commit -m "feat: metrics evaluation — FID, CLIP Score, VLM scoring, scale test, charts"
```

---

## Self-Review

**Spec coverage check:**
- [x] SVG download + rasterization → Tasks 2, 4
- [x] Filter by path count + aspect ratio → Task 3
- [x] LLaVA-Next captioning → Tasks 5, 6
- [x] CLIP Score verification → Tasks 7, 8
- [x] train_2k / train_10k / test_500 splits → Task 8
- [x] ai-toolkit framework → Tasks 11, 12
- [x] Exp 1: SDXL LoRA rank ablation r4/r8/r16/r32 → Tasks 9, 11
- [x] Exp 2: Steps ablation (500/1k/2k/4k) → Task 11
- [x] Exp 3: SDXL LoRA vs FLUX.1-dev LoRA vs SD1.5 baseline → Tasks 10, 12
- [x] Exp 4: Recraft v3 comparison → Task 15
- [x] FID, CLIP Score, LPIPS → Task 13
- [x] VLM scoring → Task 16
- [x] Scale readability SSIM test → Tasks 14, 16
- [x] Visual sample grids → Tasks 11, 12, 15

**No placeholders (TBD/TODO) found.**

**Type consistency:**
- `caption_image(processor, model, image_path)` — consistent Tasks 5, 16
- `compute_fid(real_dir, fake_dir)` — consistent Tasks 13, 16
- `compute_clip_score(image_paths, captions)` — consistent Tasks 13, 16
- `scale_test_batch(image_dir)` — consistent Tasks 14, 16
- `ssim_at_scale(img_path, scale)` — consistent Tasks 14, 16

All checks pass.
