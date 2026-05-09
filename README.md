# Logo Generation ML — diploma project

Pipeline for fine-tuning Stable Diffusion XL and FLUX.1-dev on a logo dataset
with LoRA, then comparing against SD 1.5 baseline and Recraft v3 commercial API.

7 notebooks, run in order:

| # | Notebook | Role |
|---|----------|------|
| 01 | `01_dataset_collection.ipynb` | Pull logos from HuggingFace, rasterise SVG → PNG 512×512, filter |
| 02 | `02_dataset_captioning.ipynb` | Auto-caption every image with LLaVA-Next |
| 03 | `03_dataset_verification.ipynb` | CLIP-Score sanity check + train/test split |
| 04 | `04_train_sdxl_lora.ipynb` | SDXL LoRA: rank ablation (4/8/16/32) + steps ablation |
| 05 | `05_train_flux_lora.ipynb` | FLUX.1-dev LoRA + SD 1.5 baseline |
| 06 | `06_inference_compare.ipynb` | Side-by-side: SDXL/FLUX/SD1.5/Recraft on 50 prompts |
| 07 | `07_metrics_evaluation.ipynb` | FID, CLIP Score, scale-test, VLM-rated quality |

Notebooks auto-detect Colab vs local in their first cell.

---

## Run on Google Colab (recommended)

You need: a Google account (Drive), a GitHub account, and HuggingFace account
(for FLUX.1-dev which is a gated model).

### One-time: push the project to GitHub

From this folder locally:

```powershell
# create a new repo on github.com first (call it liya_diplomCC), then:
git remote add origin https://github.com/YOUR_USERNAME/liya_diplomCC.git
git add .
git commit -m "initial commit"
git push -u origin main
```

Then edit `setup_colab.py` line 19 — change `YOUR_USERNAME` to your GitHub
handle, commit + push that one-line change.

### On Colab

1. **Pick a runtime with GPU**:
   Runtime → Change runtime type → T4 GPU (free) is enough for SDXL LoRA;
   FLUX.1-dev wants A100 (Colab Pro).
2. **Open the notebook from GitHub**:
   File → Open notebook → GitHub tab → paste your repo URL → pick
   e.g. `notebooks/01_dataset_collection.ipynb`.
3. **First cell of any notebook**: just hit Run. It will:
   - mount Google Drive
   - clone the project into `/content/drive/MyDrive/liya_diplomCC` if absent
   - install requirements
   - clone `ai-toolkit` into `/content/ai-toolkit`
4. **Authenticate HuggingFace** once per runtime (only needed for FLUX):
   ```
   !huggingface-cli login
   ```
   Get the token at <https://huggingface.co/settings/tokens>, accept the
   FLUX.1-dev license at <https://huggingface.co/black-forest-labs/FLUX.1-dev>.

> If a notebook's Cell 0 fails because `scripts/` isn't in Drive yet, run this
> in a fresh cell first:
>
> ```python
> !curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/liya_diplomCC/main/setup_colab.py | python -
> ```

### What ends up where

- `MyDrive/liya_diplomCC/` — code, configs, **and** generated `data/`, `results/`
- `/content/ai-toolkit/` — LoRA training engine (rebuilt every runtime, not in Drive)
- `~/.cache/huggingface/` — model downloads (not persisted across sessions)

Everything heavy that you want to keep across sessions must live under
`MyDrive/liya_diplomCC/` — Drive is the only persistent storage.

### Dataset size note

The default dataset (`logo-wizard/modern-logo-dataset`) has ~800 logos. After
the train/test split, ~300 pairs remain for training. The diploma plan
originally talks about 2k/10k splits; the code reflects the honest single-split
reality — see notebook 03.

---

## Run locally on Windows / macOS / Linux

Requires Python 3.11. The pipeline is GPU-heavy: at least 16 GB VRAM for SDXL,
24 GB+ for FLUX.

```powershell
# Windows (PowerShell)
Set-ExecutionPolicy -Scope Process Bypass
.\setup_local.ps1
```

```bash
# macOS / Linux
./setup_local.sh
```

The script creates `.venv311/`, installs PyTorch (CUDA 12.8 if NVIDIA, else
CPU), all Python deps, and clones `ai-toolkit` next to this folder. Then:

```powershell
.\.venv311\Scripts\Activate.ps1
jupyter lab
```

Open notebooks one at a time, run them in order. Each notebook's Cell 0
auto-detects local mode and finds the project root by looking for
`scripts/`.

---

## Project layout

```
liya_diplomCC/
├── notebooks/        # 7 notebooks, the user-facing entry point
├── scripts/          # importable helpers (caption, filter, metrics, etc.)
├── configs/          # ai-toolkit YAML configs for LoRA training
├── data/             # generated; not in git
├── results/          # generated; not in git
├── docs/             # diploma spec / plan
├── requirements.txt
├── setup_local.{ps1,sh}
└── setup_colab.py
```

All paths in configs use Colab-style `/content/drive/MyDrive/liya_diplomCC/...`
and are rewritten on the fly to local paths in the training cells.
