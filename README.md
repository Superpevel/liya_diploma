# Дипломный проект: генерация логотипов

Дообучаю SDXL и FLUX.1-dev на датасете логотипов через LoRA, потом сравниваю
с базовым SD 1.5 и коммерческим Recraft v3. Сверху ещё телеграм-бот, чтобы
можно было покликать модель руками.

В проекте 7 ноутбуков, идут по порядку:

| № | Ноутбук | Что делает |
|---|---------|------------|
| 01 | `01_dataset_collection.ipynb` | Качает логотипы с HuggingFace, переводит SVG → PNG 512×512, фильтрует |
| 02 | `02_dataset_captioning.ipynb` | Подписывает каждую картинку через LLaVA-Next |
| 03 | `03_dataset_verification.ipynb` | Прогоняет CLIP-Score проверку и делит на train/test |
| 04 | `04_train_sdxl_lora.ipynb` | Тренит SDXL LoRA: перебор ранга (4/8/16/32) и шагов |
| 05 | `05_train_flux_lora.ipynb` | FLUX.1-dev LoRA + бейзлайн SD 1.5 |
| 06 | `06_inference_compare.ipynb` | Сравнение SDXL/FLUX/SD1.5/Recraft на 50 промптах |
| 07 | `07_metrics_evaluation.ipynb` | FID, CLIP Score, scale-тест, оценка качества VLM-кой |

Ноутбуки сами понимают, где запущены — Colab или локально (это в первой ячейке).

---

## Запуск в Google Colab

Нужны: гугл-аккаунт (для Drive), GitHub и HuggingFace (FLUX.1-dev — gated модель).

### Один раз — выложить проект на GitHub

Из этой папки:

```powershell
# сначала создаём пустой репо на github.com (например liya_diplomCC), потом:
git remote add origin https://github.com/YOUR_USERNAME/liya_diplomCC.git
git add .
git commit -m "initial commit"
git push -u origin main
```

### В Colab

1. **Берём runtime с GPU**: Runtime → Change runtime type → T4 (бесплатный)
   хватает для SDXL LoRA; FLUX.1-dev хочет A100 (Colab Pro).
2. **Открываем ноутбук из гитхаба**: File → Open notebook → GitHub → URL
   репо → выбираем, например, `notebooks/01_dataset_collection.ipynb`.
3. **Первая ячейка любого ноутбука** — просто запускаем. Она:
   - монтирует Drive
   - клонит `ai-toolkit` в `/content/ai-toolkit` (если нужно)
   - ставит зависимости и проверяет sanity-imports
   - при первой установке убивает kernel — это норма, запускаем ячейку второй раз
4. **Логинимся в HuggingFace** (один раз за runtime, только если нужен FLUX):
   ```
   !huggingface-cli login
   ```
   Токен берём тут: <https://huggingface.co/settings/tokens>, лицензию
   FLUX.1-dev принимаем тут: <https://huggingface.co/black-forest-labs/FLUX.1-dev>.

### Куда что складывается

- `MyDrive/liya_diplomCC/` — код, конфиги и сгенерированные `data/`, `results/`
- `/content/ai-toolkit/` — движок для тренировки LoRA (каждый runtime заново)
- `~/.cache/huggingface/` — кэш моделей (между сессиями не сохраняется)

Всё тяжёлое, что должно пережить runtime, кладём в `MyDrive/liya_diplomCC/` —
больше места для постоянного хранения в Colab нет.

### Про размер датасета

В дефолтном `logo-wizard/modern-logo-dataset` около 800 логотипов. После
сплита train/test остаётся ~300 пар для тренировки. В планах диплома было
2k/10k — но реально в коде честно 1 сплит (см. ноутбук 03).

---

## Запуск локально (Windows / macOS / Linux)

Нужен Python 3.11. По железу: минимум 16 ГБ VRAM для SDXL, 24+ ГБ для FLUX.

```powershell
# Windows
Set-ExecutionPolicy -Scope Process Bypass
.\setup_local.ps1
```

```bash
# macOS / Linux
./setup_local.sh
```

Скрипт сам создаёт `.venv311/`, ставит торч (CUDA 12.8 если есть NVIDIA,
иначе CPU), все зависимости и клонит `ai-toolkit` рядом с этой папкой. Дальше:

```powershell
.\.venv311\Scripts\Activate.ps1
jupyter lab
```

Открываем ноутбуки по очереди. Первая ячейка сама поймёт, что мы локально, и
найдёт корень проекта по папке `scripts/`.

---

## Структура

```
liya_diplomCC/
├── notebooks/        # 7 ноутбуков — основная точка входа
├── scripts/          # переиспользуемые хелперы (капшнинг, фильтрация, метрики)
├── bot/              # телеграм-бот: SDXL/FLUX + LoRA на инференс
├── configs/          # YAML-конфиги ai-toolkit для тренировки LoRA
├── data/             # генерируется ноутбуками, не в git
├── results/          # веса LoRA и метрики, не в git
├── docs/             # план диплома, спецификации
├── requirements.txt
└── setup_local.{ps1,sh}
```

В конфигах прописаны Colab-пути вида `/content/drive/MyDrive/liya_diplomCC/...`
— ноутбук на локалке сам подменяет их при тренировке.

## Бот

В папке `bot/` лежит  телеграм-бот на aiogram 
