# Дизайн ML-части: Генерация логотипов бренда по текстовому описанию

**Дата:** 2026-05-05
**Статус:** Approved
**Область:** Датасет + файн-тюнинг LoRA + оценка моделей (сервис — отдельная фаза)

---

## Контекст и цель

Разработать ML-пайплайн для генерации логотипов по текстовому описанию. Ключевой артефакт — LoRA-адаптер поверх диффузионных моделей (SDXL, FLUX.1-dev), обученный на специализированном датасете логотипов с текстовыми описаниями. Результаты служат материалом для экспериментального раздела ВКР.

**Вычислительные ресурсы:** Google Colab Pro (A100, ~15GB VRAM).
**Хранение:** Google Drive для чекпоинтов и данных, GitHub для кода и конфигов.

---

## 1. Датасет

### Источник
SVG-Logo-Dataset с HuggingFace. Векторные SVG обеспечивают чёткие контуры и прозрачный фон — критично для качества обучающих данных.

### Пайплайн сборки

**Шаг 1 — Растеризация:**
- SVG → PNG 512×512 через `cairosvg`
- Белый фон (альфа-канал → белый)

**Шаг 2 — Фильтрация (убираем нелоготипы):**
- Удаляем SVG с ошибками рендеринга
- Удаляем слишком простые (< 3 путей) и слишком сложные (> 500 путей)
- Оставляем только aspect ratio 0.8–1.2

**Шаг 3 — Генерация описаний (LLaVA-Next):**
Промпт:
```
Describe this logo image concisely. Include: shape/geometry, color palette,
style (minimalist/detailed/geometric/organic), industry/theme if recognizable,
typography presence. Output 1-2 sentences, max 77 tokens.
```
Пример выхода: *"Minimalist circular logo with a stylized coffee cup in dark green, flat vector design, white background, sans-serif geometry."*

**Шаг 4 — VLM-верификация (случайные 200 пар из 10k):**
CLIP Score > 0.25 как порог качества описания.

### Сплиты

| Сплит | Размер | Назначение |
|-------|--------|------------|
| `train_2k` | 2 000 пар | Ablation по рангам (быстрые эксперименты) |
| `train_10k` | 10 000 пар | Финальное обучение |
| `test_500` | 500 пар | Оценка метрик (не участвует в обучении) |

### Формат
JSONL с колонками: `image_path`, `caption`, `svg_path`, `split`.

---

## 2. ML-эксперименты

**Фреймворк:** `ai-toolkit` (Ostris) — YAML-конфиги, поддержка SDXL и FLUX LoRA.
**Воспроизводимость:** seed=42 везде, все конфиги в репо.

### Эксперимент 1 — Ablation по рангу LoRA
*Модель:* SDXL. *Датасет:* `train_2k`. *Шаги:* 2000 (фиксировано).

| Конфиг | Ранг | Параметры LoRA |
|--------|------|----------------|
| A | r=4 | ~1M |
| B | r=8 | ~2M |
| C | r=16 | ~4M |
| D | r=32 | ~8M |

Метрики: FID, CLIP Score на `test_500`. Выбираем лучший ранг для Эксп.2 — приоритет по FID, CLIP Score как тай-брейкер.

### Эксперимент 2 — Ablation по числу шагов
*Модель:* SDXL. *Ранг:* лучший из Эксп.1. *Датасет:* `train_2k`.

| Конфиг | Шаги |
|--------|------|
| E | 500 |
| F | 1000 |
| G | 2000 |
| H | 4000 |

### Эксперимент 3 — Сравнение моделей (финальное)
*Датасет:* `train_10k`. *Конфигурация:* лучшая из Эксп.1+2.

| Модель | Описание |
|--------|---------|
| Baseline | SD 1.5 без файн-тюнинга, промпт-инжиниринг |
| SDXL + LoRA | Файн-тюнинг на `train_10k` |
| FLUX.1-dev + LoRA | Файн-тюнинг на `train_10k` через ai-toolkit |

### Эксперимент 4 — Сравнение с коммерческими аналогами
На тестовом наборе из 50 промптов:
- Recraft v3 API
- DALL·E 3 (если доступен API)
- Лучшая fine-tuned модель из Эксп.3

---

## 3. Метрики и оценка

### Автоматические метрики (на `test_500`)

| Метрика | Что измеряет | Библиотека |
|---------|-------------|------------|
| FID | Качество и разнообразие | `torch-fidelity` |
| CLIP Score | Text-image соответствие | `open_clip` |
| LPIPS | Перцептивное сходство | `lpips` |

### VLM-верификация (вместо user study)
LLaVA оценивает каждое сгенерированное изображение:
- *"Is this image a logo? Rate 0-5"*
- *"Does the image match this description: [caption]? Rate 0-5"*
- *"Rate the aesthetic quality of this logo 0-5"*

Даёт воспроизводимые числа, заменяет субъективный опрос.

### Тест масштабируемости (специфика логотипов)
Каждый сгенерированный логотип ресайзим до 16×16, 32×32, 64×64, 512×512 и считаем SSIM — проверяем читаемость на малых размерах и баннерах.

---

## 4. Структура проекта

```
liya_diplomCC/
├── notebooks/
│   ├── 01_dataset_collection.ipynb
│   ├── 02_dataset_captioning.ipynb
│   ├── 03_dataset_verification.ipynb
│   ├── 04_train_sdxl_lora.ipynb
│   ├── 05_train_flux_lora.ipynb
│   ├── 06_inference_compare.ipynb
│   └── 07_metrics_evaluation.ipynb
├── configs/
│   ├── sdxl_lora_r4.yaml
│   ├── sdxl_lora_r8.yaml
│   ├── sdxl_lora_r16.yaml
│   ├── sdxl_lora_r32.yaml
│   └── flux_lora_r16.yaml
├── scripts/
│   ├── svg_to_png.py
│   ├── filter_dataset.py
│   ├── caption_llava.py
│   ├── verify_dataset.py
│   ├── compute_metrics.py
│   └── scale_test.py
├── data/
│   ├── raw_svg/
│   ├── png_512/
│   ├── dataset_2k.jsonl
│   └── dataset_10k.jsonl
├── results/
│   ├── experiments/
│   └── metrics/
└── docs/
    └── superpowers/specs/
```

Данные и чекпоинты — Google Drive. В репо только код, конфиги, результаты метрик.

---

## 5. Порядок выполнения (Подход B — параллельный)

1. Собрать `train_2k` + `test_500` → сразу запустить Эксп.1 (ablation по рангу)
2. Параллельно наращивать датасет до `train_10k`
3. По итогам Эксп.1 — Эксп.2 (ablation по шагам)
4. Финальное обучение на `train_10k` (Эксп.3: SDXL + FLUX)
5. Сравнение с Recraft/DALL·E (Эксп.4)
6. Сбор всех метрик и примеров для ВКР

---

## Вне области (следующая фаза)
- FastAPI inference server
- Telegram-бот (aiogram 3.x)
- Redis / PostgreSQL
- Нагрузочное тестирование
