# Telegram-бот генерации логотипов

Бот к диплому: генерирует 4 варианта логотипа по описанию, используя
дообученные LoRA-веса из `results/experiments/`.

## Команды

| Команда     | Что делает                                                                 |
|-------------|----------------------------------------------------------------------------|
| `/start`, `/menu` | Приветствие и список команд.                                         |
| `/generate` | Запрашивает текстовый промпт, возвращает 4 варианта медиагруппой.          |
| `/feedback` | Выбор лучшего варианта из последней генерации; пишется в БД.               |
| `/history`  | Запросы за последние 24 часа со ссылками на файлы (пока живёт TTL).        |
| `/model`    | Выбор модели: SDXL r4/r16/r32 либо FLUX r16.                                |
| `/help`     | Подсказка по структуре промпта и примеры.                                  |

Дополнительно: любое текстовое сообщение длиной > 5 символов трактуется как
неявный `/generate`.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# для реальной генерации раскомментируйте torch/diffusers/... в requirements.txt
copy .env.example .env   # пропишите TELEGRAM_BOT_TOKEN
python -m bot.main
```

## Backend генерации

`bot/generator.py` лениво грузит SDXL или FLUX пайплайн под выбранную модель
и подключает LoRA из `results/experiments/...`. Поддерживаются три режима:

- `BOT_DEVICE=cuda` — реальная генерация на GPU (нужны torch+diffusers и веса
  базовых моделей через `huggingface_hub`).
- `BOT_DEVICE=cpu` — то же на CPU (очень медленно).
- `BOT_DEVICE=stub` — заглушка: рисует placeholder с промптом и seed-ом.
  Включается автоматически, если `torch` не установлен или CUDA недоступна.
  Удобно для отладки Telegram-флоу без GPU.

## Где что хранится

- `bot_storage/bot.db` — SQLite с таблицами `generations`, `variants`,
  `feedback`, `user_settings`.
- `bot_storage/images/<gen_id>/` — PNG-варианты, чистятся по TTL = 24 ч
  (фоновая задача `cleanup_loop`). По сути имитирует объектное хранилище.

## Структура

```
bot/
  main.py        # точка входа, polling
  config.py      # модели, пути, настройки
  deps.py        # синглтоны storage/generator/settings
  generator.py   # SDXL/FLUX + LoRA, либо stub
  storage.py     # SQLite + локальные файлы + TTL
  handlers.py    # /start, /generate, /feedback, /history, /model, …
  texts.py       # тексты ответов
```
