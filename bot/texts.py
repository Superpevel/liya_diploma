from __future__ import annotations

from . import config

START_TEXT = (
    "👋 Привет! Я бот для генерации логотипов на дообученной модели "
    "(SDXL/FLUX + LoRA).\n\n"
    "Доступные команды:\n"
    "• /generate — сгенерировать логотип по описанию\n"
    "• /feedback — выбрать лучший вариант из последней генерации\n"
    "• /history — мои запросы за последние 24 часа\n"
    "• /model — выбрать модель генерации\n"
    "• /help — как формулировать промпт\n"
    "• /menu — это меню\n\n"
    "Можно просто прислать описание текстом (>5 символов), без команды."
)

HELP_TEXT = (
    "📝 Как сформулировать промпт\n\n"
    "Структура: <b>что</b> + <b>стиль</b> + <b>палитра</b> + <b>отрасль</b>.\n\n"
    "Удачные примеры:\n"
    "• <code>burger logo, flat vector style, red and yellow palette, "
    "fast food brand, minimalistic, white background</code>\n"
    "• <code>coffee shop logo, hand-drawn line art, brown and cream colors, "
    "cozy and warm, vintage</code>\n"
    "• <code>tech startup logo, geometric, blue gradient, modern, abstract "
    "monogram</code>\n\n"
    "Советы:\n"
    "1. Указывайте стиль (flat, vintage, line art, 3d, gradient).\n"
    "2. Описывайте цвета явно («red and yellow», «pastel»).\n"
    "3. Добавляйте отрасль («fast food», «fintech», «kids brand»).\n"
    "4. Для иконок добавляйте «icon, white background, centered»."
)


def menu_text() -> str:
    return START_TEXT


def model_list_text(current_key: str) -> str:
    lines = ["🧠 Доступные модели:"]
    for key, cfg in config.MODELS.items():
        marker = "✅" if key == current_key else "▫️"
        lines.append(f"{marker} <b>{cfg.title}</b>")
    lines.append("\nВыбери модель кнопкой ниже.")
    return "\n".join(lines)


GENERATE_PROMPT_TEXT = (
    "✍️ Пришли текстовое описание логотипа.\n"
    "Например: <i>burger logo, flat style, red and yellow, fast food</i>.\n"
    "Подсказки по структуре — /help."
)

TOO_SHORT_TEXT = (
    "Описание слишком короткое. Нужно хотя бы 6 символов — "
    "например, <i>burger logo, flat</i>."
)

GENERATING_TEXT = "🎨 Генерирую 4 варианта, это займёт ~30–90 секунд…"

NO_LAST_GEN_TEXT = (
    "Сначала сгенерируй логотип через /generate, затем сможешь выбрать "
    "лучший вариант."
)

NO_HISTORY_TEXT = "За последние 24 часа запросов нет. Попробуй /generate."

FEEDBACK_SAVED_TEXT = "Спасибо! Выбор сохранён — он поможет дообучить модель. 🙏"
