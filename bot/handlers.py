from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Optional

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)

from . import config, deps, texts
from .storage import Generation

logger = logging.getLogger(__name__)
router = Router()

MIN_PROMPT_LEN = 6


class GenStates(StatesGroup):
    waiting_prompt = State()


def feedback_keyboard(generation: Generation) -> InlineKeyboardMarkup:
    buttons = []
    for v in generation.variants:
        buttons.append(
            InlineKeyboardButton(
                text=f"Вариант {v.idx + 1}",
                callback_data=f"fb:{generation.id}:{v.id}",
            )
        )
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def model_keyboard(current_key: str) -> InlineKeyboardMarkup:
    rows = []
    for key, cfg in config.MODELS.items():
        marker = "✅ " if key == current_key else ""
        rows.append([
            InlineKeyboardButton(
                text=f"{marker}{cfg.title}",
                callback_data=f"model:{key}",
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(CommandStart())
@router.message(Command("menu"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(texts.START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.HELP_TEXT, parse_mode="HTML")


@router.message(Command("generate"))
async def cmd_generate(message: Message, state: FSMContext) -> None:
    await state.set_state(GenStates.waiting_prompt)
    await message.answer(texts.GENERATE_PROMPT_TEXT, parse_mode="HTML")


@router.message(Command("model"))
async def cmd_model(message: Message) -> None:
    storage = deps.storage()
    current = await storage.get_user_model(message.from_user.id)
    await message.answer(
        texts.model_list_text(current),
        parse_mode="HTML",
        reply_markup=model_keyboard(current),
    )


@router.callback_query(F.data.startswith("model:"))
async def on_model_pick(call: CallbackQuery) -> None:
    storage = deps.storage()
    key = call.data.split(":", 1)[1]
    if key not in config.MODELS:
        await call.answer("Неизвестная модель", show_alert=True)
        return
    await storage.set_user_model(call.from_user.id, key)
    cfg = config.MODELS[key]
    await call.message.edit_text(
        texts.model_list_text(key),
        parse_mode="HTML",
        reply_markup=model_keyboard(key),
    )
    await call.answer(f"Выбрано: {cfg.title}")


@router.message(Command("feedback"))
async def cmd_feedback(message: Message) -> None:
    storage = deps.storage()
    last = await storage.last_generation(message.from_user.id)
    if last is None or not last.variants:
        await message.answer(texts.NO_LAST_GEN_TEXT)
        return
    prompt = html.escape(last.prompt)
    await message.answer(
        f"Выбери лучший вариант для запроса:\n<i>{prompt}</i>",
        parse_mode="HTML",
        reply_markup=feedback_keyboard(last),
    )


@router.callback_query(F.data.startswith("fb:"))
async def on_feedback(call: CallbackQuery) -> None:
    storage = deps.storage()
    _, gen_id, variant_id = call.data.split(":", 2)
    await storage.save_feedback(call.from_user.id, gen_id, variant_id)
    await call.answer(texts.FEEDBACK_SAVED_TEXT, show_alert=False)
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await call.message.answer(texts.FEEDBACK_SAVED_TEXT)


@router.message(Command("history"))
async def cmd_history(message: Message) -> None:
    storage = deps.storage()
    settings = deps.settings()
    items = await storage.recent_generations(
        message.from_user.id, settings.history_ttl_hours
    )
    if not items:
        await message.answer(texts.NO_HISTORY_TEXT)
        return
    lines = [
        f"📜 История за последние {settings.history_ttl_hours} ч "
        f"(всего {len(items)}):"
    ]
    for gen in items:
        ts = datetime.fromtimestamp(gen.created_at, tz=timezone.utc).astimezone()
        when = ts.strftime("%d.%m %H:%M")
        prompt = html.escape(gen.prompt[:80])
        model = config.MODELS.get(gen.model_key)
        model_name = model.title if model else gen.model_key
        files = [str(v.file_path) for v in gen.variants]
        files_line = (
            "\n   файлы: " + ", ".join(f"<code>{html.escape(f)}</code>"
                                       for f in files)
            if files else "\n   (файлы удалены по TTL)"
        )
        lines.append(
            f"• <b>{when}</b> · {html.escape(model_name)}\n"
            f"   <i>{prompt}</i>{files_line}"
        )
    await message.answer("\n".join(lines), parse_mode="HTML",
                         disable_web_page_preview=True)


@router.message(GenStates.waiting_prompt, F.text)
async def on_prompt_state(message: Message, state: FSMContext) -> None:
    await state.clear()
    await _run_generation(message, message.text.strip())


@router.message(F.text & ~F.text.startswith("/"))
async def on_free_text(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if len(text) <= 5:
        await message.answer(texts.TOO_SHORT_TEXT, parse_mode="HTML")
        return
    await state.clear()
    await _run_generation(message, text)


async def _run_generation(message: Message, prompt: str) -> None:
    if len(prompt) < MIN_PROMPT_LEN:
        await message.answer(texts.TOO_SHORT_TEXT, parse_mode="HTML")
        return
    storage = deps.storage()
    generator = deps.generator()
    settings = deps.settings()
    user_id = message.from_user.id
    model_key = await storage.get_user_model(user_id)
    cfg = config.MODELS[model_key]

    status = await message.answer(texts.GENERATING_TEXT)
    try:
        result = await generator.generate(
            prompt=prompt,
            model_key=model_key,
            num_variants=settings.num_variants,
        )
    except FileNotFoundError as e:
        await status.edit_text(f"❌ {e}")
        return
    except Exception as e:
        logger.exception("Generation failed")
        await status.edit_text(f"❌ Ошибка генерации: {e}")
        return

    gen = await storage.save_generation(
        user_id=user_id,
        prompt=prompt,
        model_key=model_key,
        images=result.images,
        seeds=result.seeds,
    )

    media = []
    for v in gen.variants:
        with open(v.file_path, "rb") as f:
            data = f.read()
        caption = None
        if v.idx == 0:
            caption = (
                f"🖼 4 варианта · модель: <b>{html.escape(cfg.title)}</b>\n"
                f"Запрос: <i>{html.escape(prompt)}</i>"
            )
        media.append(
            InputMediaPhoto(
                media=BufferedInputFile(
                    data, filename=f"variant_{v.idx + 1}.png"
                ),
                caption=caption,
                parse_mode="HTML" if caption else None,
            )
        )
    await message.answer_media_group(media)
    await message.answer(
        "Какой вариант лучший? Нажми /feedback или кнопку ниже:",
        reply_markup=feedback_keyboard(gen),
    )
    try:
        await status.delete()
    except Exception:
        pass
