from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from . import config, deps, storage as storage_module
from .generator import ImageGenerator
from .handlers import router

logger = logging.getLogger("bot")


BOT_COMMANDS = [
    BotCommand(command="start", description="Приветствие и меню"),
    BotCommand(command="menu", description="Показать меню"),
    BotCommand(command="generate", description="Сгенерировать логотип"),
    BotCommand(command="feedback", description="Выбрать лучший вариант"),
    BotCommand(command="history", description="История за 24 часа"),
    BotCommand(command="model", description="Сменить модель"),
    BotCommand(command="help", description="Как писать промпт"),
]


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = config.load_settings()
    storage = storage_module.Storage()
    generator = ImageGenerator(settings)
    deps.init(storage=storage, generator=generator, settings=settings)
    logger.info("Backend: %s", generator.backend_info)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    await bot.set_my_commands(BOT_COMMANDS)
    cleanup_task = asyncio.create_task(
        storage_module.cleanup_loop(storage, settings.history_ttl_hours)
    )
    try:
        await dp.start_polling(bot)
    finally:
        cleanup_task.cancel()
        await bot.session.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
