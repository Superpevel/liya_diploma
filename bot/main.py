import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from . import config, deps
from .generator import ImageGenerator
from .handlers import router
from .storage import Storage, cleanup_loop

log = logging.getLogger("bot")


COMMANDS = [
    BotCommand(command="start", description="Приветствие и меню"),
    BotCommand(command="menu", description="Показать меню"),
    BotCommand(command="generate", description="Сгенерировать логотип"),
    BotCommand(command="feedback", description="Выбрать лучший вариант"),
    BotCommand(command="history", description="История за 24 часа"),
    BotCommand(command="model", description="Сменить модель"),
    BotCommand(command="help", description="Как писать промпт"),
]


async def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = config.load_settings()
    store = Storage()
    gen = ImageGenerator(settings)
    deps.init(store, gen, settings)
    log.info("Backend: %s", gen.backend_info)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    await bot.set_my_commands(COMMANDS)
    cleaner = asyncio.create_task(cleanup_loop(store, settings.history_ttl_hours))
    try:
        await dp.start_polling(bot)
    finally:
        cleaner.cancel()
        await bot.session.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
