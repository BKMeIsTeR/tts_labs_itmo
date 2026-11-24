"""Telegram-бот для озвучивания русских фраз трактом ЛР5"""

import asyncio
import getpass
import io
import logging
import os
import sys

import av
import numpy as np
from telegram import InputFile, Update
from telegram.error import Conflict, Forbidden, InvalidToken, NetworkError, RetryAfter, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from tts_pipeline import SAMPLE_RATE, TTSPipeline, WORD

LOG = logging.getLogger(__name__)
MAX_CHARS = 2000
HELP = (
    f"Пришлите русскую фразу до {MAX_CHARS} символов, и я отвечу голосовым сообщением. "
    "Числа пишите словами, иностранные слова передавайте русскими буквами. "
    "Например, «Встречаемся завтра в пять часов!»"
)


class InputError(ValueError):
    """Подсказка, по которой пользователь может исправить текст"""


def make_voice(pipeline: TTSPipeline, text: str) -> bytes:
    if len(text) > MAX_CHARS:
        raise InputError(f"Разделите текст на сообщения до {MAX_CHARS} символов каждое")
    try:
        tokens = pipeline.tokenize(pipeline.normalize(text))
    except ValueError as error:
        raise InputError(f"{error}. Числа и иностранные слова запишите русскими словами") from error
    if not tokens:
        raise InputError("В сообщении нет слов для озвучивания. Пришлите фразу по-русски")
    # Тракт делит длинные фразы между словами. Слитная строка обходит это ограничение
    if any(len(WORD.match(token).group()) > 100 for token in tokens):
        raise InputError("В тексте есть слово длиннее 100 символов. Проверьте пробелы")

    wav, _ = pipeline(text)
    if not wav.size:
        raise InputError("Для этой записи не получилось подобрать фонемы. Попробуйте другую фразу")
    peak = float(np.max(np.abs(wav)))
    if not np.isfinite(peak) or peak == 0:
        raise RuntimeError("Синтезатор вернул тишину или некорректные отсчёты")

    # Тракт выдаёт моно при 22 050 Гц. PyAV пересчитывает частоту при кодировании Opus
    frame = av.AudioFrame.from_ndarray((wav / peak).astype("float32")[None, :],
                                       format="flt", layout="mono")
    frame.sample_rate = SAMPLE_RATE
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        stream.bit_rate = 32000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


async def load_pipeline(app: Application) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Проверьте драйвер NVIDIA и окружение TTS_ITMO_LAB4")
    LOG.info("Загружаю модели ЛР5 на %s", torch.cuda.get_device_name(0))
    app.bot_data["pipeline"] = await asyncio.to_thread(TTSPipeline, device="cuda")
    LOG.info("Модели готовы. Откройте @%s в Telegram", app.bot.username)


async def help_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def speak(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    try:
        voice = await asyncio.to_thread(make_voice, context.bot_data["pipeline"], message.text)
    except InputError as error:
        await message.reply_text(str(error))
        return
    except Exception:
        LOG.exception("Ошибка синтеза или кодирования Opus")
        await message.reply_text("Не удалось озвучить фразу. Попробуйте отправить короткое предложение")
        return
    await message.reply_voice(voice=InputFile(voice, filename="speech.ogg"), write_timeout=60)


async def unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Для озвучивания пришлите обычное текстовое сообщение. Подсказка в /help")


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, Conflict):
        LOG.error("Этот токен уже использует другой процесс. Остановите лишний экземпляр бота")
        context.application.stop_running()
        return
    if isinstance(error, Forbidden):
        return  # Пользователь заблокировал бота или запретил ответы
    LOG.warning("Ошибка Telegram: %s", type(error).__name__)
    if isinstance(update, Update) and update.message and not isinstance(error, RetryAfter):
        try:
            await update.message.reply_text("Не удалось доставить ответ. Попробуйте отправить сообщение ещё раз")
        except TelegramError:
            LOG.warning("Telegram также не принял текстовую подсказку")


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token and sys.stdin.isatty():
        token = getpass.getpass("Вставьте токен от @BotFather и нажмите Enter: ").strip()
    if not token:
        raise SystemExit("Токен пуст. Запустите скрипт в терминале или задайте TELEGRAM_BOT_TOKEN")

    # При отклонённом токене библиотека может включить его в текст исключения
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            return super().format(record).replace(token, "[токен скрыт]")

    handler = logging.StreamHandler()
    handler.setFormatter(SafeFormatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        # Последовательная обработка оставляет на видеокарте один активный запрос
        app = Application.builder().token(token).post_init(load_pipeline).concurrent_updates(False).build()
        app.add_handler(CommandHandler(["start", "help"], help_message))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, speak))
        app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.StatusUpdate.ALL, unsupported))
        app.add_error_handler(handle_error)
        app.run_polling(allowed_updates=["message"])
    except InvalidToken:
        raise SystemExit("Telegram отклонил токен. Скопируйте его целиком из @BotFather") from None
    except NetworkError:
        raise SystemExit("Не удалось подключиться к Telegram. Проверьте интернет и доступ к api.telegram.org") from None


if __name__ == "__main__":
    main()
