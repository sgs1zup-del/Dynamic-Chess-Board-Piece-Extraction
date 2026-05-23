import asyncio
import os
from pathlib import Path
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ContentType

# === НАСТРОЙКИ ===
TOKEN = "8627609761:AAF_OlPzSkeVPr7cTcejygL9c5y4Gdkjc6s"  
SAVE_DIR = Path("C:/Dynamic-Chess-Board-Piece-Extraction/dataset")
IMAGES_DIR = SAVE_DIR / "images"
FENS_FILE = SAVE_DIR / "fens.txt"

# Создаём папки
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Состояния пользователей
user_states = {}  # user_id -> {"photo_path": ..., "waiting_fen": True/False}

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот для сбора шахматного датасета.\n\n"
        "Отправь мне фото доски — я сохраню его и попрошу FEN."
    )

@dp.message(lambda msg: msg.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    
    # Генерируем имя файла
    photo_count = len(list(IMAGES_DIR.glob("img_*.jpg"))) + 1
    filename = f"img_{photo_count:03d}.jpg"
    photo_path = IMAGES_DIR / filename
    
    # Скачиваем фото
    await message.bot.download(
        message.photo[-1],  # самое большое разрешение
        destination=photo_path
    )
    
    # Сохраняем состояние
    user_states[user_id] = {
        "photo_path": str(photo_path),
        "filename": filename,
        "waiting_fen": True
    }
    
    await message.answer(
        f"Фото сохранено как `{filename}`.\n"
        f"Теперь отправь FEN позиции (можно просто текстом)."
    )

@dp.message(lambda msg: msg.text and not msg.text.startswith("/"))
async def handle_fen(message: types.Message):
    user_id = message.from_user.id
    
    if user_id not in user_states or not user_states[user_id].get("waiting_fen"):
        await message.answer("Сначала отправь фото доски!")
        return
    
    fen = message.text.strip()
    filename = user_states[user_id]["filename"]
    
    # Сохраняем в fens.txt
    with open(FENS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename} {fen}\n")
    
    # Сбрасываем состояние
    user_states[user_id]["waiting_fen"] = False
    
    await message.answer(
        f"✅ Сохранено!\n"
        f"Файл: `{filename}`\n"
        f"FEN: `{fen}`\n\n"
        f"Отправь следующее фото или /done чтобы закончить."
    )

@dp.message(Command("done"))
async def cmd_done(message: types.Message):
    user_id = message.from_user.id
    if user_id in user_states:
        del user_states[user_id]
    
    # Считаем сколько сохранено
    photo_count = len(list(IMAGES_DIR.glob("img_*.jpg")))
    
    await message.answer(
        f"Готово! Сохранено {photo_count} фото.\n"
        f"Файлы в: `{SAVE_DIR}`\n"
        f"Можешь начать обучение модели."
    )

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    photo_count = len(list(IMAGES_DIR.glob("img_*.jpg")))
    fen_count = 0
    if FENS_FILE.exists():
        fen_count = len(FENS_FILE.read_text(encoding="utf-8").strip().split("\n"))
    
    await message.answer(
        f"📊 Статус датасета:\n"
        f"Фото: {photo_count}\n"
        f"FEN записей: {fen_count}"
    )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())