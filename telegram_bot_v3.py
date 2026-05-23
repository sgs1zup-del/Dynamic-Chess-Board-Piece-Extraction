import asyncio
import os
from pathlib import Path
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command

# === НАСТРОЙКИ ===
TOKEN = "8627609761:AAF_OlPzSkeVPr7cTcejygL9c5y4Gdkjc6s"
SAVE_DIR = Path("C:/Dynamic-Chess-Board-Piece-Extraction/dataset")
IMAGES_DIR = SAVE_DIR / "images"
FENS_FILE = SAVE_DIR / "fens.txt"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

user_states = {}

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Отправь фото доски — я сохраню и попрошу FEN."
    )

@dp.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id
    
    photo_count = len(list(IMAGES_DIR.glob("img_*.jpg"))) + 1
    filename = f"img_{photo_count:03d}.jpg"
    photo_path = IMAGES_DIR / filename
    
    await message.bot.download(
        message.photo[-1],
        destination=photo_path
    )
    
    user_states[user_id] = {
        "filename": filename,
        "waiting_fen": True
    }
    
    await message.answer(
        f"Фото сохранено: `{filename}`\nПришли FEN текстом."
    )

@dp.message(F.text)
async def handle_fen(message: Message):
    user_id = message.from_user.id
    
    if user_id not in user_states or not user_states[user_id].get("waiting_fen"):
        await message.answer("Сначала отправь фото!")
        return
    
    fen = message.text.strip()
    filename = user_states[user_id]["filename"]
    
    with open(FENS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename} {fen}\n")
    
    user_states[user_id]["waiting_fen"] = False
    
    await message.answer(
        f"✅ Сохранено!\n{filename}\n{fen}\n\nОтправь следующее фото."
    )

@dp.message(Command("done"))
async def cmd_done(message: Message):
    user_id = message.from_user.id
    if user_id in user_states:
        del user_states[user_id]
    
    photo_count = len(list(IMAGES_DIR.glob("img_*.jpg")))
    await message.answer(f"Готово! Сохранено {photo_count} фото.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())