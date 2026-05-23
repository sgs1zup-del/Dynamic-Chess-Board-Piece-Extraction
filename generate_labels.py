import cv2
import numpy as np
from pathlib import Path
import chess

# Пути
BASE_DIR = Path("C:/Dynamic-Chess-Board-Piece-Extraction")
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
FENS_FILE = DATASET_DIR / "fens.txt"

LABELS_DIR.mkdir(exist_ok=True)

# Классы YOLO
PIECE_TO_CLASS = {
    'r': 5, 'n': 2, 'b': 0, 'q': 4, 'k': 1, 'p': 3,   # чёрные
    'R': 11, 'N': 8, 'B': 6, 'Q': 10, 'K': 7, 'P': 9   # белые
}

def fen_to_board(fen):
    """Парсит FEN в массив 8x8"""
    board = []
    rows = fen.split()[0].split('/')
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['.'] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)
    return board

def get_square_centers(image_path):
    """
    Получает координаты центров клеток.
    Используем упрощённый подход: находим доску и делим на 8x8.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    
    # Упрощённый метод: предполагаем, что доска занимает большую часть изображения
    # и ориентирована примерно прямо. Для точности лучше использовать square_filling.py,
    # но пока используем грубую оценку.
    
    # Находим контуры (упрощённо)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Ищем прямоугольники
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ищем самый большой прямоугольник (доска)
    board_contour = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                board_contour = approx
    
    if board_contour is None or max_area < (w * h * 0.1):
        # Если не нашли доску, используем центральную область
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        board_x, board_y = margin_x, margin_y
        board_w, board_h = w - 2*margin_x, h - 2*margin_y
    else:
        # Получаем координаты доски
        pts = board_contour.reshape(4, 2)
        x_coords = sorted([p[0] for p in pts])
        y_coords = sorted([p[1] for p in pts])
        board_x, board_y = x_coords[0], y_coords[0]
        board_w = x_coords[-1] - x_coords[0]
        board_h = y_coords[-1] - y_coords[0]
    
    # Делим доску на 8x8 клеток
    cell_w = board_w / 8
    cell_h = board_h / 8
    
    centers = {}
    for rank in range(8):      # 0-7 (8 ряд -> 1 ряд)
        for file in range(8):  # 0-7 (a -> h)
            square = f"{chr(97+file)}{8-rank}"
            center_x = board_x + file * cell_w + cell_w / 2
            center_y = board_y + rank * cell_h + cell_h / 2
            centers[square] = (center_x / w, center_y / h, cell_w / w * 0.8, cell_h / h * 0.8)
    
    return centers, (w, h)

def generate_label(image_path, fen, output_path):
    """Генерирует YOLO-разметку из FEN"""
    board = fen_to_board(fen)
    centers, img_size = get_square_centers(image_path)
    
    if centers is None:
        print(f"Не удалось обработать {image_path}")
        return
    
    labels = []
    for rank_idx, row in enumerate(board):
        for file_idx, piece in enumerate(row):
            if piece == '.':
                continue
            
            square = f"{chr(97+file_idx)}{8-rank_idx}"
            if square not in centers:
                continue
            
            cx, cy, bw, bh = centers[square]
            class_id = PIECE_TO_CLASS[piece]
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(labels))
    
    print(f"Создан: {output_path} ({len(labels)} фигур)")

def main():
    # Читаем fens.txt
    with open(FENS_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Найдено {len(lines)} записей")
    
    for line in lines:
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue
        
        filename, fen = parts
        image_path = IMAGES_DIR / filename
        label_path = LABELS_DIR / filename.replace('.jpg', '.txt').replace('.jpeg', '.txt')
        
        if not image_path.exists():
            print(f"Фото не найдено: {image_path}")
            continue
        
        generate_label(image_path, fen, label_path)
    
    print("Готово!")

if __name__ == "__main__":
    main()