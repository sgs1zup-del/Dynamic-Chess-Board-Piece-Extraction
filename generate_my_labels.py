import os
import shutil

# FEN piece to YOLO class mapping
PIECE_TO_CLASS = {
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
    'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11,
}

IMG_SIZE = 800
SQUARE_SIZE = 100

def parse_fen_board(fen_board):
    """Parse FEN board part into list of (rank, file, piece) where rank 0-7 from top."""
    pieces = []
    ranks = fen_board.split('/')
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                pieces.append((rank_idx, file_idx, ch))
                file_idx += 1
    return pieces

def fen_to_yolo(fen_board):
    pieces = parse_fen_board(fen_board)
    lines = []
    for rank_idx, file_idx, piece in pieces:
        cls = PIECE_TO_CLASS[piece]
        x_center = (file_idx * SQUARE_SIZE + SQUARE_SIZE / 2) / IMG_SIZE
        y_center = (rank_idx * SQUARE_SIZE + SQUARE_SIZE / 2) / IMG_SIZE
        w = SQUARE_SIZE / IMG_SIZE
        h = SQUARE_SIZE / IMG_SIZE
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines

def main():
    base_dir = r"C:\Dynamic-Chess-Board-Piece-Extraction"
    fens_path = os.path.join(base_dir, "dataset", "fens.txt")
    images_src = os.path.join(base_dir, "dataset", "images")
    images_dst = os.path.join(base_dir, "roboflow", "train", "images")
    labels_dst = os.path.join(base_dir, "roboflow", "train", "labels")

    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    with open(fens_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            img_name = parts[0]
            fen = parts[1] if len(parts) > 1 else ""
            # extract board part (before first space)
            fen_board = fen.split()[0]

            base_name = os.path.splitext(img_name)[0]
            src_img = os.path.join(images_src, img_name)
            dst_img = os.path.join(images_dst, img_name)
            dst_label = os.path.join(labels_dst, base_name + ".txt")

            if not os.path.exists(src_img):
                print(f"SKIP missing image: {src_img}")
                continue

            yolo_lines = fen_to_yolo(fen_board)
            with open(dst_label, 'w', encoding='utf-8') as lf:
                lf.write("\n".join(yolo_lines))
            shutil.copy2(src_img, dst_img)
            print(f"OK {img_name} -> {len(yolo_lines)} boxes")

if __name__ == "__main__":
    main()
