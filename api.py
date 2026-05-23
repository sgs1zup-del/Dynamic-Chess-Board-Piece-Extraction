"""
Flask API for chess board analysis.
Receives an image from a Flutter app and returns the FEN string.
"""

import os
import sys
import uuid
import subprocess
import math
from pathlib import Path

import cv2
import numpy as np
import chess
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
EXTRACTED_DIR = PROJECT_ROOT / "extracted-data"
SQUARE_FILLING_SCRIPT = PROJECT_ROOT / "square_filling.py"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_PATH = PROJECT_ROOT / "chess-model-yolov8m-finetuned.pt"

# ---------------------------------------------------------------------------
# YOLO model (lazy load)
# ---------------------------------------------------------------------------
_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model


# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------
CLASS_DICT = {
    0: "black-bishop",
    1: "black-king",
    2: "black-knight",
    3: "black-pawn",
    4: "black-queen",
    5: "black-rook",
    6: "white-bishop",
    7: "white-king",
    8: "white-knight",
    9: "white-pawn",
    10: "white-queen",
    11: "white-rook",
}

PIECE_MAPPING = {
    "white-pawn": chess.PAWN,
    "black-pawn": chess.PAWN,
    "white-knight": chess.KNIGHT,
    "black-knight": chess.KNIGHT,
    "white-bishop": chess.BISHOP,
    "black-bishop": chess.BISHOP,
    "white-rook": chess.ROOK,
    "black-rook": chess.ROOK,
    "white-queen": chess.QUEEN,
    "black-queen": chess.QUEEN,
    "white-king": chess.KING,
    "black-king": chess.KING,
}


# ---------------------------------------------------------------------------
# Helper functions for /analyze2
# ---------------------------------------------------------------------------

def find_board_corners(image):
    """Find the 4 corners of the chess board using contour approximation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = image.shape[0] * image.shape[1] * 0.1

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Fallback with adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def order_points(pts):
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_board(image, corners, size=800):
    """Warp the board to a square top-down view."""
    rect = order_points(corners)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped


def detect_pieces_on_warped(warped_image, model, board_size=800):
    """Run YOLO on warped image and return piece positions."""
    results = model(warped_image, verbose=False)
    square_size = board_size // 8
    cell_to_piece = {}

    for result in results:
        if result.boxes is None:
            continue
        for idx, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            x_mid = (x1 + x2) // 2
            y_mid = (y1 + y2) // 2 + 25

            col = int(x_mid // square_size)
            row = int(y_mid // square_size)

            col = max(0, min(7, col))
            row = max(0, min(7, row))

            # Map to cell number: 1=a1, 8=h1, ..., 64=h8
            # row=0 -> rank 8 (top of warped image), row=7 -> rank 1 (bottom)
            cell = (7 - row) * 8 + (col + 1)
            class_id = int(result.boxes.cls[idx])

            if cell not in cell_to_piece:
                cell_to_piece[cell] = class_id

    return cell_to_piece


def cell_dict_to_fen(cell_dict):
    """Convert cell dictionary to FEN string."""
    board = chess.Board(None)
    for cell, class_id in cell_dict.items():
        piece_name = CLASS_DICT.get(class_id)
        if piece_name is None:
            continue
        color = chess.WHITE if piece_name.startswith("white") else chess.BLACK
        piece_type = PIECE_MAPPING[piece_name]
        file_idx = (cell - 1) % 8
        rank_idx = (cell - 1) // 8
        board.set_piece_at(
            chess.square(file_idx, rank_idx), chess.Piece(piece_type, color)
        )
    return board.fen()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    Accept a chess-board photo (multipart/form-data, field name 'image')
    and return the detected FEN position.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No 'image' field in form-data"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    tmp_filename = UPLOAD_FOLDER / f"upload_{uuid.uuid4().hex}.jpg"
    image_file.save(str(tmp_filename))

    try:
        env = os.environ.copy()
        env["HEADLESS"] = "1"
        env["MPLBACKEND"] = "Agg"

        cmd = [
            sys.executable,
            str(SQUARE_FILLING_SCRIPT),
            "--image", str(tmp_filename),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.stdout:
            print("[square_filling stdout]", result.stdout)
        if result.stderr:
            print("[square_filling stderr]", result.stderr)

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "square_filling.py failed",
                "details": result.stderr[-1000:] if result.stderr else "unknown error",
            }), 500

        fen_path = EXTRACTED_DIR / "result.fen"
        if not fen_path.exists():
            return jsonify({
                "status": "error",
                "message": "FEN file not generated",
            }), 500

        fen = fen_path.read_text().strip()

        return jsonify({
            "fen": fen,
            "status": "success",
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "error",
            "message": "Processing timed out",
        }), 504

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500

    finally:
        try:
            if tmp_filename.exists():
                tmp_filename.unlink()
        except Exception:
            pass


@app.route("/analyze2", methods=["POST", "OPTIONS"])
def analyze2():
    """
    Accept a chess-board photo, detect the board via contour approximation,
    warp it to a top-down view, run YOLO, and return the detected FEN.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No 'image' field in form-data"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    tmp_filename = UPLOAD_FOLDER / f"upload2_{uuid.uuid4().hex}.jpg"
    image_file.save(str(tmp_filename))

    try:
        image = cv2.imread(str(tmp_filename))
        if image is None:
            return jsonify({"status": "error", "message": "Could not read image"}), 400

        corners = find_board_corners(image)
        if corners is None:
            return jsonify({"status": "error", "message": "Could not detect board corners"}), 400

        warped = warp_board(image, corners, size=800)

        model = get_yolo_model()
        if model is None:
            return jsonify({"status": "error", "message": "YOLO model not available"}), 500

        cell_dict = detect_pieces_on_warped(warped, model)
        fen = cell_dict_to_fen(cell_dict)

        return jsonify({"fen": fen, "status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        try:
            if tmp_filename.exists():
                tmp_filename.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Chess Vision API on http://0.0.0.0:5000")
    print("Health check: GET http://<your-ip>:5000/health")
    print("Analyze:      POST http://<your-ip>:5000/analyze")
    print("Analyze2:     POST http://<your-ip>:5000/analyze2")
    app.run(host="0.0.0.0", port=5000, debug=False)
