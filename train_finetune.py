from ultralytics import YOLO
import shutil
import os

model_path = r"C:\Dynamic-Chess-Board-Piece-Extraction\chess-model-yolov8m.pt"
data_path = r"C:\Dynamic-Chess-Board-Piece-Extraction\data.yaml"
output_path = r"C:\Dynamic-Chess-Board-Piece-Extraction\chess-model-yolov8m-finetuned.pt"

model = YOLO(model_path)

results = model.train(
    data=data_path,
    epochs=30,
    imgsz=640,
    batch=8,
    lr0=0.001,
    lrf=0.0001,
    freeze=10,
    patience=20,
    optimizer='SGD',
    project="runs/detect",
    name="finetune",
    exist_ok=True,
    verbose=True,
)

# Copy best weights to target path
best_src = os.path.join("runs", "detect", "finetune", "weights", "best.pt")
if os.path.exists(best_src):
    shutil.copy2(best_src, output_path)
    print(f"Saved fine-tuned model to {output_path}")
else:
    print("Best weights not found!")
