from ultralytics import YOLO

# === Load YOLOv8 Model ===
print("[INFO] Loading YOLOv8 model...")
model = YOLO('Weights/yolov8n.pt')  # Ensure the weights path is correct

# === Train Model ===
print("[INFO] Starting training...")
results = model.train(
    data='drone.yaml',
    epochs=5,
    imgsz=416,
    cache='Disk',
    batch=8
)

# === Validate & Extract Accuracy Metrics ===
metrics = model.val()

# === Show Accuracy Results ===
print("\n✅ Training Complete. Accuracy Metrics:")
print(f"📊 Precision:       {metrics.box.p:.4f}")
print(f"📊 Recall:          {metrics.box.r:.4f}")
print(f"📊 mAP@0.5:         {metrics.box.map50:.4f}")
print(f"📊 mAP@0.5:0.95:    {metrics.box.map:.4f}")
