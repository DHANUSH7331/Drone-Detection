import sys
import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from ultralytics import YOLO

# === CLI ARGUMENT CHECK ===
if len(sys.argv) < 2:
    print("❌ Usage: py detect.py <image_or_video_path> OR 'webcam'")
    sys.exit(1)

media_path = sys.argv[1]
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# === LOAD YOLOv8 MODEL ===
print("[INFO] Loading YOLOv8 model...")
yolo_model = YOLO("runs/detect/train/weights/best.pt")

# === LOAD ESRGAN (TFHub Captain Model) ===
print("[INFO] Loading ESRGAN super-resolution model from TFHub...")
esrgan = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

def enhance_image_with_esrgan(image_bgr):
    """Enhance image using ESRGAN"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_tensor = tf.expand_dims(image_rgb, 0)  # [1, H, W, 3]

    # Run ESRGAN
    sr_tensor = esrgan(image_tensor)
    sr_image = tf.clip_by_value(sr_tensor, 0.0, 1.0).numpy()
    sr_image = (sr_image[0] * 255).astype(np.uint8)

    # Convert back to BGR
    sr_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    return sr_bgr

# === IMAGE DETECTION ===
def detect_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    print("[INFO] Enhancing image with ESRGAN...")
    enhanced_img = enhance_image_with_esrgan(img)

    enhanced_path = os.path.join(output_dir, 'enhanced_image.jpg')
    cv2.imwrite(enhanced_path, enhanced_img)
    print(f"✅ Saved enhanced image to: {enhanced_path}")

    print("[INFO] Running YOLOv8 on enhanced image...")
    results = yolo_model(enhanced_img, verbose=False)
    annotated = results[0].plot()

    output_path = os.path.join(output_dir, 'output_result.jpg')
    cv2.imwrite(output_path, annotated)
    print(f"✅ Saved detection result to: {output_path}")

# === VIDEO DETECTION ===
def detect_video(video_path):
    print(f"[INFO] Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))

    output_path = os.path.join(output_dir, 'output_result.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    print(f"✅ Video detection complete. Saved to: {output_path}")

# === WEBCAM DETECTION ===
def detect_webcam():
    print("[INFO] Accessing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access webcam.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        results = yolo_model(frame, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 - Press Q to exit", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === DETECTION HANDLER ===
ext = os.path.splitext(media_path)[1].lower()

if media_path.lower() == 'webcam':
    detect_webcam()
elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
    detect_image(media_path)
elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
    detect_video(media_path)
else:
    print("❌ Unsupported file format.")
