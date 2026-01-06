import cv2
import os

PROJECT_DIR = r"D:\Steganography"
VIDEO_PATH = os.path.join(PROJECT_DIR, "Bali.MOV")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    capacity = h * w
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.png"), frame)
    print(f"Frame {idx} capacity: {capacity} bits")
    idx += 1

cap.release()
