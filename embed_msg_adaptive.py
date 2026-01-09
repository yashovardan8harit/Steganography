import cv2
import os
import numpy as np

PROJECT_DIR = r"D:\Steganography"
FRAME_DIR = os.path.join(PROJECT_DIR, "output_frames")
STEGO_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")
PAYLOAD_FILE = os.path.join(PROJECT_DIR, "encrypted_payload.bin")
STEGO_VIDEO = os.path.join(PROJECT_DIR, "stego_video.mp4")

os.makedirs(STEGO_DIR, exist_ok=True)

def stable_importance(frame):
    r = frame[:, :, 2].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    gray = 0.5 * r + 0.5 * g
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    return mag

with open(PAYLOAD_FILE, "rb") as f:
    payload = f.read()

bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
bit_idx = 0
total_capacity = 0

frames = sorted(os.listdir(FRAME_DIR))
fps = 30
writer = None

for fn in frames:
    frame = cv2.imread(os.path.join(FRAME_DIR, fn))
    h, w, _ = frame.shape
    capacity = (h * w) // 2

    if bit_idx < len(bits):
        imp = stable_importance(frame)
        ys, xs = np.indices((h, w))
        order = np.lexsort((xs.ravel(), ys.ravel(), -imp.ravel()))
        coords = np.column_stack(np.unravel_index(order, (h, w)))

        remaining = len(bits) - bit_idx
        to_embed = min(capacity, remaining)

        for r, c in coords[:to_embed]:
            frame[r, c, 0] = (frame[r, c, 0] & 0xFE) | bits[bit_idx]
            bit_idx += 1

    out_path = os.path.join(STEGO_DIR, fn)
    cv2.imwrite(out_path, frame)
    total_capacity += capacity

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(STEGO_VIDEO, fourcc, fps, (w, h))

    writer.write(frame)

writer.release()

print("Embedding completed correctly")
print("Total embedding capacity:", total_capacity)
print("Bits embedded:", bit_idx)
print("Adaptive LSB embedding completed")
