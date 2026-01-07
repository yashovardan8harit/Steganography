import cv2
import os
import numpy as np

PROJECT_DIR = r"D:\Steganography"
STEGO_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")
OUTPUT_BIN = os.path.join(PROJECT_DIR, "extracted_payload.bin")

def stable_importance(frame):
    r = frame[:, :, 2].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    gray = 0.5 * r + 0.5 * g

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)

    return mag


frames = sorted([
    f for f in os.listdir(STEGO_FRAMES_DIR)
    if f.lower().endswith((".png", ".jpg"))
])

bit_buffer = []
payload_len = None
bits_needed = None

for fn in frames:
    frame = cv2.imread(os.path.join(STEGO_FRAMES_DIR, fn))
    h, w, _ = frame.shape

    imp = stable_importance(frame)

    ys, xs = np.indices((h, w))
    order = np.lexsort((xs.ravel(), ys.ravel(), -imp.ravel()))
    coords = np.column_stack(np.unravel_index(order, (h, w)))

    capacity = (h * w) // 2
    used = 0

    for r, c in coords:
        if used >= capacity:
            break

        bit_buffer.append(frame[r, c, 0] & 1)
        used += 1

        if payload_len is None and len(bit_buffer) == 32:
            header = np.packbits(bit_buffer).tobytes()
            payload_len = int.from_bytes(header, "big")
            bits_needed = (payload_len + 4) * 8

        if bits_needed is not None and len(bit_buffer) >= bits_needed:
            bit_buffer = bit_buffer[:bits_needed]
            data = np.packbits(bit_buffer).tobytes()
            payload = data[4:4 + payload_len]

            with open(OUTPUT_BIN, "wb") as f:
                f.write(payload)

            print("Adaptive payload extracted")
            print("Extracted bits:", len(bit_buffer))
            print("Expected bits:", bits_needed)
            exit(0)

raise RuntimeError("Adaptive extraction incomplete")
