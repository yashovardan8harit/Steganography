import cv2
import os
import numpy as np

PROJECT_DIR = r"D:\Steganography"
STEGO_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")
OUTPUT_BIN = os.path.join(PROJECT_DIR, "extracted_payload.bin")

frames = sorted([
    f for f in os.listdir(STEGO_FRAMES_DIR)
    if f.lower().endswith(".png")
])

bit_buffer = []
payload_len = None
bits_needed = None

for fn in frames:
    frame = cv2.imread(os.path.join(STEGO_FRAMES_DIR, fn))
    h, w, _ = frame.shape
    capacity = (h * w) // 2
    count = 0

    for r in range(h):
        for c in range(w):
            if count >= capacity:
                break

            bit_buffer.append(frame[r, c, 0] & 1)
            count += 1

            if payload_len is None and len(bit_buffer) == 32:
                header_bytes = np.packbits(bit_buffer).tobytes()
                payload_len = int.from_bytes(header_bytes, "big")
                bits_needed = (payload_len + 4) * 8

            if bits_needed is not None and len(bit_buffer) >= bits_needed:
                bit_buffer = bit_buffer[:bits_needed]
                all_bytes = np.packbits(bit_buffer).tobytes()
                payload = all_bytes[4:4 + payload_len]

                with open(OUTPUT_BIN, "wb") as f:
                    f.write(payload)

                print("Payload extracted successfully")
                print("Extracted bits:", len(bit_buffer))
                print("Expected bits:", bits_needed)
                exit(0)

        if bits_needed is not None and len(bit_buffer) >= bits_needed:
            break

raise RuntimeError("Payload extraction incomplete")
