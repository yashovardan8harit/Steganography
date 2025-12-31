import cv2
import os
import numpy as np

# =========================
# Project Configuration
# =========================
PROJECT_DIR = r"D:\Steganography"
STEGO_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")
ORIGINAL_MESSAGE_FILE = os.path.join(PROJECT_DIR, "encrypted_message.txt")
OUTPUT_EXTRACTED_FILE = os.path.join(PROJECT_DIR, "extracted_encrypted_message.txt")


# =========================
# Importance Map (MUST MATCH EMBEDDING)
# =========================
def importance_map(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mag_norm = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-9) * 255).astype(np.uint8)
    return mag_norm


# =========================
# Extract bits from ONE frame
# =========================
def extract_bytes_from_frame(frame, bits_needed):
    frame_array = np.array(frame)
    h, w, _ = frame_array.shape

    gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    imp = importance_map(gray)

    coords = np.dstack(
        np.unravel_index(np.argsort(-imp.ravel()), (h, w))
    )[0]

    bits_to_read = min(bits_needed, len(coords))
    extracted_bits = []

    for i in range(bits_to_read):
        r, c = coords[i]
        extracted_bits.append(str(frame_array[r, c, 0] & 1))

    full_bytes = len(extracted_bits) // 8
    extracted_bits = extracted_bits[:full_bytes * 8]

    byte_values = [
        int(''.join(extracted_bits[i:i + 8]), 2)
        for i in range(0, len(extracted_bits), 8)
    ]

    return bytes(byte_values)


# =========================
# Multi-frame extraction (BYTE-SAFE)
# =========================
def extract_message_from_video(frames, total_bytes):
    extracted = bytearray()
    bits_needed = total_bytes * 8

    for frame in frames:
        if len(extracted) * 8 >= bits_needed:
            break

        remaining_bits = bits_needed - len(extracted) * 8
        chunk = extract_bytes_from_frame(frame, remaining_bits)
        extracted.extend(chunk)

    return extracted[:total_bytes]


# =========================
# Main Pipeline
# =========================
def main():
    # Load stego frames
    frames = []
    for filename in sorted(os.listdir(STEGO_FRAMES_DIR)):
        if filename.lower().endswith(".png"):
            frame = cv2.imread(os.path.join(STEGO_FRAMES_DIR, filename))
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No stego frames found.")

    # Load original encrypted message (for exact length)
    with open(ORIGINAL_MESSAGE_FILE, "r", encoding="ascii") as f:
        original_token = f.read().strip()

    expected_length = len(original_token)
    print(f"🔹 Expected encrypted message length: {expected_length} bytes")

    # Extract message
    extracted_bytes = extract_message_from_video(frames, expected_length)

    # Decode ONLY ONCE (Base64 is ASCII)
    extracted_token = extracted_bytes.decode("ascii")

    # Save extracted message
    with open(OUTPUT_EXTRACTED_FILE, "w", encoding="ascii") as f:
        f.write(extracted_token)

    print(f"✅ Extracted encrypted message saved to: {OUTPUT_EXTRACTED_FILE}")


if __name__ == "__main__":
    main()
