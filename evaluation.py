import os
import cv2
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy

PROJECT_DIR = r"D:\Steganography"
ORIGINAL_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames")
STEGO_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")


def calculate_frame_metrics(orig, stego):

    if orig.shape != stego.shape:
        stego = cv2.resize(stego, (orig.shape[1], orig.shape[0]))

    if np.array_equal(orig, stego):
        return None

    orig_f = orig.astype(np.float32)
    stego_f = stego.astype(np.float32)

    mse = np.mean((orig_f - stego_f) ** 2)

    if mse == 0:
        return None

    psnr = peak_signal_noise_ratio(orig, stego, data_range=255)

    ssim = structural_similarity(
        cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY),
        data_range=255
    )

    numerator = np.sum(orig_f * stego_f)
    denominator = np.sqrt(np.sum(orig_f ** 2) * np.sum(stego_f ** 2))
    ncc = numerator / denominator if denominator != 0 else 0

    entropy_orig = shannon_entropy(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY))
    entropy_stego = shannon_entropy(cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY))
    entropy_diff = abs(entropy_orig - entropy_stego)

    return mse, psnr, ssim, ncc, entropy_diff


def evaluate_video_quality():
    print("\n🎥 Evaluating Steganographic Distortion (Modified Frames Only)...")

    orig_files = sorted([f for f in os.listdir(ORIGINAL_FRAMES_DIR) if f.endswith(('.jpg', '.png'))])
    stego_files = sorted([f for f in os.listdir(STEGO_FRAMES_DIR) if f.endswith(('.jpg', '.png'))])

    if not orig_files or not stego_files:
        raise RuntimeError("Missing frames for evaluation.")

    mse_vals, psnr_vals, ssim_vals, ncc_vals, entropy_vals = [], [], [], [], []

    total = min(len(orig_files), len(stego_files))
    modified_count = 0

    for i in range(total):
        orig = cv2.imread(os.path.join(ORIGINAL_FRAMES_DIR, orig_files[i]))
        stego = cv2.imread(os.path.join(STEGO_FRAMES_DIR, stego_files[i]))

        if orig is None or stego is None:
            continue

        result = calculate_frame_metrics(orig, stego)
        if result is None:
            continue

        mse, psnr, ssim, ncc, entropy_diff = result

        mse_vals.append(mse)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        ncc_vals.append(ncc)
        entropy_vals.append(entropy_diff)

        modified_count += 1

    if modified_count == 0:
        raise RuntimeError("No modified frames detected — check embedding pipeline.")

    return {
        "modified_frames": modified_count,
        "avg_mse": np.mean(mse_vals),
        "avg_psnr": np.mean(psnr_vals),
        "avg_ssim": np.mean(ssim_vals),
        "avg_ncc": np.mean(ncc_vals),
        "avg_entropy_diff": np.mean(entropy_vals)
    }


def main():
    print("🚀 Starting Correct Steganographic Evaluation...\n")
    start = time.time()

    metrics = evaluate_video_quality()

    elapsed = time.time() - start

    print("\n📊 ===== STEGANOGRAPHIC QUALITY REPORT =====")
    print(f"🔹 Modified Frames:     {metrics['modified_frames']}")
    print(f"🔹 Avg MSE:              {metrics['avg_mse']:.6e}")
    print(f"🔹 Avg PSNR:             {metrics['avg_psnr']:.2f} dB")
    print(f"🔹 Avg SSIM:             {metrics['avg_ssim']:.6f}")
    print(f"🔹 Avg NCC:              {metrics['avg_ncc']:.6f}")
    print(f"🔹 Avg Entropy Diff:     {metrics['avg_entropy_diff']:.6e}")
    print("──────────────────────────────────────────")
    print(f"⚡ Evaluation Time:      {elapsed:.2f} seconds")
    print("==========================================\n")


if __name__ == "__main__":
    main()
