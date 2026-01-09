import os
import json
import subprocess
import matplotlib.pyplot as plt
import shutil

PROJECT_DIR = r"D:\Steganography"
MESSAGE_FILE = os.path.join(PROJECT_DIR, "message.txt")
RESULTS_FILE = os.path.join(PROJECT_DIR, "all_results.json")
STEGO_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")

base_message = "HELLO WORLD! This is a test message. "
multipliers = [10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]

all_results = []

for m in multipliers:
    msg = base_message * m

    with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
        f.write(msg)

    subprocess.run(["python", "Encryption.py"], check=True)
    subprocess.run(["python", "embed_msg.py"], check=True)
    subprocess.run(["python", "evaluation.py"], check=True)

    with open(os.path.join(PROJECT_DIR, "last_metrics.json"), "r") as f:
        metrics = json.load(f)

    metrics["payload_size"] = len(msg)
    all_results.append(metrics)

    if os.path.exists(STEGO_DIR):
        shutil.rmtree(STEGO_DIR)

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

payload = [r["payload_size"] for r in all_results]
psnr = [r["avg_psnr"] for r in all_results]
ssim = [r["avg_ssim"] for r in all_results]
mse = [r["avg_mse"] for r in all_results]
entropy = [r["avg_entropy_diff"] for r in all_results]

plt.figure()
plt.plot(payload, psnr)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average PSNR (dB)")
plt.title("Payload Size vs PSNR")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, ssim)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average SSIM")
plt.title("Payload Size vs SSIM")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, mse)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average MSE")
plt.title("Payload Size vs MSE")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, entropy)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Entropy Difference")
plt.title("Payload Size vs Entropy Difference")
plt.grid(True)
plt.show()
