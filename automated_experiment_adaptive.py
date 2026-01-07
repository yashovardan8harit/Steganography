import os
import json
import subprocess
import matplotlib.pyplot as plt

PROJECT_DIR = r"D:\Steganography"
MESSAGE_FILE = os.path.join(PROJECT_DIR, "message.txt")
ADAPTIVE_RESULTS_FILE = os.path.join(PROJECT_DIR, "adaptive_results.json")
NORMAL_RESULTS_FILE = os.path.join(PROJECT_DIR, "all_results.json")

base_message = "HELLO WORLD! This is a test message. "
multipliers = [10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]

adaptive_results = []

for m in multipliers:
    msg = base_message * m

    with open(MESSAGE_FILE, "w", encoding="utf-8") as f:
        f.write(msg)

    subprocess.run(["python", "Encryption.py"], check=True)
    subprocess.run(["python", "embed_msg_adaptive.py"], check=True)
    subprocess.run(["python", "evaluation.py"], check=True)

    with open(os.path.join(PROJECT_DIR, "last_metrics.json"), "r") as f:
        metrics = json.load(f)

    metrics["payload_size"] = len(msg)
    metrics["multiplier"] = m
    adaptive_results.append(metrics)

with open(ADAPTIVE_RESULTS_FILE, "w") as f:
    json.dump(adaptive_results, f, indent=2)

payload = [r["payload_size"] for r in adaptive_results]
psnr = [r["avg_psnr"] for r in adaptive_results]
ssim = [r["avg_ssim"] for r in adaptive_results]
mse = [r["avg_mse"] for r in adaptive_results]
entropy = [r["avg_entropy_diff"] for r in adaptive_results]

plt.figure()
plt.plot(payload, psnr)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average PSNR (dB)")
plt.title("Adaptive LSB: Payload Size vs PSNR")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, ssim)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average SSIM")
plt.title("Adaptive LSB: Payload Size vs SSIM")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, mse)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average MSE")
plt.title("Adaptive LSB: Payload Size vs MSE")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload, entropy)
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Entropy Difference")
plt.title("Adaptive LSB: Payload Size vs Entropy Difference")
plt.grid(True)
plt.show()

with open(NORMAL_RESULTS_FILE, "r") as f:
    normal_results = json.load(f)

payload_n = [r["payload_size"] for r in normal_results]
psnr_n = [r["avg_psnr"] for r in normal_results]
ssim_n = [r["avg_ssim"] for r in normal_results]

plt.figure()
plt.plot(payload_n, psnr_n, label="Normal LSB")
plt.plot(payload, psnr, label="Adaptive LSB")
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average PSNR (dB)")
plt.title("Normal vs Adaptive LSB: PSNR Comparison")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(payload_n, ssim_n, label="Normal LSB")
plt.plot(payload, ssim, label="Adaptive LSB")
plt.xlabel("Payload Size (bytes)")
plt.ylabel("Average SSIM")
plt.title("Normal vs Adaptive LSB: SSIM Comparison")
plt.legend()
plt.grid(True)
plt.show()
