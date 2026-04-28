import subprocess
import os
import sys
import shutil

PROJECT_DIR = r"D:\Steganography"
STEGO_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")

os.chdir(PROJECT_DIR)

steps = [
    ("Encryption", "Encryption_image.py"),
    ("Extract Frames", "extract_frames.py"),
    ("Embed Message", "embed_msg_adaptive.py"),
    ("Extract Message", "extract_modified_frames_adaptive.py"),
    ("Decryption", "Decryption_image.py")
]

def run_script(name, script):
    print(f"\n🔹 Running step: {name}")
    try:
        result = subprocess.run([sys.executable, script], check=True)
        print(f"✅ {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script}: {e}")
        sys.exit(1)

def main():
    print("🚀 Starting Steganography Pipeline...\n")
    for name, script in steps:
        if script == "embed_msg_adaptive.py" and os.path.exists(STEGO_DIR):
            shutil.rmtree(STEGO_DIR)
        run_script(name, script)
    print("\n🎉 All steps completed! Check output in your Steganography folder.")

if __name__ == "__main__":
    main()
