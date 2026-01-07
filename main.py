import subprocess
import os
import sys

PROJECT_DIR = r"D:\Steganography"

os.chdir(PROJECT_DIR)

steps = [
    ("Encryption", "Encryption.py"),
    ("Extract Frames", "extract_frames.py"),
    ("Embed Message", "embed_msg_adaptive.py"),
    ("Extract Message", "extract_modified_frames_adaptive.py"),
    ("Decryption", "Decryption.py")
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
        run_script(name, script)
    print("\n🎉 All steps completed! Check output in your Steganography folder.")

if __name__ == "__main__":
    main()
