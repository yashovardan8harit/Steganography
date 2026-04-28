import os
import zlib
import cv2
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
IMAGE_FILE = os.path.join(PROJECT_DIR, "secret.png")
ENCRYPTED_BIN = os.path.join(PROJECT_DIR, "encrypted_payload.bin")
PASS_FILE = os.path.join(PROJECT_DIR, "pass.txt")

KEY_LEN = 32
SALT_LEN = 16
NONCE_LEN = 12
TAG_LEN = 16

N_LOG2 = 15
R = 8
P = 1

RS_PARITY = 32
rsc = RSCodec(RS_PARITY)

def get_passphrase():
    with open(PASS_FILE, "r", encoding="utf-8") as f:
        return f.read().strip().encode()

def encrypt():
    # Read and encode image to PNG bytes
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_FILE}")
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Image encoding failed")
    plaintext = buf.tobytes()
    print("Original image bytes:", len(plaintext))

    # Compress
    compressed = zlib.compress(plaintext)
    print("Compressed size:", len(compressed))

    # Derive key
    passphrase = get_passphrase()
    salt = get_random_bytes(SALT_LEN)
    key = scrypt(passphrase, salt, KEY_LEN, N=1 << N_LOG2, r=R, p=P)

    # Encrypt
    nonce = get_random_bytes(NONCE_LEN)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(compressed)

    # Reed-Solomon encode
    rs_ciphertext = rsc.encode(ciphertext)

    # Assemble payload: salt | nonce | tag | rs_ciphertext
    payload = salt + nonce + tag + rs_ciphertext
    payload = len(payload).to_bytes(4, "big") + payload

    with open(ENCRYPTED_BIN, "wb") as f:
        f.write(payload)

    print("Encrypted payload created")
    print("Encrypted payload size:", len(payload))

if __name__ == "__main__":
    encrypt()