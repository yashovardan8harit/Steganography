import os
import zlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
PLAINTEXT_FILE = os.path.join(PROJECT_DIR, "message.txt")
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

def load_plaintext():
    if os.path.exists(PLAINTEXT_FILE):
        with open(PLAINTEXT_FILE, "rb") as f:
            return f.read()
    return b"Hello World!"

def encrypt():
    plaintext = load_plaintext()
    compressed = zlib.compress(plaintext)

    passphrase = get_passphrase()
    salt = get_random_bytes(SALT_LEN)
    key = scrypt(passphrase, salt, KEY_LEN, N=1 << N_LOG2, r=R, p=P)

    nonce = get_random_bytes(NONCE_LEN)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(compressed)

    rs_ciphertext = rsc.encode(ciphertext)

    payload = salt + nonce + tag + rs_ciphertext
    payload = len(payload).to_bytes(4, "big") + payload

    with open(ENCRYPTED_BIN, "wb") as f:
        f.write(payload)

    print("Encrypted payload created")
    print("Encrypted payload size:", len(payload))

if __name__ == "__main__":
    encrypt()
