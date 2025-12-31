import os, base64, zlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
ENCRYPTED_MESSAGE_TXT = os.path.join(PROJECT_DIR, "encrypted_message.txt")
PLAINTEXT_FILE = os.path.join(PROJECT_DIR, "message.txt")
PASS_FILE = os.path.join(PROJECT_DIR, "pass.txt")           
PASS_ENV = "STEGO_PASS"                               

N_LOG2 = 15
R = 8
P = 1
RS_PARITY = 32
rsc = RSCodec(RS_PARITY)

def get_passphrase() -> bytes:

    pw = os.environ.get(PASS_ENV)
    if pw:
        return pw.encode("utf-8")

    if os.path.exists(PASS_FILE):
        with open(PASS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().encode("utf-8")

    return input("Enter passphrase (visible): ").encode("utf-8")

def load_plaintext() -> str:
    if os.path.exists(PLAINTEXT_FILE):
        with open(PLAINTEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "This is a test message hidden in the video."*100000

def encrypt_message(plaintext: str, passphrase: bytes) -> str:

    salt = get_random_bytes(16)
    key = scrypt(passphrase, salt, key_len=32, N=1 << N_LOG2, r=R, p=P)

    pt = zlib.compress(plaintext.encode("utf-8"))

    pt_rs = rsc.encode(pt)

    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(pt_rs)

    blob = b"\x03" + bytes([N_LOG2, R, P]) + salt + nonce + tag + ct

    return base64.b64encode(blob).decode("utf-8")

def save_text(s: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

if __name__ == "__main__":
    plaintext = load_plaintext()
    pw = get_passphrase()
    token = encrypt_message(plaintext, pw)
    save_text(token, ENCRYPTED_MESSAGE_TXT)
    print(f"✅ Encrypted message saved to: {ENCRYPTED_MESSAGE_TXT}")
