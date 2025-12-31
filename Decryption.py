import os, base64, zlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
EXTRACTED_ENCRYPTED_TXT = os.path.join(PROJECT_DIR, "extracted_encrypted_message.txt")
DECRYPTED_MESSAGE_TXT   = os.path.join(PROJECT_DIR, "decrypted_message.txt")
PASS_FILE = os.path.join(PROJECT_DIR, "pass.txt")           
PASS_ENV = "STEGO_PASS"                                     

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

def decrypt_message(token_b64: str, passphrase: bytes) -> str:
    data = base64.b64decode(token_b64)
    if not data or data[0] != 3:
        raise ValueError("Unsupported envelope version or empty data.")

    N_log2 = data[1]
    r = data[2]
    p = data[3]
    salt  = data[4:20]
    nonce = data[20:32]
    tag   = data[32:48]
    ct    = data[48:]

    key = scrypt(passphrase, salt, key_len=32, N=1 << N_log2, r=r, p=p)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    pt_rs = cipher.decrypt_and_verify(ct, tag)

    decoded = rsc.decode(pt_rs)

    if isinstance(decoded, (bytes, bytearray)):
        pt_compressed = decoded
    else:
        pt_compressed = decoded[0]

    plaintext_bytes = zlib.decompress(pt_compressed)
    return plaintext_bytes.decode("utf-8")

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_text(s: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

if __name__ == "__main__":
    token = load_text(EXTRACTED_ENCRYPTED_TXT)
    pw = get_passphrase()
    msg = decrypt_message(token, pw)
    save_text(msg, DECRYPTED_MESSAGE_TXT)
    print(f"✅ Decrypted message saved to: {DECRYPTED_MESSAGE_TXT}")
