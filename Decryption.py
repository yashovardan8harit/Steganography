import os
import zlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
PAYLOAD_FILE = os.path.join(PROJECT_DIR, "extracted_payload.bin")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "decrypted_message.txt")
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

with open(PAYLOAD_FILE, "rb") as f:
    payload = f.read()

salt = payload[:SALT_LEN]
nonce = payload[SALT_LEN:SALT_LEN + NONCE_LEN]
tag = payload[SALT_LEN + NONCE_LEN:SALT_LEN + NONCE_LEN + TAG_LEN]
ciphertext = payload[SALT_LEN + NONCE_LEN + TAG_LEN:]

key = scrypt(get_passphrase(), salt, KEY_LEN, N=1 << N_LOG2, r=R, p=P)

cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
rs_encoded = cipher.decrypt_and_verify(ciphertext, tag)

decoded = rsc.decode(rs_encoded)
if isinstance(decoded, tuple):
    decoded = decoded[0]

plaintext = zlib.decompress(decoded)

with open(OUTPUT_FILE, "wb") as f:
    f.write(plaintext)

print("Message decrypted successfully")
