import os
import zlib
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from reedsolo import RSCodec

PROJECT_DIR = r"D:\Steganography"
PAYLOAD_FILE = os.path.join(PROJECT_DIR, "extracted_payload.bin")
OUTPUT_IMAGE = os.path.join(PROJECT_DIR, "recovered_image.png")
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

# Read extracted payload
with open(PAYLOAD_FILE, "rb") as f:
    payload = f.read()

# Unpack header fields
salt         = payload[:SALT_LEN]
nonce        = payload[SALT_LEN : SALT_LEN + NONCE_LEN]
tag          = payload[SALT_LEN + NONCE_LEN : SALT_LEN + NONCE_LEN + TAG_LEN]
rs_ciphertext = payload[SALT_LEN + NONCE_LEN + TAG_LEN:]

# Reed-Solomon decode (corrects up to RS_PARITY/2 byte errors)
decoded = rsc.decode(rs_ciphertext)
if isinstance(decoded, tuple):
    decoded = decoded[0]

# Derive key and decrypt
key = scrypt(get_passphrase(), salt, KEY_LEN, N=1 << N_LOG2, r=R, p=P)
cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
compressed = cipher.decrypt_and_verify(decoded, tag)

# Decompress back to raw PNG bytes
image_bytes = zlib.decompress(compressed)
print("Recovered image bytes:", len(image_bytes))

# Decode PNG bytes into an image array and save
img_array = np.frombuffer(image_bytes, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    raise RuntimeError("Image decode failed — payload may be corrupted")

cv2.imwrite(OUTPUT_IMAGE, img)
print("Image decrypted and recovered successfully:", OUTPUT_IMAGE)